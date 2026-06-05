import copy
import datetime
import json
import math
import os
import pickle
import time
from pathlib import Path

import gym
import numpy as np
import pandas as pd
from loguru import logger

import src.algorithms.ELP_DRL_MO4 as mo4_module
import src.utils.FBSUtil as FBSUtil
import src.utils.config as config
from src.algorithms.BiMO4PaperLocalSearch import BiMO4PaperLocalSearch
from src.algorithms.ELP_DRL_MO4 import ELP as MO4ELP
from src.algorithms.ELP_DRL_MO4 import MOEAD
from src.algorithms.ELP_DRL_MO4 import NSGA2
from src.algorithms.ELP_DRL_MO4 import SPEA2
from src.algorithms.ELP_DRL_MO4 import SPEA2Survival
from src.algorithms.ELP_DRL_MO4 import Problem
from src.algorithms.ELP_DRL_MO4 import StandardDQNAgent
from src.algorithms.ELP_DRL_MO4 import StandardQLearningAgent
from src.algorithms.ELP_DRL_MO4 import _ActionSequenceMutation
from src.algorithms.ELP_DRL_MO4 import _ActionSequenceSampling
from src.algorithms.ELP_DRL_MO4 import _ActionSequenceUniformCrossover
from src.algorithms.ELP_DRL_MO4 import _PYMOO_IMPORT_ERROR
from src.algorithms.ELP_DRL_MO4 import _get_initial_solution_energy
from src.algorithms.ELP_DRL_MO4 import _save_experiment_row
from src.algorithms.ELP_DRL_MO4 import _set_global_seed
from src.algorithms.ELP_DRL_MO4 import get_reference_directions
from src.algorithms.ELP_DRL_MO4 import get_termination
from src.algorithms.ELP_DRL_MO4 import minimize
from src.utils.CR_MatrixStore import CRMatrixStore
from src.utils.MO_FBSUtil_BiMO4 import MO_FBSUtil_BiMO4

np.bool8 = np.bool_


class _ActionSequenceBiMOProblem(Problem):
    def __init__(self, solver, base_solution, sequence_length, use_constraints=True):
        self.solver = solver
        self.base_solution = solver._light_clone_solution(base_solution)
        self.action_count = int(len(solver.valid_actions))
        self.use_constraints = bool(use_constraints)
        super().__init__(
            n_var=int(max(1, sequence_length)),
            n_obj=2,
            n_ieq_constr=2 if self.use_constraints else 0,
            xl=np.zeros(int(max(1, sequence_length)), dtype=int),
            xu=np.full(int(max(1, sequence_length)), self.action_count - 1, dtype=int),
            vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        sequences = np.asarray(X, dtype=int)
        sample_count = int(sequences.shape[0])
        objectives = np.zeros((sample_count, 2), dtype=float)
        constraints = np.zeros((sample_count, 2), dtype=float)
        for idx in range(sample_count):
            candidate = self.solver._evaluate_action_sequence(self.base_solution, sequences[idx])
            objective_vector = np.asarray(candidate.mo_objectives_min, dtype=float)[:2]
            is_feasible = bool(getattr(candidate, "current_is_feasible", False))
            d_inf = int(getattr(candidate, "current_d_inf", 0) or 0)
            violation = max(float(getattr(candidate, "constraint_violation", 0.0) or 0.0), 0.0)
            if self.use_constraints:
                objectives[idx, :] = objective_vector
                constraints[idx, 0] = 0.0 if is_feasible else 1.0
                constraints[idx, 1] = violation
            else:
                penalty = 0.0 if is_feasible else (1_000_000.0 + 10_000.0 * max(d_inf, 0) + violation)
                objectives[idx, :] = objective_vector + float(penalty)
        out["F"] = objectives
        if self.use_constraints:
            out["G"] = constraints


class ELP(MO4ELP):
    OBJECTIVE_DEFINITION_VERSION = MO_FBSUtil_BiMO4.OBJECTIVE_DEFINITION_VERSION
    CR_BOUNDARY_REPARTITION_ACTION_ID = 17
    CR_BOUNDARY_REPARTITION_OPERATIONS = ("align_top", "align_bottom", "split", "block_swap")

    def __init__(self, env, gbest, T, G=100, t_max=50, k=2.0, archive_limit=64, objective_weights=None):
        weights = np.asarray(objective_weights if objective_weights is not None else [0.5, 0.5], dtype=float).reshape(-1)
        if weights.size < 2:
            weights = np.pad(weights, (0, 2 - weights.size), constant_values=0.5)
        weights = np.clip(weights[:2], 0.0, None)
        if not np.any(weights > 0):
            weights = np.asarray([0.5, 0.5], dtype=float)
        weights = weights / np.sum(weights)
        self.bimo_dqn_context_enabled = _parse_env_flag("ELP_BIMO_DQN_CONTEXT_ENABLE", True)
        self.rl_context_dim = 10 if self.bimo_dqn_context_enabled else 0
        self.mo_adaptive_weights_enabled = self._parse_env_flag("ELP_BIMO_ADAPTIVE_WEIGHTS_ENABLE", True)
        self.mo_adaptive_weight_blend = float(
            min(max(_parse_env_float("ELP_BIMO_ADAPTIVE_WEIGHT_BLEND", 0.15), 0.0), 1.0)
        )
        legacy_floor = float(_parse_env_float("ELP_BIMO_ADAPTIVE_WEIGHT_FLOOR", 0.20))
        self.mo_adaptive_weight_min_component = float(
            min(max(_parse_env_float("ELP_BIMO_ADAPTIVE_WEIGHT_MIN_COMPONENT", legacy_floor), 1e-6), 0.49)
        )
        self.mo_adaptive_weight_floor = self.mo_adaptive_weight_min_component
        self.mo_adaptive_weight_refresh_interval_steps = int(
            max(1, _parse_env_int("ELP_BIMO_ADAPTIVE_WEIGHT_REFRESH_INTERVAL_STEPS", 250))
        )
        self.mo_adaptive_weight_deadband = float(max(_parse_env_float("ELP_BIMO_ADAPTIVE_WEIGHT_DEADBAND", 0.08), 0.0))
        self.mo_base_weights = self._normalize_bi_weights(weights.copy(), floor_value=self.mo_adaptive_weight_min_component)
        self.mo_weights = self.mo_base_weights.copy()
        self.mo_weight_update_count = 0
        self.mo_last_weight_target = self.mo_base_weights.copy()
        self.mo_last_weight_update_step = -10**9
        self.mo_running_min = np.asarray([math.inf, math.inf], dtype=float)
        self.mo_running_max = np.asarray([-math.inf, -math.inf], dtype=float)
        self.bimo_archive_bootstrap_enabled = _parse_env_flag("ELP_BIMO_ARCHIVE_BOOTSTRAP_ENABLE", True)
        self.bimo_archive_bootstrap_size = int(max(1, _parse_env_int("ELP_BIMO_ARCHIVE_BOOTSTRAP_SIZE", 32)))
        self.bimo_archive_bootstrap_attempt_factor = int(
            max(1, _parse_env_int("ELP_BIMO_ARCHIVE_BOOTSTRAP_ATTEMPT_FACTOR", 8))
        )
        self.bimo_archive_paperls_enabled = _parse_env_flag("ELP_BIMO_ARCHIVE_PAPERLS_ENABLE", False)
        self.bimo_archive_paperls_anchor_count = int(
            max(1, _parse_env_int("ELP_BIMO_ARCHIVE_PAPERLS_ANCHORS", 8))
        )
        self.bimo_archive_paperls_passes = int(max(1, _parse_env_int("ELP_BIMO_ARCHIVE_PAPERLS_PASSES", 2)))
        self.bimo_archive_paperls_time_limit_seconds = float(
            max(0.0, _parse_env_float("ELP_BIMO_ARCHIVE_PAPERLS_TIME_LIMIT_SECONDS", 0.0))
        )
        self.bimo_archive_paperls_reserve_seconds = float(
            max(0.0, _parse_env_float("ELP_BIMO_ARCHIVE_PAPERLS_RESERVE_SECONDS", 0.0))
        )
        self.bimo_archive_paperls_max_neighbor_evaluations = int(
            max(0, _parse_env_int("ELP_BIMO_ARCHIVE_PAPERLS_MAX_NEIGHBORS", 0))
        )
        self._bimo_archive_paperls_done = False
        self.bimo_archive_anchor_selection_enabled = _parse_env_flag("ELP_BIMO_ARCHIVE_ANCHOR_ENABLE", True)
        self.bimo_archive_anchor_min_size = int(max(1, _parse_env_int("ELP_BIMO_ARCHIVE_ANCHOR_MIN_SIZE", 2)))
        self.bimo_archive_anchor_switch_interval = int(
            max(1, _parse_env_int("ELP_BIMO_ARCHIVE_ANCHOR_SWITCH_INTERVAL_EPISODES", 1))
        )
        self.bimo_anchor_quality_weight = float(max(0.0, _parse_env_float("ELP_BIMO_ANCHOR_QUALITY_WEIGHT", 0.25)))
        self.bimo_anchor_sparse_weight = float(max(0.0, _parse_env_float("ELP_BIMO_ANCHOR_SPARSE_WEIGHT", 0.35)))
        self.bimo_anchor_extreme_weight = float(max(0.0, _parse_env_float("ELP_BIMO_ANCHOR_EXTREME_WEIGHT", 0.25)))
        self.bimo_anchor_stale_weight = float(max(0.0, _parse_env_float("ELP_BIMO_ANCHOR_STALE_WEIGHT", 0.15)))
        self.bimo_anchor_min_probability_weight = float(
            max(0.0, _parse_env_float("ELP_BIMO_ANCHOR_MIN_PROBABILITY_WEIGHT", 1e-6))
        )
        # ── 候选池配置 ──
        self.bimo_candidate_pool_enabled = _parse_env_flag("ELP_BIMO_CANDIDATE_POOL_ENABLE", True)
        self.bimo_candidate_pool_limit = int(max(1, _parse_env_int("ELP_BIMO_CANDIDATE_POOL_LIMIT", 96)))
        self.bimo_candidate_pool_bootstrap_target = int(max(1, _parse_env_int(
            "ELP_BIMO_CANDIDATE_POOL_BOOTSTRAP_TARGET",
            self.bimo_archive_bootstrap_size,
        )))
        self.bimo_candidate_pool_min_anchor_size = int(max(1, _parse_env_int("ELP_BIMO_CANDIDATE_POOL_MIN_ANCHORS", 8)))
        self.bimo_candidate_pool_runtime_max_fraction = float(
            min(max(_parse_env_float("ELP_BIMO_CANDIDATE_POOL_RUNTIME_MAX_FRACTION", 0.30), 0.0), 1.0)
        )
        self.bimo_candidate_pool_mhc_quota_fraction = float(
            min(max(_parse_env_float("ELP_BIMO_CANDIDATE_POOL_MHC_QUOTA_FRACTION", 0.30), 0.0), 1.0)
        )
        self.bimo_candidate_pool_cr_quota_fraction = float(
            min(max(_parse_env_float("ELP_BIMO_CANDIDATE_POOL_CR_QUOTA_FRACTION", 0.15), 0.0), 1.0)
        )
        self.bimo_candidate_pool_balanced_quota_fraction = float(
            min(max(_parse_env_float("ELP_BIMO_CANDIDATE_POOL_BALANCED_QUOTA_FRACTION", 0.35), 0.0), 1.0)
        )
        self.bimo_candidate_pool = []
        self._bimo_candidate_pool_keys = set()
        self._bimo_candidate_pool_visit_counts = {}
        self._bimo_candidate_pool_insert_count = 0
        self._bimo_candidate_pool_duplicate_count = 0
        # ── 奖励配置 ──
        self.bimo_reward_archive_bonus = float(_parse_env_float("ELP_BIMO_REWARD_ARCHIVE_BONUS", 0.45))
        self.bimo_reward_unaccepted_archive_bonus = float(
            _parse_env_float("ELP_BIMO_REWARD_UNACCEPTED_ARCHIVE_BONUS", 0.25)
        )
        self.bimo_reward_cr_gain_weight = float(_parse_env_float("ELP_BIMO_REWARD_CR_GAIN_WEIGHT", 0.40))
        self.bimo_reward_cr_loss_weight = float(_parse_env_float("ELP_BIMO_REWARD_CR_LOSS_WEIGHT", 0.08))
        self.bimo_reward_extreme_gain_weight = float(_parse_env_float("ELP_BIMO_REWARD_EXTREME_GAIN_WEIGHT", 0.35))
        self.bimo_reward_sparse_gain_weight = float(_parse_env_float("ELP_BIMO_REWARD_SPARSE_GAIN_WEIGHT", 0.25))
        self.bimo_reward_hv_proxy_weight = float(_parse_env_float("ELP_BIMO_REWARD_HV_PROXY_WEIGHT", 0.50))
        self.bimo_reward_clip = float(max(1.0, _parse_env_float("ELP_BIMO_REWARD_CLIP", 4.0)))
        self.bimo_cr_boundary_repartition_enabled = _parse_env_flag("ELP_BIMO_CR_BOUNDARY_REPARTITION_ENABLE", True)
        self.bimo_cr_boundary_repartition_budget = int(
            max(1, _parse_env_int("ELP_BIMO_CR_BOUNDARY_REPARTITION_BUDGET", 16))
        )
        self.bimo_cr_boundary_top_bay_pairs = int(
            max(1, _parse_env_int("ELP_BIMO_CR_BOUNDARY_TOP_BAY_PAIRS", 4))
        )
        self.bimo_cr_boundary_block_limit = int(max(1, _parse_env_int("ELP_BIMO_CR_BOUNDARY_BLOCK_LIMIT", 3)))
        self.bimo_cr_boundary_enabled_operations = set(self._parse_cr_boundary_operation_filter())
        self.bimo_cr_boundary_adaptive_operation_enabled = _parse_env_flag(
            "ELP_BIMO_CR_BOUNDARY_ADAPTIVE_OPERATION_ENABLE", False
        )
        self.bimo_cr_boundary_adaptive_min_selected = int(
            max(1, _parse_env_int("ELP_BIMO_CR_BOUNDARY_ADAPTIVE_MIN_SELECTED", 64))
        )
        self.bimo_cr_boundary_adaptive_min_weight = float(
            min(max(_parse_env_float("ELP_BIMO_CR_BOUNDARY_ADAPTIVE_MIN_WEIGHT", 0.15), 0.01), 1.0)
        )
        self.bimo_cr_boundary_adaptive_max_weight = float(
            max(
                self.bimo_cr_boundary_adaptive_min_weight,
                _parse_env_float("ELP_BIMO_CR_BOUNDARY_ADAPTIVE_MAX_WEIGHT", 2.50),
            )
        )
        self.bimo_cr_boundary_adaptive_strength = float(
            max(0.0, _parse_env_float("ELP_BIMO_CR_BOUNDARY_ADAPTIVE_STRENGTH", 1.0))
        )
        self.bimo_cr_boundary_adaptive_loss_weight = float(
            max(0.0, _parse_env_float("ELP_BIMO_CR_BOUNDARY_ADAPTIVE_LOSS_WEIGHT", 1.0))
        )
        self.bimo_cr_boundary_adaptive_accept_weight = float(
            max(0.0, _parse_env_float("ELP_BIMO_CR_BOUNDARY_ADAPTIVE_ACCEPT_WEIGHT", 0.25))
        )
        self.bimo_cr_boundary_elite_enabled = _parse_env_flag("ELP_BIMO_CR_BOUNDARY_REPARTITION_ELITE_ENABLE", True)
        self.bimo_cr_boundary_elite_trials = int(
            max(1, _parse_env_int("ELP_BIMO_CR_BOUNDARY_REPARTITION_ELITE_TRIALS", 2))
        )
        self.bimo_cr_boundary_explore_weight = float(
            max(0.0, _parse_env_float("ELP_BIMO_CR_BOUNDARY_REPARTITION_EXPLORE_WEIGHT", 1.20))
        )
        self._bimo_anchor_visit_counts = {}
        self._bimo_last_anchor_episode = -10**9
        self._bimo_last_anchor_key = None
        self._bimo_archive_bootstrap_done = False
        super().__init__(
            env=env,
            gbest=gbest,
            T=T,
            G=G,
            t_max=t_max,
            k=k,
            archive_limit=archive_limit,
            objective_weights=weights,
        )
        facility_count = int(getattr(self.env, "n", len(getattr(self.env, "areas", [])) or 0))
        self._ensure_cr_matrix_available(self.instance_name, facility_count)
        self.rel_matrix, self.cr_matrix_payload, self.cr_matrix_path = CRMatrixStore.load_matrix(
            instance_name=self.instance_name,
            expected_facility_count=facility_count,
        )
        self.dist_req_matrix = None
        # 双目标版本不复用旧的 4 目标参考前沿定义，避免指标口径错配。
        self.archive_reference_front_enabled = False
        self.archive_reference_front_payload = None
        self.archive_reference_vectors = []
        self.archive_update_count = 0
        self._reset_running_objective_bounds()
        self._reset_bimo_cr_boundary_operation_telemetry()
        self._register_bimo_cr_boundary_repartition_action()

    def _run_impl(self):
        previous_algorithm = os.getenv("ELP_EXP_ALGORITHM")
        previous_definition_version = mo4_module.MO_ReferenceFrontUtil.OBJECTIVE_DEFINITION_VERSION
        previous_mo_util = mo4_module.MO_FBSUtil
        restore_wall_time_limit = float(getattr(self, "wall_time_limit_seconds", 0.0) or 0.0)
        if not previous_algorithm:
            os.environ["ELP_EXP_ALGORITHM"] = "ELP_DRL_BiMO4"
        mo4_module.MO_ReferenceFrontUtil.OBJECTIVE_DEFINITION_VERSION = self.OBJECTIVE_DEFINITION_VERSION
        mo4_module.MO_FBSUtil = MO_FBSUtil_BiMO4
        try:
            original_wall_time_limit = restore_wall_time_limit
            main_wall_time_limit = original_wall_time_limit
            reserve_seconds = float(getattr(self, "bimo_archive_paperls_reserve_seconds", 0.0) or 0.0)
            if bool(getattr(self, "bimo_archive_paperls_enabled", False)) and original_wall_time_limit > 0.0 and reserve_seconds > 0.0:
                if reserve_seconds < original_wall_time_limit:
                    main_wall_time_limit = max(1.0, original_wall_time_limit - reserve_seconds)
                else:
                    main_wall_time_limit = max(1.0, original_wall_time_limit * 0.5)
                self.wall_time_limit_seconds = main_wall_time_limit
            result = super()._run_impl()
            self.wall_time_limit_seconds = original_wall_time_limit
            result = self._apply_bimo_archive_paperls_after_run(
                result,
                total_wall_time_limit=original_wall_time_limit,
                main_wall_time_limit=main_wall_time_limit,
            )
            self._finalize_archive_update_reporting()
            report_snapshot = self._report_representative_snapshot()
            if report_snapshot and result is not None and len(result) >= 7:
                patched_result = list(result)
                patched_result[1] = bool(getattr(report_snapshot["solution"], "current_is_feasible", False))
                patched_result[2] = copy.deepcopy(report_snapshot["solution"])
                if report_snapshot["score"] is not None:
                    patched_result[3] = float(report_snapshot["score"])
                return tuple(patched_result)
            return result
        finally:
            mo4_module.MO_FBSUtil = previous_mo_util
            mo4_module.MO_ReferenceFrontUtil.OBJECTIVE_DEFINITION_VERSION = previous_definition_version
            self.wall_time_limit_seconds = restore_wall_time_limit
            if previous_algorithm is None:
                os.environ.pop("ELP_EXP_ALGORITHM", None)

    def _reset_mo_logging_state(self):
        super()._reset_mo_logging_state()
        self.archive_update_count = 0
        self._reset_running_objective_bounds()
        base_weights = np.asarray(getattr(self, "mo_base_weights", [0.5, 0.5]), dtype=float)
        floor_value = float(getattr(self, "mo_adaptive_weight_min_component", 0.20) or 0.20)
        self.mo_base_weights = self._normalize_bi_weights(base_weights, floor_value=floor_value)
        self.mo_weights = self.mo_base_weights.copy()
        self.mo_last_weight_target = self.mo_base_weights.copy()
        self.mo_weight_update_count = 0
        self.mo_last_weight_update_step = -10**9
        self._bimo_archive_paperls_done = False
        self._reset_bimo_archive_anchor_state()

    def _reset_baseline_archive_state(self):
        super()._reset_baseline_archive_state()
        self._reset_running_objective_bounds()
        base_weights = np.asarray(getattr(self, "mo_base_weights", [0.5, 0.5]), dtype=float)
        floor_value = float(getattr(self, "mo_adaptive_weight_min_component", 0.20) or 0.20)
        self.mo_base_weights = self._normalize_bi_weights(base_weights, floor_value=floor_value)
        self.mo_weights = self.mo_base_weights.copy()
        self.mo_last_weight_target = self.mo_base_weights.copy()
        self.mo_weight_update_count = 0
        self.mo_last_weight_update_step = -10**9
        self._reset_bimo_archive_anchor_state(reset_bootstrap=False)
        self._reset_bimo_candidate_pool()

    def _reset_bimo_candidate_pool(self):
        self.bimo_candidate_pool = []
        self._bimo_candidate_pool_keys = set()
        self._bimo_candidate_pool_visit_counts = {}
        self._bimo_candidate_pool_insert_count = 0
        self._bimo_candidate_pool_duplicate_count = 0

    def _ensure_bimo_candidate_pool_state(self):
        if not hasattr(self, "bimo_candidate_pool"):
            self.bimo_candidate_pool = []
        if not hasattr(self, "_bimo_candidate_pool_keys"):
            self._bimo_candidate_pool_keys = {
                self._bimo_solution_key(candidate)
                for candidate in (getattr(self, "bimo_candidate_pool", []) or [])
            }
        if not hasattr(self, "_bimo_candidate_pool_visit_counts"):
            self._bimo_candidate_pool_visit_counts = {}
        if not hasattr(self, "_bimo_candidate_pool_insert_count"):
            self._bimo_candidate_pool_insert_count = 0
        if not hasattr(self, "_bimo_candidate_pool_duplicate_count"):
            self._bimo_candidate_pool_duplicate_count = 0

    def _reset_running_objective_bounds(self):
        self.mo_running_min = np.asarray([math.inf, math.inf], dtype=float)
        self.mo_running_max = np.asarray([-math.inf, -math.inf], dtype=float)

    @classmethod
    def _parse_cr_boundary_operation_filter(cls):
        env_name = "ELP_BIMO_CR_BOUNDARY_REPARTITION_OPERATIONS"
        raw_value = os.getenv(env_name)
        if raw_value is None:
            return tuple(cls.CR_BOUNDARY_REPARTITION_OPERATIONS)
        operations = []
        seen = set()
        allowed = set(cls.CR_BOUNDARY_REPARTITION_OPERATIONS)
        for token in str(raw_value).split(","):
            operation = token.strip().lower()
            if not operation:
                continue
            if operation not in allowed:
                allowed_text = ", ".join(cls.CR_BOUNDARY_REPARTITION_OPERATIONS)
                raise ValueError(
                    f"环境变量 {env_name} 包含非法 Action17 子操作: {operation!r}，可选值: {allowed_text}"
                )
            if operation not in seen:
                operations.append(operation)
                seen.add(operation)
        if not operations:
            raise ValueError(f"环境变量 {env_name} 至少要包含一个 Action17 子操作。")
        return tuple(operations)

    def _reset_bimo_archive_anchor_state(self, reset_bootstrap=True):
        self._bimo_anchor_visit_counts = {}
        self._bimo_last_anchor_episode = -10**9
        self._bimo_last_anchor_key = None
        if bool(reset_bootstrap):
            self._bimo_archive_bootstrap_done = False

    def _register_bimo_cr_boundary_repartition_action(self):
        if not bool(getattr(self, "bimo_cr_boundary_repartition_enabled", True)):
            return
        if str(os.getenv("ELP_MO_BASELINE_ALGO", "") or "").strip():
            return

        action_id = int(self.CR_BOUNDARY_REPARTITION_ACTION_ID)
        self.action_labels[action_id] = "cr_boundary_repartition"
        if action_id not in self.valid_actions:
            self.valid_actions.append(action_id)

        if action_id not in self.action_telemetry:
            self.action_telemetry[action_id] = {
                "name": self.action_labels[action_id],
                "selected": 0,
                "accepted": 0,
                "improved": 0,
                "global_best_hits": 0,
                "delta_sum": 0.0,
                "accepted_delta_sum": 0.0,
                "elite_selected": 0,
                "elite_accepted": 0,
                "elite_improved": 0,
                "elite_global_best_hits": 0,
                "elite_delta_sum": 0.0,
                "elite_accepted_delta_sum": 0.0,
            }
        else:
            self.action_telemetry[action_id]["name"] = self.action_labels[action_id]

        if hasattr(self, "action_base_explore_weights"):
            self.action_base_explore_weights[action_id] = float(
                getattr(self, "bimo_cr_boundary_explore_weight", 1.20)
            )
        for counter_name in ("action_recent_selected", "action_recent_accepted", "action_recent_gbest"):
            counter = getattr(self, counter_name, None)
            if isinstance(counter, dict):
                counter.setdefault(action_id, 0)

        if bool(getattr(self, "bimo_cr_boundary_elite_enabled", True)):
            if action_id not in self.elite_actions:
                self.elite_actions.insert(0, action_id)
            self.elite_action_trials[action_id] = int(
                max(1, getattr(self, "bimo_cr_boundary_elite_trials", 2) or 2)
            )

    def _bimo_cr_boundary_operation_stats_template(self):
        stats = {"generated": 0}
        for prefix in ("", "elite_"):
            stats.update(
                {
                    f"{prefix}selected": 0,
                    f"{prefix}accepted": 0,
                    f"{prefix}improved": 0,
                    f"{prefix}global_best_hits": 0,
                    f"{prefix}archive_would_change": 0,
                    f"{prefix}archive_changed": 0,
                    f"{prefix}relation_score_sum": 0.0,
                    f"{prefix}cr_gain_sum": 0.0,
                    f"{prefix}cr_gain_norm_sum": 0.0,
                    f"{prefix}mhc_loss_sum": 0.0,
                    f"{prefix}mhc_loss_norm_sum": 0.0,
                    f"{prefix}hv_gain_proxy_sum": 0.0,
                    f"{prefix}extreme_gain_sum": 0.0,
                    f"{prefix}sparse_distance_sum": 0.0,
                }
            )
        return stats

    def _reset_bimo_cr_boundary_operation_telemetry(self):
        self.bimo_cr_boundary_operation_telemetry = {
            operation: self._bimo_cr_boundary_operation_stats_template()
            for operation in (*self.CR_BOUNDARY_REPARTITION_OPERATIONS, "unknown")
        }
        self._bimo_cr_boundary_selected_records = []
        self._bimo_last_cr_boundary_selected_by_phase = {}
        self._bimo_last_cr_boundary_accepted_by_phase = {}
        self._last_generated_cr_boundary_info = None

    def _ensure_bimo_cr_boundary_operation_telemetry(self):
        telemetry = getattr(self, "bimo_cr_boundary_operation_telemetry", None)
        if not isinstance(telemetry, dict):
            self._reset_bimo_cr_boundary_operation_telemetry()
            telemetry = self.bimo_cr_boundary_operation_telemetry
        for operation in (*self.CR_BOUNDARY_REPARTITION_OPERATIONS, "unknown"):
            telemetry.setdefault(operation, self._bimo_cr_boundary_operation_stats_template())
        if not isinstance(getattr(self, "_bimo_cr_boundary_selected_records", None), list):
            self._bimo_cr_boundary_selected_records = []
        if not isinstance(getattr(self, "_bimo_last_cr_boundary_selected_by_phase", None), dict):
            self._bimo_last_cr_boundary_selected_by_phase = {}
        if not isinstance(getattr(self, "_bimo_last_cr_boundary_accepted_by_phase", None), dict):
            self._bimo_last_cr_boundary_accepted_by_phase = {}
        return telemetry

    def _bimo_cr_boundary_normalize_operation(self, operation):
        operation = str(operation or "").strip()
        if operation in self.CR_BOUNDARY_REPARTITION_OPERATIONS:
            return operation
        return "unknown"

    def _bimo_cr_boundary_operation_stats(self, operation):
        telemetry = self._ensure_bimo_cr_boundary_operation_telemetry()
        operation = self._bimo_cr_boundary_normalize_operation(operation)
        return operation, telemetry[operation]

    def _bimo_cr_boundary_enabled_operation_list(self):
        enabled_operations = set(
            getattr(self, "bimo_cr_boundary_enabled_operations", set(self.CR_BOUNDARY_REPARTITION_OPERATIONS))
            or set()
        )
        return tuple(
            operation
            for operation in self.CR_BOUNDARY_REPARTITION_OPERATIONS
            if operation in enabled_operations
        )

    def _bimo_cr_boundary_operation_activity(self, operation):
        operation, stats = self._bimo_cr_boundary_operation_stats(operation)
        selected = int(stats.get("selected", 0) or 0) + int(stats.get("elite_selected", 0) or 0)
        accepted = int(stats.get("accepted", 0) or 0) + int(stats.get("elite_accepted", 0) or 0)
        archive_changed = int(stats.get("archive_changed", 0) or 0) + int(
            stats.get("elite_archive_changed", 0) or 0
        )
        global_best_hits = int(stats.get("global_best_hits", 0) or 0) + int(
            stats.get("elite_global_best_hits", 0) or 0
        )
        cr_gain_norm_sum = float(stats.get("cr_gain_norm_sum", 0.0) or 0.0) + float(
            stats.get("elite_cr_gain_norm_sum", 0.0) or 0.0
        )
        mhc_loss_norm_sum = float(stats.get("mhc_loss_norm_sum", 0.0) or 0.0) + float(
            stats.get("elite_mhc_loss_norm_sum", 0.0) or 0.0
        )
        selected_denominator = max(selected, 1)
        return {
            "operation": operation,
            "selected": selected,
            "accepted": accepted,
            "archive_changed": archive_changed,
            "global_best_hits": global_best_hits,
            "cr_gain_norm_sum": cr_gain_norm_sum,
            "mhc_loss_norm_sum": mhc_loss_norm_sum,
            "avg_cr_gain_norm": cr_gain_norm_sum / selected_denominator,
            "avg_mhc_loss_norm": mhc_loss_norm_sum / selected_denominator,
        }

    def _bimo_cr_boundary_operation_productivity(self, activity):
        accept_weight = float(getattr(self, "bimo_cr_boundary_adaptive_accept_weight", 0.25) or 0.0)
        return (
            float(activity.get("archive_changed", 0) or 0)
            + float(activity.get("global_best_hits", 0) or 0)
            + accept_weight * float(activity.get("accepted", 0) or 0)
        )

    def _bimo_cr_boundary_operation_adaptive_weight(self, operation):
        if not bool(getattr(self, "bimo_cr_boundary_adaptive_operation_enabled", False)):
            return 1.0
        operation = self._bimo_cr_boundary_normalize_operation(operation)
        enabled_operations = self._bimo_cr_boundary_enabled_operation_list()
        if operation not in enabled_operations:
            return 0.0

        activity = self._bimo_cr_boundary_operation_activity(operation)
        min_selected = int(max(1, getattr(self, "bimo_cr_boundary_adaptive_min_selected", 64) or 64))
        if int(activity["selected"]) < min_selected:
            return 1.0

        activities = [self._bimo_cr_boundary_operation_activity(item) for item in enabled_operations]
        total_selected = sum(int(item["selected"]) for item in activities)
        if total_selected < min_selected:
            return 1.0

        total_productivity = sum(self._bimo_cr_boundary_operation_productivity(item) for item in activities)
        if total_productivity <= 1e-12:
            return 1.0

        operation_productivity_rate = self._bimo_cr_boundary_operation_productivity(activity) / max(
            int(activity["selected"]), 1
        )
        global_productivity_rate = total_productivity / max(total_selected, 1)
        relative_score = operation_productivity_rate / max(global_productivity_rate, 1e-12)

        total_loss_norm = sum(float(item["mhc_loss_norm_sum"]) for item in activities)
        global_loss_norm = total_loss_norm / max(total_selected, 1)
        operation_loss_norm = float(activity["avg_mhc_loss_norm"])
        if global_loss_norm > 1e-12 and operation_loss_norm > global_loss_norm:
            loss_ratio = operation_loss_norm / global_loss_norm
            loss_weight = float(getattr(self, "bimo_cr_boundary_adaptive_loss_weight", 1.0) or 0.0)
            relative_score /= 1.0 + loss_weight * max(loss_ratio - 1.0, 0.0)

        total_cr_gain_norm = sum(float(item["cr_gain_norm_sum"]) for item in activities)
        global_cr_gain_norm = total_cr_gain_norm / max(total_selected, 1)
        operation_cr_gain_norm = float(activity["avg_cr_gain_norm"])
        if operation_cr_gain_norm > global_cr_gain_norm:
            relative_score *= 1.0 + min(operation_cr_gain_norm - global_cr_gain_norm, 1.0)

        strength = float(getattr(self, "bimo_cr_boundary_adaptive_strength", 1.0) or 0.0)
        if strength > 0.0:
            relative_score = relative_score**strength
        min_weight = float(getattr(self, "bimo_cr_boundary_adaptive_min_weight", 0.15) or 0.15)
        max_weight = float(getattr(self, "bimo_cr_boundary_adaptive_max_weight", 2.50) or 2.50)
        return float(np.clip(relative_score, min_weight, max(max_weight, min_weight)))

    def _bimo_cr_boundary_operation_budget_caps(self, enabled_operations, budget):
        operations = tuple(
            operation
            for operation in self.CR_BOUNDARY_REPARTITION_OPERATIONS
            if operation in set(enabled_operations or ())
        )
        budget = int(max(1, budget))
        if not operations:
            return {}
        if not bool(getattr(self, "bimo_cr_boundary_adaptive_operation_enabled", False)):
            return {operation: budget for operation in operations}

        weights = {
            operation: max(float(self._bimo_cr_boundary_operation_adaptive_weight(operation)), 0.0)
            for operation in operations
        }
        total_weight = float(sum(weights.values()))
        if total_weight <= 1e-12:
            return {operation: budget for operation in operations}
        return {
            operation: max(1, int(math.ceil(budget * weights[operation] / total_weight)))
            for operation in operations
        }

    def _bimo_cr_boundary_candidate_selection_key(self, base_key, candidate_info, operation):
        if not bool(getattr(self, "bimo_cr_boundary_adaptive_operation_enabled", False)):
            return base_key, 1.0, 0.0
        adaptive_weight = float(self._bimo_cr_boundary_operation_adaptive_weight(operation))
        archive_bonus = 1.0 if bool(candidate_info.get("archive_would_change", False)) else 0.0
        cr_gain_norm = float(candidate_info.get("cr_gain_norm", 0.0) or 0.0)
        mhc_loss_norm = float(candidate_info.get("mhc_loss_norm", 0.0) or 0.0)
        hv_gain_proxy = float(candidate_info.get("hv_gain_proxy", 0.0) or 0.0)
        extreme_gain = float(candidate_info.get("extreme_gain", 0.0) or 0.0)
        sparse_distance = float(candidate_info.get("sparse_distance", 0.0) or 0.0)
        adaptive_score = adaptive_weight * (
            4.0 * archive_bonus
            + 2.0 * cr_gain_norm
            + 1.5 * hv_gain_proxy
            + extreme_gain
            + 0.5 * sparse_distance
            - mhc_loss_norm
        )
        return (base_key[0], float(adaptive_score), *tuple(base_key[1:])), adaptive_weight, float(adaptive_score)

    @staticmethod
    def _bimo_cost_matches(left, right):
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            return bool(np.isfinite(left_value) == np.isfinite(right_value))
        tolerance = 1e-8 * max(abs(left_value), abs(right_value), 1.0)
        return abs(left_value - right_value) <= tolerance

    @staticmethod
    def _bimo_safe_cost_value(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return value
        return numeric if np.isfinite(numeric) else value

    def _bimo_add_cr_boundary_metric_sums(self, stats, prefix, info):
        metric_keys = (
            ("relation_score", "relation_score_sum"),
            ("cr_gain", "cr_gain_sum"),
            ("cr_gain_norm", "cr_gain_norm_sum"),
            ("mhc_loss", "mhc_loss_sum"),
            ("mhc_loss_norm", "mhc_loss_norm_sum"),
            ("hv_gain_proxy", "hv_gain_proxy_sum"),
            ("extreme_gain", "extreme_gain_sum"),
            ("sparse_distance", "sparse_distance_sum"),
        )
        for info_key, stat_key in metric_keys:
            value = info.get(info_key, 0.0) if isinstance(info, dict) else 0.0
            try:
                value = float(value or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if np.isfinite(value):
                stats[f"{prefix}{stat_key}"] += value

    def _record_bimo_cr_boundary_generated_encodings(self, encodings):
        self._ensure_bimo_cr_boundary_operation_telemetry()
        for encoding in encodings or []:
            operation, stats = self._bimo_cr_boundary_operation_stats(
                encoding.get("operation", "unknown") if isinstance(encoding, dict) else "unknown"
            )
            stats["generated"] += 1

    def _record_bimo_cr_boundary_selection(self, action_idx, previous_cost, next_cost, phase):
        if int(action_idx) != int(self.CR_BOUNDARY_REPARTITION_ACTION_ID):
            return None
        info = getattr(self, "_last_generated_cr_boundary_info", None)
        if not isinstance(info, dict):
            return None
        operation, stats = self._bimo_cr_boundary_operation_stats(info.get("operation", "unknown"))
        prefix = self._phase_prefix(phase)
        stats[f"{prefix}selected"] += 1
        if bool(info.get("archive_would_change", False)):
            stats[f"{prefix}archive_would_change"] += 1
        self._bimo_add_cr_boundary_metric_sums(stats, prefix, info)
        previous_value = self._bimo_safe_cost_value(previous_cost)
        next_value = self._bimo_safe_cost_value(next_cost)
        record = {
            "action_idx": int(action_idx),
            "phase": str(phase),
            "operation": operation,
            "previous_cost": previous_value,
            "next_cost": next_value,
            "accepted_recorded": False,
        }
        records = self._bimo_cr_boundary_selected_records
        records.append(record)
        if len(records) > 4096:
            del records[:2048]
        self._bimo_last_cr_boundary_selected_by_phase[(str(phase), int(action_idx))] = record
        return record

    def _match_bimo_cr_boundary_selected_record(self, action_idx, previous_cost, next_cost, phase):
        if int(action_idx) != int(self.CR_BOUNDARY_REPARTITION_ACTION_ID):
            return None
        phase = str(phase)
        records = getattr(self, "_bimo_cr_boundary_selected_records", [])
        for record in reversed(records):
            if bool(record.get("accepted_recorded", False)):
                continue
            if int(record.get("action_idx", -1)) != int(action_idx) or str(record.get("phase", "")) != phase:
                continue
            if self._bimo_cost_matches(record.get("previous_cost"), previous_cost) and self._bimo_cost_matches(
                record.get("next_cost"), next_cost
            ):
                record["accepted_recorded"] = True
                return record
        return self._bimo_last_cr_boundary_selected_by_phase.get((phase, int(action_idx)))

    def _record_bimo_cr_boundary_acceptance(self, action_idx, previous_cost, next_cost, improved, phase):
        record = self._match_bimo_cr_boundary_selected_record(action_idx, previous_cost, next_cost, phase)
        if not isinstance(record, dict):
            return None
        operation, stats = self._bimo_cr_boundary_operation_stats(record.get("operation", "unknown"))
        prefix = self._phase_prefix(phase)
        stats[f"{prefix}accepted"] += 1
        if bool(improved):
            stats[f"{prefix}improved"] += 1
        record["operation"] = operation
        self._bimo_last_cr_boundary_accepted_by_phase[(str(phase), int(action_idx))] = record
        return record

    def _record_bimo_cr_boundary_global_best(self, action_idx, phase):
        if int(action_idx) != int(self.CR_BOUNDARY_REPARTITION_ACTION_ID):
            return None
        phase = str(phase)
        meta = getattr(self, "_last_transition_meta", {}) or {}
        operation = None
        if isinstance(meta, dict) and bool(meta.get("cr_boundary_repartition_used", False)):
            operation = meta.get("cr_boundary_repartition_operation", "unknown")
        if operation is None:
            record = self._bimo_last_cr_boundary_accepted_by_phase.get((phase, int(action_idx)))
            if isinstance(record, dict):
                operation = record.get("operation", "unknown")
        if operation is None:
            return None
        operation, stats = self._bimo_cr_boundary_operation_stats(operation)
        prefix = self._phase_prefix(phase)
        stats[f"{prefix}global_best_hits"] += 1
        stats[f"{prefix}archive_changed"] += 1
        return operation

    def _record_action_selection(self, action_idx, previous_cost, next_cost, global_best=False, phase="main"):
        delta = super()._record_action_selection(
            action_idx,
            previous_cost,
            next_cost,
            global_best=global_best,
            phase=phase,
        )
        self._record_bimo_cr_boundary_selection(action_idx, previous_cost, next_cost, phase)
        return delta

    def _record_action_acceptance(self, action_idx, previous_cost, next_cost, improved=False, phase="main"):
        super()._record_action_acceptance(
            action_idx,
            previous_cost,
            next_cost,
            improved=improved,
            phase=phase,
        )
        self._record_bimo_cr_boundary_acceptance(action_idx, previous_cost, next_cost, improved, phase)

    def _record_action_global_best(self, action_idx, phase="main"):
        super()._record_action_global_best(action_idx, phase=phase)
        self._record_bimo_cr_boundary_global_best(action_idx, phase)

    def get_bimo_cr_boundary_operation_telemetry(self):
        telemetry = self._ensure_bimo_cr_boundary_operation_telemetry()
        summary = {}
        for operation in (*self.CR_BOUNDARY_REPARTITION_OPERATIONS, "unknown"):
            stats = telemetry.get(operation, {})
            item = dict(stats)
            has_activity = bool(item.get("generated", 0))
            for prefix in ("", "elite_"):
                selected = int(item.get(f"{prefix}selected", 0) or 0)
                accepted = int(item.get(f"{prefix}accepted", 0) or 0)
                has_activity = has_activity or selected > 0 or accepted > 0
                item[f"{prefix}accept_rate"] = 0.0 if selected == 0 else accepted / selected
                item[f"{prefix}archive_would_change_rate"] = (
                    0.0 if selected == 0 else int(item.get(f"{prefix}archive_would_change", 0) or 0) / selected
                )
                item[f"{prefix}archive_change_rate"] = (
                    0.0 if selected == 0 else int(item.get(f"{prefix}archive_changed", 0) or 0) / selected
                )
                for metric in (
                    "relation_score",
                    "cr_gain",
                    "cr_gain_norm",
                    "mhc_loss",
                    "mhc_loss_norm",
                    "hv_gain_proxy",
                    "extreme_gain",
                    "sparse_distance",
                ):
                    total = float(item.get(f"{prefix}{metric}_sum", 0.0) or 0.0)
                    item[f"{prefix}avg_{metric}"] = 0.0 if selected == 0 else total / selected
            item["adaptive_weight"] = self._bimo_cr_boundary_operation_adaptive_weight(operation)
            if has_activity:
                summary[operation] = item
        return summary

    def get_action_telemetry(self):
        summary = super().get_action_telemetry()
        action_id = int(self.CR_BOUNDARY_REPARTITION_ACTION_ID)
        if action_id in summary:
            summary[action_id]["operation_telemetry"] = self.get_bimo_cr_boundary_operation_telemetry()
        return summary

    def format_bimo_cr_boundary_operation_telemetry(self):
        telemetry = self.get_bimo_cr_boundary_operation_telemetry()
        lines = []
        for operation in (*self.CR_BOUNDARY_REPARTITION_OPERATIONS, "unknown"):
            stats = telemetry.get(operation)
            if not stats:
                continue
            lines.append(
                (
                    "Action 17 op={operation} | gen={generated} | "
                    "main sel={selected} acc={accepted} ({accept_rate:.1%}) gbest={gbest} "
                    "arch_pred={archive_pred} arch={archive_changed} avg_cr_gain={avg_cr_gain:.2f} "
                    "avg_mhc_loss={avg_mhc_loss:.2f} | "
                    "elite sel={elite_selected} acc={elite_accepted} ({elite_accept_rate:.1%}) "
                    "gbest={elite_gbest} arch_pred={elite_archive_pred} arch={elite_archive_changed} "
                    "avg_cr_gain={elite_avg_cr_gain:.2f} avg_mhc_loss={elite_avg_mhc_loss:.2f} | "
                    "adaptive_w={adaptive_weight:.3f} avg_mhc_loss_norm={avg_mhc_loss_norm:.4f}"
                ).format(
                    operation=operation,
                    generated=int(stats.get("generated", 0) or 0),
                    selected=int(stats.get("selected", 0) or 0),
                    accepted=int(stats.get("accepted", 0) or 0),
                    accept_rate=float(stats.get("accept_rate", 0.0) or 0.0),
                    gbest=int(stats.get("global_best_hits", 0) or 0),
                    archive_pred=int(stats.get("archive_would_change", 0) or 0),
                    archive_changed=int(stats.get("archive_changed", 0) or 0),
                    avg_cr_gain=float(stats.get("avg_cr_gain", 0.0) or 0.0),
                    avg_mhc_loss=float(stats.get("avg_mhc_loss", 0.0) or 0.0),
                    elite_selected=int(stats.get("elite_selected", 0) or 0),
                    elite_accepted=int(stats.get("elite_accepted", 0) or 0),
                    elite_accept_rate=float(stats.get("elite_accept_rate", 0.0) or 0.0),
                    elite_gbest=int(stats.get("elite_global_best_hits", 0) or 0),
                    elite_archive_pred=int(stats.get("elite_archive_would_change", 0) or 0),
                    elite_archive_changed=int(stats.get("elite_archive_changed", 0) or 0),
                    elite_avg_cr_gain=float(stats.get("elite_avg_cr_gain", 0.0) or 0.0),
                    elite_avg_mhc_loss=float(stats.get("elite_avg_mhc_loss", 0.0) or 0.0),
                    adaptive_weight=float(stats.get("adaptive_weight", 1.0) or 1.0),
                    avg_mhc_loss_norm=float(stats.get("avg_mhc_loss_norm", 0.0) or 0.0),
                )
            )
        return lines

    def format_action_telemetry(self):
        return super().format_action_telemetry() + self.format_bimo_cr_boundary_operation_telemetry()

    def _update_running_objective_bounds(self, objectives_min):
        vector = np.asarray(objectives_min, dtype=float).reshape(-1)[:2]
        if vector.size < 2 or not np.all(np.isfinite(vector)):
            return
        if not hasattr(self, "mo_running_min") or not hasattr(self, "mo_running_max"):
            self._reset_running_objective_bounds()
        self.mo_running_min = np.minimum(self.mo_running_min, vector)
        self.mo_running_max = np.maximum(self.mo_running_max, vector)

    @staticmethod
    def _repo_root_path():
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _facility_count_from_instance_file(instance_name):
        target_instance = str(instance_name or "").strip()
        if not target_instance:
            raise ValueError("实例名不能为空，无法生成 CR 矩阵。")
        instance_path = Path(config.FILE_PATH)
        if not instance_path.exists():
            raise FileNotFoundError(f"缺少实例数据文件，无法生成 CR 矩阵: {instance_path}")
        with instance_path.open("rb") as file_obj:
            problems, _flows, _layouts, _lengths, _widths, _sizes = pickle.load(file_obj)
        if target_instance not in problems:
            valid_instances = sorted(str(key) for key in problems.keys())
            raise ValueError(f"实例 {target_instance} 不存在，无法生成 CR 矩阵。可选实例: {valid_instances}")
        facility_count = int(problems[target_instance])
        if facility_count <= 0:
            raise ValueError(f"实例 {target_instance} 的设施数非法: {facility_count}")
        return facility_count

    @classmethod
    def _ensure_cr_matrix_available(cls, instance_name, facility_count=None):
        path = CRMatrixStore.build_path(instance_name=instance_name)
        if path.exists():
            return path
        count = int(facility_count) if facility_count is not None else cls._facility_count_from_instance_file(instance_name)
        if count <= 0:
            raise ValueError(f"无法为实例 {instance_name} 生成 CR 矩阵，设施数非法: {count}")
        matrix = CRMatrixStore.generate_matrix(count)
        output_path = CRMatrixStore.save_matrix(instance_name=instance_name, matrix=matrix, overwrite=False)
        logger.info("CR matrix generated for instance {} | facilities={} | path={}", instance_name, count, output_path)
        return output_path

    def _resolved_result_path(self, relative_path):
        if not relative_path:
            return None
        return self._repo_root_path() / str(relative_path).replace("/", os.sep)

    def _effective_archive_update_count(self):
        counter_value = int(getattr(self, "archive_update_count", 0) or 0)
        trace_bucket = getattr(self, "_mo_total_counters", {}) or {}
        bucket_value = int(trace_bucket.get("archiveChanges", 0) or 0)
        return max(counter_value, bucket_value)

    def _patch_run_summary_file(self, archive_update_count):
        run_summary_path = self._resolved_result_path(getattr(self.mo_run_summary, "get", lambda *_: None)("runSummaryPath"))
        if run_summary_path is None or not run_summary_path.exists():
            return
        payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload.pop("gbestUpdateCount", None)
            payload["archiveUpdateCount"] = int(archive_update_count)
            run_summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _patch_action_stats_file(self, archive_update_count):
        action_stats_path = self._resolved_result_path(getattr(self.mo_run_summary, "get", lambda *_: None)("actionStatsPath"))
        if action_stats_path is None or not action_stats_path.exists():
            return
        payload = json.loads(action_stats_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            meta_section = payload.get("meta")
            if isinstance(meta_section, dict):
                meta_section.pop("gbestUpdateCount", None)
                meta_section["archiveUpdateCount"] = int(archive_update_count)
            overall_section = payload.get("overall")
            if isinstance(overall_section, dict):
                overall_section.pop("gbestUpdateCount", None)
                overall_section["archiveUpdateCount"] = int(archive_update_count)
                overall_section["archiveChanges"] = int(archive_update_count)
            action_stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _patch_summary_csv(self, archive_update_count):
        algorithm_name = str(
            self.mo_run_summary.get("algorithm")
            or os.getenv("ELP_EXP_ALGORITHM", "ELP_DRL_BiMO4")
            or "ELP_DRL_BiMO4"
        )
        summary_csv_path = Path(config.RESULT_PATH) / "mo_runs_summary" / f"{self.instance_name}-{algorithm_name}.csv"
        if not summary_csv_path.exists() or not isinstance(self.mo_run_summary, dict):
            return
        frame = pd.read_csv(summary_csv_path, encoding="utf-8-sig")
        if "runId" not in frame.columns:
            return
        run_id = str(self.mo_run_summary.get("runId") or "")
        row_mask = frame["runId"].astype(str) == run_id
        if not row_mask.any():
            return
        if "gbestUpdateCount" in frame.columns:
            frame = frame.drop(columns=["gbestUpdateCount"])
        if "archiveUpdateCount" not in frame.columns:
            frame["archiveUpdateCount"] = pd.NA
        frame.loc[row_mask, "archiveUpdateCount"] = int(archive_update_count)
        frame.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    @staticmethod
    def _solution_array_payload(solution):
        solution_array = getattr(getattr(solution, "fbs_model", None), "array_2d", None)
        if hasattr(solution_array, "tolist"):
            solution_array = solution_array.tolist()
        return solution_array

    @staticmethod
    def _normalize_bi_weights(weights, floor_value=0.20):
        vector = np.asarray(weights, dtype=float).reshape(-1)
        if vector.size < 2:
            vector = np.pad(vector, (0, 2 - vector.size), constant_values=0.5)
        vector = np.clip(vector[:2], float(floor_value), None)
        total = float(np.sum(vector))
        if total <= 0.0 or not np.isfinite(total):
            return np.asarray([0.5, 0.5], dtype=float)
        return vector / total

    def _compute_adaptive_weight_target(self):
        if not bool(getattr(self, "mo_adaptive_weights_enabled", False)):
            return self.mo_base_weights.copy()
        if len(self.pareto_archive) < 2:
            return self.mo_base_weights.copy()
        normalized, _, _ = MO_FBSUtil_BiMO4._normalized_archive_matrix(
            self.pareto_archive,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
        )
        if normalized.shape[0] < 2:
            return self.mo_base_weights.copy()
        rows = []
        for idx in range(normalized.shape[0]):
            delta = normalized - normalized[idx]
            norms = np.linalg.norm(delta, axis=1)
            norms[idx] = np.inf
            nearest = float(np.min(norms))
            rows.append((nearest, idx))
        rows.sort(key=lambda item: item[0], reverse=True)
        sparse_vector = np.asarray(normalized[int(rows[0][1])], dtype=float)
        target = 1.0 - sparse_vector
        target = np.maximum(target, float(getattr(self, "mo_adaptive_weight_min_component", 0.20)))
        return self._normalize_bi_weights(
            target,
            floor_value=getattr(self, "mo_adaptive_weight_min_component", 0.20),
        )

    def _refresh_dynamic_weights(self):
        floor_value = float(getattr(self, "mo_adaptive_weight_min_component", 0.20) or 0.20)
        self.mo_base_weights = self._normalize_bi_weights(
            getattr(self, "mo_base_weights", [0.5, 0.5]),
            floor_value=floor_value,
        )
        self.mo_weights = self._normalize_bi_weights(
            getattr(self, "mo_weights", self.mo_base_weights),
            floor_value=floor_value,
        )
        if not bool(getattr(self, "mo_adaptive_weights_enabled", False)):
            self.mo_weights = self.mo_base_weights.copy()
            self.mo_last_weight_target = self.mo_base_weights.copy()
            return self.mo_weights

        current_step = int(max(getattr(self, "_trace_global_step", 0) or 0, 0))
        last_update_step = int(getattr(self, "mo_last_weight_update_step", -10**9) or -10**9)
        refresh_interval = int(max(1, int(getattr(self, "mo_adaptive_weight_refresh_interval_steps", 250) or 250)))
        if int(getattr(self, "mo_weight_update_count", 0) or 0) > 0 and (current_step - last_update_step) < refresh_interval:
            return self.mo_weights

        target = self._compute_adaptive_weight_target()
        self.mo_last_weight_target = np.asarray(target, dtype=float)
        current = self._normalize_bi_weights(
            getattr(self, "mo_weights", self.mo_base_weights),
            floor_value=floor_value,
        )
        deadband = float(max(getattr(self, "mo_adaptive_weight_deadband", 0.08) or 0.08, 0.0))
        if float(np.sum(np.abs(target - current))) < deadband:
            return self.mo_weights
        blend = float(getattr(self, "mo_adaptive_weight_blend", 0.15) or 0.15)
        updated = (1.0 - blend) * current + blend * target
        self.mo_weights = self._normalize_bi_weights(
            updated,
            floor_value=floor_value,
        )
        self.mo_weight_update_count = int(getattr(self, "mo_weight_update_count", 0) or 0) + 1
        self.mo_last_weight_update_step = current_step
        return self.mo_weights

    def _dynamic_weight_refresh_due(self):
        if not bool(getattr(self, "mo_adaptive_weights_enabled", False)):
            return False
        if int(getattr(self, "mo_weight_update_count", 0) or 0) <= 0:
            return False
        current_step = int(max(getattr(self, "_trace_global_step", 0) or 0, 0))
        last_update_step = int(getattr(self, "mo_last_weight_update_step", -10**9) or -10**9)
        refresh_interval = int(max(1, int(getattr(self, "mo_adaptive_weight_refresh_interval_steps", 250) or 250)))
        return bool((current_step - last_update_step) >= refresh_interval)

    def _bimo_normalized_objective_vector(self, solution):
        objectives = getattr(solution, "mo_objectives_min", None)
        if objectives is None:
            return np.zeros(2, dtype=float)
        normalized = MO_FBSUtil_BiMO4.normalize_objective_vector(
            objectives,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
        )
        if normalized is None:
            normalized = MO_FBSUtil_BiMO4.normalize_with_running_bounds(
                objectives,
                running_min=self.mo_running_min,
                running_max=self.mo_running_max,
            )
        if normalized is None:
            return np.zeros(2, dtype=float)
        normalized = np.asarray(normalized, dtype=float).reshape(-1)[:2]
        if normalized.size < 2:
            normalized = np.pad(normalized, (0, 2 - normalized.size), constant_values=0.0)
        normalized[~np.isfinite(normalized)] = 0.0
        return np.clip(normalized, 0.0, 1.0)

    def _bimo_archive_distance_features(self, solution):
        candidates = self._bimo_archive_anchor_candidates()
        current_vector = self._bimo_normalized_objective_vector(solution)
        if not candidates:
            return 1.0, 1.0

        distances = []
        for candidate in candidates:
            candidate_vector = self._bimo_normalized_objective_vector(candidate)
            distance = float(np.linalg.norm(current_vector - candidate_vector, ord=2))
            if distance > 1e-12:
                distances.append(distance)
        if not distances:
            return 0.0, 0.0
        nearest = float(min(distances))
        farthest = float(max(distances))
        scale = math.sqrt(2.0)
        return float(np.clip(nearest / scale, 0.0, 1.0)), float(np.clip(farthest / scale, 0.0, 1.0))

    def _agent_state_context(self, solution):
        if not bool(getattr(self, "bimo_dqn_context_enabled", False)):
            return None

        normalized = self._bimo_normalized_objective_vector(solution)
        nearest_distance, farthest_distance = self._bimo_archive_distance_features(solution)
        representative_distance = 1.0
        if self.representative_solution is not None:
            rep_vector = self._bimo_normalized_objective_vector(self.representative_solution)
            representative_distance = float(np.linalg.norm(normalized - rep_vector, ord=2) / math.sqrt(2.0))
        representative_distance = float(np.clip(representative_distance, 0.0, 1.0))

        archive_ratio = float(
            np.clip(
                len(getattr(self, "pareto_archive", []) or []) / float(max(1, int(getattr(self, "archive_limit", 1) or 1))),
                0.0,
                1.0,
            )
        )
        weights = np.asarray(getattr(self, "mo_weights", [0.5, 0.5]), dtype=float).reshape(-1)
        if weights.size < 2:
            weights = np.pad(weights, (0, 2 - weights.size), constant_values=0.5)
        weights = weights[:2]
        weight_sum = float(np.sum(np.clip(weights, 0.0, None)))
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            weights = np.asarray([0.5, 0.5], dtype=float)
        else:
            weights = np.clip(weights, 0.0, None) / weight_sum

        context = np.asarray(
            [
                normalized[0],  # MHC 归一化值，越小越接近 MHC 极端。
                1.0 - normalized[1],  # CR 归一化收益，越大越接近高 CR 极端。
                1.0 - normalized[0],
                1.0 - normalized[1],
                nearest_distance,
                farthest_distance,
                representative_distance,
                archive_ratio,
                weights[0],
                weights[1],
            ],
            dtype=np.float32,
        )
        context[~np.isfinite(context)] = 0.0
        return np.clip(context, 0.0, 1.0)

    @staticmethod
    def _bimo_normalized_hv_2d(points, reference=(1.1, 1.1)):
        matrix = np.asarray(points, dtype=float)
        if matrix.size == 0:
            return 0.0
        matrix = matrix.reshape(-1, 2)
        finite_mask = np.all(np.isfinite(matrix), axis=1)
        matrix = matrix[finite_mask]
        if matrix.size == 0:
            return 0.0

        reference = np.asarray(reference, dtype=float).reshape(-1)[:2]
        if reference.size < 2 or not np.all(np.isfinite(reference)):
            reference = np.asarray([1.1, 1.1], dtype=float)
        reference = np.maximum(reference, 1e-9)
        matrix = np.clip(matrix, 0.0, reference)

        nondominated = []
        for idx, point in enumerate(matrix):
            dominated = False
            for other_idx, other in enumerate(matrix):
                if idx == other_idx:
                    continue
                if np.all(other <= point + 1e-12) and np.any(other < point - 1e-12):
                    dominated = True
                    break
            if not dominated:
                nondominated.append(point)
        if not nondominated:
            return 0.0

        points = np.asarray(nondominated, dtype=float)
        points = points[np.argsort(points[:, 0])]
        hv = 0.0
        best_y = float(reference[1])
        for point in points:
            x = float(point[0])
            y = float(point[1])
            if y < best_y - 1e-12:
                hv += max(float(reference[0]) - x, 0.0) * max(best_y - y, 0.0)
                best_y = y
        reference_area = float(reference[0] * reference[1])
        if reference_area <= 0.0:
            return 0.0
        return float(np.clip(hv / reference_area, 0.0, 1.0))

    def _candidate_archive_quality_features(self, candidate):
        if candidate is None or not getattr(candidate, "current_is_feasible", False):
            return {"sparse_distance": 0.0, "extreme_gain": 0.0, "hv_gain_proxy": 0.0}
        objectives = getattr(candidate, "mo_objectives_min", None)
        if objectives is None:
            return {"sparse_distance": 0.0, "extreme_gain": 0.0, "hv_gain_proxy": 0.0}

        candidate_vector = np.asarray(objectives, dtype=float).reshape(-1)[:2]
        if candidate_vector.size < 2 or not np.all(np.isfinite(candidate_vector)):
            return {"sparse_distance": 0.0, "extreme_gain": 0.0, "hv_gain_proxy": 0.0}

        archive_vectors = []
        for item in getattr(self, "pareto_archive", []) or []:
            item_vector = getattr(item, "mo_objectives_min", None)
            if item_vector is None:
                continue
            item_vector = np.asarray(item_vector, dtype=float).reshape(-1)[:2]
            if item_vector.size == 2 and np.all(np.isfinite(item_vector)):
                archive_vectors.append(item_vector)

        if not archive_vectors:
            return {"sparse_distance": 1.0, "extreme_gain": 1.0, "hv_gain_proxy": 1.0}

        archive_matrix = np.asarray(archive_vectors, dtype=float)
        combined = np.vstack([archive_matrix, candidate_vector.reshape(1, 2)])
        ideal = np.min(combined, axis=0)
        nadir = np.max(combined, axis=0)
        span = np.maximum(nadir - ideal, 1e-12)
        archive_normalized = np.clip((archive_matrix - ideal) / span, 0.0, 1.0)
        candidate_normalized = np.clip((candidate_vector - ideal) / span, 0.0, 1.0)

        distances = np.linalg.norm(archive_normalized - candidate_normalized, axis=1)
        nearest = float(np.min(distances)) if distances.size else math.sqrt(2.0)
        sparse_distance = float(np.clip(nearest / math.sqrt(2.0), 0.0, 1.0))
        archive_min = np.min(archive_normalized, axis=0)
        extreme_gain = float(np.max(np.maximum(archive_min - candidate_normalized, 0.0)))
        archive_hv = self._bimo_normalized_hv_2d(archive_normalized)
        candidate_hv = self._bimo_normalized_hv_2d(
            np.vstack([archive_normalized, candidate_normalized.reshape(1, 2)])
        )
        hv_gain_proxy = float(np.clip(candidate_hv - archive_hv, 0.0, 1.0))
        return {
            "sparse_distance": sparse_distance,
            "extreme_gain": float(np.clip(extreme_gain, 0.0, 1.0)),
            "hv_gain_proxy": hv_gain_proxy,
        }

    @staticmethod
    def _relative_gain(current_value, candidate_value, maximize):
        try:
            current = float(current_value)
            candidate = float(candidate_value)
        except Exception:
            return 0.0
        if not np.isfinite(current) or not np.isfinite(candidate):
            return 0.0
        delta = candidate - current if bool(maximize) else current - candidate
        denom = max(abs(current), abs(candidate), 1.0)
        return float(np.clip(delta / denom, -1.0, 1.0))

    def _accept_candidate_with_context(self, current_solution, candidate_solution):
        accept, probability, current_tilde, candidate_tilde = super()._accept_candidate_with_context(
            current_solution,
            candidate_solution,
        )
        meta = dict(getattr(self, "_last_transition_meta", {}) or {})
        quality_features = self._candidate_archive_quality_features(candidate_solution)
        meta.update(
            {
                "current_mhc": float(getattr(current_solution, "MHC", math.inf)),
                "candidate_mhc": float(getattr(candidate_solution, "MHC", math.inf)),
                "current_cr": float(getattr(current_solution, "CR", 0.0)),
                "candidate_cr": float(getattr(candidate_solution, "CR", 0.0)),
                "mhc_relative_gain": self._relative_gain(
                    getattr(current_solution, "MHC", math.inf),
                    getattr(candidate_solution, "MHC", math.inf),
                    maximize=False,
                ),
                "cr_relative_gain": self._relative_gain(
                    getattr(current_solution, "CR", 0.0),
                    getattr(candidate_solution, "CR", 0.0),
                    maximize=True,
                ),
                "archive_sparse_distance": float(quality_features["sparse_distance"]),
                "archive_extreme_gain": float(quality_features["extreme_gain"]),
                "archive_hv_gain_proxy": float(quality_features["hv_gain_proxy"]),
            }
        )
        cr_boundary_info = getattr(candidate_solution, "bimo_cr_boundary_repartition_info", None)
        if isinstance(cr_boundary_info, dict):
            meta.update(
                {
                    "cr_boundary_repartition_used": True,
                    "cr_boundary_repartition_operation": str(cr_boundary_info.get("operation", "")),
                    "cr_boundary_repartition_relation_score": float(cr_boundary_info.get("relation_score", 0.0) or 0.0),
                    "cr_boundary_repartition_cr_gain": float(cr_boundary_info.get("cr_gain", 0.0) or 0.0),
                    "cr_boundary_repartition_mhc_loss": float(cr_boundary_info.get("mhc_loss", 0.0) or 0.0),
                    "cr_boundary_repartition_archive_would_change": bool(
                        cr_boundary_info.get("archive_would_change", False)
                    ),
                    "cr_boundary_repartition_bay_pair_idx": int(cr_boundary_info.get("bay_pair_idx", -1)),
                }
            )
        self._last_transition_meta = meta
        return accept, probability, current_tilde, candidate_tilde

    def _archive_candidate_before_current_update(self, candidate, accept):
        meta = dict(getattr(self, "_last_transition_meta", {}) or {})
        # 可行候选先入候选池（即使被支配也不丢失）
        if candidate is not None and getattr(candidate, "current_is_feasible", False):
            self._observe_bimo_candidate_pool(candidate, source="main_candidate")
        if not bool(meta.get("archive_would_change", False)):
            meta["archive_inserted_before_accept"] = False
            self._last_transition_meta = meta
            return False
        if candidate is None or not bool(getattr(candidate, "current_is_feasible", False)):
            meta["archive_would_change"] = False
            meta["archive_inserted_before_accept"] = False
            self._last_transition_meta = meta
            return False

        archive_changed = bool(self._observe_feasible_state(candidate))
        meta["archive_would_change"] = bool(archive_changed)
        meta["archive_inserted_before_accept"] = bool(archive_changed)
        self._last_transition_meta = meta
        if archive_changed:
            self._record_mo_event(
                "candidate_archived_before_current_update",
                acceptedAsCurrent=bool(accept),
                candidateMhc=self._safe_float(getattr(candidate, "MHC", None)),
                candidateCr=self._safe_float(getattr(candidate, "CR", None)),
                candidateDecisionScore=self._safe_float(getattr(candidate, "decision_score", None)),
                archiveSize=int(len(getattr(self, "pareto_archive", []) or [])),
            )
        return bool(archive_changed)

    def _compute_transition_reward(
        self,
        previous_cost,
        next_cost,
        previous_d_inf,
        next_d_inf,
        previous_best_feasible,
        accept,
    ):
        reward = super()._compute_transition_reward(
            previous_cost,
            next_cost,
            previous_d_inf,
            next_d_inf,
            previous_best_feasible,
            accept,
        )
        meta = dict(getattr(self, "_last_transition_meta", {}) or {})

        archive_would_change = bool(meta.get("archive_would_change", False))
        if archive_would_change:
            reward += float(getattr(self, "bimo_reward_archive_bonus", 0.45) or 0.0)
            reward += float(getattr(self, "bimo_reward_extreme_gain_weight", 0.35) or 0.0) * float(
                meta.get("archive_extreme_gain", 0.0) or 0.0
            )
            reward += float(getattr(self, "bimo_reward_sparse_gain_weight", 0.25) or 0.0) * float(
                meta.get("archive_sparse_distance", 0.0) or 0.0
            )
            reward += float(getattr(self, "bimo_reward_hv_proxy_weight", 0.50) or 0.0) * float(
                meta.get("archive_hv_gain_proxy", 0.0) or 0.0
            )
            if not bool(accept):
                reward += float(getattr(self, "bimo_reward_unaccepted_archive_bonus", 0.25) or 0.0)

        cr_gain = float(meta.get("cr_relative_gain", 0.0) or 0.0)
        if cr_gain >= 0.0:
            reward += float(getattr(self, "bimo_reward_cr_gain_weight", 0.40) or 0.0) * cr_gain
        else:
            reward += float(getattr(self, "bimo_reward_cr_loss_weight", 0.08) or 0.0) * cr_gain

        reward_clip = float(max(1.0, getattr(self, "bimo_reward_clip", 4.0) or 4.0))
        return float(np.clip(reward, -reward_clip, reward_clip))

    def _report_representative_snapshot(self):
        report_solution, report_score, report_index = MO_FBSUtil_BiMO4.select_knee_solution(
            self.pareto_archive,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
        )
        if report_solution is None:
            report_solution = self.representative_solution
            report_score = float(getattr(self, "representative_decision_score", math.inf))
            report_index = self.representative_archive_index
        if report_solution is None:
            return None
        return {
            "solution": copy.deepcopy(report_solution),
            "score": self._safe_float(report_score),
            "index": None if report_index is None else int(report_index),
            "mhc": self._safe_float(getattr(report_solution, "MHC", None)),
            "cr": self._safe_float(getattr(report_solution, "CR", None)),
            "solution_array": self._solution_array_payload(report_solution),
        }

    def _patch_archive_json_with_report_representative(self, report_snapshot):
        if not report_snapshot or not isinstance(self.mo_run_summary, dict):
            return
        archive_path = self._resolved_result_path(self.mo_run_summary.get("paretoArchivePath"))
        if archive_path is None or not archive_path.exists():
            return
        payload = json.loads(archive_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        payload["representativeArchiveIndex"] = None if report_snapshot["index"] is None else int(report_snapshot["index"]) + 1
        payload["representativeDecisionScore"] = report_snapshot["score"]
        payload["objectiveWeights"] = np.asarray(self.mo_weights, dtype=float).tolist()
        payload["baseObjectiveWeights"] = np.asarray(self.mo_base_weights, dtype=float).tolist()
        payload["adaptiveWeightTarget"] = np.asarray(self.mo_last_weight_target, dtype=float).tolist()
        payload["dqnContextMode"] = "bimo_objective_context" if self.bimo_dqn_context_enabled else "disabled"
        payload["dqnContextDim"] = int(getattr(self, "rl_context_dim", 0) or 0)
        archive_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _patch_run_summary_with_report_representative(self, report_snapshot):
        if not report_snapshot or not isinstance(self.mo_run_summary, dict):
            return
        run_summary_path = self._resolved_result_path(self.mo_run_summary.get("runSummaryPath"))
        if run_summary_path is None or not run_summary_path.exists():
            return
        payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        payload["searchRepresentativeArchiveIndex"] = None if self.representative_archive_index is None else int(self.representative_archive_index) + 1
        payload["searchRepresentativeDecisionScore"] = self._safe_float(self.representative_decision_score)
        payload["searchRepMhc"] = self._safe_float(getattr(self.representative_solution, "MHC", None)) if self.representative_solution is not None else None
        payload["searchRepCr"] = self._safe_float(getattr(self.representative_solution, "CR", None)) if self.representative_solution is not None else None
        payload["representativeArchiveIndex"] = None if report_snapshot["index"] is None else int(report_snapshot["index"]) + 1
        payload["representativeDecisionScore"] = report_snapshot["score"]
        payload["decisionScore"] = report_snapshot["score"]
        payload["repMhc"] = report_snapshot["mhc"]
        payload["repCr"] = report_snapshot["cr"]
        payload["representativeSolution"] = report_snapshot["solution_array"]
        payload["objectiveWeights"] = np.asarray(self.mo_weights, dtype=float).tolist()
        payload["baseObjectiveWeights"] = np.asarray(self.mo_base_weights, dtype=float).tolist()
        payload["adaptiveWeightTarget"] = np.asarray(self.mo_last_weight_target, dtype=float).tolist()
        payload["adaptiveWeightsEnabled"] = bool(self.mo_adaptive_weights_enabled)
        payload["adaptiveWeightBlend"] = self._safe_float(self.mo_adaptive_weight_blend)
        payload["adaptiveWeightUpdates"] = int(self.mo_weight_update_count)
        payload["dqnContextMode"] = "bimo_objective_context" if self.bimo_dqn_context_enabled else "disabled"
        payload["dqnContextDim"] = int(getattr(self, "rl_context_dim", 0) or 0)
        # 候选池统计
        payload["candidatePoolSize"] = self._bimo_candidate_pool_size()
        payload["candidatePoolLimit"] = int(getattr(self, "bimo_candidate_pool_limit", 96) or 96)
        payload["candidatePoolInsertions"] = int(getattr(self, "_bimo_candidate_pool_insert_count", 0) or 0)
        payload["candidatePoolDuplicates"] = int(getattr(self, "_bimo_candidate_pool_duplicate_count", 0) or 0)
        payload["candidatePoolBootstrapTarget"] = int(getattr(self, "bimo_candidate_pool_bootstrap_target", 32) or 32)
        run_summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _patch_summary_csv_with_report_representative(self, report_snapshot):
        if not report_snapshot or not isinstance(self.mo_run_summary, dict):
            return
        algorithm_name = str(
            self.mo_run_summary.get("algorithm")
            or os.getenv("ELP_EXP_ALGORITHM", "ELP_DRL_BiMO4")
            or "ELP_DRL_BiMO4"
        )
        summary_csv_path = Path(config.RESULT_PATH) / "mo_runs_summary" / f"{self.instance_name}-{algorithm_name}.csv"
        if not summary_csv_path.exists():
            return
        frame = pd.read_csv(summary_csv_path, encoding="utf-8-sig")
        if "runId" not in frame.columns:
            return
        row_mask = frame["runId"].astype(str) == str(self.mo_run_summary.get("runId") or "")
        if not row_mask.any():
            return
        for column in (
            "searchRepresentativeArchiveIndex",
            "searchRepresentativeDecisionScore",
            "searchRepMhc",
            "searchRepCr",
            "adaptiveWeightsEnabled",
            "adaptiveWeightBlend",
            "adaptiveWeightUpdates",
            "objectiveWeights",
            "baseObjectiveWeights",
            "adaptiveWeightTarget",
            "dqnContextMode",
            "dqnContextDim",
        ):
            if column not in frame.columns:
                frame[column] = pd.NA
        frame.loc[row_mask, "searchRepresentativeArchiveIndex"] = (
            None if self.representative_archive_index is None else int(self.representative_archive_index) + 1
        )
        frame.loc[row_mask, "searchRepresentativeDecisionScore"] = self._safe_float(self.representative_decision_score)
        frame.loc[row_mask, "searchRepMhc"] = self._safe_float(getattr(self.representative_solution, "MHC", None)) if self.representative_solution is not None else None
        frame.loc[row_mask, "searchRepCr"] = self._safe_float(getattr(self.representative_solution, "CR", None)) if self.representative_solution is not None else None
        if "representativeArchiveIndex" in frame.columns:
            frame.loc[row_mask, "representativeArchiveIndex"] = (
                None if report_snapshot["index"] is None else int(report_snapshot["index"]) + 1
            )
        if "representativeDecisionScore" in frame.columns:
            frame.loc[row_mask, "representativeDecisionScore"] = report_snapshot["score"]
        if "decisionScore" in frame.columns:
            frame.loc[row_mask, "decisionScore"] = report_snapshot["score"]
        if "repMhc" in frame.columns:
            frame.loc[row_mask, "repMhc"] = report_snapshot["mhc"]
        if "repCr" in frame.columns:
            frame.loc[row_mask, "repCr"] = report_snapshot["cr"]
        frame.loc[row_mask, "adaptiveWeightsEnabled"] = bool(self.mo_adaptive_weights_enabled)
        frame.loc[row_mask, "adaptiveWeightBlend"] = self._safe_float(self.mo_adaptive_weight_blend)
        frame.loc[row_mask, "adaptiveWeightUpdates"] = int(self.mo_weight_update_count)
        frame.loc[row_mask, "objectiveWeights"] = json.dumps(np.asarray(self.mo_weights, dtype=float).tolist(), ensure_ascii=False)
        frame.loc[row_mask, "baseObjectiveWeights"] = json.dumps(np.asarray(self.mo_base_weights, dtype=float).tolist(), ensure_ascii=False)
        frame.loc[row_mask, "adaptiveWeightTarget"] = json.dumps(np.asarray(self.mo_last_weight_target, dtype=float).tolist(), ensure_ascii=False)
        frame.loc[row_mask, "dqnContextMode"] = (
            "bimo_objective_context" if self.bimo_dqn_context_enabled else "disabled"
        )
        frame.loc[row_mask, "dqnContextDim"] = int(getattr(self, "rl_context_dim", 0) or 0)
        # 候选池字段
        for col in (
            "candidatePoolSize",
            "candidatePoolLimit",
            "candidatePoolInsertions",
            "candidatePoolDuplicates",
            "candidatePoolBootstrapTarget",
        ):
            if col not in frame.columns:
                frame[col] = pd.NA
        frame.loc[row_mask, "candidatePoolSize"] = self._bimo_candidate_pool_size()
        frame.loc[row_mask, "candidatePoolLimit"] = int(getattr(self, "bimo_candidate_pool_limit", 96) or 96)
        frame.loc[row_mask, "candidatePoolInsertions"] = int(getattr(self, "_bimo_candidate_pool_insert_count", 0) or 0)
        frame.loc[row_mask, "candidatePoolDuplicates"] = int(getattr(self, "_bimo_candidate_pool_duplicate_count", 0) or 0)
        frame.loc[row_mask, "candidatePoolBootstrapTarget"] = int(getattr(self, "bimo_candidate_pool_bootstrap_target", 32) or 32)
        frame.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    def _apply_report_representative_reporting(self):
        report_snapshot = self._report_representative_snapshot()
        if not report_snapshot:
            return None
        if isinstance(self.mo_run_summary, dict):
            self.mo_run_summary["searchRepresentativeArchiveIndex"] = (
                None if self.representative_archive_index is None else int(self.representative_archive_index) + 1
            )
            self.mo_run_summary["searchRepresentativeDecisionScore"] = self._safe_float(self.representative_decision_score)
            self.mo_run_summary["searchRepMhc"] = (
                self._safe_float(getattr(self.representative_solution, "MHC", None))
                if self.representative_solution is not None
                else None
            )
            self.mo_run_summary["searchRepCr"] = (
                self._safe_float(getattr(self.representative_solution, "CR", None))
                if self.representative_solution is not None
                else None
            )
            self.mo_run_summary["representativeArchiveIndex"] = (
                None if report_snapshot["index"] is None else int(report_snapshot["index"]) + 1
            )
            self.mo_run_summary["representativeDecisionScore"] = report_snapshot["score"]
            self.mo_run_summary["decisionScore"] = report_snapshot["score"]
            self.mo_run_summary["repMhc"] = report_snapshot["mhc"]
            self.mo_run_summary["repCr"] = report_snapshot["cr"]
            self.mo_run_summary["representativeSolution"] = report_snapshot["solution_array"]
            self.mo_run_summary["objectiveWeights"] = np.asarray(self.mo_weights, dtype=float).tolist()
            self.mo_run_summary["baseObjectiveWeights"] = np.asarray(self.mo_base_weights, dtype=float).tolist()
            self.mo_run_summary["adaptiveWeightTarget"] = np.asarray(self.mo_last_weight_target, dtype=float).tolist()
            self.mo_run_summary["adaptiveWeightsEnabled"] = bool(self.mo_adaptive_weights_enabled)
            self.mo_run_summary["adaptiveWeightBlend"] = self._safe_float(self.mo_adaptive_weight_blend)
            self.mo_run_summary["adaptiveWeightUpdates"] = int(self.mo_weight_update_count)
            self.mo_run_summary["dqnContextMode"] = (
                "bimo_objective_context" if self.bimo_dqn_context_enabled else "disabled"
            )
            self.mo_run_summary["dqnContextDim"] = int(getattr(self, "rl_context_dim", 0) or 0)
        if isinstance(self.last_run_payload, dict):
            self.last_run_payload["search_representative_archive_index"] = (
                None if self.representative_archive_index is None else int(self.representative_archive_index)
            )
            self.last_run_payload["search_representative_decision_score"] = self._safe_float(self.representative_decision_score)
            self.last_run_payload["search_rep_mhc"] = (
                self._safe_float(getattr(self.representative_solution, "MHC", None))
                if self.representative_solution is not None
                else None
            )
            self.last_run_payload["search_rep_cr"] = (
                self._safe_float(getattr(self.representative_solution, "CR", None))
                if self.representative_solution is not None
                else None
            )
            self.last_run_payload["rep_mhc"] = report_snapshot["mhc"]
            self.last_run_payload["rep_cr"] = report_snapshot["cr"]
            self.last_run_payload["decision_score"] = report_snapshot["score"]
            self.last_run_payload["report_representative_archive_index"] = (
                None if report_snapshot["index"] is None else int(report_snapshot["index"])
            )
            self.last_run_payload["objective_weights"] = np.asarray(self.mo_weights, dtype=float).tolist()
            self.last_run_payload["base_objective_weights"] = np.asarray(self.mo_base_weights, dtype=float).tolist()
            self.last_run_payload["adaptive_weight_target"] = np.asarray(self.mo_last_weight_target, dtype=float).tolist()
            self.last_run_payload["adaptive_weights_enabled"] = bool(self.mo_adaptive_weights_enabled)
            self.last_run_payload["adaptive_weight_updates"] = int(self.mo_weight_update_count)
            self.last_run_payload["dqn_context_mode"] = (
                "bimo_objective_context" if self.bimo_dqn_context_enabled else "disabled"
            )
            self.last_run_payload["dqn_context_dim"] = int(getattr(self, "rl_context_dim", 0) or 0)
        self._patch_archive_json_with_report_representative(report_snapshot)
        self._patch_run_summary_with_report_representative(report_snapshot)
        self._patch_summary_csv_with_report_representative(report_snapshot)
        return report_snapshot

    def _finalize_archive_update_reporting(self):
        archive_update_count = self._effective_archive_update_count()
        if isinstance(self.mo_run_summary, dict):
            self.mo_run_summary.pop("gbestUpdateCount", None)
            self.mo_run_summary["archiveUpdateCount"] = int(archive_update_count)
        if isinstance(self.last_run_payload, dict):
            self.last_run_payload.pop("gbest_update_count", None)
            self.last_run_payload.pop("gbestUpdateCount", None)
            self.last_run_payload["archive_update_count"] = int(archive_update_count)
        self._patch_run_summary_file(archive_update_count)
        self._patch_action_stats_file(archive_update_count)
        self._patch_summary_csv(archive_update_count)
        self._apply_report_representative_reporting()

    def _compute_reference_front_metrics(self):
        archive_hypervolume = self._safe_float(
            MO_FBSUtil_BiMO4.archive_hypervolume(self.pareto_archive, ideal=self.mo_ideal, nadir=self.mo_nadir)
        )
        archive_spacing = self._safe_float(
            MO_FBSUtil_BiMO4.archive_spacing(self.pareto_archive, ideal=self.mo_ideal, nadir=self.mo_nadir)
        )
        return {
            "archive_hypervolume": archive_hypervolume,
            "archive_spacing": archive_spacing,
            "archive_igd": None,
            "reference_front_path": None,
            "reference_front_size": None,
            "reference_front_archive_count": None,
            "archive_hypervolume_mode": "self_reference_2d",
            "archive_hypervolume_reference_point": None,
        }

    def _bimo_archive_paperls_effective_time_limit(self):
        configured_limit = float(getattr(self, "bimo_archive_paperls_time_limit_seconds", 0.0) or 0.0)
        if configured_limit > 0.0:
            return configured_limit
        reserve_seconds = float(getattr(self, "bimo_archive_paperls_reserve_seconds", 0.0) or 0.0)
        return max(0.0, reserve_seconds)

    def _append_unique_bimo_archive_paperls_anchor(self, anchors, seen, candidate):
        if candidate is None or not bool(getattr(candidate, "current_is_feasible", False)):
            return False
        key = self._bimo_solution_key(candidate)
        if key in seen:
            return False
        seen.add(key)
        anchors.append(copy.deepcopy(candidate))
        return True

    def _bimo_ranked_anchor_candidates(self, candidates):
        valid = []
        seen = set()
        for candidate in list(candidates or []):
            if candidate is None or not bool(getattr(candidate, "current_is_feasible", False)):
                continue
            if getattr(candidate, "mo_objectives_min", None) is None:
                continue
            key = self._bimo_solution_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            valid.append(candidate)
        if not valid:
            return []

        ranked = []
        ranked_seen = set()

        def append_candidate(candidate):
            key = self._bimo_solution_key(candidate)
            if key in ranked_seen:
                return
            ranked_seen.add(key)
            ranked.append(candidate)

        for objective_idx in range(2):
            extreme = min(
                valid,
                key=lambda item: float(
                    np.asarray(getattr(item, "mo_objectives_min", [math.inf, math.inf]), dtype=float)[objective_idx]
                ),
            )
            append_candidate(extreme)

        for candidate in self._bimo_sparse_anchor_candidates(valid, len(valid)):
            append_candidate(candidate)

        for candidate in sorted(
            valid,
            key=lambda item: (
                float(getattr(item, "decision_score", math.inf)),
                float(getattr(item, "MHC", math.inf)),
                -float(getattr(item, "CR", 0.0)),
            ),
        ):
            append_candidate(candidate)
        return ranked

    def _bimo_anchor_source_counts(self, anchors):
        archive_keys = {
            self._bimo_solution_key(candidate)
            for candidate in (getattr(self, "pareto_archive", []) or [])
            if getattr(candidate, "current_is_feasible", False)
        }
        pool_keys = {
            self._bimo_solution_key(candidate)
            for candidate in (getattr(self, "bimo_candidate_pool", []) or [])
            if getattr(candidate, "current_is_feasible", False)
        }
        archive_count = 0
        pool_count = 0
        unknown_count = 0
        for anchor in list(anchors or []):
            key = self._bimo_solution_key(anchor)
            if key in archive_keys:
                archive_count += 1
            elif key in pool_keys:
                pool_count += 1
            else:
                unknown_count += 1
        return archive_count, pool_count, unknown_count

    def _bimo_quality_gated_paperls_pool_candidates(self, archive_candidates):
        """仅允许能改变当前 Pareto 档案的候选池解参与最终 PaperLS。"""
        archive_keys = {
            self._bimo_solution_key(candidate)
            for candidate in list(archive_candidates or [])
            if getattr(candidate, "current_is_feasible", False)
        }
        gated = []
        seen = set()
        for candidate in self._bimo_ranked_anchor_candidates(
            self._bimo_anchor_candidate_pool(include_archive=False, include_candidate_pool=True)
        ):
            key = self._bimo_solution_key(candidate)
            if key in archive_keys or key in seen:
                continue
            comparison_archive = list(archive_candidates or []) + list(gated)
            _preview, would_change, _removed = MO_FBSUtil_BiMO4.update_pareto_archive(
                comparison_archive,
                candidate,
                max_size=int(max(1, getattr(self, "archive_limit", 64) or 64)),
                clone_fn=lambda item: item,
                require_candidate_retained=True,
                ideal=getattr(self, "mo_ideal", None),
                nadir=getattr(self, "mo_nadir", None),
            )
            if not bool(would_change):
                continue
            seen.add(key)
            gated.append(candidate)
        return gated

    def _bimo_archive_paperls_anchor_pool(self):
        anchor_limit = int(max(1, getattr(self, "bimo_archive_paperls_anchor_count", 8) or 8))
        anchors = []
        seen = set()

        # 1) 优先 representative / best feasible
        for candidate in (self.representative_solution, self.best_feasible_solution):
            self._append_unique_bimo_archive_paperls_anchor(anchors, seen, candidate)
            if len(anchors) >= anchor_limit:
                return anchors[:anchor_limit]

        # 2) 主线策略：最终 PaperLS 以 Pareto archive 为主，避免被候选池中的被支配解稀释。
        archive_candidates = self._bimo_anchor_candidate_pool(include_archive=True, include_candidate_pool=False)
        for candidate in self._bimo_ranked_anchor_candidates(archive_candidates):
            self._append_unique_bimo_archive_paperls_anchor(anchors, seen, candidate)
            if len(anchors) >= anchor_limit:
                return anchors[:anchor_limit]

        # 3) 只有能改变当前档案的候选池解才补位。
        for candidate in self._bimo_quality_gated_paperls_pool_candidates(archive_candidates):
            self._append_unique_bimo_archive_paperls_anchor(anchors, seen, candidate)
            if len(anchors) >= anchor_limit:
                return anchors[:anchor_limit]
        return anchors[:anchor_limit]

    def _run_bimo_archive_paperls_intensification(self):
        stats = {
            "enabled": bool(getattr(self, "bimo_archive_paperls_enabled", False)),
            "anchorsRequested": int(max(1, getattr(self, "bimo_archive_paperls_anchor_count", 8) or 8)),
            "anchorsSelected": 0,
            "anchorsUsed": 0,
            "selectedAnchorsFromArchive": 0,
            "selectedAnchorsFromCandidatePool": 0,
            "selectedAnchorsFromUnknown": 0,
            "anchorsFromArchive": 0,
            "anchorsFromCandidatePool": 0,
            "anchorsFromUnknown": 0,
            "anchorPolicy": "archive_first_quality_gated_pool",
            "archiveSizeBefore": int(len(getattr(self, "pareto_archive", []) or [])),
            "archiveSizeAfter": int(len(getattr(self, "pareto_archive", []) or [])),
            "candidatePoolSizeBefore": self._bimo_candidate_pool_size(),
            "candidatePoolSizeAfter": self._bimo_candidate_pool_size(),
            "archiveInsertions": 0,
            "acceptedMoves": 0,
            "neighborEvaluations": 0,
            "passes": int(max(1, getattr(self, "bimo_archive_paperls_passes", 2) or 2)),
            "timeLimitSeconds": self._safe_float(self._bimo_archive_paperls_effective_time_limit()),
            "maxNeighborEvaluations": int(
                max(0, getattr(self, "bimo_archive_paperls_max_neighbor_evaluations", 0) or 0)
            ),
            "perAnchorTimeLimitSeconds": 0.0,
            "perAnchorMaxNeighborEvaluations": 0,
            "stoppedByTime": False,
            "stoppedByGlobalTime": False,
            "stoppedByEvaluationLimit": False,
            "anchorDiagnostics": [],
            "runtimeSeconds": 0.0,
        }
        if not bool(getattr(self, "bimo_archive_paperls_enabled", False)):
            return stats
        if bool(getattr(self, "_bimo_archive_paperls_done", False)):
            return stats
        self._bimo_archive_paperls_done = True
        if not getattr(self, "pareto_archive", None):
            return stats

        self._refresh_archive_state()
        anchors = self._bimo_archive_paperls_anchor_pool()
        if not anchors:
            return stats

        archive_keys = {
            self._bimo_solution_key(candidate)
            for candidate in (getattr(self, "pareto_archive", []) or [])
            if getattr(candidate, "current_is_feasible", False)
        }
        pool_keys = {
            self._bimo_solution_key(candidate)
            for candidate in (getattr(self, "bimo_candidate_pool", []) or [])
            if getattr(candidate, "current_is_feasible", False)
        }

        stats["anchorsSelected"] = int(len(anchors))
        selected_archive_count, selected_pool_count, selected_unknown_count = self._bimo_anchor_source_counts(anchors)
        stats["selectedAnchorsFromArchive"] = int(selected_archive_count)
        stats["selectedAnchorsFromCandidatePool"] = int(selected_pool_count)
        stats["selectedAnchorsFromUnknown"] = int(selected_unknown_count)

        start_counter = int(getattr(self, "archive_update_count", 0) or 0)
        total_time_limit = float(stats["timeLimitSeconds"] or 0.0)
        total_eval_limit = int(stats["maxNeighborEvaluations"] or 0)
        selected_anchor_count = int(len(anchors))
        per_anchor_time_limit = total_time_limit / selected_anchor_count if total_time_limit > 0.0 else 0.0
        per_anchor_eval_limit = (
            int(math.ceil(total_eval_limit / selected_anchor_count))
            if total_eval_limit > 0 and selected_anchor_count > 0
            else 0
        )
        stats["perAnchorTimeLimitSeconds"] = self._safe_float(per_anchor_time_limit)
        stats["perAnchorMaxNeighborEvaluations"] = int(per_anchor_eval_limit)
        total_neighbor_evaluations = 0
        total_accepted_moves = 0
        any_anchor_stopped_by_time = False
        global_started_at = time.perf_counter()

        def anchor_source_for_key(anchor_key):
            if anchor_key in archive_keys:
                return "archive"
            if anchor_key in pool_keys:
                return "candidate_pool"
            return "unknown"

        def solution_objective_payload(solution):
            objectives = getattr(solution, "mo_objectives_min", None)
            return {
                "mhc": self._safe_float(getattr(solution, "MHC", None)),
                "cr": self._safe_float(getattr(solution, "CR", None)),
                "decisionScore": self._safe_float(self._bimo_candidate_score(solution)),
                "objective0": (
                    self._safe_float(np.asarray(objectives, dtype=float).reshape(-1)[0])
                    if objectives is not None and np.asarray(objectives, dtype=float).size >= 1
                    else None
                ),
                "objective1": (
                    self._safe_float(np.asarray(objectives, dtype=float).reshape(-1)[1])
                    if objectives is not None and np.asarray(objectives, dtype=float).size >= 2
                    else None
                ),
            }

        for anchor_index, anchor in enumerate(anchors):
            elapsed = time.perf_counter() - global_started_at
            if total_time_limit > 0.0 and elapsed >= total_time_limit:
                stats["stoppedByGlobalTime"] = True
                break
            remaining_time = max(total_time_limit - elapsed, 0.0) if total_time_limit > 0.0 else 0.0
            anchor_time_limit = (
                min(per_anchor_time_limit, remaining_time)
                if total_time_limit > 0.0
                else 0.0
            )
            if total_time_limit > 0.0 and anchor_time_limit <= 1e-9:
                stats["stoppedByGlobalTime"] = True
                break

            if total_eval_limit > 0:
                remaining_evaluations = max(total_eval_limit - total_neighbor_evaluations, 0)
                if remaining_evaluations <= 0:
                    stats["stoppedByEvaluationLimit"] = True
                    break
                anchor_eval_limit = min(per_anchor_eval_limit, remaining_evaluations)
            else:
                anchor_eval_limit = 0

            anchor_key = self._bimo_solution_key(anchor)
            anchor_source = anchor_source_for_key(anchor_key)
            if anchor_source == "archive":
                stats["anchorsFromArchive"] += 1
            elif anchor_source == "candidate_pool":
                stats["anchorsFromCandidatePool"] += 1
            else:
                stats["anchorsFromUnknown"] += 1

            anchor_archive_updates_before = int(getattr(self, "archive_update_count", 0) or 0)
            anchor_diag = {
                "anchorIndex": int(anchor_index + 1),
                "source": anchor_source,
                "timeLimitSeconds": self._safe_float(anchor_time_limit),
                "maxNeighborEvaluations": int(anchor_eval_limit),
                "archiveSizeBefore": int(len(getattr(self, "pareto_archive", []) or [])),
                "candidatePoolSizeBefore": self._bimo_candidate_pool_size(),
                "anchor": solution_objective_payload(anchor),
            }
            local_search = BiMO4PaperLocalSearch(
                self,
                passes=stats["passes"],
                time_limit_seconds=float(anchor_time_limit),
                max_neighbor_evaluations=int(anchor_eval_limit),
            )
            refined = local_search.local_search(copy.deepcopy(anchor))
            if refined is not None and bool(getattr(refined, "current_is_feasible", False)):
                local_search._observe_candidate(refined)
            summary = local_search.summary()
            total_neighbor_evaluations += int(summary["neighborEvaluations"])
            total_accepted_moves += int(summary["acceptedMoves"])
            any_anchor_stopped_by_time = bool(any_anchor_stopped_by_time or summary["stoppedByTime"])
            anchor_archive_updates_after = int(getattr(self, "archive_update_count", 0) or 0)
            anchor_diag.update(
                {
                    "archiveSizeAfter": int(len(getattr(self, "pareto_archive", []) or [])),
                    "candidatePoolSizeAfter": self._bimo_candidate_pool_size(),
                    "archiveInsertions": int(max(anchor_archive_updates_after - anchor_archive_updates_before, 0)),
                    "acceptedMoves": int(summary["acceptedMoves"]),
                    "neighborEvaluations": int(summary["neighborEvaluations"]),
                    "stoppedByTime": bool(summary["stoppedByTime"]),
                    "stoppedByEvaluationLimit": bool(summary["stoppedByEvaluationLimit"]),
                    "runtimeSeconds": self._safe_float(summary["runtimeSeconds"]),
                    "refined": (
                        solution_objective_payload(refined)
                        if refined is not None and bool(getattr(refined, "current_is_feasible", False))
                        else None
                    ),
                }
            )
            stats["anchorDiagnostics"].append(anchor_diag)
            stats["anchorsUsed"] += 1

        self._refresh_archive_state()
        if self.representative_solution is not None:
            self.best_feasible_solution = copy.deepcopy(self.representative_solution)
            self.gbest = copy.deepcopy(self.representative_solution)
            self.true_gbest = copy.deepcopy(self.representative_solution)
            self.best_feasible_cost = float(self.representative_decision_score)
            self.best_energy = float(self.representative_decision_score)
        stats["archiveSizeAfter"] = int(len(getattr(self, "pareto_archive", []) or []))
        stats["candidatePoolSizeAfter"] = self._bimo_candidate_pool_size()
        stats["archiveInsertions"] = int(max(int(getattr(self, "archive_update_count", 0) or 0) - start_counter, 0))
        stats["acceptedMoves"] = int(total_accepted_moves)
        stats["neighborEvaluations"] = int(total_neighbor_evaluations)
        if total_time_limit > 0.0 and stats["anchorsUsed"] < stats["anchorsSelected"]:
            stats["stoppedByGlobalTime"] = bool((time.perf_counter() - global_started_at) >= total_time_limit)
        if total_eval_limit > 0 and total_neighbor_evaluations >= total_eval_limit:
            stats["stoppedByEvaluationLimit"] = True
        stats["stoppedByTime"] = bool(any_anchor_stopped_by_time or stats["stoppedByGlobalTime"])
        stats["runtimeSeconds"] = float(time.perf_counter() - global_started_at)
        logger.info(
            "BiMO4 PaperLS archive intensification | anchors={} | evals={} | "
            "archive_size {} -> {} | pool_size {} -> {} | source archive/pool/unknown={}/{}/{} | "
            "inserts={} | per_anchor={:.2f}s | runtime={:.2f}s",
            stats["anchorsUsed"],
            stats["neighborEvaluations"],
            stats["archiveSizeBefore"],
            stats["archiveSizeAfter"],
            stats["candidatePoolSizeBefore"],
            stats["candidatePoolSizeAfter"],
            stats["anchorsFromArchive"],
            stats["anchorsFromCandidatePool"],
            stats["anchorsFromUnknown"],
            stats["archiveInsertions"],
            stats["perAnchorTimeLimitSeconds"],
            stats["runtimeSeconds"],
        )
        return stats

    def _patch_bimo_archive_paperls_run_summary_file(self, fields):
        if not isinstance(self.mo_run_summary, dict):
            return
        run_summary_path = self._resolved_result_path(self.mo_run_summary.get("runSummaryPath"))
        if run_summary_path is None or not run_summary_path.exists():
            return
        payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload.update(fields)
            run_summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _patch_bimo_archive_paperls_summary_csv(self, fields):
        if not isinstance(self.mo_run_summary, dict):
            return
        algorithm_name = str(
            self.mo_run_summary.get("algorithm")
            or os.getenv("ELP_EXP_ALGORITHM", "ELP_DRL_BiMO4")
            or "ELP_DRL_BiMO4"
        )
        summary_csv_path = Path(config.RESULT_PATH) / "mo_runs_summary" / f"{self.instance_name}-{algorithm_name}.csv"
        if not summary_csv_path.exists():
            return
        frame = pd.read_csv(summary_csv_path, encoding="utf-8-sig")
        if "runId" not in frame.columns:
            return
        row_mask = frame["runId"].astype(str) == str(self.mo_run_summary.get("runId") or "")
        if not row_mask.any():
            return
        for column, value in fields.items():
            if column not in frame.columns:
                frame[column] = pd.NA
            frame.loc[row_mask, column] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        frame.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    def _update_bimo_archive_paperls_reporting(
        self,
        start_time,
        end_time,
        fast_time,
        archive_path,
        reference_metrics,
        stats,
        total_wall_time_limit,
        main_wall_time_limit,
    ):
        runtime_seconds = None
        if start_time is not None and end_time is not None:
            runtime_seconds = float((end_time - start_time).total_seconds())
        summary_fields = {
            "endTime": None if end_time is None else end_time.isoformat(),
            "runtimeSeconds": runtime_seconds,
            "archiveSize": int(len(getattr(self, "pareto_archive", []) or [])),
            "paretoArchivePath": archive_path,
            "archiveHypervolume": reference_metrics["archive_hypervolume"],
            "archiveSpacing": reference_metrics["archive_spacing"],
            "archiveIgd": reference_metrics["archive_igd"],
            "referenceFrontPath": reference_metrics["reference_front_path"],
            "referenceFrontSize": reference_metrics["reference_front_size"],
            "referenceArchiveCount": reference_metrics["reference_front_archive_count"],
            "archiveHypervolumeMode": reference_metrics["archive_hypervolume_mode"],
            "archiveHypervolumeReferencePoint": reference_metrics["archive_hypervolume_reference_point"],
            "wallTimeLimitSeconds": self._safe_float(total_wall_time_limit),
            "mainWallTimeLimitSeconds": self._safe_float(main_wall_time_limit),
            "archivePaperLsEnabled": bool(stats.get("enabled", False)),
            "archivePaperLsStats": stats,
        }
        if isinstance(self.mo_run_summary, dict):
            self.mo_run_summary.update(summary_fields)
        if isinstance(self.last_run_payload, dict):
            self.last_run_payload.update(
                {
                    "pareto_archive_path": archive_path,
                    "pareto_size": int(len(getattr(self, "pareto_archive", []) or [])),
                    "archive_hypervolume": reference_metrics["archive_hypervolume"],
                    "archive_spacing": reference_metrics["archive_spacing"],
                    "archive_igd": reference_metrics["archive_igd"],
                    "reference_front_path": reference_metrics["reference_front_path"],
                    "reference_front_size": reference_metrics["reference_front_size"],
                    "reference_front_archive_count": reference_metrics["reference_front_archive_count"],
                    "archive_hypervolume_mode": reference_metrics["archive_hypervolume_mode"],
                    "archive_hypervolume_reference_point": reference_metrics["archive_hypervolume_reference_point"],
                    "runtime_seconds": runtime_seconds,
                    "wall_time_limit_seconds": self._safe_float(total_wall_time_limit),
                    "main_wall_time_limit_seconds": self._safe_float(main_wall_time_limit),
                    "archive_paperls_enabled": bool(stats.get("enabled", False)),
                    "archive_paperls_stats": stats,
                }
            )
        self._patch_bimo_archive_paperls_run_summary_file(summary_fields)
        self._patch_bimo_archive_paperls_summary_csv(summary_fields)

    def _apply_bimo_archive_paperls_after_run(self, result, total_wall_time_limit, main_wall_time_limit):
        if not bool(getattr(self, "bimo_archive_paperls_enabled", False)):
            return result
        if result is None or len(result) < 7:
            return result

        start_time = result[4]
        fast_time = result[6]
        stats = self._run_bimo_archive_paperls_intensification()
        if not bool(stats.get("enabled", False)):
            return result

        end_time = datetime.datetime.now()
        if int(stats.get("archiveInsertions", 0) or 0) > 0:
            fast_time = end_time
        self._refresh_archive_state()
        run_algorithm = os.getenv("ELP_EXP_ALGORITHM", "ELP_DRL_BiMO4")
        archive_path = self._save_pareto_archive(start_time, algorithm_name=run_algorithm)
        reference_metrics = self._compute_reference_front_metrics()
        self._update_bimo_archive_paperls_reporting(
            start_time=start_time,
            end_time=end_time,
            fast_time=fast_time,
            archive_path=archive_path,
            reference_metrics=reference_metrics,
            stats=stats,
            total_wall_time_limit=total_wall_time_limit,
            main_wall_time_limit=main_wall_time_limit,
        )

        best_solution = copy.deepcopy(self.representative_solution) if self.representative_solution is not None else result[2]
        best_energy = (
            float(self.representative_decision_score)
            if self.representative_solution is not None
            else float(result[3])
        )
        is_valid = bool(getattr(best_solution, "current_is_feasible", False))
        patched_result = list(result)
        patched_result[1] = is_valid
        patched_result[2] = best_solution
        patched_result[3] = best_energy
        patched_result[5] = end_time
        patched_result[6] = fast_time
        return tuple(patched_result)

    def _bimo_noop_candidate(self, solution):
        candidate = self._light_clone_solution(solution)
        self._evaluate_solution(candidate)
        return candidate

    @staticmethod
    def _bimo_encoding_key(permutation, bay):
        return (
            tuple(int(value) for value in np.asarray(permutation, dtype=int).reshape(-1).tolist()),
            tuple(int(value) for value in np.asarray(bay, dtype=int).reshape(-1).tolist()),
        )

    def _bimo_current_boundary_adjacency(self, solution, facility_count):
        try:
            fac_x = np.asarray(getattr(solution, "fac_x", []), dtype=float).reshape(-1)
            fac_y = np.asarray(getattr(solution, "fac_y", []), dtype=float).reshape(-1)
            fac_b = np.asarray(getattr(solution, "fac_b", []), dtype=float).reshape(-1)
            fac_h = np.asarray(getattr(solution, "fac_h", []), dtype=float).reshape(-1)
        except Exception:
            return np.zeros((facility_count, facility_count), dtype=float)
        if min(fac_x.size, fac_y.size, fac_b.size, fac_h.size) < int(facility_count):
            return np.zeros((facility_count, facility_count), dtype=float)
        return MO_FBSUtil_BiMO4.get_boundary_adjacency_matrix(
            fac_x[:facility_count],
            fac_y[:facility_count],
            fac_b[:facility_count],
            fac_h[:facility_count],
            facility_count,
        )

    def _bimo_archive_would_change(self, candidate):
        if candidate is None or not bool(getattr(candidate, "current_is_feasible", False)):
            return False
        _preview, archive_would_change, _removed = MO_FBSUtil_BiMO4.update_pareto_archive(
            getattr(self, "pareto_archive", []) or [],
            candidate,
            max_size=int(max(1, getattr(self, "archive_limit", 64) or 64)),
            clone_fn=lambda item: item,
            require_candidate_retained=True,
            ideal=getattr(self, "mo_ideal", None),
            nadir=getattr(self, "mo_nadir", None),
        )
        return bool(archive_would_change)

    def _bimo_cr_relation_candidate_key(self, current_solution, candidate, relation_score):
        current_cr = float(getattr(current_solution, "CR", 0.0) or 0.0)
        candidate_cr = float(getattr(candidate, "CR", 0.0) or 0.0)
        current_mhc = float(getattr(current_solution, "MHC", math.inf))
        candidate_mhc = float(getattr(candidate, "MHC", math.inf))
        if not np.isfinite(current_cr):
            current_cr = 0.0
        if not np.isfinite(candidate_cr):
            candidate_cr = 0.0
        if not np.isfinite(current_mhc):
            current_mhc = candidate_mhc if np.isfinite(candidate_mhc) else 1.0
        if not np.isfinite(candidate_mhc):
            candidate_mhc = current_mhc + max(abs(current_mhc), 1.0)
        cr_gain = candidate_cr - current_cr
        cr_gain_norm = cr_gain / max(abs(current_cr), abs(candidate_cr), 1.0)
        mhc_loss = max(candidate_mhc - current_mhc, 0.0)
        mhc_loss_norm = mhc_loss / max(abs(current_mhc), abs(candidate_mhc), 1.0)
        archive_would_change = self._bimo_archive_would_change(candidate)
        quality_features = self._candidate_archive_quality_features(candidate)
        decision_score = float(getattr(candidate, "decision_score", math.inf))
        if not np.isfinite(decision_score):
            decision_score = math.inf
        return (
            1 if bool(getattr(candidate, "current_is_feasible", False)) else 0,
            1 if archive_would_change else 0,
            float(cr_gain_norm),
            float(cr_gain),
            float(quality_features.get("hv_gain_proxy", 0.0) or 0.0),
            float(quality_features.get("extreme_gain", 0.0) or 0.0),
            float(quality_features.get("sparse_distance", 0.0) or 0.0),
            -float(mhc_loss_norm),
            -float(decision_score),
            float(relation_score),
        ), {
            "archive_would_change": bool(archive_would_change),
            "candidate_cr": float(candidate_cr),
            "cr_gain": float(cr_gain),
            "cr_gain_norm": float(cr_gain_norm),
            "mhc_loss": float(mhc_loss),
            "mhc_loss_norm": float(mhc_loss_norm),
            "hv_gain_proxy": float(quality_features.get("hv_gain_proxy", 0.0) or 0.0),
            "extreme_gain": float(quality_features.get("extreme_gain", 0.0) or 0.0),
            "sparse_distance": float(quality_features.get("sparse_distance", 0.0) or 0.0),
        }

    @staticmethod
    def _bimo_move_in_sequence(sequence, facility, position):
        updated = [int(item) for item in sequence if int(item) != int(facility)]
        insert_position = int(min(max(int(position), 0), len(updated)))
        updated.insert(insert_position, int(facility))
        return updated

    def _bimo_cr_boundary_pair_candidates(self, solution, bay_structure):
        relation_matrix = np.asarray(getattr(self, "rel_matrix", None), dtype=float)
        if relation_matrix.ndim != 2 or relation_matrix.shape[0] != relation_matrix.shape[1]:
            raise ValueError("CR 关系矩阵必须是方阵，无法执行 cr_boundary_repartition。")
        if len(bay_structure) < 2:
            return []

        permutation = np.asarray(solution.fbs_model.permutation, dtype=int).reshape(-1)
        facility_count = int(np.max(permutation)) if permutation.size else 0
        if facility_count <= 0:
            return []
        if relation_matrix.shape[0] < facility_count:
            raise ValueError(
                f"CR 关系矩阵规模不足: matrix={relation_matrix.shape}, facility_count={facility_count}"
            )

        active_relation = np.maximum(relation_matrix[:facility_count, :facility_count], 0.0)
        adjacency = self._bimo_current_boundary_adjacency(solution, facility_count)
        rows = []
        for bay_idx in range(len(bay_structure) - 1):
            left_bay = [int(item) for item in bay_structure[bay_idx]]
            right_bay = [int(item) for item in bay_structure[bay_idx + 1]]
            if not left_bay or not right_bay:
                continue

            weighted_missing_score = 0.0
            total_score = 0.0
            pair_rows = []
            for left_facility in left_bay:
                for right_facility in right_bay:
                    relation = float(active_relation[left_facility - 1, right_facility - 1])
                    if relation <= 0.0:
                        continue
                    total_score += relation
                    missing_weight = max(1.0 - float(adjacency[left_facility - 1, right_facility - 1]), 0.0)
                    weighted_missing_score += relation * missing_weight
                    pair_rows.append((relation * max(missing_weight, 1e-6), relation, left_facility, right_facility))

            if not pair_rows:
                continue
            pair_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
            rows.append(
                {
                    "bay_idx": int(bay_idx),
                    "score": float(weighted_missing_score if weighted_missing_score > 1e-12 else total_score),
                    "total_score": float(total_score),
                    "top_pairs": pair_rows[: min(3, len(pair_rows))],
                }
            )

        rows.sort(key=lambda item: (item["score"], item["total_score"]), reverse=True)
        limit = int(max(1, getattr(self, "bimo_cr_boundary_top_bay_pairs", 4) or 4))
        return rows[:limit]

    def _bimo_append_boundary_encoding(self, encodings, seen, base_structure, bay_idx, left_bay, right_bay, metadata):
        if not left_bay or not right_bay:
            return False
        candidate_structure = self._copy_bay_structure(base_structure)
        candidate_structure[int(bay_idx)] = [int(item) for item in left_bay]
        candidate_structure[int(bay_idx) + 1] = [int(item) for item in right_bay]
        candidate_perm, candidate_bay = FBSUtil.arrayToPermutation(candidate_structure)
        candidate_key = self._bimo_encoding_key(candidate_perm, candidate_bay)
        if candidate_key in seen:
            return False
        seen.add(candidate_key)
        encodings.append(
            {
                **metadata,
                "permutation": np.asarray(candidate_perm, dtype=int),
                "bay": np.asarray(candidate_bay, dtype=int),
            }
        )
        return True

    def _bimo_cr_boundary_repartition_encodings(self, solution):
        permutation = np.asarray(solution.fbs_model.permutation, dtype=int).reshape(-1)
        bay = np.asarray(solution.fbs_model.bay, dtype=int).reshape(-1)
        if permutation.size != bay.size or permutation.size < 2:
            return []

        bay_structure = self._copy_bay_structure(FBSUtil.permutationToArray(permutation, bay))
        if len(bay_structure) < 2:
            return []

        base_key = self._bimo_encoding_key(permutation, bay)
        seen = {base_key}
        encodings = []
        budget = int(max(1, getattr(self, "bimo_cr_boundary_repartition_budget", 16) or 16))
        block_limit = int(max(1, getattr(self, "bimo_cr_boundary_block_limit", 3) or 3))
        enabled_operations = set(
            getattr(self, "bimo_cr_boundary_enabled_operations", set(self.CR_BOUNDARY_REPARTITION_OPERATIONS))
            or set()
        )
        operation_caps = self._bimo_cr_boundary_operation_budget_caps(enabled_operations, budget)
        operation_counts = {operation: 0 for operation in operation_caps}

        def append_operation(operation, target_left_bay, target_right_bay, metadata):
            if operation not in enabled_operations:
                return False
            if operation_counts.get(operation, 0) >= operation_caps.get(operation, budget):
                return False
            appended = self._bimo_append_boundary_encoding(
                encodings,
                seen,
                bay_structure,
                bay_idx,
                target_left_bay,
                target_right_bay,
                metadata,
            )
            if appended:
                operation_counts[operation] = operation_counts.get(operation, 0) + 1
            return appended

        for bay_pair in self._bimo_cr_boundary_pair_candidates(solution, bay_structure):
            bay_idx = int(bay_pair["bay_idx"])
            left_bay = [int(item) for item in bay_structure[bay_idx]]
            right_bay = [int(item) for item in bay_structure[bay_idx + 1]]
            merged = left_bay + right_bay
            if len(merged) < 2:
                continue

            split_candidates = {len(left_bay), max(1, len(merged) // 2)}
            for _weighted_score, relation_score, left_facility, right_facility in bay_pair["top_pairs"]:
                left_position = merged.index(int(left_facility))
                right_position = merged.index(int(right_facility))
                low_position = min(left_position, right_position)
                high_position = max(left_position, right_position)
                for split_idx in range(low_position + 1, high_position + 1):
                    if 0 < split_idx < len(merged):
                        split_candidates.add(int(split_idx))

                if "align_top" in enabled_operations:
                    aligned_top_left = self._bimo_move_in_sequence(left_bay, left_facility, 0)
                    aligned_top_right = self._bimo_move_in_sequence(right_bay, right_facility, 0)
                    append_operation(
                        "align_top",
                        aligned_top_left,
                        aligned_top_right,
                        {
                            "operation": "align_top",
                            "relation_score": float(relation_score),
                            "bay_pair_idx": int(bay_idx),
                            "left_facility": int(left_facility),
                            "right_facility": int(right_facility),
                        },
                    )
                    if len(encodings) >= budget:
                        return encodings

                if "align_bottom" in enabled_operations:
                    aligned_bottom_left = self._bimo_move_in_sequence(left_bay, left_facility, len(left_bay))
                    aligned_bottom_right = self._bimo_move_in_sequence(right_bay, right_facility, len(right_bay))
                    append_operation(
                        "align_bottom",
                        aligned_bottom_left,
                        aligned_bottom_right,
                        {
                            "operation": "align_bottom",
                            "relation_score": float(relation_score),
                            "bay_pair_idx": int(bay_idx),
                            "left_facility": int(left_facility),
                            "right_facility": int(right_facility),
                        },
                    )
                    if len(encodings) >= budget:
                        return encodings

            if "split" in enabled_operations:
                for split_idx in sorted(split_candidates):
                    if not 0 < int(split_idx) < len(merged):
                        continue
                    if append_operation(
                        "split",
                        merged[: int(split_idx)],
                        merged[int(split_idx) :],
                        {
                            "operation": "split",
                            "relation_score": float(bay_pair["score"]),
                            "bay_pair_idx": int(bay_idx),
                        },
                    ):
                        if len(encodings) >= budget:
                            return encodings

            if "block_swap" in enabled_operations:
                for left_block_size in range(1, min(block_limit, len(left_bay)) + 1):
                    left_prefix = left_bay[:-left_block_size]
                    left_suffix = left_bay[-left_block_size:]
                    for right_block_size in range(1, min(block_limit, len(right_bay)) + 1):
                        right_prefix = right_bay[:right_block_size]
                        right_suffix = right_bay[right_block_size:]
                        new_left = left_prefix + right_prefix
                        new_right = left_suffix + right_suffix
                        if not new_left or not new_right:
                            continue
                        if append_operation(
                            "block_swap",
                            new_left,
                            new_right,
                            {
                                "operation": "block_swap",
                                "relation_score": float(bay_pair["score"]),
                                "bay_pair_idx": int(bay_idx),
                                "left_block_size": int(left_block_size),
                                "right_block_size": int(right_block_size),
                            },
                        ):
                            if len(encodings) >= budget:
                                return encodings
        return encodings

    def _generate_bimo_cr_boundary_repartition_candidate(self, solution):
        encodings = self._bimo_cr_boundary_repartition_encodings(solution)
        if not encodings:
            self._last_generated_cr_boundary_info = None
            return self._bimo_noop_candidate(solution)
        self._record_bimo_cr_boundary_generated_encodings(encodings)

        best_candidate = None
        best_key = None
        best_info = None
        for encoding in encodings:
            candidate = self._prepare_light_clone_with_encoding(
                solution,
                encoding["permutation"],
                encoding["bay"],
            )
            self._evaluate_solution(candidate)
            candidate_key, candidate_info = self._bimo_cr_relation_candidate_key(
                solution,
                candidate,
                relation_score=encoding["relation_score"],
            )
            if best_key is None or candidate_key > best_key:
                best_candidate = candidate
                best_key = candidate_key
                best_info = {**encoding, **candidate_info}

        if best_candidate is None:
            self._last_generated_cr_boundary_info = None
            return self._bimo_noop_candidate(solution)
        self._last_generated_cr_boundary_info = dict(best_info or {})
        best_candidate.bimo_cr_boundary_repartition_info = {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in (best_info or {}).items()
        }
        return best_candidate

    def _score_candidate_encoding(self, permutation, bay, solution):
        metrics = FBSUtil._evaluate_candidate_encoding_fast(
            permutation,
            bay,
            solution.areas,
            solution.H,
            solution.F,
            solution.aspect_limits,
            v_worst=self.mo_worst_feasible_mhc,
            k_penalty=self.k_penalty,
            distance_metric="manhattan",
        )
        objectives_raw = MO_FBSUtil_BiMO4.calculate_objectives(
            metrics["fac_x"],
            metrics["fac_y"],
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["mhc"],
            len(metrics["fac_x"]),
            rel_matrix=self.rel_matrix,
        )
        objectives_min = MO_FBSUtil_BiMO4.to_minimization(objectives_raw)
        self._update_running_objective_bounds(objectives_min)
        constraint_violation = MO_FBSUtil_BiMO4.calculate_total_constraint_violation(
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["lower_bounds"],
            metrics["upper_bounds"],
        )
        search_energy = MO_FBSUtil_BiMO4.search_energy(
            objectives_min,
            is_feasible=bool(metrics["is_feasible"]),
            d_inf=int(metrics["d_inf"]),
            total_violation=constraint_violation,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
            running_min=self.mo_running_min,
            running_max=self.mo_running_max,
        )
        return (
            float(search_energy),
            int(metrics["d_inf"]),
            float(constraint_violation),
            float(metrics["mhc"]),
            np.asarray(permutation, dtype=int),
            np.asarray(bay, dtype=int),
        )

    def generate_candidate_by_action(self, solution, action_idx):
        if (
            int(action_idx) == int(self.CR_BOUNDARY_REPARTITION_ACTION_ID)
            and bool(getattr(self, "bimo_cr_boundary_repartition_enabled", True))
        ):
            return self._generate_bimo_cr_boundary_repartition_candidate(solution)
        return super().generate_candidate_by_action(solution, action_idx)

    def _refresh_solution_search_metrics(self, solution):
        objectives_min = getattr(solution, "mo_objectives_min", None)
        if objectives_min is None:
            return float(getattr(solution, "fitness", math.inf))

        self._update_running_objective_bounds(objectives_min)
        proxy_energy = MO_FBSUtil_BiMO4.surrogate_energy(
            objectives_min,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
            running_min=self.mo_running_min,
            running_max=self.mo_running_max,
        )
        decision_score = MO_FBSUtil_BiMO4.decision_score(
            objectives_min,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
            running_min=self.mo_running_min,
            running_max=self.mo_running_max,
        )
        total_violation = float(getattr(solution, "constraint_violation", 0.0) or 0.0)
        search_energy = MO_FBSUtil_BiMO4.search_energy(
            objectives_min,
            is_feasible=bool(getattr(solution, "current_is_feasible", False)),
            d_inf=int(getattr(solution, "current_d_inf", 0) or 0),
            total_violation=total_violation,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
            running_min=self.mo_running_min,
            running_max=self.mo_running_max,
        )

        solution.proxy_energy = float(proxy_energy)
        solution.decision_score = float(decision_score)
        solution.fitness = float(search_energy)
        solution.best_feasible_cost = getattr(self, "best_feasible_cost", math.inf)
        solution.best_fitness = getattr(self, "best_feasible_cost", math.inf)
        solution.worst_feasible_cost = getattr(self, "worst_feasible_cost", None)
        solution.current_v_worst = getattr(self, "worst_feasible_cost", None)
        return float(search_energy)

    # ═══════════════════════════════════════════════════════════════
    # 候选池方法
    # ═══════════════════════════════════════════════════════════════

    def _bimo_candidate_pool_size(self):
        self._ensure_bimo_candidate_pool_state()
        return int(len(getattr(self, "bimo_candidate_pool", []) or []))

    def _observe_bimo_candidate_pool(self, candidate, source="unknown"):
        """将可行候选插入候选池（不改变 Pareto 档案）。"""
        self._ensure_bimo_candidate_pool_state()
        if not getattr(self, "bimo_candidate_pool_enabled", True):
            return False
        if candidate is None or not getattr(candidate, "current_is_feasible", False):
            return False
        if getattr(candidate, "mo_objectives_min", None) is None:
            return False

        key = self._bimo_solution_key(candidate)
        if key in self._bimo_candidate_pool_keys:
            self._bimo_candidate_pool_duplicate_count += 1
            return False

        cloned = copy.deepcopy(candidate)
        cloned_score = getattr(cloned, "decision_score", math.inf)
        try:
            cloned_score = float(cloned_score)
        except Exception:
            cloned_score = math.inf
        if not np.isfinite(cloned_score):
            self._refresh_solution_search_metrics(cloned)
        self.bimo_candidate_pool.append(cloned)
        self._bimo_candidate_pool_keys.add(key)
        self._bimo_candidate_pool_insert_count += 1
        self._prune_bimo_candidate_pool()
        return True

    def _set_bimo_candidate_pool(self, keep):
        """同步候选池内容、去重键和访问计数。"""
        self.bimo_candidate_pool = list(keep or [])
        self._bimo_candidate_pool_keys = {self._bimo_solution_key(candidate) for candidate in self.bimo_candidate_pool}
        old_visits = getattr(self, "_bimo_candidate_pool_visit_counts", {}) or {}
        self._bimo_candidate_pool_visit_counts = {
            key: int(old_visits.get(key, 0) or 0)
            for key in self._bimo_candidate_pool_keys
        }

    def _prune_bimo_candidate_pool(self):
        """限制候选池大小，按 MHC / CR / 均衡 / 稀疏配额保留解。"""
        self._ensure_bimo_candidate_pool_state()
        limit = int(getattr(self, "bimo_candidate_pool_limit", 96) or 96)
        pool = getattr(self, "bimo_candidate_pool", []) or []
        if len(pool) <= limit:
            return

        feasible = []
        feasible_keys = set()
        for candidate in pool:
            if not getattr(candidate, "current_is_feasible", False):
                continue
            if getattr(candidate, "mo_objectives_min", None) is None:
                continue
            key = self._bimo_solution_key(candidate)
            if key in feasible_keys:
                continue
            feasible_keys.add(key)
            feasible.append(candidate)
        if len(feasible) <= limit:
            self._set_bimo_candidate_pool(feasible)
            return

        keep = []
        seen = set()

        def objective_value(candidate, objective_idx, default):
            objectives = np.asarray(getattr(candidate, "mo_objectives_min", [default, default]), dtype=float).reshape(-1)
            if objectives.size <= objective_idx or not np.isfinite(objectives[objective_idx]):
                return float(default)
            return float(objectives[objective_idx])

        def append_candidates(candidates, quota):
            quota = int(max(0, quota or 0))
            appended = 0
            for candidate in list(candidates or []):
                if len(keep) >= limit or appended >= quota:
                    break
                key = self._bimo_solution_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                keep.append(candidate)
                appended += 1

        mhc_quota = int(max(1, round(limit * float(getattr(self, "bimo_candidate_pool_mhc_quota_fraction", 0.30) or 0.0))))
        cr_quota = int(max(1, round(limit * float(getattr(self, "bimo_candidate_pool_cr_quota_fraction", 0.15) or 0.0))))
        balanced_quota = int(max(1, round(limit * float(getattr(self, "bimo_candidate_pool_balanced_quota_fraction", 0.35) or 0.0))))

        # MHC 极端：保护低物流成本端，避免候选池被高 CR 解挤满。
        append_candidates(
            sorted(feasible, key=lambda item: objective_value(item, 0, math.inf)),
            mhc_quota,
        )
        # CR 极端：CR 是最大化目标，在最小化向量中对应第二目标越小越好。
        append_candidates(
            sorted(feasible, key=lambda item: objective_value(item, 1, math.inf)),
            cr_quota,
        )
        # 均衡质量：保留代表性低 decision_score 解。
        append_candidates(
            sorted(
                feasible,
                key=lambda item: (
                    self._bimo_candidate_score(item),
                    float(getattr(item, "MHC", math.inf)),
                    -float(getattr(item, "CR", 0.0)),
                ),
            ),
            balanced_quota,
        )

        if len(keep) < limit:
            remaining = limit - len(keep)
            remaining_candidates = [candidate for candidate in feasible if self._bimo_solution_key(candidate) not in seen]
            append_candidates(self._bimo_sparse_anchor_candidates(remaining_candidates, len(remaining_candidates)), remaining)

        if len(keep) < limit:
            remaining = limit - len(keep)
            append_candidates(
                sorted(
                    feasible,
                    key=lambda item: (
                        self._bimo_candidate_score(item),
                        float(getattr(item, "MHC", math.inf)),
                        -float(getattr(item, "CR", 0.0)),
                    ),
                ),
                remaining,
            )

        self._set_bimo_candidate_pool(keep[:limit])

    def _bimo_anchor_candidate_pool(self, include_archive=True, include_candidate_pool=True):
        """返回联合 anchor 候选（Pareto archive + candidate pool，去重）。"""
        combined = []
        seen = set()
        if include_archive:
            for c in (getattr(self, "pareto_archive", []) or []):
                if not getattr(c, "current_is_feasible", False):
                    continue
                key = self._bimo_solution_key(c)
                if key not in seen:
                    seen.add(key)
                    combined.append(c)
        if include_candidate_pool:
            for c in (getattr(self, "bimo_candidate_pool", []) or []):
                if not getattr(c, "current_is_feasible", False):
                    continue
                key = self._bimo_solution_key(c)
                if key not in seen:
                    seen.add(key)
                    combined.append(c)
        return combined

    def _bimo_sparse_anchor_candidates(self, candidates, count):
        """从给定候选集合中选择目标空间最稀疏的 anchor。"""
        count = int(max(0, count or 0))
        valid = [
            candidate
            for candidate in list(candidates or [])
            if getattr(candidate, "current_is_feasible", False)
            and getattr(candidate, "mo_objectives_min", None) is not None
        ]
        if count <= 0 or len(valid) <= 1:
            return []
        normalized, _, _ = MO_FBSUtil_BiMO4._normalized_archive_matrix(
            valid,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
        )
        if normalized.shape[0] <= 1:
            return []
        rows = []
        for idx in range(normalized.shape[0]):
            delta = normalized - normalized[idx]
            norms = np.linalg.norm(delta, axis=1)
            norms[idx] = np.inf
            nearest = float(np.min(norms))
            rows.append((nearest, idx))
        rows.sort(key=lambda item: item[0], reverse=True)
        return [valid[int(idx)] for _, idx in rows[:count]]

    def _refresh_archive_state(self):
        feasible_archive = [candidate for candidate in self.pareto_archive if getattr(candidate, "current_is_feasible", False)]
        self.pareto_archive = feasible_archive
        self.mo_ideal, self.mo_nadir = MO_FBSUtil_BiMO4.compute_ideal_nadir(self.pareto_archive)
        self._refresh_dynamic_weights()

        for candidate in self.pareto_archive:
            self._refresh_solution_search_metrics(candidate)

        representative, decision_score, archive_index = MO_FBSUtil_BiMO4.select_representative_solution(
            self.pareto_archive,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
        )
        self.representative_solution = copy.deepcopy(representative) if representative is not None else None
        self.representative_decision_score = float(decision_score)
        self.representative_archive_index = archive_index
        if self.representative_solution is not None:
            self._refresh_solution_search_metrics(self.representative_solution)

    def _build_elite_anchor_pool(self):
        anchors = []
        max_anchor_count = int(max(1, self.mo_elite_anchor_count))

        primary_candidates = [self.best_feasible_solution]
        if bool(getattr(self, "mo_elite_include_representative", True)):
            primary_candidates.append(self.representative_solution)
        for candidate in self._sorted_elite_candidates(primary_candidates):
            self._append_unique_elite_anchor(anchors, candidate)
            break

        archive_candidates = self._bimo_anchor_candidate_pool(include_archive=True, include_candidate_pool=False)
        pool_candidates = self._bimo_anchor_candidate_pool(include_archive=False, include_candidate_pool=True)
        ranked_archive = self._bimo_ranked_anchor_candidates(archive_candidates)
        ranked_pool = self._bimo_ranked_anchor_candidates(pool_candidates)

        # 精英局部搜索同样 Pareto-first；候选池仅在档案 anchor 不足时补位。
        preferred_candidates = ranked_archive if ranked_archive else ranked_pool
        extreme_candidates = []
        if preferred_candidates:
            if bool(getattr(self, "mo_elite_include_extremes", True)):
                objective_count = 2
                for objective_idx in range(objective_count):
                    extreme_candidate = min(
                        preferred_candidates,
                        key=lambda item: float(
                            np.asarray(getattr(item, "mo_objectives_min", [math.inf] * objective_count), dtype=float)[
                                objective_idx
                            ]
                        ),
                    )
                    extreme_candidates.append(extreme_candidate)
                if extreme_candidates and len(anchors) < max_anchor_count:
                    offset = int(
                        (
                            int(getattr(self, "_trace_global_step", 0) or 0)
                            + int(getattr(self, "elite_trigger_count", 0) or 0)
                        )
                        % len(extreme_candidates)
                    )
                    rotated_extremes = extreme_candidates[offset:] + extreme_candidates[:offset]
                    for extreme_candidate in rotated_extremes:
                        before_count = len(anchors)
                        self._append_unique_elite_anchor(anchors, extreme_candidate)
                        if len(anchors) > before_count:
                            break
            if bool(getattr(self, "mo_elite_include_sparse", True)):
                for sparse_candidate in self._bimo_sparse_anchor_candidates(
                    preferred_candidates,
                    self.mo_elite_sparse_anchor_count,
                ):
                    self._append_unique_elite_anchor(anchors, sparse_candidate)
                    if len(anchors) >= max_anchor_count:
                        break

        if len(anchors) < max_anchor_count:
            fill_candidates = []
            fill_candidates.extend(primary_candidates)
            fill_candidates.extend(extreme_candidates)
            fill_candidates.extend(ranked_archive)
            fill_candidates.extend(ranked_pool)
            for candidate in self._sorted_elite_candidates(fill_candidates):
                self._append_unique_elite_anchor(anchors, candidate)
                if len(anchors) >= max_anchor_count:
                    break

        return anchors[:max_anchor_count]

    @staticmethod
    def _bimo_solution_key(solution):
        model = getattr(solution, "fbs_model", None)
        permutation = getattr(model, "permutation", None)
        bay = getattr(model, "bay", None)
        if permutation is not None and bay is not None:
            return (
                "layout",
                tuple(np.asarray(permutation, dtype=int).reshape(-1).tolist()),
                tuple(np.asarray(bay, dtype=int).reshape(-1).tolist()),
            )
        objectives = getattr(solution, "mo_objectives_min", None)
        if objectives is not None:
            vector = np.asarray(objectives, dtype=float).reshape(-1)[:2]
            if vector.size == 2 and np.all(np.isfinite(vector)):
                return ("objectives", tuple(np.round(vector, 12).tolist()))
        return ("object", id(solution))

    def _bimo_candidate_score(self, solution):
        for attr_name in ("decision_score", "proxy_energy", "fitness"):
            value = getattr(solution, attr_name, math.inf)
            try:
                value = float(value)
            except Exception:
                continue
            if np.isfinite(value):
                return float(value)
        objectives = getattr(solution, "mo_objectives_min", None)
        if objectives is None:
            return math.inf
        return float(
            MO_FBSUtil_BiMO4.decision_score(
                objectives,
                ideal=self.mo_ideal,
                nadir=self.mo_nadir,
                weights=self.mo_weights,
                running_min=self.mo_running_min,
                running_max=self.mo_running_max,
            )
        )

    @staticmethod
    def _scale_minimize_scores(values):
        vector = np.asarray(values, dtype=float).reshape(-1)
        finite_mask = np.isfinite(vector)
        if not np.any(finite_mask):
            return np.ones(vector.shape[0], dtype=float)
        finite_values = vector[finite_mask]
        worst_value = float(np.max(finite_values))
        vector = np.where(finite_mask, vector, worst_value)
        best_value = float(np.min(vector))
        span = float(np.max(vector) - best_value)
        if span <= 1e-12:
            return np.ones(vector.shape[0], dtype=float)
        return np.clip(1.0 - (vector - best_value) / span, 0.0, 1.0)

    @staticmethod
    def _scale_maximize_scores(values):
        vector = np.asarray(values, dtype=float).reshape(-1)
        finite_mask = np.isfinite(vector)
        if not np.any(finite_mask):
            return np.zeros(vector.shape[0], dtype=float)
        vector = np.where(finite_mask, vector, 0.0)
        max_value = float(np.max(vector))
        min_value = float(np.min(vector))
        span = max_value - min_value
        if span <= 1e-12:
            return np.ones(vector.shape[0], dtype=float) if max_value > 0.0 else np.zeros(vector.shape[0], dtype=float)
        return np.clip((vector - min_value) / span, 0.0, 1.0)

    def _bimo_archive_anchor_candidates(self):
        archive_candidates = self._bimo_anchor_candidate_pool(include_archive=True, include_candidate_pool=False)
        pool_candidates = self._bimo_anchor_candidate_pool(include_archive=False, include_candidate_pool=True)
        if not archive_candidates:
            return self._bimo_ranked_anchor_candidates(pool_candidates)

        archive_keys = {self._bimo_solution_key(candidate) for candidate in archive_candidates}
        pool_ranked = [
            candidate
            for candidate in self._bimo_ranked_anchor_candidates(pool_candidates)
            if self._bimo_solution_key(candidate) not in archive_keys
        ]
        max_fraction = float(getattr(self, "bimo_candidate_pool_runtime_max_fraction", 0.30) or 0.0)
        if max_fraction <= 0.0:
            pool_cap = 0
        elif max_fraction >= 1.0:
            pool_cap = len(pool_ranked)
        else:
            pool_cap = int(math.floor(len(archive_candidates) * max_fraction / max(1e-12, 1.0 - max_fraction)))
        min_anchor_count = int(max(1, getattr(self, "bimo_candidate_pool_min_anchor_size", 8) or 8))
        if len(archive_candidates) + pool_cap < min_anchor_count:
            pool_cap = max(pool_cap, min_anchor_count - len(archive_candidates))
        pool_cap = int(max(0, min(pool_cap, len(pool_ranked))))
        return list(archive_candidates) + pool_ranked[:pool_cap]

    def _bimo_anchor_selection_weights(self, candidates):
        count = int(len(candidates))
        if count <= 0:
            return np.asarray([], dtype=float), {}

        normalized, _, _ = MO_FBSUtil_BiMO4._normalized_archive_matrix(
            candidates,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
        )
        if normalized.shape[0] != count:
            normalized, _, _ = MO_FBSUtil_BiMO4._normalized_archive_matrix(candidates)
        if normalized.shape[0] != count:
            normalized = np.zeros((count, 2), dtype=float)

        quality = self._scale_minimize_scores([self._bimo_candidate_score(candidate) for candidate in candidates])

        if count <= 1:
            sparse = np.zeros(count, dtype=float)
        else:
            distance_matrix = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
            np.fill_diagonal(distance_matrix, np.inf)
            nearest = np.min(distance_matrix, axis=1)
            sparse = self._scale_maximize_scores(nearest)

        if normalized.size == 0:
            extreme = np.zeros(count, dtype=float)
        else:
            # 归一化最小化空间中越接近任一目标极端，越应被保留为搜索锚点。
            extreme = np.maximum(1.0 - normalized[:, 0], 1.0 - normalized[:, 1])
            extreme = np.clip(extreme, 0.0, 1.0)

        visit_counts = np.asarray(
            [
                int((getattr(self, "_bimo_anchor_visit_counts", {}) or {}).get(self._bimo_solution_key(candidate), 0))
                for candidate in candidates
            ],
            dtype=float,
        )
        stale = 1.0 / (1.0 + np.maximum(visit_counts, 0.0))

        weights = (
            float(getattr(self, "bimo_anchor_quality_weight", 0.25) or 0.0) * quality
            + float(getattr(self, "bimo_anchor_sparse_weight", 0.35) or 0.0) * sparse
            + float(getattr(self, "bimo_anchor_extreme_weight", 0.25) or 0.0) * extreme
            + float(getattr(self, "bimo_anchor_stale_weight", 0.15) or 0.0) * stale
        )
        min_weight = float(getattr(self, "bimo_anchor_min_probability_weight", 1e-6) or 0.0)
        weights = np.maximum(np.asarray(weights, dtype=float), min_weight)
        weights[~np.isfinite(weights)] = min_weight
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones(count, dtype=float)

        components = {
            "quality": quality,
            "sparse": sparse,
            "extreme": extreme,
            "stale": stale,
            "visit_counts": visit_counts,
        }
        return weights, components

    def _select_bimo_archive_anchor(self):
        candidates = self._bimo_archive_anchor_candidates()
        if len(candidates) < int(max(1, getattr(self, "bimo_archive_anchor_min_size", 2))):
            return None, {}

        weights, components = self._bimo_anchor_selection_weights(candidates)
        probabilities = weights / float(np.sum(weights))
        selected_index = int(np.random.choice(len(candidates), p=probabilities))
        selected = candidates[selected_index]
        details = {
            "candidate_count": int(len(candidates)),
            "selected_index": int(selected_index),
            "probability": float(probabilities[selected_index]),
            "quality": float(components["quality"][selected_index]),
            "sparse": float(components["sparse"][selected_index]),
            "extreme": float(components["extreme"][selected_index]),
            "stale": float(components["stale"][selected_index]),
            "visit_count": int(components["visit_counts"][selected_index]),
        }
        return selected, details

    def _restart_from_bimo_archive_anchor(self, episode_idx):
        if not bool(getattr(self, "bimo_archive_anchor_selection_enabled", True)):
            return False
        selected, details = self._select_bimo_archive_anchor()
        if selected is None:
            return False

        self.s = copy.deepcopy(selected)
        self._evaluate_solution(self.s)
        self.current_energy = self.s.fitness
        self.no_improve_steps = 0
        key = self._bimo_solution_key(selected)
        self._bimo_anchor_visit_counts[key] = int(self._bimo_anchor_visit_counts.get(key, 0)) + 1
        self._bimo_last_anchor_key = key
        self._bimo_last_anchor_episode = int(episode_idx)
        self._record_mo_event(
            "bimo_archive_anchor_selected",
            anchorEpisode=int(episode_idx) + 1,
            archiveSize=int(len(getattr(self, "pareto_archive", []) or [])),
            anchorMhc=self._safe_float(getattr(self.s, "MHC", None)),
            anchorCr=self._safe_float(getattr(self.s, "CR", None)),
            anchorDecisionScore=self._safe_float(getattr(self.s, "decision_score", None)),
            **details,
        )
        return True

    def _prepare_episode_start(self, episode_idx):
        super()._prepare_episode_start(episode_idx)
        if not bool(getattr(self, "bimo_archive_anchor_selection_enabled", True)):
            return
        interval = int(max(1, getattr(self, "bimo_archive_anchor_switch_interval", 1) or 1))
        last_episode = int(getattr(self, "_bimo_last_anchor_episode", -10**9) or -10**9)
        if int(episode_idx) != 0 and (int(episode_idx) - last_episode) < interval:
            return
        self._restart_from_bimo_archive_anchor(episode_idx)

    def _observe_bimo_bootstrap_candidate(self, candidate):
        self._evaluate_solution(candidate)
        if not bool(getattr(candidate, "current_is_feasible", False)):
            return False
        return bool(self._observe_feasible_state(candidate))

    def _bootstrap_bimo_initial_archive(self, max_attempts=None):
        target_archive_size = int(max(1, getattr(self, "bimo_archive_bootstrap_size", 32) or 32))
        pool_target = int(max(1, getattr(self, "bimo_candidate_pool_bootstrap_target", target_archive_size) or target_archive_size))
        attempt_factor = int(max(1, getattr(self, "bimo_archive_bootstrap_attempt_factor", 8) or 8))
        requested_attempts = 0 if max_attempts is None else int(max_attempts)
        max_total_attempts = int(max(pool_target * attempt_factor, requested_attempts, pool_target))

        self._reset_baseline_archive_state()
        self._reset_bimo_archive_anchor_state(reset_bootstrap=False)
        self._ensure_bimo_candidate_pool_state()

        attempts = 0
        inserted_count = 0
        pool_inserted = 0
        best_candidate = self._light_clone_solution(self.s)
        best_score = math.inf

        initial_pool_before = self._bimo_candidate_pool_size()
        if self._observe_bimo_bootstrap_candidate(best_candidate):
            inserted_count += 1
            best_score = self._bimo_candidate_score(best_candidate)
            if self._bimo_candidate_pool_size() <= initial_pool_before and self._observe_bimo_candidate_pool(
                best_candidate,
                source="bootstrap_seed",
            ):
                pool_inserted += 1

        recipes = list(getattr(self, "bootstrap_recipes", []) or [])
        if not recipes:
            recipes = [getattr(self, "light_restart_recipe", []), getattr(self, "diversify_recipe", [])]
        recipes = [list(recipe) for recipe in recipes if recipe is not None]
        restart_interval = max(1, len(recipes) + 1)

        # 终止条件：候选池达到目标（同时 Pareto archive 也会随之增长）
        while self._bimo_candidate_pool_size() < pool_target and attempts < max_total_attempts:
            attempts += 1
            if attempts % restart_interval == 0:
                candidate = copy.deepcopy(self.env)
                candidate.reset()
                self._evaluate_solution(candidate)
            else:
                base_candidates = self._bimo_anchor_candidate_pool(
                    include_archive=True, include_candidate_pool=True
                )
                if base_candidates:
                    base = base_candidates[int(np.random.randint(0, len(base_candidates)))]
                else:
                    base = best_candidate
                recipe = recipes[(attempts - 1) % len(recipes)] if recipes else []
                candidate = self._generate_candidate_by_recipe(base, recipe)

            candidate_score = self._bimo_candidate_score(candidate)
            if candidate_score < best_score:
                best_candidate = copy.deepcopy(candidate)
                best_score = float(candidate_score)

            if getattr(candidate, "current_is_feasible", False):
                before_size = len(self.pareto_archive)
                pool_before = self._bimo_candidate_pool_size()
                changed = self._observe_feasible_state(candidate)
                if self._bimo_candidate_pool_size() <= pool_before:
                    self._observe_bimo_candidate_pool(candidate, source="bootstrap_candidate")
                if changed or len(self.pareto_archive) > before_size:
                    inserted_count += 1
                if self._bimo_candidate_pool_size() > pool_before:
                    pool_inserted += 1

        if self.pareto_archive:
            self._refresh_archive_state()
            self.s = copy.deepcopy(self.representative_solution or self.pareto_archive[0])
            self._evaluate_solution(self.s)
            self.current_energy = self.s.fitness
        else:
            self.s = copy.deepcopy(best_candidate)
            self._evaluate_solution(self.s)
            self.current_energy = self.s.fitness

        self._record_mo_event(
            "bimo_archive_bootstrap",
            targetSize=int(target_archive_size),
            poolTargetSize=int(pool_target),
            attempts=int(attempts),
            maxAttempts=int(max_total_attempts),
            insertedCount=int(inserted_count),
            candidatePoolInsertedCount=int(pool_inserted),
            archiveSize=int(len(getattr(self, "pareto_archive", []) or [])),
            candidatePoolSize=self._bimo_candidate_pool_size(),
            success=bool(self.pareto_archive),
        )
        return bool(self.pareto_archive)

    def _bootstrap_until_first_feasible(self, max_attempts=None):
        if bool(getattr(self, "bimo_archive_bootstrap_enabled", True)) and not bool(
            getattr(self, "_bimo_archive_bootstrap_done", False)
        ):
            self._bimo_archive_bootstrap_done = True
            if self._bootstrap_bimo_initial_archive(max_attempts=max_attempts):
                return self._activate_main_search_from_feasible()
        return super()._bootstrap_until_first_feasible(max_attempts=max_attempts)

    def _sync_solution_metrics(self, solution, metrics):
        solution.fac_x = metrics["fac_x"]
        solution.fac_y = metrics["fac_y"]
        solution.fac_b = metrics["fac_b"]
        solution.fac_h = metrics["fac_h"]
        solution.fac_aspect_ratio = metrics["fac_aspect_ratio"]
        solution.lower_bounds = metrics["lower_bounds"]
        solution.upper_bounds = metrics["upper_bounds"]
        solution.aspect_limits = metrics["aspect_limits"]
        solution.infeasible_mask = metrics["infeasible_mask"]
        solution.D = metrics["D"]
        solution.TM = metrics["TM"]
        solution.MHC = float(metrics["mhc"])
        solution.CR = float(metrics["cr"])
        solution.DR = 0.0
        solution.AR = 0.0
        solution.raw_cost = float(metrics["cost"])
        solution.mo_objectives_raw = np.asarray(metrics["mo_objectives_raw"], dtype=float)
        solution.mo_objectives_min = np.asarray(metrics["mo_objectives_min"], dtype=float)
        solution.constraint_violation = float(metrics["constraint_violation"])
        solution.current_d_inf = int(metrics["d_inf"])
        solution.current_is_feasible = bool(metrics["is_feasible"])
        solution.feasible_solution_count = getattr(self, "feasible_solution_count", 0)
        solution.best_feasible_cost = getattr(self, "best_feasible_cost", math.inf)
        solution.worst_feasible_cost = getattr(self, "worst_feasible_cost", None)
        solution.best_fitness = getattr(self, "best_feasible_cost", math.inf)
        solution.current_v_worst = getattr(self, "worst_feasible_cost", None)
        self._refresh_solution_search_metrics(solution)
        solution.state = solution.constructState()

    def _evaluate_solution(self, solution):
        metrics = FBSUtil.evaluate_layout_fast(
            solution.fbs_model,
            solution.areas,
            solution.H,
            solution.F,
            solution.aspect_limits,
            v_worst=self.mo_worst_feasible_mhc,
            k_penalty=self.k_penalty,
            distance_metric="manhattan",
        )
        objectives_raw = MO_FBSUtil_BiMO4.calculate_objectives(
            metrics["fac_x"],
            metrics["fac_y"],
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["mhc"],
            len(metrics["fac_x"]),
            rel_matrix=self.rel_matrix,
        )
        constraint_violation = MO_FBSUtil_BiMO4.calculate_total_constraint_violation(
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["lower_bounds"],
            metrics["upper_bounds"],
        )
        objectives_min = MO_FBSUtil_BiMO4.to_minimization(objectives_raw)
        self._update_running_objective_bounds(objectives_min)
        metrics.update(
            {
                "cr": float(objectives_raw[1]),
                "dr": 0.0,
                "ar": 0.0,
                "mo_objectives_raw": np.asarray(objectives_raw, dtype=float),
                "mo_objectives_min": objectives_min,
                "constraint_violation": float(constraint_violation),
            }
        )
        self._sync_solution_metrics(solution, metrics)
        return metrics

    def _observe_feasible_state(self, solution):
        if not getattr(solution, "current_is_feasible", False):
            self._last_archive_observation = {"archive_changed": False, "rep_changed": False}
            return False

        # 候选池插入（先于 Pareto 档案更新）
        self._observe_bimo_candidate_pool(solution, source="observe_feasible")

        self.feasible_solution_count += 1
        self.mo_worst_feasible_mhc = (
            float(solution.MHC)
            if self.mo_worst_feasible_mhc is None
            else max(float(self.mo_worst_feasible_mhc), float(solution.MHC))
        )

        previous_rep_vector = None
        previous_rep_score = float(self.representative_decision_score)
        if self.representative_solution is not None and getattr(self.representative_solution, "mo_objectives_min", None) is not None:
            previous_rep_vector = np.asarray(self.representative_solution.mo_objectives_min, dtype=float)

        updated_archive, inserted, removed = MO_FBSUtil_BiMO4.update_pareto_archive(
            self.pareto_archive,
            solution,
            max_size=self.archive_limit,
            clone_fn=copy.deepcopy,
        )
        archive_changed = bool(inserted)
        if archive_changed:
            self.pareto_archive = updated_archive
        elif not self.pareto_archive and getattr(solution, "current_is_feasible", False):
            self.pareto_archive = [copy.deepcopy(solution)]
            archive_changed = True
        if archive_changed:
            self.archive_update_count = int(getattr(self, "archive_update_count", 0) or 0) + 1

        if archive_changed or self._dynamic_weight_refresh_due():
            self._refresh_archive_state()
        self._refresh_solution_search_metrics(solution)

        current_rep_vector = None
        if self.representative_solution is not None and getattr(self.representative_solution, "mo_objectives_min", None) is not None:
            current_rep_vector = np.asarray(self.representative_solution.mo_objectives_min, dtype=float)
        rep_changed = False
        if previous_rep_vector is None and current_rep_vector is not None:
            rep_changed = True
        elif previous_rep_vector is not None and current_rep_vector is None:
            rep_changed = True
        elif previous_rep_vector is not None and current_rep_vector is not None:
            rep_changed = not np.allclose(previous_rep_vector, current_rep_vector, atol=1e-9, rtol=1e-7)
        rep_score_changed = abs(float(self.representative_decision_score) - previous_rep_score) > 1e-12
        rep_changed = bool(rep_changed or rep_score_changed)

        if self.representative_solution is not None and (
            rep_changed or self.best_feasible_solution is None
        ):
            self.best_feasible_solution = copy.deepcopy(self.representative_solution)
            self.gbest = copy.deepcopy(self.representative_solution)
            self.true_gbest = copy.deepcopy(self.representative_solution)
            self.best_feasible_cost = float(self.representative_decision_score)
            self.best_energy = float(self.representative_decision_score)
        elif (
            self.representative_solution is None
            and np.isfinite(getattr(solution, "decision_score", math.inf))
            and (
                self.best_feasible_solution is None
                or float(solution.decision_score) < float(getattr(self, "best_feasible_cost", math.inf))
            )
        ):
            self.best_feasible_solution = copy.deepcopy(solution)
            self.gbest = copy.deepcopy(solution)
            self.true_gbest = copy.deepcopy(solution)
            self.best_feasible_cost = float(solution.decision_score)
            self.best_energy = float(solution.decision_score)

        self._last_archive_observation = {
            "archive_changed": bool(archive_changed),
            "rep_changed": bool(rep_changed),
            "removed_count": int(removed),
        }
        return bool(archive_changed)

    def _archive_item_payload(self, solution, index):
        solution_array = getattr(getattr(solution, "fbs_model", None), "array_2d", None)
        if hasattr(solution_array, "tolist"):
            solution_array = solution_array.tolist()
        return {
            "index": int(index),
            "decisionScore": float(getattr(solution, "decision_score", math.inf)),
            "searchEnergy": float(getattr(solution, "fitness", math.inf)),
            "mhc": float(getattr(solution, "MHC", math.inf)),
            "cr": float(getattr(solution, "CR", 0.0)),
            "dInf": int(getattr(solution, "current_d_inf", 0) or 0),
            "constraintViolation": float(getattr(solution, "constraint_violation", 0.0) or 0.0),
            "isFeasible": bool(getattr(solution, "current_is_feasible", False)),
            "moObjectivesRaw": np.asarray(getattr(solution, "mo_objectives_raw", []), dtype=float).tolist(),
            "moObjectivesMin": np.asarray(getattr(solution, "mo_objectives_min", []), dtype=float).tolist(),
            "solution": solution_array,
        }

    def _save_pareto_archive(self, start_time, algorithm_name=None):
        if not self.pareto_archive:
            self.pareto_archive_path = None
            return None

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        archive_dir = os.path.join(os.path.abspath(config.RESULT_PATH), "pareto_archives")
        os.makedirs(archive_dir, exist_ok=True)
        timestamp = (
            start_time.strftime("%Y%m%d_%H%M%S_%f")
            if start_time is not None
            else datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        algo_tag = self._normalize_algorithm_tag(algorithm_name)
        archive_path = os.path.join(archive_dir, f"{self.instance_name}-{algo_tag}-{timestamp}.json")
        payload = {
            "instance": self.instance_name,
            "algorithm": algo_tag,
            "objectiveDefinitionVersion": self.OBJECTIVE_DEFINITION_VERSION,
            "objectiveNames": ["MHC", "CR"],
            "objectiveDirections": ["min", "max"],
            "generatedAt": datetime.datetime.now().isoformat(timespec="seconds"),
            "archiveSize": len(self.pareto_archive),
            "representativeArchiveIndex": None if self.representative_archive_index is None else int(self.representative_archive_index) + 1,
            "representativeDecisionScore": None if not np.isfinite(self.representative_decision_score) else float(self.representative_decision_score),
            "items": [self._archive_item_payload(solution, index + 1) for index, solution in enumerate(self.pareto_archive)],
        }
        with open(archive_path, "w", encoding="utf-8") as file_obj:
            import json
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        self.pareto_archive_path = os.path.relpath(archive_path, repo_root).replace("\\", "/")
        return self.pareto_archive_path

    def _ensure_pymoo_available(self):
        if NSGA2 is not None and MOEAD is not None and SPEA2 is not None:
            return
        raise ImportError(
            "缺少 pymoo 依赖，无法运行 NSGA-II/MOEA-D/SPEA2 双目标基线。请先安装 `pymoo>=0.6.1`。"
        ) from _PYMOO_IMPORT_ERROR

    def run_moea_baseline(
        self,
        algorithm_name,
        population_size=64,
        generations=80,
        sequence_length=None,
        seed=None,
    ):
        algo_key = str(algorithm_name or "").strip().lower().replace("-", "").replace("/", "")
        previous_mo_util = mo4_module.MO_FBSUtil
        mo4_module.MO_FBSUtil = MO_FBSUtil_BiMO4
        try:
            self._ensure_pymoo_available()
            if algo_key not in {"nsga2", "moead", "spea2"}:
                raise ValueError(f"Unsupported baseline algorithm for BiMO: {algorithm_name}")

            population_size = int(max(8, population_size))
            generations = int(max(1, generations))
            sequence_length = int(max(1, sequence_length if sequence_length is not None else self.t_max))
            run_seed = None if seed is None else int(seed)
            wall_time_limit_seconds = float(max(getattr(self, "wall_time_limit_seconds", 0.0) or 0.0, 0.0))

            self._reset_baseline_archive_state()
            start_time = datetime.datetime.now()
            fast_time = start_time

            base_solution = self._light_clone_solution(self.s)
            self._evaluate_solution(base_solution)
            self._observe_feasible_state(base_solution)

            problem = _ActionSequenceBiMOProblem(
                solver=self,
                base_solution=base_solution,
                sequence_length=sequence_length,
                use_constraints=(algo_key != "moead"),
            )
            sampling = _ActionSequenceSampling()
            crossover = _ActionSequenceUniformCrossover(swap_prob=0.5)
            mutation = _ActionSequenceMutation(mutation_prob=1.0 / float(max(1, sequence_length)))

            if algo_key == "nsga2":
                algorithm = NSGA2(
                    pop_size=population_size,
                    sampling=sampling,
                    crossover=crossover,
                    mutation=mutation,
                    eliminate_duplicates=True,
                )
                effective_population = population_size
            elif algo_key == "spea2":
                algorithm = SPEA2(
                    pop_size=population_size,
                    sampling=sampling,
                    crossover=crossover,
                    mutation=mutation,
                    survival=SPEA2Survival(normalize=False),
                    eliminate_duplicates=True,
                )
                effective_population = population_size
            else:
                n_partitions = self._compute_moead_partitions(population_size, objective_count=2)
                ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=n_partitions)
                algorithm = MOEAD(
                    ref_dirs=ref_dirs,
                    n_neighbors=min(20, max(2, len(ref_dirs) - 1)),
                    prob_neighbor_mating=0.7,
                    sampling=sampling,
                    crossover=crossover,
                    mutation=mutation,
                )
                effective_population = int(len(ref_dirs))

            termination = get_termination("n_gen", generations)
            termination_mode = "n_gen"
            if wall_time_limit_seconds > 0.0:
                from pymoo.core.termination import TerminateIfAny

                termination = TerminateIfAny(
                    get_termination("n_gen", generations),
                    get_termination("time", wall_time_limit_seconds),
                )
                termination_mode = "n_gen_or_time"

            optimize_start = time.perf_counter()
            result = minimize(
                problem,
                algorithm,
                termination=termination,
                seed=run_seed,
                save_history=False,
                verbose=False,
            )
            optimize_runtime_seconds = float(max(time.perf_counter() - optimize_start, 0.0))
            actual_generations = int(
                max(
                    1,
                    int(
                        getattr(
                            getattr(result, "algorithm", None),
                            "n_gen",
                            getattr(algorithm, "n_gen", generations),
                        )
                        or generations
                    ),
                )
            )
            wall_time_terminated = bool(
                wall_time_limit_seconds > 0.0
                and optimize_runtime_seconds + 1e-9 >= max(wall_time_limit_seconds - 1.0, wall_time_limit_seconds * 0.98)
            )

            best_observed = self.best_feasible_cost
            for sequence in self._collect_result_sequences(result):
                candidate = self._evaluate_action_sequence(base_solution, sequence)
                changed = self._observe_feasible_state(candidate)
                if changed and np.isfinite(self.best_feasible_cost) and self.best_feasible_cost < best_observed:
                    best_observed = float(self.best_feasible_cost)
                    fast_time = datetime.datetime.now()

            self._refresh_archive_state()
            end_time = datetime.datetime.now()
            iteration_count = int(effective_population * actual_generations)

            best_solution = self.best_feasible_solution if self.best_feasible_solution is not None else self._light_clone_solution(base_solution)
            if self.representative_solution is not None:
                best_solution = copy.deepcopy(self.representative_solution)
            is_valid = bool(getattr(best_solution, "current_is_feasible", False))
            best_energy = float(getattr(best_solution, "decision_score", math.inf))
            if not np.isfinite(best_energy):
                best_energy = float(self.best_feasible_cost if np.isfinite(self.best_feasible_cost) else best_solution.fitness)

            stable_decision_score = self._safe_float(getattr(best_solution, "proxy_energy", None))
            archive_algo_name = f"MO_BASELINE_{algo_key.upper()}"
            archive_path = self._save_pareto_archive(start_time, algorithm_name=archive_algo_name)
            reference_metrics = self._compute_reference_front_metrics()
            self.last_run_payload = {
                "pareto_archive_path": archive_path,
                "pareto_size": len(self.pareto_archive),
                "rep_mhc": None if best_solution is None else float(getattr(best_solution, "MHC", math.inf)),
                "rep_cr": None if best_solution is None else float(getattr(best_solution, "CR", 0.0)),
                "rep_dr": None,
                "rep_ar": None,
                "decision_score": self._safe_float(best_energy),
                "stable_decision_score": stable_decision_score,
                "archive_hypervolume": reference_metrics["archive_hypervolume"],
                "archive_spacing": reference_metrics["archive_spacing"],
                "archive_igd": None,
                "reference_front_path": None,
                "reference_front_size": None,
                "reference_front_archive_count": None,
                "archive_hypervolume_mode": reference_metrics["archive_hypervolume_mode"],
                "archive_hypervolume_reference_point": None,
                "mo_run_id": None,
                "mo_bundle_dir": None,
                "mo_trace_path": None,
                "mo_events_path": None,
                "mo_action_stats_path": None,
                "mo_run_summary_path": None,
                "wall_time_terminated": bool(wall_time_terminated),
                "wall_time_limit_seconds": self._safe_float(wall_time_limit_seconds),
                "runtime_seconds": self._safe_float(optimize_runtime_seconds),
                "baseline_algorithm": algo_key.upper(),
                "baseline_population": int(effective_population),
                "baseline_generations": int(actual_generations),
                "baseline_generations_requested": int(generations),
                "baseline_sequence_length": int(sequence_length),
                "baseline_seed": run_seed,
                "baseline_termination_mode": termination_mode,
            }
            return iteration_count, is_valid, best_solution, best_energy, start_time, end_time, fast_time
        finally:
            mo4_module.MO_FBSUtil = previous_mo_util


def _format_summary_metrics(solver, best_energy):
    payload = getattr(solver, "last_run_payload", {}) or {}
    hv = payload.get("archive_hypervolume")
    spacing = payload.get("archive_spacing")

    def _fmt(value):
        return "NA" if value is None else f"{float(value):.6f}"

    return (
        f"representative decision score: {float(best_energy):.6f} | "
        f"HV: {_fmt(hv)} | Spacing: {_fmt(spacing)}"
    )


def _parse_env_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw.strip())
    except Exception as exc:
        raise ValueError(f"环境变量 {name} 必须是整数，当前值: {raw!r}") from exc


def _parse_env_float(name, default):
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw.strip())
    except Exception as exc:
        raise ValueError(f"环境变量 {name} 必须是浮点数，当前值: {raw!r}") from exc


def _parse_env_flag(name, default):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "y"}:
        return True
    if value in {"0", "false", "no", "off", "n"}:
        return False
    raise ValueError(f"环境变量 {name} 必须是布尔值，当前值: {raw!r}")


def _parse_env_int_list(name):
    raw = os.getenv(name)
    if raw is None:
        return []
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except Exception as exc:
            raise ValueError(f"环境变量 {name} 包含非法整数: {token!r}") from exc
    return values


def _preflight_required_files(instance_name=None):
    required_paths = [Path(config.FILE_PATH)]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        formatted = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"预检查失败，缺少以下必需文件：\n{formatted}")
    if instance_name:
        ELP._ensure_cr_matrix_available(instance_name)


if __name__ == "__main__":
    exp_instance = os.getenv("ELP_EXP_INSTANCE", "Du62")
    _preflight_required_files(exp_instance)
    baseline_algo = os.getenv("ELP_MO_BASELINE_ALGO", "").strip().lower()
    baseline_enabled = baseline_algo in {"nsga2", "moead", "spea2"}
    default_algorithm = f"MO_BASELINE_{baseline_algo.upper()}" if baseline_enabled else "ELP_DRL_BiMO4"
    exp_algorithm = os.getenv("ELP_EXP_ALGORITHM", default_algorithm)
    default_remark = "WarmStart(GA)+ELP+Bi-objective Pareto archive (MHC min, CR max)"
    exp_remark = os.getenv("ELP_EXP_REMARK", default_remark)
    exp_number = _parse_env_int("ELP_EXP_NUMBER", 30)
    is_exp = _parse_env_flag("ELP_IS_EXP", True)

    G = _parse_env_int("ELP_G", 1000)
    t_max = _parse_env_int("ELP_T_MAX", 300)
    T_initial = _parse_env_float("ELP_T_INITIAL", 10000.0)
    k_hist = _parse_env_float("ELP_K_HIST", 10.0)
    base_seed = _parse_env_int("ELP_BASE_SEED", 20260427)
    fixed_seeds = _parse_env_int_list("ELP_FIXED_SEEDS")
    if fixed_seeds:
        exp_number = len(fixed_seeds)

    baseline_population = _parse_env_int("ELP_MO_BASELINE_POP", 64)
    baseline_generations = _parse_env_int("ELP_MO_BASELINE_GEN", 80)
    baseline_sequence_length = _parse_env_int("ELP_MO_BASELINE_SEQ_LEN", t_max)

    def _run_once(run_index):
        if fixed_seeds:
            run_seed = int(fixed_seeds[run_index])
        else:
            run_seed = int(base_seed + run_index)
        strict_determinism = _set_global_seed(run_seed)
        logger.info(f"Experiment seed: {run_seed} | strict_determinism: {strict_determinism}")
        env = gym.make("FbsEnv-v0", instance=exp_instance)
        try:
            env.reset(seed=run_seed)
        except TypeError:
            env.reset()
        except Exception:
            env.reset()
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        initial_gbest = copy.deepcopy(base_env)
        logger.info(f"Initial solution energy: {_get_initial_solution_energy(base_env)}")
        solver = ELP(
            env=base_env,
            gbest=initial_gbest,
            T=T_initial,
            G=G,
            t_max=t_max,
            k=k_hist,
        )
        if baseline_enabled:
            return solver, solver.run_moea_baseline(
                algorithm_name=baseline_algo,
                population_size=baseline_population,
                generations=baseline_generations,
                sequence_length=baseline_sequence_length,
                seed=run_seed,
            )
        return solver, solver.run()

    if is_exp:
        for i in range(exp_number):
            logger.info(f"Starting experiment {i + 1} for {exp_algorithm}")
            try:
                elp_solver, result_tuple = _run_once(i)
                total_iter, is_valid, best_sol, best_energy, start, end, fast = result_tuple
                logger.info(f"Experiment {i + 1} complete | {_format_summary_metrics(elp_solver, best_energy)}")
                if not baseline_enabled:
                    for telemetry_line in elp_solver.format_action_telemetry():
                        logger.info(f"Telemetry | {telemetry_line}")
                _save_experiment_row(
                    exp_instance,
                    exp_algorithm,
                    exp_remark,
                    total_iter,
                    is_valid,
                    best_sol,
                    best_energy,
                    start,
                    end,
                    fast,
                    elp_solver,
                )
            except Exception as exc:
                logger.exception(f"Experiment {i + 1} failed: {exc}")
                raise
    else:
        elp_solver, result_tuple = _run_once(0)
        total_iter, is_valid, best_sol, best_energy, start, end, fast = result_tuple
        print(f"Single run complete | {_format_summary_metrics(elp_solver, best_energy)}")
        if not baseline_enabled:
            for telemetry_line in elp_solver.format_action_telemetry():
                print(telemetry_line)
        _save_experiment_row(
            exp_instance,
            exp_algorithm,
            exp_remark,
            total_iter,
            is_valid,
            best_sol,
            best_energy,
            start,
            end,
            fast,
            elp_solver,
        )

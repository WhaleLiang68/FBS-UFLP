import copy
import datetime
import json
import math
import os
from pathlib import Path

import gym
import numpy as np
from loguru import logger

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except Exception:
    SummaryWriter = None
    _TENSORBOARD_AVAILABLE = False

import src
import src.utils.ExperimentsUtil as ExperimentsUtil
import src.utils.FBSUtil as FBSUtil
import src.utils.MO_ExperimentsUtil as MO_ExperimentsUtil
import src.utils.MO_ReferenceFrontUtil as MO_ReferenceFrontUtil
import src.utils.config as config
from src.utils.FBSModel import FBSModel
from src.algorithms.ELP_DRL_Standard import ELP as StandardELP
from src.algorithms.ELP_DRL_Standard import StandardDQNAgent
from src.algorithms.ELP_DRL_Standard import StandardQLearningAgent
from src.algorithms.ELP_DRL_Standard import _get_initial_solution_energy
from src.algorithms.ELP_DRL_Standard import _set_global_seed
from src.utils.MO_DataGenerator import MO_DataGenerator
from src.utils.MO_FBSUtil_MO4 import MO_FBSUtil

_PYMOO_IMPORT_ERROR = None

try:
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.spea2 import SPEA2, SPEA2Survival
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.core.problem import Problem
    from pymoo.core.sampling import Sampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.util.ref_dirs import get_reference_directions
except Exception as exc:
    MOEAD = None
    NSGA2 = None
    SPEA2 = None
    SPEA2Survival = None
    Crossover = object
    Mutation = object
    Problem = object
    Sampling = object
    minimize = None
    get_termination = None
    get_reference_directions = None
    _PYMOO_IMPORT_ERROR = exc

np.bool8 = np.bool_


class _ActionSequenceSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        n_samples = int(max(1, n_samples))
        lower = np.asarray(problem.xl, dtype=int)
        upper = np.asarray(problem.xu, dtype=int)
        random_values = np.random.randint(lower[None, :], upper[None, :] + 1, size=(n_samples, problem.n_var))
        return random_values.astype(int)


class _ActionSequenceUniformCrossover(Crossover):
    def __init__(self, swap_prob=0.5):
        super().__init__(2, 2)
        self.swap_prob = float(np.clip(swap_prob, 0.0, 1.0))

    def _do(self, problem, X, **kwargs):
        n_matings = int(X.shape[1])
        n_var = int(X.shape[2])
        parent_a = X[0].astype(int)
        parent_b = X[1].astype(int)
        swap_mask = np.random.rand(n_matings, n_var) < self.swap_prob
        child_a = np.where(swap_mask, parent_b, parent_a)
        child_b = np.where(swap_mask, parent_a, parent_b)
        return np.stack([child_a, child_b], axis=0).astype(int)


class _ActionSequenceMutation(Mutation):
    def __init__(self, mutation_prob=None):
        super().__init__()
        self.mutation_prob = mutation_prob

    def _do(self, problem, X, **kwargs):
        values = np.asarray(X, dtype=int).copy()
        rows, cols = values.shape
        per_gene_prob = self.mutation_prob
        if per_gene_prob is None:
            per_gene_prob = 1.0 / float(max(1, problem.n_var))
        per_gene_prob = float(np.clip(per_gene_prob, 0.0, 1.0))
        mutate_mask = np.random.rand(rows, cols) < per_gene_prob
        if np.any(mutate_mask):
            lower = np.asarray(problem.xl, dtype=int)[None, :]
            upper = np.asarray(problem.xu, dtype=int)[None, :]
            random_values = np.random.randint(lower, upper + 1, size=(rows, cols))
            values[mutate_mask] = random_values[mutate_mask]
        return values


class _ActionSequenceMOProblem(Problem):
    def __init__(self, solver, base_solution, sequence_length, use_constraints=True):
        self.solver = solver
        self.base_solution = solver._light_clone_solution(base_solution)
        self.action_count = int(len(solver.valid_actions))
        self.use_constraints = bool(use_constraints)
        super().__init__(
            n_var=int(max(1, sequence_length)),
            n_obj=4,
            n_ieq_constr=2 if self.use_constraints else 0,
            xl=np.zeros(int(max(1, sequence_length)), dtype=int),
            xu=np.full(int(max(1, sequence_length)), self.action_count - 1, dtype=int),
            vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        sequences = np.asarray(X, dtype=int)
        sample_count = int(sequences.shape[0])
        objectives = np.zeros((sample_count, 4), dtype=float)
        constraints = np.zeros((sample_count, 2), dtype=float)
        for idx in range(sample_count):
            candidate = self.solver._evaluate_action_sequence(self.base_solution, sequences[idx])
            objective_vector = np.asarray(candidate.mo_objectives_min, dtype=float)[:4]
            is_feasible = bool(getattr(candidate, "current_is_feasible", False))
            d_inf = int(getattr(candidate, "current_d_inf", 0) or 0)
            violation = max(float(getattr(candidate, "constraint_violation", 0.0) or 0.0), 0.0)
            if self.use_constraints:
                objectives[idx, :] = objective_vector
                constraints[idx, 0] = 0.0 if is_feasible else 1.0
                constraints[idx, 1] = violation
            else:
                # MOEA/D 在 pymoo 中不支持显式约束，改为统一罚函数保持可行性优先。
                penalty = 0.0 if is_feasible else (1_000_000.0 + 10_000.0 * max(d_inf, 0) + violation)
                objectives[idx, :] = objective_vector + float(penalty)
        out["F"] = objectives
        if self.use_constraints:
            out["G"] = constraints


class ELP(StandardELP):
    """Multi-objective ELP variant that keeps a Pareto archive and representative solution."""

    def __init__(self, env, gbest, T, G=100, t_max=50, k=2.0, archive_limit=64, objective_weights=None):
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.instance_name = str(getattr(base_env, "instance", "UNKNOWN") or "UNKNOWN")
        self.mo_weights = np.asarray(
            objective_weights if objective_weights is not None else [0.25, 0.25, 0.25, 0.25],
            dtype=float,
        ).reshape(-1)
        if self.mo_weights.size < 4:
            self.mo_weights = np.pad(self.mo_weights, (0, 4 - self.mo_weights.size), constant_values=0.25)
        self.mo_weights = self.mo_weights[:4]
        if not np.any(self.mo_weights > 0):
            self.mo_weights = np.full(4, 0.25, dtype=float)
        self.mo_weights = self.mo_weights / np.sum(self.mo_weights)

        self.archive_limit = int(max(4, archive_limit))
        self.fast_segment_insert_budget = 8
        self.fast_segment_insert_segment_lengths = (2, 3)
        self.pareto_archive = []
        self.representative_solution = None
        self.representative_decision_score = math.inf
        self.representative_archive_index = None
        self.mo_ideal = None
        self.mo_nadir = None
        self.pareto_archive_path = None
        self.last_run_payload = None
        self.mo_run_summary = None
        self.mo_worst_feasible_mhc = None
        self._last_transition_meta = {}
        self.wall_time_limit_seconds = max(
            0.0,
            float(os.getenv("ELP_WALL_TIME_LIMIT_SECONDS", "0") or 0.0),
        )
        self.mo_trace_interval = int(max(1, int(os.getenv("ELP_MO_TRACE_INTERVAL", "1000"))))
        self.agent_mode = None
        self.mo_nondominated_accept_cap_high = float(os.getenv("ELP_MO_ND_ACCEPT_CAP_HIGH", "0.90"))
        self.mo_nondominated_accept_cap_low = float(os.getenv("ELP_MO_ND_ACCEPT_CAP_LOW", "0.08"))
        self.mo_nondominated_accept_cap_low = max(0.0, min(self.mo_nondominated_accept_cap_low, 1.0))
        self.mo_nondominated_accept_cap_high = max(0.0, min(self.mo_nondominated_accept_cap_high, 1.0))
        if self.mo_nondominated_accept_cap_high < self.mo_nondominated_accept_cap_low:
            self.mo_nondominated_accept_cap_high = self.mo_nondominated_accept_cap_low
        # 非支配接受率分段参数：前期保探索，后期收敛
        self.mo_nondominated_accept_cap_early_floor = float(os.getenv("ELP_MO_ND_ACCEPT_CAP_EARLY_FLOOR", "0.55"))
        self.mo_nondominated_accept_cap_mid_floor = float(os.getenv("ELP_MO_ND_ACCEPT_CAP_MID_FLOOR", "0.25"))
        self.mo_nondominated_accept_cap_late_max_start = float(
            os.getenv("ELP_MO_ND_ACCEPT_CAP_LATE_MAX_START", "0.35")
        )
        self.mo_nondominated_accept_cap_late_max_end = float(
            os.getenv("ELP_MO_ND_ACCEPT_CAP_LATE_MAX_END", "0.18")
        )
        self.mo_nondominated_accept_cap_early_floor = max(
            self.mo_nondominated_accept_cap_low,
            min(self.mo_nondominated_accept_cap_early_floor, self.mo_nondominated_accept_cap_high),
        )
        self.mo_nondominated_accept_cap_mid_floor = max(
            self.mo_nondominated_accept_cap_low,
            min(self.mo_nondominated_accept_cap_mid_floor, self.mo_nondominated_accept_cap_high),
        )
        self.mo_nondominated_accept_cap_late_max_start = max(
            self.mo_nondominated_accept_cap_low,
            min(self.mo_nondominated_accept_cap_late_max_start, self.mo_nondominated_accept_cap_high),
        )
        self.mo_nondominated_accept_cap_late_max_end = max(
            self.mo_nondominated_accept_cap_low,
            min(self.mo_nondominated_accept_cap_late_max_end, self.mo_nondominated_accept_cap_high),
        )
        if self.mo_nondominated_accept_cap_late_max_start < self.mo_nondominated_accept_cap_late_max_end:
            self.mo_nondominated_accept_cap_late_max_start = self.mo_nondominated_accept_cap_late_max_end
        # 后段“非入档候选”门控：仅对互不支配且不改档案的候选降接受概率
        self.mo_late_nonarchive_gate_start = float(os.getenv("ELP_MO_LATE_NOARCHIVE_GATE_START", "0.88"))
        self.mo_late_nonarchive_prob_scale_end = float(os.getenv("ELP_MO_LATE_NOARCHIVE_PROB_SCALE_END", "0.70"))
        self.mo_late_nonarchive_prob_cap_start = float(os.getenv("ELP_MO_LATE_NOARCHIVE_PROB_CAP_START", "0.50"))
        self.mo_late_nonarchive_prob_cap_end = float(os.getenv("ELP_MO_LATE_NOARCHIVE_PROB_CAP_END", "0.28"))
        self.mo_late_nonarchive_stagnation_windows = int(
            max(1, int(os.getenv("ELP_MO_LATE_NOARCHIVE_STAGNATION_WINDOWS", "4")))
        )
        self.mo_late_nonarchive_stagnation_ramp_windows = int(
            max(
                1,
                int(
                    os.getenv(
                        "ELP_MO_LATE_NOARCHIVE_STAGNATION_RAMP_WINDOWS",
                        str(self.mo_late_nonarchive_stagnation_windows),
                    )
                ),
            )
        )
        self.mo_late_nonarchive_stagnation_min_progress = float(
            os.getenv("ELP_MO_LATE_NOARCHIVE_STAGNATION_MIN_PROGRESS", "0.70")
        )
        self.mo_late_nonarchive_gate_enabled = self._parse_env_flag(
            "ELP_MO_LATE_NOARCHIVE_GATE_ENABLE",
            False,
        )
        self.mo_late_nonarchive_gate_start = min(max(self.mo_late_nonarchive_gate_start, 0.0), 1.0)
        self.mo_late_nonarchive_prob_scale_end = min(max(self.mo_late_nonarchive_prob_scale_end, 0.0), 1.0)
        self.mo_late_nonarchive_prob_cap_start = min(max(self.mo_late_nonarchive_prob_cap_start, 0.0), 1.0)
        self.mo_late_nonarchive_prob_cap_end = min(max(self.mo_late_nonarchive_prob_cap_end, 0.0), 1.0)
        self.mo_late_nonarchive_stagnation_min_progress = min(
            max(self.mo_late_nonarchive_stagnation_min_progress, 0.0),
            1.0,
        )
        if self.mo_late_nonarchive_prob_cap_start < self.mo_late_nonarchive_prob_cap_end:
            self.mo_late_nonarchive_prob_cap_start = self.mo_late_nonarchive_prob_cap_end
        self.local_search_cooldown_steps = int(max(1, int(os.getenv("ELP_MO_LOCAL_SEARCH_COOLDOWN", "24"))))
        self.local_search_min_rel_improvement = max(
            0.0,
            float(os.getenv("ELP_MO_LOCAL_SEARCH_MIN_REL_IMPROVE", "0.01")),
        )
        self.local_search_disable_after_progress = float(os.getenv("ELP_MO_LOCAL_SEARCH_DISABLE_AFTER", "0.80"))
        self.local_search_disable_after_progress = min(max(self.local_search_disable_after_progress, 0.0), 1.0)
        self.last_local_search_step = -10**9
        self.local_search_backoff_enable = self._parse_env_flag("ELP_MO_LOCAL_SEARCH_BACKOFF_ENABLE", True)
        self.local_search_backoff_exp_cap = int(max(0, int(os.getenv("ELP_MO_LOCAL_SEARCH_BACKOFF_EXP_CAP", "4"))))
        self.local_search_cooldown_max_steps = int(
            max(
                self.local_search_cooldown_steps,
                int(
                    os.getenv(
                        "ELP_MO_LOCAL_SEARCH_COOLDOWN_MAX",
                        str(self.local_search_cooldown_steps * 16),
                    )
                ),
            )
        )
        self.archive_quality_gate_when_full = self._parse_env_flag("ELP_MO_ARCHIVE_QUALITY_GATE", True)
        self.archive_quality_hv_tol = max(0.0, float(os.getenv("ELP_MO_ARCHIVE_HV_TOL", "1e-10")))
        self.archive_quality_spacing_tol = max(0.0, float(os.getenv("ELP_MO_ARCHIVE_SPACING_TOL", "1e-8")))
        self.archive_quality_igd_tol = max(0.0, float(os.getenv("ELP_MO_ARCHIVE_IGD_TOL", "1e-8")))
        self.archive_spacing_guard_when_full = self._parse_env_flag("ELP_MO_ARCHIVE_SPACING_GUARD", True)
        self.archive_spacing_guard_rel_tol = max(0.0, float(os.getenv("ELP_MO_ARCHIVE_SPACING_GUARD_REL_TOL", "0.20")))
        self.archive_spacing_guard_hv_gain_rel = max(0.0, float(os.getenv("ELP_MO_ARCHIVE_SPACING_GUARD_HV_GAIN_REL", "0.02")))
        self.archive_require_candidate_retained = self._parse_env_flag("ELP_MO_ARCHIVE_REQUIRE_RETAINED", True)
        self.archive_reference_front_enabled = self._parse_env_flag("ELP_MO_ARCHIVE_REFERENCE_ENABLE", True)
        self.archive_reference_front_payload = None
        self.archive_reference_vectors = []
        self.archive_fixed_hv_reference_margin = max(
            0.0,
            float(os.getenv("ELP_MO_ARCHIVE_FIXED_HV_MARGIN", "0.1")),
        )
        # MO4 的在线阶段禁止重型满档质量门控，精确 HV/IGD 只用于最终评估。
        self.archive_quality_gate_when_full = False
        self.archive_spacing_guard_when_full = False

        self.mo_elite_multi_anchor_enabled = self._parse_env_flag("ELP_MO_ELITE_MULTI_ANCHOR_ENABLE", True)
        self.mo_elite_anchor_count = int(max(1, int(os.getenv("ELP_MO_ELITE_ANCHOR_COUNT", "3"))))
        self.mo_elite_sparse_anchor_count = int(max(0, int(os.getenv("ELP_MO_ELITE_SPARSE_ANCHOR_COUNT", "1"))))
        self.mo_elite_include_representative = self._parse_env_flag("ELP_MO_ELITE_INCLUDE_REPRESENTATIVE", True)
        self.mo_elite_include_extremes = self._parse_env_flag("ELP_MO_ELITE_INCLUDE_EXTREMES", True)
        self.mo_elite_include_sparse = self._parse_env_flag("ELP_MO_ELITE_INCLUDE_SPARSE", True)
        self.mo_elite_boundary_candidate_enabled = self._parse_env_flag("ELP_MO4_ELITE_BOUNDARY_CANDIDATE_ENABLE", True)
        self.mo_elite_boundary_min_distance = max(
            0.0,
            float(os.getenv("ELP_MO4_ELITE_BOUNDARY_MIN_DISTANCE", "0.045")),
        )
        self.mo_elite_boundary_rep_slack = max(
            0.0,
            float(os.getenv("ELP_MO4_ELITE_BOUNDARY_REP_SLACK", "0.22")),
        )
        self.mo_elite_boundary_extreme_gain_floor = max(
            0.0,
            float(os.getenv("ELP_MO4_ELITE_BOUNDARY_EXTREME_GAIN_FLOOR", "1e-4")),
        )
        self.mo_elite_boundary_preview_budget = int(
            max(0, int(os.getenv("ELP_MO4_ELITE_BOUNDARY_PREVIEW_BUDGET", "2")))
        )
        self.mo_elite_boundary_min_archive_size = int(
            max(0, int(os.getenv("ELP_MO4_ELITE_BOUNDARY_MIN_ARCHIVE_SIZE", "12")))
        )
        self.mo_elite_boundary_accept_budget = int(
            os.getenv("ELP_MO4_ELITE_BOUNDARY_ACCEPT_BUDGET", "-1")
        )
        self.mo_elite_boundary_quality_gate_enabled = self._parse_env_flag(
            "ELP_MO4_ELITE_BOUNDARY_QUALITY_GATE_ENABLE",
            False,
        )
        self.mo_elite_max_candidates_before_full = int(
            max(0, int(os.getenv("ELP_MO4_ELITE_MAX_CANDIDATES_BEFORE_FULL", "0")))
        )
        self.mo_elite_max_candidates_after_full = int(
            max(0, int(os.getenv("ELP_MO4_ELITE_MAX_CANDIDATES_PER_TRIGGER", "0")))
        )
        self.mo_elite_boundary_early_min_score = max(
            0.0,
            float(os.getenv("ELP_MO4_ELITE_BOUNDARY_EARLY_MIN_SCORE", "0.08")),
        )
        self.mo_elite_boundary_min_score = max(
            0.0,
            float(os.getenv("ELP_MO4_ELITE_BOUNDARY_MIN_SCORE", "0.12")),
        )
        self.mo_elite_boundary_strong_score = max(
            0.0,
            float(os.getenv("ELP_MO4_ELITE_BOUNDARY_STRONG_SCORE", "0.24")),
        )
        self.mo_elite_boundary_quality_rep_slack = max(
            0.0,
            float(os.getenv("ELP_MO4_ELITE_BOUNDARY_QUALITY_REP_SLACK", "0.18")),
        )

        facility_count = int(getattr(base_env, "n", len(getattr(base_env, "areas", [])) or 0))
        self.rel_matrix, self.dist_req_matrix = MO_DataGenerator.load_or_generate_data(
            facility_count,
            instance_name=self.instance_name,
        )

        super().__init__(env=env, gbest=gbest, T=T, G=G, t_max=t_max, k=k)
        self._reset_mo_logging_state()
        self.temperature_floor_samples = 16
        self.action_labels[10] = "fast_segment_insert"
        if 10 in self.action_telemetry:
            self.action_telemetry[10]["name"] = "fast_segment_insert"
        # MO4 不在在线主循环中使用参考前沿，只在最终评估阶段计算 IGD。

    @staticmethod
    def _parse_env_flag(name, default):
        raw_value = os.getenv(name)
        if raw_value is None:
            return bool(default)
        normalized = str(raw_value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)

    @staticmethod
    def _copy_bay_structure(bay_structure):
        return [
            [
                int(facility)
                for facility in (
                    current_bay.tolist() if isinstance(current_bay, np.ndarray) else current_bay
                )
            ]
            for current_bay in bay_structure
        ]

    @staticmethod
    def _safe_float(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return float(value)

    def _elapsed_seconds_since(self, start_time):
        if start_time is None:
            return None
        return (datetime.datetime.now() - start_time).total_seconds()

    def _wall_time_limit_reached(self, start_time):
        limit_seconds = float(getattr(self, "wall_time_limit_seconds", 0.0) or 0.0)
        if limit_seconds <= 0.0:
            return False
        elapsed_seconds = self._elapsed_seconds_since(start_time)
        return elapsed_seconds is not None and elapsed_seconds >= limit_seconds

    def _refresh_archive_reference_front(self, force_rebuild=False):
        if not bool(getattr(self, "archive_reference_front_enabled", False)):
            self.archive_reference_front_payload = None
            self.archive_reference_vectors = []
            return None
        payload = MO_ReferenceFrontUtil.ensure_instance_reference_front(
            self.instance_name,
            result_root=config.RESULT_PATH,
            force_rebuild=bool(force_rebuild),
        )
        self.archive_reference_front_payload = payload
        self.archive_reference_vectors = [
            np.asarray(item.get("moObjectivesMin"), dtype=float)
            for item in (payload.get("items") or [])
            if isinstance(item, dict) and item.get("moObjectivesMin") is not None
        ]
        return payload

    def _compute_reference_front_metrics(self):
        reference_payload = self._refresh_archive_reference_front(force_rebuild=True)
        archive_hypervolume = self._compute_fixed_reference_hypervolume(
            reference_payload=reference_payload,
            strict=True,
        )
        archive_spacing = self._compute_reference_front_spacing(
            reference_payload=reference_payload,
            strict=True,
        )
        archive_igd = self._safe_float(
            MO_ReferenceFrontUtil.compute_archive_igd(self.pareto_archive, reference_payload)
        )
        return {
            "archive_hypervolume": archive_hypervolume,
            "archive_spacing": archive_spacing,
            "archive_igd": archive_igd,
            "reference_front_path": None if reference_payload is None else str(reference_payload.get("referenceFrontPath") or ""),
            "reference_front_size": None if reference_payload is None else int(reference_payload.get("referenceFrontSize") or 0),
            "reference_front_archive_count": None if reference_payload is None else int(reference_payload.get("sourceArchiveCount") or 0),
            "archive_hypervolume_mode": "fixed_reference_front",
            "archive_hypervolume_reference_point": [
                1.0 + float(getattr(self, "archive_fixed_hv_reference_margin", 0.1) or 0.0)
            ] * 4,
        }

    def _reference_front_ideal_nadir(self, reference_payload, strict=False):
        if not reference_payload:
            if strict:
                raise RuntimeError("公共参考前沿为空，无法计算固定口径 HV/Spacing。")
            return None, None

        ideal = reference_payload.get("ideal")
        nadir = reference_payload.get("nadir")
        if ideal is None or nadir is None:
            if strict:
                raise RuntimeError("公共参考前沿缺少 ideal/nadir，无法计算固定口径 HV/Spacing。")
            return None, None

        ideal = np.asarray(ideal, dtype=float).reshape(-1)
        nadir = np.asarray(nadir, dtype=float).reshape(-1)
        if ideal.size < 4 or nadir.size < 4 or not np.all(np.isfinite(ideal[:4])) or not np.all(np.isfinite(nadir[:4])):
            if strict:
                raise RuntimeError("公共参考前沿 ideal/nadir 非法，无法计算固定口径 HV/Spacing。")
            return None, None
        return ideal[:4], nadir[:4]

    def _compute_fixed_reference_hypervolume(self, archive=None, reference_payload=None, strict=False):
        archive = self.pareto_archive if archive is None else archive
        ideal, nadir = self._reference_front_ideal_nadir(reference_payload, strict=strict)
        if ideal is None or nadir is None:
            return None
        normalized, _, _ = MO_FBSUtil._normalized_archive_matrix(archive, ideal=ideal, nadir=nadir)
        if normalized.size == 0:
            return 0.0
        margin = float(getattr(self, "archive_fixed_hv_reference_margin", 0.1) or 0.0)
        reference_point = np.full(int(normalized.shape[1]), 1.0 + margin, dtype=float)
        return self._safe_float(MO_FBSUtil._union_hypervolume(normalized, reference_point))

    def _compute_reference_front_spacing(self, archive=None, reference_payload=None, strict=False):
        archive = self.pareto_archive if archive is None else archive
        ideal, nadir = self._reference_front_ideal_nadir(reference_payload, strict=strict)
        if ideal is None or nadir is None:
            return None
        return self._safe_float(MO_FBSUtil.archive_spacing(archive, ideal=ideal, nadir=nadir))

    def _current_best_metric_label(self):
        return 'representative decision score'

    def _extra_training_progress_text(self):
        if not getattr(self, "pareto_archive", None):
            return ''
        if self.archive_reference_front_payload is None:
            # 仅用于进度日志展示 IGD，不参与在线接受或档案门控。
            self._refresh_archive_reference_front(force_rebuild=False)

        def _fmt_metric(value):
            return 'NA' if value is None else f'{value:.6f}'

        archive_hv = self._compute_fixed_reference_hypervolume(
            reference_payload=self.archive_reference_front_payload,
            strict=False,
        )
        archive_spacing = self._compute_reference_front_spacing(
            reference_payload=self.archive_reference_front_payload,
            strict=False,
        )
        archive_igd = self._safe_float(
            MO_ReferenceFrontUtil.compute_archive_igd(
                self.pareto_archive,
                self.archive_reference_front_payload,
            )
        )
        return (
            f" | HV: {_fmt_metric(archive_hv)}"
            f" | IGD: {_fmt_metric(archive_igd)}"
            f" | Spacing: {_fmt_metric(archive_spacing)}"
        )

    def _reference_vectors_for_archive(self):
        vectors = list(getattr(self, "archive_reference_vectors", []) or [])
        return vectors if vectors else None

    def _append_unique_elite_anchor(self, anchors, candidate):
        if candidate is None:
            return
        if not getattr(candidate, "current_is_feasible", False) or not np.isfinite(getattr(candidate, "fitness", np.inf)):
            return
        candidate_vector = getattr(candidate, "mo_objectives_min", None)
        for existing in anchors:
            if MO_FBSUtil._duplicate_objectives(candidate, existing):
                return
        anchors.append(copy.deepcopy(candidate))

    def _sorted_elite_candidates(self, candidates):
        return sorted(
            [candidate for candidate in candidates if candidate is not None],
            key=lambda item: (
                float(getattr(item, "decision_score", math.inf)),
                float(getattr(item, "fitness", math.inf)),
            ),
        )

    def _elite_sparse_archive_candidates(self, count):
        if int(count) <= 0 or len(self.pareto_archive) <= 1:
            return []
        normalized, _, _ = MO_FBSUtil._normalized_archive_matrix(
            self.pareto_archive,
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
        return [self.pareto_archive[idx] for _, idx in rows[: int(count)]]

    def _build_elite_anchor_pool(self):
        anchors = []
        max_anchor_count = int(max(1, self.mo_elite_anchor_count))

        primary_candidates = [self.best_feasible_solution]
        if bool(getattr(self, "mo_elite_include_representative", True)):
            primary_candidates.append(self.representative_solution)
        for candidate in self._sorted_elite_candidates(primary_candidates):
            self._append_unique_elite_anchor(anchors, candidate)
            break

        extreme_candidates = []
        if self.pareto_archive:
            if bool(getattr(self, "mo_elite_include_extremes", True)):
                for objective_idx in range(4):
                    extreme_candidate = min(
                        self.pareto_archive,
                        key=lambda item: float(getattr(item, "mo_objectives_min", [math.inf] * 4)[objective_idx]),
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
                for sparse_candidate in self._elite_sparse_archive_candidates(self.mo_elite_sparse_anchor_count):
                    self._append_unique_elite_anchor(anchors, sparse_candidate)
                    if len(anchors) >= max_anchor_count:
                        break

        if len(anchors) < max_anchor_count:
            fill_candidates = []
            fill_candidates.extend(primary_candidates)
            fill_candidates.extend(extreme_candidates)
            fill_candidates.extend(self._elite_sparse_archive_candidates(max_anchor_count))
            for candidate in self._sorted_elite_candidates(fill_candidates):
                self._append_unique_elite_anchor(anchors, candidate)
                if len(anchors) >= max_anchor_count:
                    break

        return anchors[:max_anchor_count]

    def _elite_boundary_candidate_info(self, base, candidate, require_archive_preview=True):
        info = {
            "eligible": False,
            "boundaryScore": 0.0,
            "nearestDistance": 0.0,
            "extremeGain": 0.0,
            "decisionScore": math.inf,
        }
        if not bool(getattr(self, "mo_elite_boundary_candidate_enabled", True)):
            return info
        if candidate is None or not getattr(candidate, "current_is_feasible", False):
            return info
        if not np.isfinite(getattr(candidate, "fitness", np.inf)):
            return info
        if len(self.pareto_archive) < int(getattr(self, "mo_elite_boundary_min_archive_size", 0) or 0):
            return info

        quality_ideal = self.mo_ideal
        quality_nadir = self.mo_nadir
        if quality_ideal is None or quality_nadir is None:
            quality_ideal, quality_nadir = MO_FBSUtil.compute_ideal_nadir(list(self.pareto_archive) + [candidate])

        candidate_vector = MO_FBSUtil.normalize_objective_vector(
            getattr(candidate, "mo_objectives_min", None),
            ideal=quality_ideal,
            nadir=quality_nadir,
        )
        if candidate_vector is None:
            return info

        candidate_score = MO_FBSUtil.decision_score(
            getattr(candidate, "mo_objectives_min", None),
            ideal=quality_ideal,
            nadir=quality_nadir,
            weights=self.mo_weights,
        )
        info["decisionScore"] = float(candidate_score)

        reference_score = self.representative_decision_score
        if not np.isfinite(reference_score):
            reference_score = MO_FBSUtil.decision_score(
                getattr(base, "mo_objectives_min", None),
                ideal=quality_ideal,
                nadir=quality_nadir,
                weights=self.mo_weights,
            )
        if np.isfinite(reference_score) and candidate_score > reference_score + float(self.mo_elite_boundary_rep_slack):
            return info

        normalized_archive, _, _ = MO_FBSUtil._normalized_archive_matrix(
            self.pareto_archive,
            ideal=quality_ideal,
            nadir=quality_nadir,
        )
        if normalized_archive.size == 0:
            nearest_distance = math.inf
            extreme_gain = math.inf
        else:
            distances = np.linalg.norm(normalized_archive - candidate_vector, axis=1)
            nearest_distance = float(np.min(distances)) if distances.size else math.inf
            current_min = np.min(normalized_archive, axis=0)
            extreme_gain = float(np.max(np.maximum(current_min - candidate_vector, 0.0)))

        if not np.isfinite(nearest_distance):
            nearest_distance = float(self.mo_elite_boundary_min_distance)
        if not np.isfinite(extreme_gain):
            extreme_gain = float(self.mo_elite_boundary_extreme_gain_floor)

        boundary_score = float(nearest_distance + 2.0 * extreme_gain)
        rough_eligible = bool(
            nearest_distance >= float(self.mo_elite_boundary_min_distance)
            or extreme_gain >= float(self.mo_elite_boundary_extreme_gain_floor)
        )
        if not rough_eligible:
            return info

        if bool(require_archive_preview):
            archive_preview, archive_would_change, _ = MO_FBSUtil.update_pareto_archive(
                self.pareto_archive,
                candidate,
                max_size=self.archive_limit,
                clone_fn=lambda item: item,
                quality_gate_when_full=False,
                require_candidate_retained=True,
                ideal=quality_ideal,
                nadir=quality_nadir,
            )
            _ = archive_preview
            if not archive_would_change:
                return info

        info.update(
            {
                "eligible": True,
                "boundaryScore": boundary_score,
                "nearestDistance": float(nearest_distance),
                "extremeGain": float(extreme_gain),
            }
        )
        return info

    def _elite_boundary_quality_decision(self, boundary_info):
        decision = {
            "qualityEligible": False,
            "budgetExempt": False,
            "qualityMinScore": 0.0,
            "qualityReason": "disabled",
        }
        if not bool((boundary_info or {}).get("eligible", False)):
            decision["qualityReason"] = "not_boundary"
            return decision
        if not bool(getattr(self, "mo_elite_boundary_quality_gate_enabled", True)):
            decision.update(
                {
                    "qualityEligible": True,
                    "budgetExempt": False,
                    "qualityReason": "gate_disabled",
                }
            )
            return decision

        boundary_score = float((boundary_info or {}).get("boundaryScore", 0.0) or 0.0)
        strong_score = float(getattr(self, "mo_elite_boundary_strong_score", 0.0) or 0.0)
        archive_full = len(self.pareto_archive) >= int(getattr(self, "archive_limit", 0) or 0)
        min_score = float(
            getattr(
                self,
                "mo_elite_boundary_min_score" if archive_full else "mo_elite_boundary_early_min_score",
                0.0,
            )
            or 0.0
        )
        min_score = min(min_score, strong_score) if strong_score > 0.0 else min_score
        decision["qualityMinScore"] = float(min_score)

        if boundary_score + 1e-12 < min_score:
            decision["qualityReason"] = "score_below_min"
            return decision

        budget_exempt = bool(strong_score > 0.0 and boundary_score >= strong_score)
        if not budget_exempt:
            candidate_score = float((boundary_info or {}).get("decisionScore", math.inf))
            reference_score = float(getattr(self, "representative_decision_score", math.inf))
            if np.isfinite(candidate_score) and np.isfinite(reference_score):
                slack = float(getattr(self, "mo_elite_boundary_quality_rep_slack", 0.0) or 0.0)
                if candidate_score > reference_score + slack:
                    decision["qualityReason"] = "decision_slack"
                    return decision

        decision.update(
            {
                "qualityEligible": True,
                "budgetExempt": budget_exempt,
                "qualityReason": "strong_boundary" if budget_exempt else "quality_pass",
            }
        )
        return decision

    def _elite_candidate_eval_limit(self):
        archive_limit = int(getattr(self, "archive_limit", 0) or 0)
        archive_size = len(getattr(self, "pareto_archive", []) or [])
        if archive_limit > 0 and archive_size >= archive_limit:
            return int(max(0, int(getattr(self, "mo_elite_max_candidates_after_full", 0) or 0)))
        return int(max(0, int(getattr(self, "mo_elite_max_candidates_before_full", 0) or 0)))

    def _elite_intensification(self, fast_time):
        if not bool(getattr(self, "mo_elite_multi_anchor_enabled", True)):
            return super()._elite_intensification(fast_time)

        anchors = self._build_elite_anchor_pool()
        if not anchors:
            return fast_time, False

        self.elite_trigger_count += 1
        improved_any = False
        initial_best_cost = float(self.best_feasible_cost) if np.isfinite(self.best_feasible_cost) else math.inf
        try:
            boundary_accept_budget = int(getattr(self, "mo_elite_boundary_accept_budget", -1))
        except (TypeError, ValueError):
            boundary_accept_budget = -1
        boundary_accept_limited = boundary_accept_budget >= 0
        boundary_accept_used = 0
        quality_gate_enabled = bool(getattr(self, "mo_elite_boundary_quality_gate_enabled", False))
        elite_candidate_limit = int(max(0, self._elite_candidate_eval_limit()))
        elite_candidate_evaluations = 0
        elite_rounds_used = 0
        elite_anchors_visited = 0
        elite_stopped_by_budget = False

        for _ in range(self.elite_max_rounds):
            if elite_stopped_by_budget:
                break
            anchors = self._build_elite_anchor_pool()
            if not anchors:
                break

            elite_rounds_used += 1
            round_improved = False
            for anchor in anchors:
                if elite_stopped_by_budget:
                    break
                base = copy.deepcopy(anchor)
                self._evaluate_solution(base)
                if not base.current_is_feasible or not np.isfinite(base.fitness):
                    continue
                elite_anchors_visited += 1

                best_candidate = None
                best_action_idx = None
                best_candidate_cost = float(base.fitness)
                best_candidate_key = None
                best_candidate_boundary_info = None
                best_candidate_scalar_improved = False
                boundary_preview_budget = int(max(0, int(getattr(self, "mo_elite_boundary_preview_budget", 0) or 0)))
                boundary_preview_used = 0

                for action_idx in self.elite_actions:
                    if elite_stopped_by_budget:
                        break
                    trial_count = self.elite_action_trials.get(action_idx, 1)
                    for _trial in range(trial_count):
                        if elite_candidate_limit > 0 and elite_candidate_evaluations >= elite_candidate_limit:
                            elite_stopped_by_budget = True
                            break
                        candidate = self.generate_candidate_by_action(base, action_idx)
                        elite_candidate_evaluations += 1
                        self._record_action_selection(
                            action_idx,
                            base.fitness,
                            candidate.fitness,
                            phase="elite",
                        )
                        if not candidate.current_is_feasible or not np.isfinite(candidate.fitness):
                            continue

                        scalar_gain = float(base.fitness) - float(candidate.fitness)
                        scalar_gain_rel = scalar_gain / max(abs(float(base.fitness)), 1e-12)
                        scalar_improved = scalar_gain > 1e-12
                        boundary_info = self._elite_boundary_candidate_info(
                            base,
                            candidate,
                            require_archive_preview=False,
                        )
                        if bool(boundary_info.get("eligible", False)) and boundary_preview_used < boundary_preview_budget:
                            checked_boundary_info = self._elite_boundary_candidate_info(
                                base,
                                candidate,
                                require_archive_preview=True,
                            )
                            boundary_preview_used += 1
                            if bool(checked_boundary_info.get("eligible", False)):
                                boundary_info = checked_boundary_info
                            else:
                                boundary_info["eligible"] = False
                        boundary_eligible = bool(boundary_info.get("eligible", False))
                        if (
                            boundary_eligible
                            and not scalar_improved
                            and (quality_gate_enabled or boundary_accept_limited)
                        ):
                            boundary_quality = self._elite_boundary_quality_decision(boundary_info)
                            boundary_info.update(boundary_quality)
                            budget_exempt = bool(boundary_quality.get("budgetExempt", False))
                            # 质量门槛和接受预算均为显式实验开关，默认不改变 budgeted 行为。
                            if quality_gate_enabled and not bool(boundary_quality.get("qualityEligible", False)):
                                boundary_info["eligible"] = False
                                boundary_eligible = False
                            elif (
                                boundary_accept_limited
                                and not budget_exempt
                                and boundary_accept_used >= boundary_accept_budget
                            ):
                                boundary_info["eligible"] = False
                                boundary_info["qualityReason"] = "budget_exhausted"
                                boundary_eligible = False
                        elif boundary_eligible and (quality_gate_enabled or boundary_accept_limited):
                            boundary_info.update({
                                "qualityEligible": True,
                                "budgetExempt": True,
                                "qualityMinScore": 0.0,
                                "qualityReason": "scalar_improved",
                            })
                        if not scalar_improved and not boundary_eligible:
                            continue

                        if quality_gate_enabled:
                            boundary_quality_score = float(boundary_info.get("boundaryScore", 0.0) or 0.0)
                            if not scalar_improved:
                                candidate_decision_score = float(boundary_info.get("decisionScore", math.inf))
                                reference_score = float(getattr(self, "representative_decision_score", math.inf))
                                if np.isfinite(candidate_decision_score) and np.isfinite(reference_score):
                                    boundary_quality_score -= 0.10 * max(candidate_decision_score - reference_score, 0.0)
                            candidate_key = (
                                1 if boundary_eligible else 0,
                                1 if bool(boundary_info.get("budgetExempt", False)) else 0,
                                float(boundary_quality_score),
                                1 if scalar_improved else 0,
                                float(scalar_gain_rel),
                                -float(boundary_info.get("decisionScore", math.inf)),
                            )
                        else:
                            candidate_key = (
                                1 if boundary_eligible else 0,
                                float(boundary_info.get("boundaryScore", 0.0) or 0.0),
                                1 if scalar_improved else 0,
                                float(scalar_gain_rel),
                                -float(boundary_info.get("decisionScore", math.inf)),
                            )
                        if best_candidate_key is None or candidate_key > best_candidate_key:
                            best_candidate = candidate
                            best_action_idx = action_idx
                            best_candidate_cost = float(candidate.fitness)
                            best_candidate_key = candidate_key
                            best_candidate_boundary_info = boundary_info
                            best_candidate_scalar_improved = bool(scalar_improved)

                if best_candidate is None or not (
                    bool(best_candidate_scalar_improved)
                    or bool((best_candidate_boundary_info or {}).get("eligible", False))
                ):
                    continue

                round_improved = True
                improved_any = True
                self._record_action_acceptance(
                    best_action_idx,
                    base.fitness,
                    best_candidate_cost,
                    improved=bool(best_candidate_scalar_improved),
                    phase="elite",
                )
                if bool((best_candidate_boundary_info or {}).get("eligible", False)):
                    boundary_only_selected = not bool(best_candidate_scalar_improved)
                    boundary_budget_exempt = bool((best_candidate_boundary_info or {}).get("budgetExempt", False))
                    if boundary_accept_limited and boundary_only_selected and not boundary_budget_exempt:
                        boundary_accept_used += 1
                    self._record_mo_event(
                        "elite_boundary_candidate",
                        action=int(best_action_idx),
                        scalarImproved=bool(best_candidate_scalar_improved),
                        boundaryScore=self._safe_float(best_candidate_boundary_info.get("boundaryScore")),
                        nearestDistance=self._safe_float(best_candidate_boundary_info.get("nearestDistance")),
                        extremeGain=self._safe_float(best_candidate_boundary_info.get("extremeGain")),
                        decisionScore=self._safe_float(best_candidate_boundary_info.get("decisionScore")),
                        boundaryAcceptUsed=int(boundary_accept_used),
                        boundaryAcceptBudget=int(boundary_accept_budget),
                        boundaryAcceptLimited=bool(boundary_accept_limited),
                        boundaryBudgetExempt=bool(boundary_budget_exempt),
                        qualityMinScore=self._safe_float(best_candidate_boundary_info.get("qualityMinScore")),
                        qualityReason=str(best_candidate_boundary_info.get("qualityReason", "")),
                    )
                archive_improved = self._observe_feasible_state(best_candidate)
                if archive_improved:
                    self._record_action_global_best(best_action_idx, phase="elite")
                    fast_time = datetime.datetime.now()

            if not round_improved:
                break

        if elite_candidate_limit > 0:
            self._record_mo_event(
                "elite_budget",
                candidateEvaluations=int(elite_candidate_evaluations),
                candidateLimit=int(elite_candidate_limit),
                roundsUsed=int(elite_rounds_used),
                anchorsVisited=int(elite_anchors_visited),
                stoppedByBudget=bool(elite_stopped_by_budget),
                improved=bool(improved_any),
            )

        if improved_any:
            final_state = None
            if self.best_feasible_solution is not None:
                final_state = copy.deepcopy(self.best_feasible_solution)
            elif self.representative_solution is not None:
                final_state = copy.deepcopy(self.representative_solution)
            elif anchors:
                final_state = copy.deepcopy(anchors[0])
            if final_state is not None:
                self.s = final_state
                self.current_energy = self.s.fitness
                self.no_improve_steps = 0
                self._update_histogram(self.current_energy)
                self.energy_history.append(self.current_energy)
                self.modified_energy_history.append(self._tilde_energy(self.current_energy))

            final_best_cost = float(self.best_feasible_cost) if np.isfinite(self.best_feasible_cost) else initial_best_cost
            final_gain = max(initial_best_cost - final_best_cost, 0.0) if np.isfinite(initial_best_cost) else 0.0
            self.elite_improvement_count += 1
            self.elite_total_gain += float(final_gain)
        return fast_time, improved_any

    @staticmethod
    def _make_trace_bucket():
        return {
            "steps": 0,
            "accepted": 0,
            "rejected": 0,
            "candidateDominatesAccepts": 0,
            "candidateDominatedRejects": 0,
            "nondominatedAccepts": 0,
            "nondominatedRejects": 0,
            "archiveChanges": 0,
            "representativeChanges": 0,
            "rewardSum": 0.0,
            "rewardCount": 0,
            "lossSum": 0.0,
            "lossCount": 0,
            "acceptProbabilitySum": 0.0,
            "acceptProbabilityCount": 0,
            "localSearchTriggers": 0,
            "localSearchImprovements": 0,
            "diversifyAttempts": 0,
            "diversifyAccepts": 0,
            "reheatTriggers": 0,
            "actionSelected": {},
            "actionAccepted": {},
            "actionImproved": {},
        }

    def _reset_mo_logging_state(self):
        self.mo_recorder = None
        self.mo_run_summary = None
        self._run_start_time = None
        self._trace_global_step = 0
        self._trace_episode_index = -1
        self._trace_step_index = -1
        self._current_action_context = {}
        self._last_archive_observation = {"archive_changed": False, "rep_changed": False}
        self._last_loss_value = None
        self._last_effective_archive_change_step = 0
        self.local_search_failure_streak = 0
        self._mo_total_counters = self._make_trace_bucket()
        self._mo_window_counters = self._make_trace_bucket()

    @staticmethod
    def _bump_named_counter(counter_map, key, amount=1):
        key = str(key)
        counter_map[key] = int(counter_map.get(key, 0)) + int(amount)

    def _action_counter_key(self, action_idx):
        if action_idx is None:
            return "unknown"
        action_idx = int(action_idx)
        return f"{action_idx}:{self.action_labels.get(action_idx, str(action_idx))}"

    def _set_action_context(self, phase, action_idx=None, action_name=None):
        if action_name is None and action_idx is not None:
            action_name = self.action_labels.get(int(action_idx), str(action_idx))
        self._current_action_context = {
            "phase": str(phase),
            "action_idx": None if action_idx is None else int(action_idx),
            "action_name": str(action_name or phase),
        }

    def _clear_action_context(self):
        self._current_action_context = {}

    def _record_action_selection(self, action_idx, previous_cost, next_cost, global_best=False, phase="main"):
        delta = super()._record_action_selection(action_idx, previous_cost, next_cost, global_best=global_best, phase=phase)
        self._set_action_context(phase=phase, action_idx=action_idx)
        action_key = self._action_counter_key(action_idx)
        self._bump_named_counter(self._mo_total_counters["actionSelected"], action_key)
        self._bump_named_counter(self._mo_window_counters["actionSelected"], action_key)
        return delta

    def _record_action_acceptance(self, action_idx, previous_cost, next_cost, improved=False, phase="main"):
        super()._record_action_acceptance(action_idx, previous_cost, next_cost, improved=improved, phase=phase)
        action_key = self._action_counter_key(action_idx)
        self._bump_named_counter(self._mo_total_counters["actionAccepted"], action_key)
        self._bump_named_counter(self._mo_window_counters["actionAccepted"], action_key)
        if improved:
            self._bump_named_counter(self._mo_total_counters["actionImproved"], action_key)
            self._bump_named_counter(self._mo_window_counters["actionImproved"], action_key)

    def _record_mo_event(self, event_type, **fields):
        if getattr(self, "mo_recorder", None) is None:
            return
        elapsed_seconds = None
        if getattr(self, "_run_start_time", None) is not None:
            elapsed_seconds = (datetime.datetime.now() - self._run_start_time).total_seconds()
        context = dict(getattr(self, "_current_action_context", {}) or {})
        payload = {
            "globalStep": int(getattr(self, "_trace_global_step", 0) or 0),
            "episode": None if getattr(self, "_trace_episode_index", -1) < 0 else int(self._trace_episode_index) + 1,
            "stepInEpisode": None if getattr(self, "_trace_step_index", -1) < 0 else int(self._trace_step_index) + 1,
            "elapsedSeconds": self._safe_float(elapsed_seconds),
            "phase": context.get("phase"),
            "actionIdx": context.get("action_idx"),
            "actionName": context.get("action_name"),
            "temperature": self._safe_float(getattr(self, "T", None)),
            "temperatureMin": self._safe_float(getattr(self, "T_min", None)),
            "archiveSize": len(getattr(self, "pareto_archive", []) or []),
            "representativeDecisionScore": self._safe_float(getattr(self, "representative_decision_score", None)),
        }
        payload.update(fields)
        self.mo_recorder.record_event(event_type, payload)

    def _ensure_mo_logging_state(self):
        if not hasattr(self, "_mo_total_counters") or not hasattr(self, "_mo_window_counters"):
            self._reset_mo_logging_state()

    def _increment_archive_counters(self, archive_changed=False, rep_changed=False):
        self._ensure_mo_logging_state()
        for bucket in (self._mo_total_counters, self._mo_window_counters):
            bucket["archiveChanges"] += int(bool(archive_changed))
            bucket["representativeChanges"] += int(bool(rep_changed))

    def _note_local_search(self, improved=False):
        self._ensure_mo_logging_state()
        for bucket in (self._mo_total_counters, self._mo_window_counters):
            bucket["localSearchTriggers"] += 1
            if improved:
                bucket["localSearchImprovements"] += 1

    def _note_diversification(self, accepted=False):
        self._ensure_mo_logging_state()
        for bucket in (self._mo_total_counters, self._mo_window_counters):
            bucket["diversifyAttempts"] += 1
            if accepted:
                bucket["diversifyAccepts"] += 1

    def _note_reheat(self):
        self._ensure_mo_logging_state()
        for bucket in (self._mo_total_counters, self._mo_window_counters):
            bucket["reheatTriggers"] += 1

    def _update_transition_counters(self, accept, probability, reward, loss_value):
        self._ensure_mo_logging_state()
        comparison = int((getattr(self, "_last_transition_meta", {}) or {}).get("comparison", 0))
        for bucket in (self._mo_total_counters, self._mo_window_counters):
            bucket["steps"] += 1
            bucket["accepted"] += int(bool(accept))
            bucket["rejected"] += int(not accept)
            bucket["acceptProbabilitySum"] += float(probability)
            bucket["acceptProbabilityCount"] += 1
            bucket["rewardSum"] += float(reward)
            bucket["rewardCount"] += 1
            if loss_value is not None and np.isfinite(loss_value):
                bucket["lossSum"] += float(loss_value)
                bucket["lossCount"] += 1
            if comparison < 0 and accept:
                bucket["candidateDominatesAccepts"] += 1
            elif comparison > 0 and not accept:
                bucket["candidateDominatedRejects"] += 1
            elif comparison == 0 and accept:
                bucket["nondominatedAccepts"] += 1
            elif comparison == 0 and not accept:
                bucket["nondominatedRejects"] += 1

    def _build_bucket_snapshot(self, bucket):
        reward_count = int(bucket.get("rewardCount", 0) or 0)
        loss_count = int(bucket.get("lossCount", 0) or 0)
        prob_count = int(bucket.get("acceptProbabilityCount", 0) or 0)
        return {
            "steps": int(bucket.get("steps", 0) or 0),
            "accepted": int(bucket.get("accepted", 0) or 0),
            "rejected": int(bucket.get("rejected", 0) or 0),
            "candidateDominatesAccepts": int(bucket.get("candidateDominatesAccepts", 0) or 0),
            "candidateDominatedRejects": int(bucket.get("candidateDominatedRejects", 0) or 0),
            "nondominatedAccepts": int(bucket.get("nondominatedAccepts", 0) or 0),
            "nondominatedRejects": int(bucket.get("nondominatedRejects", 0) or 0),
            "archiveChanges": int(bucket.get("archiveChanges", 0) or 0),
            "representativeChanges": int(bucket.get("representativeChanges", 0) or 0),
            "localSearchTriggers": int(bucket.get("localSearchTriggers", 0) or 0),
            "localSearchImprovements": int(bucket.get("localSearchImprovements", 0) or 0),
            "diversifyAttempts": int(bucket.get("diversifyAttempts", 0) or 0),
            "diversifyAccepts": int(bucket.get("diversifyAccepts", 0) or 0),
            "reheatTriggers": int(bucket.get("reheatTriggers", 0) or 0),
            "meanReward": None if reward_count == 0 else float(bucket["rewardSum"] / reward_count),
            "meanLoss": None if loss_count == 0 else float(bucket["lossSum"] / loss_count),
            "meanAcceptProbability": None if prob_count == 0 else float(bucket["acceptProbabilitySum"] / prob_count),
            "actionSelected": dict(bucket.get("actionSelected", {}) or {}),
            "actionAccepted": dict(bucket.get("actionAccepted", {}) or {}),
            "actionImproved": dict(bucket.get("actionImproved", {}) or {}),
        }

    def _solution_trace_fields(self, solution, prefix):
        if solution is None:
            return {
                f"{prefix}Feasible": None,
                f"{prefix}SearchEnergy": None,
                f"{prefix}DecisionScore": None,
                f"{prefix}ProxyEnergy": None,
                f"{prefix}Mhc": None,
                f"{prefix}Cr": None,
                f"{prefix}Dr": None,
                f"{prefix}Ar": None,
                f"{prefix}DInf": None,
                f"{prefix}ConstraintViolation": None,
            }
        return {
            f"{prefix}Feasible": bool(getattr(solution, "current_is_feasible", False)),
            f"{prefix}SearchEnergy": self._safe_float(getattr(solution, "fitness", None)),
            f"{prefix}DecisionScore": self._safe_float(getattr(solution, "decision_score", None)),
            f"{prefix}ProxyEnergy": self._safe_float(getattr(solution, "proxy_energy", None)),
            f"{prefix}Mhc": self._safe_float(getattr(solution, "MHC", None)),
            f"{prefix}Cr": self._safe_float(getattr(solution, "CR", None)),
            f"{prefix}Dr": self._safe_float(getattr(solution, "DR", None)),
            f"{prefix}Ar": self._safe_float(getattr(solution, "AR", None)),
            f"{prefix}DInf": int(getattr(solution, "current_d_inf", 0) or 0),
            f"{prefix}ConstraintViolation": self._safe_float(getattr(solution, "constraint_violation", None)),
        }

    def _record_trace_snapshot(self, global_step, total_steps, episode_idx, step_idx, agent):
        if getattr(self, "mo_recorder", None) is None:
            return
        if not self.mo_recorder.should_record_trace(global_step, total_steps=total_steps):
            return
        elapsed_seconds = None
        if getattr(self, "_run_start_time", None) is not None:
            elapsed_seconds = (datetime.datetime.now() - self._run_start_time).total_seconds()
        payload = {
            "globalStep": int(global_step),
            "episode": int(episode_idx) + 1,
            "stepInEpisode": int(step_idx) + 1,
            "progressRatio": float(global_step / max(1, total_steps)),
            "elapsedSeconds": self._safe_float(elapsed_seconds),
            "temperature": self._safe_float(getattr(self, "T", None)),
            "temperatureMin": self._safe_float(getattr(self, "T_min", None)),
            "acceptRateWindow": self._safe_float(getattr(self, "accept_rate_window", None)),
            "noImproveSteps": int(getattr(self, "no_improve_steps", 0) or 0),
            "feasibleSolutionCount": int(getattr(self, "feasible_solution_count", 0) or 0),
            "gbestUpdateCount": int(getattr(self, "gbest_update_count", 0) or 0),
            "diversificationCount": int(getattr(self, "diversification_count", 0) or 0),
            "reheatTriggerCount": int(getattr(self, "reheat_trigger_count", 0) or 0),
            "eliteTriggerCount": int(getattr(self, "elite_trigger_count", 0) or 0),
            "eliteImprovementCount": int(getattr(self, "elite_improvement_count", 0) or 0),
            "archiveSize": len(getattr(self, "pareto_archive", []) or []),
            "nonArchiveStagnationWindows": self._safe_float(self._nonarchive_stagnation_windows()),
            "lastEffectiveArchiveChangeStep": int(getattr(self, "_last_effective_archive_change_step", 0) or 0),
            "localSearchFailureStreak": int(getattr(self, "local_search_failure_streak", 0) or 0),
            "localSearchEffectiveCooldown": int(self._effective_local_search_cooldown_steps()),
            "epsilon": self._safe_float(getattr(agent, "epsilon", None)),
            "latestLoss": self._safe_float(getattr(self, "_last_loss_value", None)),
            "agentTotalSteps": int(getattr(agent, "total_steps", 0) or 0) if hasattr(agent, "total_steps") else None,
            "agentOptimizeSteps": int(getattr(agent, "optimize_steps", 0) or 0) if hasattr(agent, "optimize_steps") else None,
            "window": self._build_bucket_snapshot(self._mo_window_counters),
            "cumulative": self._build_bucket_snapshot(self._mo_total_counters),
        }
        payload.update(self._solution_trace_fields(getattr(self, "s", None), "current"))
        payload.update(self._solution_trace_fields(getattr(self, "representative_solution", None), "representative"))
        self.mo_recorder.record_trace(payload)
        self._mo_window_counters = self._make_trace_bucket()

    def _build_action_stats_payload(self, agent, total_iter):
        return {
            "meta": {
                "runId": None if getattr(self, "mo_recorder", None) is None else self.mo_recorder.run_id,
                "instance": self.instance_name,
                "algorithm": "ELP_DRL_MO",
                "agentMode": getattr(self, "agent_mode", None),
                "iterations": int(total_iter),
            },
            "overall": {
                **self._build_bucket_snapshot(self._mo_total_counters),
                "feasibleSolutionCount": int(getattr(self, "feasible_solution_count", 0) or 0),
                "gbestUpdateCount": int(getattr(self, "gbest_update_count", 0) or 0),
                "archiveSize": len(getattr(self, "pareto_archive", []) or []),
                "reheatTriggerCount": int(getattr(self, "reheat_trigger_count", 0) or 0),
                "diversificationCount": int(getattr(self, "diversification_count", 0) or 0),
                "eliteTriggerCount": int(getattr(self, "elite_trigger_count", 0) or 0),
                "eliteImprovementCount": int(getattr(self, "elite_improvement_count", 0) or 0),
                "epsilonEnd": self._safe_float(getattr(agent, "epsilon", None)),
                "agentTotalSteps": int(getattr(agent, "total_steps", 0) or 0) if hasattr(agent, "total_steps") else None,
                "agentOptimizeSteps": int(getattr(agent, "optimize_steps", 0) or 0) if hasattr(agent, "optimize_steps") else None,
                "representativeDecisionScore": self._safe_float(getattr(self, "representative_decision_score", None)),
            },
            "actions": self.get_action_telemetry(),
        }

    def _light_clone_solution(self, solution):
        clone = copy.copy(solution)
        for key, value in solution.__dict__.items():
            setattr(clone, key, value)
        clone.fbs_model = FBSModel(solution.fbs_model.permutation, solution.fbs_model.bay)
        return clone

    def _prepare_light_clone_with_encoding(self, solution, permutation, bay):
        clone = self._light_clone_solution(solution)
        clone.fbs_model.permutation = np.asarray(permutation, dtype=int).tolist()
        clone.fbs_model.bay = np.asarray(bay, dtype=int).tolist()
        return clone

    def _restore_solution_snapshot(self, solution, snapshot):
        for key, value in snapshot.__dict__.items():
            setattr(solution, key, value)
        solution.fbs_model = FBSModel(snapshot.fbs_model.permutation, snapshot.fbs_model.bay)
        return solution

    def _candidate_from_light_clone(self, solution, action_idx):
        candidate = self._light_clone_solution(solution)
        recipe = self.action_recipes[action_idx]
        self._apply_recipe(candidate, recipe)
        self._evaluate_solution(candidate)
        return candidate

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
        objectives_raw = MO_FBSUtil.calculate_objectives(
            metrics["fac_x"],
            metrics["fac_y"],
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["mhc"],
            len(metrics["fac_x"]),
            rel_matrix=self.rel_matrix,
            dist_req_matrix=self.dist_req_matrix,
            aspect_limits=metrics["aspect_limits"],
        )
        objectives_min = MO_FBSUtil.to_minimization(objectives_raw)
        constraint_violation = MO_FBSUtil.calculate_total_constraint_violation(
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["lower_bounds"],
            metrics["upper_bounds"],
        )
        search_energy = MO_FBSUtil.search_energy(
            objectives_min,
            is_feasible=bool(metrics["is_feasible"]),
            d_inf=int(metrics["d_inf"]),
            total_violation=constraint_violation,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
        )
        return (
            float(search_energy),
            int(metrics["d_inf"]),
            float(constraint_violation),
            float(metrics["mhc"]),
            np.asarray(permutation, dtype=int),
            np.asarray(bay, dtype=int),
        )

    def _apply_fast_segment_insert(self, solution):
        permutation = np.asarray(solution.fbs_model.permutation, dtype=int).copy()
        bay = np.asarray(solution.fbs_model.bay, dtype=int).copy()
        bay_structure = self._copy_bay_structure(FBSUtil.permutationToArray(permutation, bay))
        if not bay_structure:
            return False

        candidate_pool = []
        seen = set()
        max_candidates = int(max(1, self.fast_segment_insert_budget))
        max_attempts = max_candidates * 6
        eligible_bays = [
            bay_idx
            for bay_idx, current_bay in enumerate(bay_structure)
            if any(len(current_bay) > segment_length for segment_length in self.fast_segment_insert_segment_lengths)
        ]
        if not eligible_bays:
            return False

        attempts = 0
        while len(candidate_pool) < max_candidates and attempts < max_attempts:
            attempts += 1
            bay_idx = int(np.random.choice(eligible_bays))
            current_bay = bay_structure[bay_idx]
            feasible_lengths = [
                segment_length
                for segment_length in self.fast_segment_insert_segment_lengths
                if len(current_bay) > segment_length
            ]
            if not feasible_lengths:
                continue

            segment_length = int(np.random.choice(feasible_lengths))
            start_idx = int(np.random.randint(0, len(current_bay) - segment_length + 1))
            segment = current_bay[start_idx : start_idx + segment_length]
            remaining = current_bay[:start_idx] + current_bay[start_idx + segment_length :]
            insert_choices = [insert_idx for insert_idx in range(len(remaining) + 1) if insert_idx != start_idx]
            if not insert_choices:
                continue

            insert_idx = int(np.random.choice(insert_choices))
            candidate_bay = remaining[:insert_idx] + segment + remaining[insert_idx:]
            if candidate_bay == current_bay:
                continue

            candidate_structure = self._copy_bay_structure(bay_structure)
            candidate_structure[bay_idx] = candidate_bay
            candidate_perm, candidate_bay_flags = FBSUtil.arrayToPermutation(candidate_structure)
            candidate_key = (
                tuple(np.asarray(candidate_perm, dtype=int).tolist()),
                tuple(np.asarray(candidate_bay_flags, dtype=int).tolist()),
            )
            if candidate_key in seen:
                continue

            seen.add(candidate_key)
            candidate_pool.append(self._score_candidate_encoding(candidate_perm, candidate_bay_flags, solution))

        if not candidate_pool:
            return False

        candidate_pool.sort(key=lambda item: item[:4])
        best_perm = candidate_pool[0][4]
        best_bay = candidate_pool[0][5]
        solution.fbs_model.permutation = best_perm.tolist()
        solution.fbs_model.bay = best_bay.tolist()
        return True

    def _apply_recipe(self, solution, recipe):
        layout_dirty = False
        for primitive_action in recipe:
            if primitive_action == 3 and layout_dirty:
                self._evaluate_solution(solution)
                layout_dirty = False
            if primitive_action == 10:
                layout_dirty = self._apply_fast_segment_insert(solution) or layout_dirty
                continue
            solution._apply_action(solution.actions[primitive_action])
            layout_dirty = True

    def generate_candidate_by_action(self, solution, action_idx):
        return self._candidate_from_light_clone(solution, action_idx)

    def _generate_candidate_by_recipe(self, solution, recipe):
        candidate = self._light_clone_solution(solution)
        self._apply_recipe(candidate, recipe)
        self._evaluate_solution(candidate)
        return candidate

    def _greedy_local_search(self, solution):
        current_cost = float(getattr(solution, "fitness", np.inf))
        accepted_snapshot = self._light_clone_solution(solution)
        improved = True

        while improved:
            improved = False

            bay_structure = FBSUtil.permutationToArray(
                solution.fbs_model.permutation,
                solution.fbs_model.bay,
            )
            n_bays = len(bay_structure)

            for i in range(n_bays - 1):
                bay_structure[i], bay_structure[i + 1] = bay_structure[i + 1], bay_structure[i]
                new_perm, new_bay = FBSUtil.arrayToPermutation(bay_structure)
                solution.fbs_model.permutation = np.asarray(new_perm, dtype=int).tolist()
                solution.fbs_model.bay = np.asarray(new_bay, dtype=int).tolist()
                self._evaluate_solution(solution)

                if (
                    solution.current_is_feasible
                    and np.isfinite(solution.fitness)
                    and solution.fitness < current_cost
                ):
                    current_cost = float(solution.fitness)
                    self._observe_feasible_state(solution)
                    accepted_snapshot = self._light_clone_solution(solution)
                    improved = True
                else:
                    bay_structure[i], bay_structure[i + 1] = bay_structure[i + 1], bay_structure[i]
                    self._restore_solution_snapshot(solution, accepted_snapshot)

            bay_structure = FBSUtil.permutationToArray(
                solution.fbs_model.permutation,
                solution.fbs_model.bay,
            )

            for bay_idx, bay in enumerate(bay_structure):
                bay = list(bay)
                n_fac = len(bay)
                if n_fac < 2:
                    continue

                for j in range(n_fac - 1):
                    new_bay_structure = self._copy_bay_structure(bay_structure)
                    new_bay_structure[bay_idx][j], new_bay_structure[bay_idx][j + 1] = (
                        new_bay_structure[bay_idx][j + 1],
                        new_bay_structure[bay_idx][j],
                    )
                    new_perm, new_bay_arr = FBSUtil.arrayToPermutation(new_bay_structure)
                    solution.fbs_model.permutation = np.asarray(new_perm, dtype=int).tolist()
                    solution.fbs_model.bay = np.asarray(new_bay_arr, dtype=int).tolist()
                    self._evaluate_solution(solution)

                    if (
                        solution.current_is_feasible
                        and np.isfinite(solution.fitness)
                        and solution.fitness < current_cost
                    ):
                        current_cost = float(solution.fitness)
                        self._observe_feasible_state(solution)
                        accepted_snapshot = self._light_clone_solution(solution)
                        bay_structure = FBSUtil.permutationToArray(
                            solution.fbs_model.permutation,
                            solution.fbs_model.bay,
                        )
                        bay = list(bay_structure[bay_idx]) if bay_idx < len(bay_structure) else []
                        n_fac = len(bay)
                        improved = True
                    else:
                        self._restore_solution_snapshot(solution, accepted_snapshot)

        return solution

    def _refresh_solution_search_metrics(self, solution):
        objectives_min = getattr(solution, "mo_objectives_min", None)
        if objectives_min is None:
            return float(getattr(solution, "fitness", math.inf))

        proxy_energy = MO_FBSUtil.surrogate_energy(
            objectives_min,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
        )
        decision_score = MO_FBSUtil.decision_score(
            objectives_min,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
        )
        total_violation = float(getattr(solution, "constraint_violation", 0.0) or 0.0)
        search_energy = MO_FBSUtil.search_energy(
            objectives_min,
            is_feasible=bool(getattr(solution, "current_is_feasible", False)),
            d_inf=int(getattr(solution, "current_d_inf", 0) or 0),
            total_violation=total_violation,
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
            weights=self.mo_weights,
        )

        solution.proxy_energy = float(proxy_energy)
        solution.decision_score = float(decision_score)
        solution.fitness = float(search_energy)
        solution.best_feasible_cost = getattr(self, "best_feasible_cost", math.inf)
        solution.best_fitness = getattr(self, "best_feasible_cost", math.inf)
        solution.worst_feasible_cost = getattr(self, "worst_feasible_cost", None)
        solution.current_v_worst = getattr(self, "worst_feasible_cost", None)
        return float(search_energy)

    def _refresh_archive_state(self):
        feasible_archive = [candidate for candidate in self.pareto_archive if getattr(candidate, "current_is_feasible", False)]
        self.pareto_archive = feasible_archive
        self.mo_ideal, self.mo_nadir = MO_FBSUtil.compute_ideal_nadir(self.pareto_archive)

        for candidate in self.pareto_archive:
            self._refresh_solution_search_metrics(candidate)

        representative, decision_score, archive_index = MO_FBSUtil.select_representative_solution(
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
        solution.DR = float(metrics["dr"])
        solution.AR = float(metrics["ar"])
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
        objectives_raw = MO_FBSUtil.calculate_objectives(
            metrics["fac_x"],
            metrics["fac_y"],
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["mhc"],
            len(metrics["fac_x"]),
            rel_matrix=self.rel_matrix,
            dist_req_matrix=self.dist_req_matrix,
            aspect_limits=metrics["aspect_limits"],
        )
        constraint_violation = MO_FBSUtil.calculate_total_constraint_violation(
            metrics["fac_b"],
            metrics["fac_h"],
            metrics["lower_bounds"],
            metrics["upper_bounds"],
        )
        objectives_min = MO_FBSUtil.to_minimization(objectives_raw)
        metrics.update(
            {
                "cr": float(objectives_raw[1]),
                "dr": float(objectives_raw[2]),
                "ar": float(objectives_raw[3]),
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

        updated_archive, inserted, removed = MO_FBSUtil.update_pareto_archive(
            self.pareto_archive,
            solution,
            max_size=self.archive_limit,
            clone_fn=copy.deepcopy,
            quality_gate_when_full=self.archive_quality_gate_when_full,
            quality_hv_tol=self.archive_quality_hv_tol,
            quality_spacing_tol=self.archive_quality_spacing_tol,
            spacing_guard_when_full=bool(getattr(self, "archive_spacing_guard_when_full", False)),
            spacing_guard_rel_tol=float(getattr(self, "archive_spacing_guard_rel_tol", 0.0) or 0.0),
            spacing_guard_hv_gain_rel=float(getattr(self, "archive_spacing_guard_hv_gain_rel", 0.0) or 0.0),
            require_candidate_retained=bool(getattr(self, "archive_require_candidate_retained", False)),
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
        )
        archive_changed = bool(inserted)
        if archive_changed:
            self.pareto_archive = updated_archive
        elif not self.pareto_archive and getattr(solution, "current_is_feasible", False):
            self.pareto_archive = [copy.deepcopy(solution)]
            archive_changed = True

        self._refresh_archive_state()
        self._refresh_solution_search_metrics(solution)

        if self.representative_solution is not None:
            self.best_feasible_solution = copy.deepcopy(self.representative_solution)
            self.gbest = copy.deepcopy(self.representative_solution)
            self.true_gbest = copy.deepcopy(self.representative_solution)
            self.best_feasible_cost = float(self.representative_decision_score)
            self.best_energy = float(self.representative_decision_score)
        elif np.isfinite(getattr(solution, "decision_score", math.inf)):
            self.best_feasible_solution = copy.deepcopy(solution)
            self.gbest = copy.deepcopy(solution)
            self.true_gbest = copy.deepcopy(solution)
            self.best_feasible_cost = float(solution.decision_score)
            self.best_energy = float(solution.decision_score)

        feasible_cost = float(getattr(solution, "fitness", math.inf))
        self.worst_feasible_cost = (
            feasible_cost if self.worst_feasible_cost is None else max(float(self.worst_feasible_cost), feasible_cost)
        )

        rep_changed = False
        if self.representative_solution is not None:
            current_rep_vector = np.asarray(self.representative_solution.mo_objectives_min, dtype=float)
            if previous_rep_vector is None:
                rep_changed = True
            elif not np.allclose(current_rep_vector, previous_rep_vector, atol=1e-9, rtol=1e-7):
                rep_changed = True
            elif float(self.representative_decision_score) < previous_rep_score - 1e-12:
                rep_changed = True
            if rep_changed:
                self.gbest_update_count += 1
                self.best_history.append(float(self.representative_decision_score))

        self._last_archive_observation = {
            "archive_changed": bool(archive_changed),
            "rep_changed": bool(rep_changed),
            "inserted": bool(inserted),
            "removedCount": int(removed or 0),
        }
        if archive_changed or rep_changed:
            observed_step = int(max(getattr(self, "_trace_global_step", 0) or 0, 0))
            if int(getattr(self, "_trace_step_index", -1) or -1) >= 0:
                observed_step += 1
            self._last_effective_archive_change_step = max(
                int(getattr(self, "_last_effective_archive_change_step", 0) or 0),
                int(observed_step),
            )
        if archive_changed or rep_changed:
            self._increment_archive_counters(archive_changed=archive_changed, rep_changed=rep_changed)
        if archive_changed:
            self._record_mo_event(
                "archive_update",
                inserted=bool(inserted),
                removedCount=int(removed or 0),
                candidateDecisionScore=self._safe_float(getattr(solution, "decision_score", None)),
                candidateMhc=self._safe_float(getattr(solution, "MHC", None)),
                candidateCr=self._safe_float(getattr(solution, "CR", None)),
                candidateDr=self._safe_float(getattr(solution, "DR", None)),
                candidateAr=self._safe_float(getattr(solution, "AR", None)),
            )
        if rep_changed and self.representative_solution is not None:
            self._record_mo_event(
                "representative_update",
                representativeArchiveIndex=None if self.representative_archive_index is None else int(self.representative_archive_index) + 1,
                representativeDecisionScore=self._safe_float(self.representative_decision_score),
                representativeMhc=self._safe_float(getattr(self.representative_solution, "MHC", None)),
                representativeCr=self._safe_float(getattr(self.representative_solution, "CR", None)),
                representativeDr=self._safe_float(getattr(self.representative_solution, "DR", None)),
                representativeAr=self._safe_float(getattr(self.representative_solution, "AR", None)),
            )

        for env_obj in (solution, getattr(self, "s", None), self.gbest, self.true_gbest):
            if env_obj is None:
                continue
            env_obj.feasible_solution_count = self.feasible_solution_count
            env_obj.best_feasible_cost = self.best_feasible_cost
            env_obj.worst_feasible_cost = self.worst_feasible_cost
            env_obj.best_fitness = self.best_feasible_cost
            env_obj.current_v_worst = self.worst_feasible_cost
            if getattr(env_obj, "mo_objectives_min", None) is not None:
                self._refresh_solution_search_metrics(env_obj)
        return bool(archive_changed or rep_changed)

    def _layout_score(self, solution):
        proxy_score = float(getattr(solution, "decision_score", math.inf))
        return (
            0 if getattr(solution, "current_is_feasible", False) else 1,
            int(getattr(solution, "current_d_inf", 0)),
            float(getattr(solution, "constraint_violation", 0.0) or 0.0),
            proxy_score,
        )

    def _sample_temperature_floor(self, n_samples=None, target_accept=None):
        n_samples = int(n_samples or self.temperature_floor_samples)
        target_accept = float(target_accept or self.temperature_floor_target_accept)
        fallback = max(self.T_initial * 0.05, 1.0)
        cap = max(self.T_initial * self.temperature_floor_cap_ratio, fallback)

        if self.best_feasible_solution is None or not np.isfinite(getattr(self.best_feasible_solution, "fitness", np.inf)):
            return float(fallback)
        if not 0.0 < target_accept < 1.0:
            target_accept = self.temperature_floor_target_accept

        base_solution = self._light_clone_solution(self.best_feasible_solution)
        allowed_actions = [
            table_idx
            for table_idx, action_idx in enumerate(self.valid_actions)
            if action_idx in self.temperature_floor_action_ids
        ]
        if not allowed_actions:
            allowed_actions = self._get_allowed_action_indices(base_solution)
        if not allowed_actions:
            return float(fallback)

        sample_budget = max(1, min(int(n_samples), len(allowed_actions)))
        sampled_action_table_indices = np.random.permutation(allowed_actions)[:sample_budget]

        current_tilde = self._tilde_energy(base_solution.fitness)
        deltas = []
        for action_table_idx in sampled_action_table_indices:
            action_idx = self.valid_actions[int(action_table_idx)]
            recipe = self.action_recipes[action_idx]
            candidate = self._generate_candidate_by_recipe(base_solution, recipe)
            candidate_tilde = self._tilde_energy(candidate.fitness)
            delta = candidate_tilde - current_tilde
            if np.isfinite(delta) and delta > 0:
                deltas.append(float(delta))

        if not deltas:
            return float(fallback)

        delta_q = float(np.percentile(deltas, self.temperature_floor_quantile))
        sampled_floor = -delta_q / math.log(target_accept)
        if not np.isfinite(sampled_floor):
            return float(fallback)
        return float(min(cap, max(sampled_floor, fallback)))

    def _proxy_gap_band(self, solution):
        if self.representative_solution is None:
            return 3
        current_score = float(getattr(solution, "decision_score", math.inf))
        best_score = float(getattr(self, "representative_decision_score", math.inf))
        if not np.isfinite(current_score) or not np.isfinite(best_score):
            return 3
        gap_ratio = max(current_score - best_score, 0.0) / max(abs(best_score), 1e-8)
        if gap_ratio <= 0.01:
            return 0
        if gap_ratio <= 0.05:
            return 1
        if gap_ratio <= 0.15:
            return 2
        return 3

    def _archive_size_band(self):
        archive_size = len(self.pareto_archive)
        if archive_size <= 1:
            return 0
        if archive_size <= 4:
            return 1
        if archive_size <= 8:
            return 2
        return 3

    def _archive_crowding_band(self, solution):
        if not getattr(solution, "current_is_feasible", False) or len(self.pareto_archive) < 2:
            return 3
        current_vector = MO_FBSUtil.normalize_objective_vector(
            getattr(solution, "mo_objectives_min", None),
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
        )
        if current_vector is None:
            return 3

        distances = []
        for candidate in self.pareto_archive:
            candidate_vector = MO_FBSUtil.normalize_objective_vector(
                getattr(candidate, "mo_objectives_min", None),
                ideal=self.mo_ideal,
                nadir=self.mo_nadir,
            )
            if candidate_vector is None:
                continue
            distance = float(np.linalg.norm(current_vector - candidate_vector, ord=2))
            if distance > 1e-12:
                distances.append(distance)
        if not distances:
            return 0
        nearest = min(distances)
        if nearest <= 0.05:
            return 0
        if nearest <= 0.15:
            return 1
        if nearest <= 0.30:
            return 2
        return 3

    def state_encoder(self, solution):
        d_band = self._d_inf_band(solution.current_d_inf)
        t_band = self._temperature_band()
        proxy_band = self._proxy_gap_band(solution)
        archive_band = self._archive_size_band()
        stagnation_band = self._stagnation_band()
        hist_band = self._histogram_band(solution.fitness)
        crowding_band = self._archive_crowding_band(solution)
        state = d_band
        for band in (t_band, proxy_band, archive_band, stagnation_band, hist_band, crowding_band):
            state = state * 4 + band
        return state

    def _agent_state_context(self, solution):
        return None

    def _nondominated_acceptance_cap(self):
        progress_ratio = min(max(float(getattr(self, "current_progress_ratio", 0.0) or 0.0), 0.0), 1.0)
        temperature_ratio = min(
            max(float(getattr(self, "T", 0.0) or 0.0) / max(float(getattr(self, "T_initial", 1.0) or 1.0), 1e-8), 0.0),
            1.0,
        )
        adaptive_factor = max(0.0, 0.75 * (1.0 - progress_ratio) + 0.25 * temperature_ratio)
        cap_value = self.mo_nondominated_accept_cap_low + (
            self.mo_nondominated_accept_cap_high - self.mo_nondominated_accept_cap_low
        ) * adaptive_factor
        # 前中期保持足够探索，避免过早收缩
        if progress_ratio <= 0.30:
            cap_value = max(cap_value, self.mo_nondominated_accept_cap_early_floor)
        elif progress_ratio <= 0.70:
            cap_value = max(cap_value, self.mo_nondominated_accept_cap_mid_floor)
        else:
            # 后 30% 逐步收紧上限，提高收敛稳定性
            late_progress = min(max((progress_ratio - 0.70) / 0.30, 0.0), 1.0)
            late_stage_cap = self.mo_nondominated_accept_cap_late_max_start + (
                self.mo_nondominated_accept_cap_late_max_end - self.mo_nondominated_accept_cap_late_max_start
            ) * late_progress
            cap_value = min(cap_value, late_stage_cap)
        return float(np.clip(cap_value, self.mo_nondominated_accept_cap_low, self.mo_nondominated_accept_cap_high))

    def _should_trigger_local_search(self, global_step, archive_improved, relative_improvement):
        if int(global_step) - int(getattr(self, "last_local_search_step", -10**9)) < int(
            self._effective_local_search_cooldown_steps()
        ):
            return False
        if bool(archive_improved):
            return True
        if float(getattr(self, "current_progress_ratio", 0.0) or 0.0) >= self.local_search_disable_after_progress:
            return False
        return float(relative_improvement) >= float(self.local_search_min_rel_improvement)

    def _effective_local_search_cooldown_steps(self):
        base_cooldown = int(max(1, int(getattr(self, "local_search_cooldown_steps", 1) or 1)))
        if not bool(getattr(self, "local_search_backoff_enable", True)):
            return base_cooldown
        failure_streak = int(max(0, int(getattr(self, "local_search_failure_streak", 0) or 0)))
        exp_cap = int(max(0, int(getattr(self, "local_search_backoff_exp_cap", 0) or 0)))
        multiplier = 2 ** min(failure_streak, exp_cap)
        cooldown = int(base_cooldown * max(1, multiplier))
        max_steps = int(max(base_cooldown, int(getattr(self, "local_search_cooldown_max_steps", base_cooldown) or base_cooldown)))
        return int(min(cooldown, max_steps))

    def _update_local_search_backoff(self, improved):
        # 局部搜索连续失败时指数退避，成功时立即重置
        if bool(improved):
            self.local_search_failure_streak = 0
            return
        self.local_search_failure_streak = int(getattr(self, "local_search_failure_streak", 0) or 0) + 1

    def _nonarchive_stagnation_windows(self):
        current_step = int(max(getattr(self, "_trace_global_step", 0) or 0, 0))
        last_change_step = int(max(getattr(self, "_last_effective_archive_change_step", 0) or 0, 0))
        stale_steps = max(current_step - last_change_step, 0)
        return float(stale_steps) / float(max(1, int(getattr(self, "mo_trace_interval", 1000) or 1000)))

    def _apply_late_nonarchive_gate(self, probability, archive_would_change):
        """仅在后段停滞期对非入档候选进行概率门控，降低无效漂移。"""
        probability = float(np.clip(probability, 0.0, 1.0))
        if not bool(getattr(self, "mo_late_nonarchive_gate_enabled", False)):
            return probability, False, 1.0, 1.0, 0.0, self._nonarchive_stagnation_windows()
        if bool(archive_would_change):
            return probability, False, 1.0, 1.0, 0.0, self._nonarchive_stagnation_windows()
        gate_start = float(getattr(self, "mo_late_nonarchive_gate_start", 0.88) or 0.88)
        progress_ratio = min(max(float(getattr(self, "current_progress_ratio", 0.0) or 0.0), 0.0), 1.0)
        if progress_ratio < float(self.mo_late_nonarchive_stagnation_min_progress):
            return probability, False, 1.0, 1.0, 0.0, self._nonarchive_stagnation_windows()

        stagnation_windows = self._nonarchive_stagnation_windows()
        trigger_windows = float(max(1, int(self.mo_late_nonarchive_stagnation_windows)))
        if stagnation_windows < trigger_windows:
            return probability, False, 1.0, 1.0, 0.0, stagnation_windows

        progress_phase = 0.0
        if gate_start < 1.0:
            progress_phase = min(max((progress_ratio - gate_start) / max(1.0 - gate_start, 1e-8), 0.0), 1.0)
        stagnation_phase = min(
            max(
                (stagnation_windows - trigger_windows + 1.0)
                / float(max(1, int(self.mo_late_nonarchive_stagnation_ramp_windows))),
                0.0,
            ),
            1.0,
        )
        late_phase = min(max(max(progress_phase, stagnation_phase), 0.0), 1.0)
        scale = 1.0 - (1.0 - float(self.mo_late_nonarchive_prob_scale_end)) * float(late_phase)
        cap = float(self.mo_late_nonarchive_prob_cap_start) + (
            float(self.mo_late_nonarchive_prob_cap_end) - float(self.mo_late_nonarchive_prob_cap_start)
        ) * float(late_phase)
        gated_probability = float(min(probability * scale, cap))
        return float(np.clip(gated_probability, 0.0, 1.0)), True, float(scale), float(cap), float(late_phase), float(stagnation_windows)

    def _accept_candidate_with_context(self, current_solution, candidate_solution):
        comparison = MO_FBSUtil.compare_solution_quality(candidate_solution, current_solution)
        current_tilde = self._tilde_energy(current_solution.fitness)
        candidate_tilde = self._tilde_energy(candidate_solution.fitness)
        archive_preview, archive_would_change, _ = MO_FBSUtil.update_pareto_archive(
            self.pareto_archive,
            candidate_solution,
            max_size=self.archive_limit,
            clone_fn=lambda item: item,
            quality_gate_when_full=self.archive_quality_gate_when_full,
            quality_hv_tol=self.archive_quality_hv_tol,
            quality_spacing_tol=self.archive_quality_spacing_tol,
            spacing_guard_when_full=bool(getattr(self, "archive_spacing_guard_when_full", False)),
            spacing_guard_rel_tol=float(getattr(self, "archive_spacing_guard_rel_tol", 0.0) or 0.0),
            spacing_guard_hv_gain_rel=float(getattr(self, "archive_spacing_guard_hv_gain_rel", 0.0) or 0.0),
            require_candidate_retained=bool(getattr(self, "archive_require_candidate_retained", False)),
            ideal=self.mo_ideal,
            nadir=self.mo_nadir,
        )
        _ = archive_preview

        if comparison < 0:
            accept = True
            probability = 1.0
            raw_probability = 1.0
            probability_cap = 1.0
            late_gate_applied = False
            late_gate_scale = 1.0
            late_gate_cap = 1.0
            late_gate_phase = 0.0
            late_gate_stagnation_windows = self._nonarchive_stagnation_windows()
        elif comparison > 0:
            accept = False
            probability = 0.0
            raw_probability = 0.0
            probability_cap = 0.0
            late_gate_applied = False
            late_gate_scale = 1.0
            late_gate_cap = 0.0
            late_gate_phase = 0.0
            late_gate_stagnation_windows = self._nonarchive_stagnation_windows()
        else:
            if candidate_tilde <= current_tilde + 1e-12:
                raw_probability = 1.0
                probability_cap = 1.0
                probability = 1.0
                (
                    probability,
                    late_gate_applied,
                    late_gate_scale,
                    late_gate_cap,
                    late_gate_phase,
                    late_gate_stagnation_windows,
                ) = self._apply_late_nonarchive_gate(
                    probability,
                    archive_would_change=archive_would_change,
                )
                accept = bool(np.random.rand() < probability)
            else:
                exponent = (current_tilde - candidate_tilde) / max(self.T, 1e-12)
                exponent = max(min(exponent, 700.0), -700.0)
                raw_probability = float(min(1.0, math.exp(exponent)))
                probability_cap = self._nondominated_acceptance_cap()
                probability = float(min(raw_probability, probability_cap))
                (
                    probability,
                    late_gate_applied,
                    late_gate_scale,
                    late_gate_cap,
                    late_gate_phase,
                    late_gate_stagnation_windows,
                ) = self._apply_late_nonarchive_gate(
                    probability,
                    archive_would_change=archive_would_change,
                )
                accept = bool(np.random.rand() < probability)

        self._last_transition_meta = {
            "comparison": comparison,
            "archive_would_change": bool(archive_would_change),
            "raw_probability": float(raw_probability),
            "probability_cap": float(probability_cap),
            "late_nonarchive_gate_applied": bool(late_gate_applied),
            "late_nonarchive_gate_scale": float(late_gate_scale),
            "late_nonarchive_gate_cap": float(late_gate_cap),
            "late_nonarchive_gate_phase": float(late_gate_phase),
            "late_nonarchive_stagnation_windows": float(late_gate_stagnation_windows),
            "probability_after_gate": float(probability),
            "current_proxy": float(getattr(current_solution, "proxy_energy", current_solution.fitness)),
            "candidate_proxy": float(getattr(candidate_solution, "proxy_energy", candidate_solution.fitness)),
            "current_violation": float(getattr(current_solution, "constraint_violation", 0.0) or 0.0),
            "candidate_violation": float(getattr(candidate_solution, "constraint_violation", 0.0) or 0.0),
            "current_d_inf": int(getattr(current_solution, "current_d_inf", 0) or 0),
            "candidate_d_inf": int(getattr(candidate_solution, "current_d_inf", 0) or 0),
            "accepted": bool(accept),
        }
        return bool(accept), float(probability), float(current_tilde), float(candidate_tilde)

    def _accept_candidate(self, current_cost, candidate_cost):
        candidate = getattr(self, "_pending_candidate", None)
        current = getattr(self, "s", None)
        if candidate is None or current is None:
            return super()._accept_candidate(current_cost, candidate_cost)
        return self._accept_candidate_with_context(current, candidate)

    def _archive_candidate_before_current_update(self, candidate, accept):
        return False

    def _compute_transition_reward(
        self,
        previous_cost,
        next_cost,
        previous_d_inf,
        next_d_inf,
        previous_best_feasible,
        accept,
    ):
        meta = dict(getattr(self, "_last_transition_meta", {}) or {})
        reward = 0.0
        comparison = int(meta.get("comparison", 0))
        if comparison < 0:
            reward += 1.2
        elif comparison > 0:
            reward -= 0.8
        else:
            proxy_delta = float(meta.get("current_proxy", previous_cost) - meta.get("candidate_proxy", next_cost))
            reward += float(np.clip(proxy_delta, -1.0, 1.0))

        if accept and meta.get("archive_would_change", False):
            reward += 1.0
        elif (not accept) and meta.get("archive_would_change", False):
            reward -= 0.15

        reward += 0.30 * (float(previous_d_inf) - float(next_d_inf))
        reward += 0.10 * float(
            np.clip(meta.get("current_violation", 0.0) - meta.get("candidate_violation", 0.0), -5.0, 5.0)
        )
        if previous_d_inf > 0 and next_d_inf == 0:
            reward += 0.50
        if accept and np.isfinite(next_cost) and np.isfinite(previous_best_feasible) and next_cost < previous_best_feasible:
            reward += 0.30
        if not accept:
            reward -= 0.15
        return float(np.clip(reward, -3.0, 3.0))

    def _current_best_energy_for_logging(self):
        if np.isfinite(self.representative_decision_score):
            return float(self.representative_decision_score)
        return super()._current_best_energy_for_logging()

    def _attempt_diversification(self, global_step, force=False):
        if (not force) and (global_step - self.last_diversify_step < max(10, self.t_max // 4)):
            return
        previous_context = dict(getattr(self, "_current_action_context", {}) or {})
        self._set_action_context("diversify", action_name="diversify_recipe")
        candidate = copy.deepcopy(self.s)
        self._apply_recipe(candidate, self.diversify_recipe)
        self._evaluate_solution(candidate)
        previous_best = self.best_feasible_cost
        previous_cost = self.s.fitness
        previous_d_inf = self.s.current_d_inf
        self._pending_candidate = candidate
        accept, prob, _, _ = self._accept_candidate(self.s.fitness, candidate.fitness)
        self._pending_candidate = None
        self.prob_history.append(prob)
        candidate_archive_improved = self._archive_candidate_before_current_update(candidate, accept=accept)
        if accept:
            self.s = candidate
            self.current_energy = self.s.fitness
            archive_improved = bool(candidate_archive_improved)
            if not archive_improved:
                archive_improved = self._observe_feasible_state(self.s)
            improved = bool(archive_improved)
            reward = self._compute_transition_reward(
                previous_cost,
                self.s.fitness,
                previous_d_inf,
                self.s.current_d_inf,
                previous_best,
                accept=True,
            )
            if improved:
                self.no_improve_steps = 0
            else:
                self.no_improve_steps += 1
        else:
            improved = bool(candidate_archive_improved)
            if improved:
                self.no_improve_steps = 0
            else:
                self.no_improve_steps += 1
            reward = self._compute_transition_reward(
                previous_cost,
                self.s.fitness,
                previous_d_inf,
                self.s.current_d_inf,
                previous_best,
                accept=False,
            )
        self._note_diversification(accepted=accept)
        self._record_mo_event(
            "diversification",
            forced=bool(force),
            accepted=bool(accept),
            improved=bool(improved),
            probability=self._safe_float(prob),
            reward=self._safe_float(reward),
        )
        if accept:
            self._update_histogram(self.s.fitness)
        else:
            self._update_histogram(self.current_energy)
        self.modified_energy_history.append(self._tilde_energy(self.s.fitness))
        self.energy_history.append(self.s.fitness)
        self.last_diversify_step = global_step
        self.diversification_count += 1
        self.T = max(self.T, self.T_min)
        self._current_action_context = previous_context
        return reward

    def _attempt_reheating(self, global_step, total_steps, fast_time):
        if not self.reheat_enabled:
            return fast_time, False
        if self.reheat_episode_count >= self.reheat_max_per_episode:
            return fast_time, False
        progress_ratio = float(global_step) / float(max(1, total_steps))
        if progress_ratio > self.reheat_progress_cap_ratio:
            return fast_time, False
        if self.no_improve_steps < self.reheat_no_improve_threshold:
            return fast_time, False
        if self.accept_rate_window > self.reheat_accept_rate_threshold:
            return fast_time, False
        if self.T >= self.reheat_temp_gate_ratio * self.T_initial:
            return fast_time, False
        if global_step - self.last_reheat_step < self.reheat_cooldown_steps:
            return fast_time, False

        previous_temperature = float(self.T)
        target_low = self.reheat_target_low_ratio * self.T_initial
        target_high = self.reheat_target_high_ratio * self.T_initial
        self.T = min(target_high, max(self.T, target_low))

        no_improve_before = self.no_improve_steps
        self.last_reheat_step = global_step
        self.reheat_episode_count += 1
        self.reheat_trigger_count += 1
        self.no_improve_steps = 0
        self._note_reheat()
        self._record_mo_event(
            "reheat_triggered",
            previousTemperature=self._safe_float(previous_temperature),
            newTemperature=self._safe_float(self.T),
            noImproveBefore=int(no_improve_before),
            acceptRateWindow=self._safe_float(self.accept_rate_window),
            progressRatio=float(progress_ratio),
        )

        self._attempt_diversification(global_step, force=True)
        fast_time, _ = self._elite_intensification(fast_time)
        if self.enable_reheat_logging:
            logger.info(
                f"Reheat triggered | step: {global_step} | T: {previous_temperature:.6f} -> {self.T:.6f} | "
                f"accept_rate_window: {self.accept_rate_window:.3f} | no_improve: {no_improve_before}"
            )
        return fast_time, True


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
            "dr": float(getattr(solution, "DR", 0.0)),
            "ar": float(getattr(solution, "AR", 0.0)),
            "dInf": int(getattr(solution, "current_d_inf", 0) or 0),
            "constraintViolation": float(getattr(solution, "constraint_violation", 0.0) or 0.0),
            "isFeasible": bool(getattr(solution, "current_is_feasible", False)),
            "moObjectivesRaw": np.asarray(getattr(solution, "mo_objectives_raw", []), dtype=float).tolist(),
            "moObjectivesMin": np.asarray(getattr(solution, "mo_objectives_min", []), dtype=float).tolist(),
            "solution": solution_array,
        }

    @staticmethod
    def _normalize_algorithm_tag(name):
        raw = str(name or "ELP_DRL_MO")
        cleaned = "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in raw)
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("_") or "ELP_DRL_MO"

    def _save_pareto_archive(self, start_time, algorithm_name=None):
        if not self.pareto_archive:
            self.pareto_archive_path = None
            return None

        repo_root = Path(__file__).resolve().parents[2]
        archive_dir = Path(config.RESULT_PATH) / "pareto_archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = start_time.strftime("%Y%m%d_%H%M%S_%f") if start_time is not None else datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        algo_tag = self._normalize_algorithm_tag(algorithm_name)
        filename = f"{self.instance_name}-{algo_tag}-{timestamp}.json"
        archive_path = archive_dir / filename

        payload = {
            "instance": self.instance_name,
            "algorithm": algo_tag,
            "objectiveDefinitionVersion": MO_ReferenceFrontUtil.OBJECTIVE_DEFINITION_VERSION,
            "arSatisfactionMode": "paper_triangular",
            "arLowerAspectRatio": 1.0,
            "arOptimalAspectRatio": 1.5,
            "generatedAt": datetime.datetime.now().isoformat(timespec="seconds"),
            "archiveSize": len(self.pareto_archive),
            "representativeArchiveIndex": None if self.representative_archive_index is None else int(self.representative_archive_index) + 1,
            "representativeDecisionScore": None if not np.isfinite(self.representative_decision_score) else float(self.representative_decision_score),
            "items": [self._archive_item_payload(solution, index + 1) for index, solution in enumerate(self.pareto_archive)],
        }
        archive_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.pareto_archive_path = archive_path.resolve().relative_to(repo_root).as_posix()
        return self.pareto_archive_path

    def _ensure_pymoo_available(self):
        if NSGA2 is not None and MOEAD is not None and SPEA2 is not None:
            return
        raise ImportError(
            "缺少 pymoo 依赖，无法运行 NSGA-II/MOEA-D/SPEA2 基线。请先安装 `pymoo>=0.6.1`。"
        ) from _PYMOO_IMPORT_ERROR

    def _evaluate_action_sequence(self, base_solution, action_sequence):
        candidate = self._light_clone_solution(base_solution)
        action_count = int(max(1, len(self.valid_actions)))
        for action_token in np.asarray(action_sequence, dtype=int).reshape(-1):
            table_idx = int(np.clip(action_token, 0, action_count - 1))
            action_idx = self.valid_actions[table_idx]
            candidate = self.generate_candidate_by_action(candidate, action_idx)
        return candidate

    def _reset_baseline_archive_state(self):
        self.pareto_archive = []
        self.representative_solution = None
        self.representative_decision_score = math.inf
        self.representative_archive_index = None
        self.mo_ideal = None
        self.mo_nadir = None
        self.pareto_archive_path = None
        self.best_feasible_cost = math.inf
        self.worst_feasible_cost = None
        self.best_feasible_solution = None
        self.feasible_solution_count = 0
        self.mo_worst_feasible_mhc = None
        self.gbest_update_count = 0

    @staticmethod
    def _compute_moead_partitions(population_size, objective_count=4):
        objective_count = int(max(2, objective_count))
        population_size = int(max(2, population_size))
        partitions = 1
        while math.comb(partitions + objective_count - 1, objective_count - 1) < population_size:
            partitions += 1
        return partitions

    def _collect_result_sequences(self, result):
        sequences = []
        population = getattr(result, "pop", None)
        if population is not None:
            pop_x = population.get("X")
            if pop_x is not None:
                pop_x = np.asarray(pop_x, dtype=int)
                if pop_x.ndim == 1:
                    pop_x = pop_x.reshape(1, -1)
                sequences.extend(pop_x.tolist())
        if not sequences and getattr(result, "X", None) is not None:
            result_x = np.asarray(result.X, dtype=int)
            if result_x.ndim == 1:
                result_x = result_x.reshape(1, -1)
            sequences.extend(result_x.tolist())
        return sequences

    @staticmethod
    def _baseline_float_env(name, default):
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            return float(default)
        try:
            return float(str(raw).strip())
        except Exception:
            return float(default)

    def _baseline_solution_score(self, solution):
        if solution is None:
            return math.inf
        objectives = getattr(solution, "mo_objectives_min", None)
        if objectives is not None:
            try:
                score = MO_FBSUtil.surrogate_energy(
                    objectives,
                    ideal=self.mo_ideal,
                    nadir=self.mo_nadir,
                    weights=self.mo_weights,
                )
                if np.isfinite(score):
                    return float(score)
            except Exception:
                pass
        for attr in ("decision_score", "proxy_energy", "fitness"):
            value = getattr(solution, attr, math.inf)
            try:
                value = float(value)
            except Exception:
                continue
            if np.isfinite(value):
                return float(value)
        return math.inf

    def _baseline_solution_rank_key(self, solution):
        if solution is None:
            return (1, math.inf, math.inf, math.inf)
        is_feasible, d_inf, violation = MO_FBSUtil.constraint_signature(solution)
        return (
            0 if is_feasible else 1,
            int(max(d_inf, 0)),
            float(max(violation, 0.0)),
            self._baseline_solution_score(solution),
        )

    def _baseline_solution_preferred(self, candidate, incumbent):
        if candidate is None:
            return False
        if incumbent is None:
            return True
        quality_cmp = MO_FBSUtil.compare_solution_quality(candidate, incumbent)
        if quality_cmp < 0:
            return True
        if quality_cmp > 0:
            return False
        return self._baseline_solution_score(candidate) < self._baseline_solution_score(incumbent) - 1e-12

    def _truncate_pso_guide_archive(self, entries, limit):
        limit = int(max(1, limit))
        entries = list(entries or [])
        while len(entries) > limit:
            candidates = [entry["candidate"] for entry in entries]
            normalized, _, _ = MO_FBSUtil._normalized_archive_matrix(
                candidates,
                ideal=self.mo_ideal,
                nadir=self.mo_nadir,
            )
            if normalized.shape[0] != len(entries) or normalized.shape[0] <= 1:
                remove_idx = max(range(len(entries)), key=lambda idx: self._baseline_solution_score(entries[idx]["candidate"]))
                entries.pop(remove_idx)
                continue
            distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            nearest = np.min(distances, axis=1)
            remove_idx = min(
                range(len(entries)),
                key=lambda idx: (
                    float(nearest[idx]) if np.isfinite(nearest[idx]) else math.inf,
                    -self._baseline_solution_score(entries[idx]["candidate"]),
                ),
            )
            entries.pop(remove_idx)
        return entries

    def _update_pso_guide_archive(self, entries, position, candidate, limit):
        if not getattr(candidate, "current_is_feasible", False):
            return list(entries or [])
        objectives = getattr(candidate, "mo_objectives_min", None)
        if objectives is None:
            return list(entries or [])

        new_entry = {
            "position": np.asarray(position, dtype=float).copy(),
            "candidate": candidate,
        }
        updated = []
        dominated_or_duplicate = False
        replaced_duplicate = False
        for entry in list(entries or []):
            other = entry["candidate"]
            other_objectives = getattr(other, "mo_objectives_min", None)
            if other_objectives is None:
                continue
            if MO_FBSUtil._duplicate_objectives(candidate, other):
                dominated_or_duplicate = True
                if not replaced_duplicate and self._baseline_solution_preferred(candidate, other):
                    updated.append(new_entry)
                    replaced_duplicate = True
                else:
                    updated.append(entry)
                continue
            if MO_FBSUtil.pareto_dominates(other_objectives, objectives):
                dominated_or_duplicate = True
                updated.append(entry)
                continue
            if MO_FBSUtil.pareto_dominates(objectives, other_objectives):
                continue
            updated.append(entry)

        if not dominated_or_duplicate:
            updated.append(new_entry)
        return self._truncate_pso_guide_archive(updated, limit)

    def _sample_pso_guide_position(self, guide_entries, pbest_positions, pbest_candidates, rng):
        if guide_entries:
            guide_idx = int(rng.integers(0, len(guide_entries)))
            return np.asarray(guide_entries[guide_idx]["position"], dtype=float).copy()
        valid_indices = [idx for idx, candidate in enumerate(pbest_candidates) if candidate is not None]
        if valid_indices:
            best_idx = min(valid_indices, key=lambda idx: self._baseline_solution_rank_key(pbest_candidates[idx]))
            return np.asarray(pbest_positions[best_idx], dtype=float).copy()
        upper = max(0, len(self.valid_actions) - 1)
        return rng.uniform(0.0, float(upper), size=int(pbest_positions.shape[1]))

    def _run_pso_baseline(
        self,
        population_size=64,
        generations=80,
        sequence_length=None,
        seed=None,
    ):
        population_size = int(max(8, population_size))
        generations = int(max(1, generations))
        sequence_length = int(max(1, sequence_length if sequence_length is not None else self.t_max))
        run_seed = None if seed is None else int(seed)
        rng = np.random.default_rng(run_seed)

        action_count = int(max(1, len(self.valid_actions)))
        action_upper = float(max(0, action_count - 1))
        inertia = float(np.clip(self._baseline_float_env("ELP_MO_PSO_INERTIA", 0.72), 0.0, 1.5))
        cognitive = float(max(0.0, self._baseline_float_env("ELP_MO_PSO_C1", 1.49)))
        social = float(max(0.0, self._baseline_float_env("ELP_MO_PSO_C2", 1.49)))
        vmax_ratio = float(np.clip(self._baseline_float_env("ELP_MO_PSO_VMAX_RATIO", 0.50), 0.0, 2.0))
        vmax = max(1.0, action_upper * vmax_ratio) if action_count > 1 else 0.0
        default_mutation_prob = 1.0 / float(max(1, sequence_length))
        mutation_prob = float(np.clip(self._baseline_float_env("ELP_MO_PSO_MUTATION_PROB", default_mutation_prob), 0.0, 1.0))
        guide_limit = int(max(self.archive_limit, population_size))

        self._reset_baseline_archive_state()
        start_time = datetime.datetime.now()
        fast_time = start_time

        base_solution = self._light_clone_solution(self.s)
        self._evaluate_solution(base_solution)
        self._observe_feasible_state(base_solution)

        positions = rng.integers(0, action_count, size=(population_size, sequence_length)).astype(float)
        velocities = (
            rng.uniform(-vmax, vmax, size=(population_size, sequence_length))
            if vmax > 0.0
            else np.zeros((population_size, sequence_length), dtype=float)
        )
        pbest_positions = positions.copy()
        pbest_candidates = [None for _ in range(population_size)]
        guide_entries = []
        best_observed = self.best_feasible_cost
        evaluations = 0

        for generation in range(generations):
            for particle_idx in range(population_size):
                sequence = np.rint(positions[particle_idx]).astype(int)
                sequence = np.clip(sequence, 0, action_count - 1)
                candidate = self._evaluate_action_sequence(base_solution, sequence)
                evaluations += 1
                changed = self._observe_feasible_state(candidate)
                if changed and np.isfinite(self.best_feasible_cost) and self.best_feasible_cost < best_observed:
                    best_observed = float(self.best_feasible_cost)
                    fast_time = datetime.datetime.now()
                if self._baseline_solution_preferred(candidate, pbest_candidates[particle_idx]):
                    pbest_candidates[particle_idx] = candidate
                    pbest_positions[particle_idx] = positions[particle_idx].copy()
                guide_entries = self._update_pso_guide_archive(
                    guide_entries,
                    positions[particle_idx],
                    candidate,
                    guide_limit,
                )

            if generation >= generations - 1:
                break
            for particle_idx in range(population_size):
                guide_position = self._sample_pso_guide_position(guide_entries, pbest_positions, pbest_candidates, rng)
                r1 = rng.random(sequence_length)
                r2 = rng.random(sequence_length)
                velocities[particle_idx] = (
                    inertia * velocities[particle_idx]
                    + cognitive * r1 * (pbest_positions[particle_idx] - positions[particle_idx])
                    + social * r2 * (guide_position - positions[particle_idx])
                )
                if vmax > 0.0:
                    velocities[particle_idx] = np.clip(velocities[particle_idx], -vmax, vmax)
                positions[particle_idx] = np.clip(positions[particle_idx] + velocities[particle_idx], 0.0, action_upper)
                if mutation_prob > 0.0 and action_count > 1:
                    mutate_mask = rng.random(sequence_length) < mutation_prob
                    if np.any(mutate_mask):
                        positions[particle_idx, mutate_mask] = rng.integers(0, action_count, size=int(np.sum(mutate_mask)))
                        velocities[particle_idx, mutate_mask] = 0.0

        self._refresh_archive_state()
        end_time = datetime.datetime.now()
        iteration_count = int(evaluations)

        best_solution = self.best_feasible_solution if self.best_feasible_solution is not None else self._light_clone_solution(base_solution)
        if self.representative_solution is not None:
            best_solution = copy.deepcopy(self.representative_solution)
        is_valid = bool(getattr(best_solution, "current_is_feasible", False))
        best_energy = float(getattr(best_solution, "decision_score", math.inf))
        if not np.isfinite(best_energy):
            best_energy = float(self.best_feasible_cost if np.isfinite(self.best_feasible_cost) else best_solution.fitness)

        stable_decision_score = self._safe_float(getattr(best_solution, "proxy_energy", None))
        archive_path = self._save_pareto_archive(start_time, algorithm_name="MO_BASELINE_PSO")
        reference_metrics = self._compute_reference_front_metrics()
        archive_hypervolume = reference_metrics["archive_hypervolume"]
        archive_spacing = reference_metrics["archive_spacing"]
        self.last_run_payload = {
            "pareto_archive_path": archive_path,
            "pareto_size": len(self.pareto_archive),
            "rep_mhc": None if best_solution is None else float(getattr(best_solution, "MHC", math.inf)),
            "rep_cr": None if best_solution is None else float(getattr(best_solution, "CR", 0.0)),
            "rep_dr": None if best_solution is None else float(getattr(best_solution, "DR", 0.0)),
            "rep_ar": None if best_solution is None else float(getattr(best_solution, "AR", 0.0)),
            "decision_score": self._safe_float(best_energy),
            "stable_decision_score": stable_decision_score,
            "archive_hypervolume": archive_hypervolume,
            "archive_spacing": archive_spacing,
            "archive_igd": reference_metrics["archive_igd"],
            "reference_front_path": reference_metrics["reference_front_path"],
            "reference_front_size": reference_metrics["reference_front_size"],
            "reference_front_archive_count": reference_metrics["reference_front_archive_count"],
            "archive_hypervolume_mode": reference_metrics["archive_hypervolume_mode"],
            "archive_hypervolume_reference_point": reference_metrics["archive_hypervolume_reference_point"],
            "mo_run_id": None,
            "mo_bundle_dir": None,
            "mo_trace_path": None,
            "mo_events_path": None,
            "mo_action_stats_path": None,
            "mo_run_summary_path": None,
            "baseline_algorithm": "PSO",
            "baseline_population": int(population_size),
            "baseline_generations": int(generations),
            "baseline_sequence_length": int(sequence_length),
            "baseline_seed": run_seed,
            "baseline_pso_inertia": float(inertia),
            "baseline_pso_c1": float(cognitive),
            "baseline_pso_c2": float(social),
            "baseline_pso_vmax_ratio": float(vmax_ratio),
            "baseline_pso_mutation_prob": float(mutation_prob),
            "baseline_pso_guide_limit": int(guide_limit),
        }
        return iteration_count, is_valid, best_solution, best_energy, start_time, end_time, fast_time

    def run_moea_baseline(
        self,
        algorithm_name,
        population_size=64,
        generations=80,
        sequence_length=None,
        seed=None,
    ):
        algo_key = str(algorithm_name or "").strip().lower().replace("-", "").replace("/", "")
        if algo_key in {"pso", "mopso"}:
            return self._run_pso_baseline(
                population_size=population_size,
                generations=generations,
                sequence_length=sequence_length,
                seed=seed,
            )

        self._ensure_pymoo_available()
        if algo_key not in {"nsga2", "moead", "spea2"}:
            raise ValueError(f"Unsupported baseline algorithm: {algorithm_name}")

        population_size = int(max(8, population_size))
        generations = int(max(1, generations))
        sequence_length = int(max(1, sequence_length if sequence_length is not None else self.t_max))
        run_seed = None if seed is None else int(seed)

        self._reset_baseline_archive_state()
        start_time = datetime.datetime.now()
        fast_time = start_time

        base_solution = self._light_clone_solution(self.s)
        self._evaluate_solution(base_solution)
        self._observe_feasible_state(base_solution)

        problem = _ActionSequenceMOProblem(
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
            n_partitions = self._compute_moead_partitions(population_size, objective_count=4)
            ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=n_partitions)
            algorithm = MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=min(20, max(2, len(ref_dirs) - 1)),
                prob_neighbor_mating=0.7,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
            )
            effective_population = int(len(ref_dirs))

        result = minimize(
            problem,
            algorithm,
            termination=get_termination("n_gen", generations),
            seed=run_seed,
            save_history=False,
            verbose=False,
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
        iteration_count = int(effective_population * generations)

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
        archive_hypervolume = reference_metrics["archive_hypervolume"]
        archive_spacing = reference_metrics["archive_spacing"]
        self.last_run_payload = {
            "pareto_archive_path": archive_path,
            "pareto_size": len(self.pareto_archive),
            "rep_mhc": None if best_solution is None else float(getattr(best_solution, "MHC", math.inf)),
            "rep_cr": None if best_solution is None else float(getattr(best_solution, "CR", 0.0)),
            "rep_dr": None if best_solution is None else float(getattr(best_solution, "DR", 0.0)),
            "rep_ar": None if best_solution is None else float(getattr(best_solution, "AR", 0.0)),
            "decision_score": self._safe_float(best_energy),
            "stable_decision_score": stable_decision_score,
            "archive_hypervolume": archive_hypervolume,
            "archive_spacing": archive_spacing,
            "archive_igd": reference_metrics["archive_igd"],
            "reference_front_path": reference_metrics["reference_front_path"],
            "reference_front_size": reference_metrics["reference_front_size"],
            "reference_front_archive_count": reference_metrics["reference_front_archive_count"],
            "archive_hypervolume_mode": reference_metrics["archive_hypervolume_mode"],
            "archive_hypervolume_reference_point": reference_metrics["archive_hypervolume_reference_point"],
            "mo_run_id": None,
            "mo_bundle_dir": None,
            "mo_trace_path": None,
            "mo_events_path": None,
            "mo_action_stats_path": None,
            "mo_run_summary_path": None,
            "baseline_algorithm": algo_key.upper(),
            "baseline_population": int(effective_population),
            "baseline_generations": int(generations),
            "baseline_sequence_length": int(sequence_length),
            "baseline_seed": run_seed,
        }
        return iteration_count, is_valid, best_solution, best_energy, start_time, end_time, fast_time

    def _run_impl(self):
        start_time = datetime.datetime.now()
        fast_time = start_time
        self._reset_mo_logging_state()
        self._run_start_time = start_time

        agent_mode = os.getenv("ELP_RL_AGENT", "dqn").strip().lower()
        self.agent_mode = agent_mode
        run_algorithm = os.getenv("ELP_EXP_ALGORITHM", "ELP_DRL_MO4")
        run_remark = os.getenv("ELP_EXP_REMARK", "")
        self.mo_recorder = MO_ExperimentsUtil.MOExperimentRecorder(
            instance=self.instance_name,
            algorithm=run_algorithm,
            start_time=start_time,
            trace_interval=self.mo_trace_interval,
            remark=run_remark,
            result_root=config.RESULT_PATH,
        )

        if agent_mode == "qlearning":
            agent = StandardQLearningAgent(s_dim=16384, a_dim=len(self.valid_actions))
        else:
            agent = StandardDQNAgent(
                s_dim=16384,
                a_dim=len(self.valid_actions),
                epsilon=0.50,
                epsilon_min=self.dqn_epsilon_min,
                epsilon_decay=self.dqn_epsilon_decay,
                gamma=0.95,
                lr=3e-4,
                batch_size=64,
                replay_capacity=50000,
                warmup_steps=self.dqn_warmup_steps,
                update_every=4,
                target_update_every=self.dqn_target_update_every,
                idqn_k=2,
                embedding_dim=32,
                hidden_dim=64,
                context_dim=int(max(0, int(getattr(self, "rl_context_dim", 0) or 0))),
                grad_clip=10.0,
            )
        if not self._bootstrap_until_first_feasible():
            logger.warning("Failed to seed a feasible archive before ELP search.")
        if np.isfinite(self.best_feasible_cost):
            self.best_energy = self.best_feasible_cost
            self.current_energy = self.s.fitness

        global_step = 0
        total_steps = max(1, self.G * self.t_max)
        next_progress_marker_idx = 0
        wall_time_terminated = False

        # TensorBoard writer (optional, independent of dashboard)
        tb_writer = None
        tb_log_dir = None
        if _TENSORBOARD_AVAILABLE and os.environ.get("ELP_TENSORBOARD_ENABLE", "").strip() in {"1", "true", "yes", "on"}:
            tb_log_dir = os.path.join(
                config.RESULT_PATH, "tensorboard_logs",
                run_algorithm or "unknown",
                f"{instance_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            os.makedirs(tb_log_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_log_dir)
            logger.info(f"TensorBoard logging enabled: {tb_log_dir}")
        for episode in range(self.G):
            if self._wall_time_limit_reached(start_time):
                wall_time_terminated = True
                self._record_mo_event(
                    "wall_time_stop",
                    totalIterations=int(global_step),
                    wallTimeLimitSeconds=self._safe_float(getattr(self, "wall_time_limit_seconds", None)),
                    elapsedSeconds=self._safe_float(self._elapsed_seconds_since(start_time)),
                    progressRatio=self._safe_float(float(global_step) / float(max(1, total_steps))),
                )
                break
            if self.worst_feasible_cost is None and not self._bootstrap_until_first_feasible(max_attempts=max(200, 2 * self.t_max)):
                logger.warning(f"Episode {episode}: feasible archive still unavailable.")
                continue
            episode_best_before = self.best_feasible_cost
            self._prepare_episode_start(episode)
            for step_idx in range(self.t_max):
                if self._wall_time_limit_reached(start_time):
                    wall_time_terminated = True
                    self._record_mo_event(
                        "wall_time_stop",
                        totalIterations=int(global_step),
                        wallTimeLimitSeconds=self._safe_float(getattr(self, "wall_time_limit_seconds", None)),
                        elapsedSeconds=self._safe_float(self._elapsed_seconds_since(start_time)),
                        progressRatio=self._safe_float(float(global_step) / float(max(1, total_steps))),
                    )
                    break
                self._trace_global_step = global_step
                self._trace_episode_index = episode
                self._trace_step_index = step_idx
                self.current_progress_ratio = float(global_step) / float(max(1, total_steps))
                current_state_idx = self.state_encoder(self.s)
                current_state_context = self._agent_state_context(self.s)
                allowed_actions = self._get_allowed_action_indices(self.s)
                if isinstance(agent, StandardDQNAgent):
                    action_table_idx = agent.select_action(
                        current_state_idx,
                        allowed_actions=allowed_actions,
                        context_features=current_state_context,
                    )
                else:
                    action_table_idx = agent.select_action(current_state_idx, allowed_actions=allowed_actions)
                real_action_idx = self.valid_actions[action_table_idx]
                previous_cost = self.s.fitness
                previous_d_inf = self.s.current_d_inf
                previous_best_feasible = self.best_feasible_cost

                candidate = self.generate_candidate_by_action(self.s, real_action_idx)
                self._record_action_selection(
                    real_action_idx,
                    previous_cost,
                    candidate.fitness,
                    phase="main",
                )
                self._pending_candidate = candidate
                accept, prob, _, _ = self._accept_candidate(self.s.fitness, candidate.fitness)
                self._pending_candidate = None
                self.prob_history.append(prob)
                self._record_acceptance(accept)
                candidate_archive_improved = self._archive_candidate_before_current_update(candidate, accept=accept)
                improved = False

                if accept:
                    self.s = candidate
                    self.current_energy = self.s.fitness
                    archive_improved = bool(candidate_archive_improved)
                    if not archive_improved:
                        archive_improved = self._observe_feasible_state(self.s)
                    if self._last_transition_meta:
                        self._last_transition_meta["archive_would_change"] = bool(archive_improved)
                    if archive_improved:
                        self._record_action_global_best(real_action_idx, phase="main")
                    accepted_improved = bool(
                        np.isfinite(previous_cost)
                        and np.isfinite(self.s.fitness)
                        and self.s.fitness < previous_cost
                    )
                    self._record_action_acceptance(
                        real_action_idx,
                        previous_cost,
                        self.s.fitness,
                        improved=accepted_improved,
                        phase="main",
                    )
                    improved = bool(archive_improved)
                    relative_improvement = 0.0
                    if np.isfinite(previous_cost) and abs(previous_cost) > 1e-12 and np.isfinite(self.s.fitness):
                        relative_improvement = max((previous_cost - self.s.fitness) / abs(previous_cost), 0.0)
                    trigger_local_search = self._should_trigger_local_search(
                        global_step=global_step,
                        archive_improved=improved,
                        relative_improvement=relative_improvement,
                    )
                    self._update_histogram(self.s.fitness)
                    if improved:
                        self.no_improve_steps = 0
                        fast_time = datetime.datetime.now()
                    else:
                        self.no_improve_steps += 1

                    if trigger_local_search:
                        self.last_local_search_step = int(global_step)
                        local_search_best_before = self.best_feasible_cost
                        local_search_cost_before = self.s.fitness
                        previous_context = dict(getattr(self, "_current_action_context", {}) or {})
                        self._set_action_context("local_search", action_name="greedy_local_search")
                        self.s = self._greedy_local_search(self.s)
                        self.current_energy = self.s.fitness
                        local_search_improved = bool(
                            np.isfinite(self.best_feasible_cost)
                            and self.best_feasible_cost < local_search_best_before
                        )
                        self._note_local_search(improved=local_search_improved)
                        self._update_local_search_backoff(local_search_improved)
                        self._record_mo_event(
                            "local_search",
                            improved=local_search_improved,
                            beforeSearchEnergy=self._safe_float(local_search_cost_before),
                            afterSearchEnergy=self._safe_float(self.s.fitness),
                            failureStreak=int(getattr(self, "local_search_failure_streak", 0) or 0),
                            effectiveCooldown=int(self._effective_local_search_cooldown_steps()),
                        )
                        self._current_action_context = previous_context
                        if self.s.current_is_feasible and self.s.fitness < previous_best_feasible:
                            improved = True

                else:
                    self._update_histogram(self.s.fitness)
                    if candidate_archive_improved:
                        self._record_action_global_best(real_action_idx, phase="main")
                        improved = True
                        self.no_improve_steps = 0
                    else:
                        self.no_improve_steps += 1

                if improved:
                    self.no_improve_steps = 0
                    fast_time = datetime.datetime.now()
                    fast_time, _ = self._elite_intensification(fast_time)

                gbest_recomputed = bool(
                    np.isfinite(self.best_feasible_cost)
                    and self.best_feasible_cost < previous_best_feasible
                )
                if gbest_recomputed:
                    self.T_min = self._sample_temperature_floor()
                    self._record_mo_event(
                        "temperature_floor_update",
                        bestDecisionScore=self._safe_float(self.best_feasible_cost),
                        temperatureMin=self._safe_float(self.T_min),
                    )

                next_state_idx = self.state_encoder(self.s)
                next_state_context = self._agent_state_context(self.s)
                allowed_next_actions = self._get_allowed_action_indices(self.s)
                reward = self._compute_transition_reward(
                    previous_cost,
                    self.s.fitness,
                    previous_d_inf,
                    self.s.current_d_inf,
                    previous_best_feasible,
                    accept,
                )
                done_flag = step_idx == self.t_max - 1
                if isinstance(agent, StandardDQNAgent):
                    loss_value = agent.update_Q(
                        current_state_idx,
                        action_table_idx,
                        reward,
                        next_state_idx,
                        done=done_flag,
                        allowed_next_actions=allowed_next_actions,
                        state_context=current_state_context,
                        next_state_context=next_state_context,
                    )
                else:
                    loss_value = agent.update_Q(
                        current_state_idx,
                        action_table_idx,
                        reward,
                        next_state_idx,
                        done=done_flag,
                        allowed_next_actions=allowed_next_actions,
                    )
                self._last_loss_value = loss_value
                self._update_transition_counters(accept, prob, reward, loss_value)
                self.modified_energy_history.append(self._tilde_energy(self.s.fitness))
                self.energy_history.append(self.s.fitness)
                global_step += 1
                self._trace_global_step = global_step
                while (
                    next_progress_marker_idx < len(self.progress_markers)
                    and (global_step / total_steps) >= self.progress_markers[next_progress_marker_idx]
                ):
                    self._log_training_progress(self.progress_markers[next_progress_marker_idx], start_time)
                    next_progress_marker_idx += 1

                self._record_trace_snapshot(global_step, total_steps, episode, step_idx, agent)
                fast_time, reheated = self._attempt_reheating(global_step, total_steps, fast_time)
                if (not reheated) and self.no_improve_steps >= self.diversify_trigger_no_improve:
                    self._attempt_diversification(global_step)
                self._clear_action_context()

            finalize_episode = getattr(agent, "finalize_episode", None)
            if callable(finalize_episode):
                finalize_episode()

            if wall_time_terminated:
                break

            self.T = max(self.T * self.cooling_per_step, self.T_min)

            if np.isfinite(self.best_feasible_cost) and self.best_feasible_cost < episode_best_before:
                self.episodes_without_improvement = 0
            else:
                self.episodes_without_improvement += 1
            agent.decay_epsilon()

            # TensorBoard per-episode logging
            if tb_writer is not None:
                episode_idx = int(episode)
                tb_writer.add_scalar("Episode/decisionScore", self._safe_float(self.best_feasible_cost), episode_idx)
                tb_writer.add_scalar("Episode/paretoArchiveSize", int(len(self.pareto_archive)), episode_idx)
                tb_writer.add_scalar("Episode/temperature", self._safe_float(self.T), episode_idx)
                tb_writer.add_scalar("Episode/epsilon", self._safe_float(getattr(agent, "epsilon", None)), episode_idx)
                tb_writer.add_scalar("Episode/episodesWithoutImprovement", int(self.episodes_without_improvement), episode_idx)
                if hasattr(self, "_last_loss_value") and self._last_loss_value is not None:
                    tb_writer.add_scalar("Training/loss", float(self._last_loss_value), episode_idx)
                # Archive objective stats
                if self.pareto_archive:
                    mhc_vals = [float(getattr(sol, "MHC", 0) or 0) for sol in self.pareto_archive]
                    cr_vals = [float(getattr(sol, "CR", 0) or 0) for sol in self.pareto_archive]
                    if mhc_vals:
                        tb_writer.add_scalar("Archive/medianMHC", float(np.median(mhc_vals)), episode_idx)
                    if cr_vals:
                        tb_writer.add_scalar("Archive/medianCR", float(np.median(cr_vals)), episode_idx)

        while (not wall_time_terminated) and next_progress_marker_idx < len(self.progress_markers):
            self._log_training_progress(self.progress_markers[next_progress_marker_idx], start_time)
            next_progress_marker_idx += 1

        end_time = datetime.datetime.now()
        self._record_trace_snapshot(global_step, total_steps, max(self.G - 1, 0), max(self.t_max - 1, 0), agent)
        self._refresh_archive_state()

        best_solution = self.best_feasible_solution if self.best_feasible_solution is not None else copy.deepcopy(self.s)
        is_valid = bool(getattr(best_solution, "current_is_feasible", False))
        best_energy = float(self.best_feasible_cost if np.isfinite(self.best_feasible_cost) else best_solution.fitness)
        if self.representative_solution is not None:
            best_solution = copy.deepcopy(self.representative_solution)
            best_energy = float(self.representative_decision_score)
            is_valid = bool(getattr(best_solution, "current_is_feasible", False))
        representative_stable_score = None
        if best_solution is not None:
            representative_stable_score = self._safe_float(getattr(best_solution, "proxy_energy", None))
            if representative_stable_score is None and getattr(best_solution, "mo_objectives_min", None) is not None:
                representative_stable_score = self._safe_float(
                    MO_FBSUtil.surrogate_energy(
                        best_solution.mo_objectives_min,
                        ideal=self.mo_ideal,
                        nadir=self.mo_nadir,
                        weights=self.mo_weights,
                    )
                )
        archive_path = self._save_pareto_archive(start_time, algorithm_name=run_algorithm)
        reference_metrics = self._compute_reference_front_metrics()
        archive_hypervolume = reference_metrics["archive_hypervolume"]
        archive_spacing = reference_metrics["archive_spacing"]
        solution_array = getattr(getattr(best_solution, "fbs_model", None), "array_2d", None)
        if hasattr(solution_array, "tolist"):
            solution_array = solution_array.tolist()
        best_result_seconds = None if fast_time is None else (fast_time - start_time).total_seconds()
        self._record_mo_event(
            "run_completed",
            totalIterations=int(global_step),
            runtimeSeconds=self._safe_float((end_time - start_time).total_seconds()),
            bestResultSeconds=self._safe_float(best_result_seconds),
            archivePath=archive_path,
            wallTimeTerminated=bool(wall_time_terminated),
            wallTimeLimitSeconds=self._safe_float(getattr(self, "wall_time_limit_seconds", None)),
        )
        action_stats = self._build_action_stats_payload(agent, global_step)
        run_summary = {
            "startTime": start_time.isoformat(),
            "fastTime": None if fast_time is None else fast_time.isoformat(),
            "endTime": end_time.isoformat(),
            "runtimeSeconds": (end_time - start_time).total_seconds(),
            "bestResultSeconds": best_result_seconds,
            "iterations": int(global_step),
            "wallTimeTerminated": bool(wall_time_terminated),
            "wallTimeLimitSeconds": self._safe_float(getattr(self, "wall_time_limit_seconds", None)),
            "isValid": bool(is_valid),
            "gbestUpdateCount": int(self.gbest_update_count),
            "feasibleSolutionCount": int(self.feasible_solution_count),
            "archiveSize": int(len(self.pareto_archive)),
            "representativeArchiveIndex": None if self.representative_archive_index is None else int(self.representative_archive_index) + 1,
            "decisionScore": self._safe_float(best_energy),
            "stableDecisionScore": representative_stable_score,
            "archiveHypervolume": archive_hypervolume,
            "archiveSpacing": archive_spacing,
            "archiveIgd": reference_metrics["archive_igd"],
            "referenceFrontPath": reference_metrics["reference_front_path"],
            "referenceFrontSize": reference_metrics["reference_front_size"],
            "referenceArchiveCount": reference_metrics["reference_front_archive_count"],
            "archiveHypervolumeMode": reference_metrics["archive_hypervolume_mode"],
            "archiveHypervolumeReferencePoint": reference_metrics["archive_hypervolume_reference_point"],
            "repMhc": None if best_solution is None else self._safe_float(getattr(best_solution, "MHC", None)),
            "repCr": None if best_solution is None else self._safe_float(getattr(best_solution, "CR", None)),
            "repDr": None if best_solution is None else self._safe_float(getattr(best_solution, "DR", None)),
            "repAr": None if best_solution is None else self._safe_float(getattr(best_solution, "AR", None)),
            "paretoArchivePath": archive_path,
            "objectiveDefinitionVersion": MO_ReferenceFrontUtil.OBJECTIVE_DEFINITION_VERSION,
            "arSatisfactionMode": "paper_triangular",
            "agentMode": agent_mode,
            "epsilonEnd": self._safe_float(getattr(agent, "epsilon", None)),
            "agentOptimizeSteps": int(getattr(agent, "optimize_steps", 0) or 0) if hasattr(agent, "optimize_steps") else None,
            "agentTotalSteps": int(getattr(agent, "total_steps", 0) or 0) if hasattr(agent, "total_steps") else None,
            "G": int(self.G),
            "tMax": int(self.t_max),
            "archiveLimit": int(self.archive_limit),
            "traceInterval": int(self.mo_trace_interval),
            "objectiveWeights": np.asarray(self.mo_weights, dtype=float).tolist(),
            "representativeSolution": solution_array,
        }
        self.mo_run_summary = self.mo_recorder.finalize(run_summary, action_stats)
        self.last_run_payload = {
            "pareto_archive_path": archive_path,
            "pareto_size": len(self.pareto_archive),
            "rep_mhc": None if best_solution is None else float(getattr(best_solution, "MHC", math.inf)),
            "rep_cr": None if best_solution is None else float(getattr(best_solution, "CR", 0.0)),
            "rep_dr": None if best_solution is None else float(getattr(best_solution, "DR", 0.0)),
            "rep_ar": None if best_solution is None else float(getattr(best_solution, "AR", 0.0)),
            "decision_score": None if not np.isfinite(best_energy) else float(best_energy),
            "stable_decision_score": representative_stable_score,
            "archive_hypervolume": archive_hypervolume,
            "archive_spacing": archive_spacing,
            "archive_igd": reference_metrics["archive_igd"],
            "reference_front_path": reference_metrics["reference_front_path"],
            "reference_front_size": reference_metrics["reference_front_size"],
            "reference_front_archive_count": reference_metrics["reference_front_archive_count"],
            "archive_hypervolume_mode": reference_metrics["archive_hypervolume_mode"],
            "archive_hypervolume_reference_point": reference_metrics["archive_hypervolume_reference_point"],
            "wall_time_terminated": bool(wall_time_terminated),
            "wall_time_limit_seconds": self._safe_float(getattr(self, "wall_time_limit_seconds", None)),
            "mo_run_id": self.mo_run_summary.get("runId"),
            "mo_bundle_dir": self.mo_run_summary.get("bundleDir"),
            "mo_trace_path": self.mo_run_summary.get("tracePath"),
            "mo_events_path": self.mo_run_summary.get("eventsPath"),
            "mo_action_stats_path": self.mo_run_summary.get("actionStatsPath"),
            "mo_run_summary_path": self.mo_run_summary.get("runSummaryPath"),
        }
        if tb_writer is not None:
            tb_writer.add_scalar("Final/decisionScore", self._safe_float(best_energy), 0)
            tb_writer.add_scalar("Final/paretoArchiveSize", int(len(self.pareto_archive)), 0)
            tb_writer.add_scalar("Final/totalSteps", int(global_step), 0)
            tb_writer.close()
            logger.info(f"TensorBoard logs saved to: {tb_log_dir}")

        return global_step, is_valid, best_solution, best_energy, start_time, end_time, fast_time


def _save_experiment_row(exp_instance, exp_algorithm, exp_remark, total_iter, is_valid, best_sol, best_energy, start, end, fast, solver):
    extra_fields = dict(getattr(solver, "last_run_payload", {}) or {})
    MO_ExperimentsUtil.save_legacy_mo_experiment_result(
        exp_instance=exp_instance,
        exp_algorithm=exp_algorithm,
        exp_iterations=total_iter,
        exp_solution=best_sol.fbs_model.array_2d,
        exp_fitness=best_energy,
        exp_start_time=start,
        exp_fast_time=fast,
        exp_end_time=end,
        exp_is_valid_aspect_ratio=is_valid,
        exp_remark=exp_remark,
        exp_gbest_updates=solver.gbest_update_count,
        exp_extra_fields=extra_fields,
    )


if __name__ == "__main__":
    def _format_summary_metrics(solver, best_energy):
        payload = getattr(solver, "last_run_payload", {}) or {}
        hv = payload.get("archive_hypervolume")
        igd = payload.get("archive_igd")
        spacing = payload.get("archive_spacing")

        def _fmt(value):
            return "NA" if value is None else f"{float(value):.6f}"

        return (
            f"representative decision score: {float(best_energy):.6f} | "
            f"HV: {_fmt(hv)} | IGD: {_fmt(igd)} | Spacing: {_fmt(spacing)}"
        )

    def _parse_env_int(name, default):
        raw = os.getenv(name)
        if raw is None:
            return int(default)
        try:
            return int(raw.strip())
        except Exception:
            return int(default)

    def _parse_env_float(name, default):
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        try:
            return float(raw.strip())
        except Exception:
            return float(default)

    def _parse_env_flag(name, default):
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        return raw.strip().lower() in {"1", "true", "yes", "on", "y"}

    exp_instance = os.getenv("ELP_EXP_INSTANCE", "Du62")
    baseline_algo = os.getenv("ELP_MO_BASELINE_ALGO", "").strip().lower()
    baseline_enabled = baseline_algo in {"nsga2", "moead", "spea2", "pso", "mopso"}
    default_algorithm = f"MO_BASELINE_{baseline_algo.upper()}" if baseline_enabled else "ELP_DRL_MO4"
    exp_algorithm = os.getenv("ELP_EXP_ALGORITHM", default_algorithm)
    default_remark = (
        "MO baseline on action-sequence encoding"
        if baseline_enabled
        else "WarmStart(GA)+ELP+Pareto archive with representative solution"
    )
    exp_remark = os.getenv("ELP_EXP_REMARK", default_remark)
    exp_number = _parse_env_int("ELP_EXP_NUMBER", 30)
    is_exp = _parse_env_flag("ELP_IS_EXP", True)

    G = _parse_env_int("ELP_G", 1000)
    t_max = _parse_env_int("ELP_T_MAX", 300)
    T_initial = _parse_env_float("ELP_T_INITIAL", 10000.0)
    k_hist = _parse_env_float("ELP_K_HIST", 10.0)
    base_seed = _parse_env_int("ELP_BASE_SEED", 20260427)

    baseline_population = _parse_env_int("ELP_MO_BASELINE_POP", 64)
    baseline_generations = _parse_env_int("ELP_MO_BASELINE_GEN", 80)
    baseline_sequence_length = _parse_env_int("ELP_MO_BASELINE_SEQ_LEN", t_max)

    def _run_once(run_index):
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

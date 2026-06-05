import copy
import datetime
import math
import os
import time

import gym
import numpy as np
from loguru import logger

import src
import src.utils.FBSUtil as FBSUtil
from src.algorithms.ELP_DRL_BiMO4 import ELP as _BiMO4ELP
from src.algorithms.ELP_DRL_BiMO4 import (
    _format_summary_metrics as _bimo4_format_summary,
)
from src.algorithms.ELP_DRL_BiMO4 import (
    _get_initial_solution_energy,
    _parse_env_flag,
    _parse_env_float,
    _parse_env_int,
    _save_experiment_row,
    _set_global_seed,
)
from src.algorithms.ELP_DRL_MO4 import MO_ReferenceFrontUtil
from src.utils.FBSModel import FBSModel
from src.utils.FBSUtil import arrayToPermutation, permutationToArray
from src.utils.MO_FBSUtil_BiMO4 import MO_FBSUtil_BiMO4


class ELP(_BiMO4ELP):
    """GRASP baseline for BiMO4 with dual local search backends.

    Two local search backends:
    - paper_adapted (default): Paper-style interchange-only local search with
      DBLS (dominance-based) + AOLS (single-objective alternating) steps.
    - engineered: Reuses existing action neighborhoods (facility_swap,
      bay_swap, repair, bay_shuffle, flow_guided_swap, segment_insert,
      cross_bay_relocate)

    Constructive phase: C1_GR (MHC/GR), C2_RG (MHC/RG), C3_GR (CR/GR), C4_RG (CR/RG)
    """

    def __init__(self, env, gbest, T, G=100, t_max=50, k=2.0, archive_limit=64, objective_weights=None):
        super().__init__(
            env=env,
            gbest=gbest,
            T=T,
            G=G,
            t_max=t_max,
            k=k,
            archive_limit=archive_limit,
            objective_weights=objective_weights,
        )
        # ELP_DRL_MO 父类会将 mo_weights 填充为 4 维，双目标上下文必须强制回到 2 维。
        self.mo_base_weights = self._normalize_bi_weights(
            getattr(self, "mo_base_weights", [0.5, 0.5]), floor_value=0.05
        )
        self.mo_weights = self.mo_base_weights.copy()
        self.mo_last_weight_target = self.mo_base_weights.copy()
        self._reset_running_objective_bounds()
        self.grasp_alpha = float(_parse_env_float("ELP_GRASP_ALPHA", 0.45))
        self.grasp_constructive_mix = self._normalize_constructive_mix(
            [
                _parse_env_float("ELP_GRASP_PC1", 0.30),
                _parse_env_float("ELP_GRASP_PC2", 0.20),
                _parse_env_float("ELP_GRASP_PC3", 0.30),
                _parse_env_float("ELP_GRASP_PC4", 0.20),
            ]
        )
        self._backend_raw = os.getenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted").strip()
        if self._backend_raw not in {"paper_adapted", "engineered"}:
            logger.warning(
                "Unknown ELP_GRASP_LOCAL_SEARCH_BACKEND={}, falling back to paper_adapted",
                self._backend_raw,
            )
            self._backend_raw = "paper_adapted"
        self.local_search_backend = self._backend_raw
        self.grasp_local_search_passes = int(max(1, _parse_env_int("ELP_GRASP_LOCAL_SEARCH_PASSES", 2)))
        self.grasp_refine_steps = int(max(1, _parse_env_int("ELP_GRASP_REFINEMENT_STEPS", self.t_max)))
        self.grasp_archive_seed_trials = int(max(1, _parse_env_int("ELP_GRASP_ARCHIVE_SEED_TRIALS", 4)))
        self.wall_time_limit = float(max(0.0, _parse_env_float("ELP_WALL_TIME_LIMIT_SECONDS", 0.0)))
        self._wall_time_terminated = False

        backend_suffix = "PAPERLS" if self.local_search_backend == "paper_adapted" else "ACTIONLS"
        self.default_run_algorithm = f"ELP_DRL_BiMO4_GRASP_{backend_suffix}"
        if self.local_search_backend == "paper_adapted":
            self.default_run_remark = "Adapted GRASP baseline for BiMO4 with paper-style interchange local search"
        else:
            self.default_run_remark = "Adapted GRASP baseline for BiMO4 with engineered action local search"

    # ------------------------------------------------------------------
    # 双目标权重修正：父类链可能将 mo_weights 填充为 4 维，此处强制回到 2 维。
    # ------------------------------------------------------------------
    def _refresh_dynamic_weights(self):
        self.mo_weights = self._normalize_bi_weights(
            getattr(self, "mo_weights", [0.5, 0.5]), floor_value=0.05
        )
        self.mo_base_weights = self._normalize_bi_weights(
            getattr(self, "mo_base_weights", [0.5, 0.5]), floor_value=0.05
        )
        return super()._refresh_dynamic_weights()

    # ------------------------------------------------------------------
    # 环境变量解析辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_float(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return float(value)

    @staticmethod
    def _normalize_constructive_mix(values):
        mix = np.asarray(values, dtype=float).reshape(-1)
        if mix.size != 4:
            raise ValueError("GRASP 构造器配比必须正好包含 4 项。")
        mix = np.clip(mix, 0.0, None)
        if float(np.sum(mix)) <= 0.0:
            mix = np.asarray([0.30, 0.20, 0.30, 0.20], dtype=float)
        return (mix / np.sum(mix)).tolist()

    @staticmethod
    def _repo_root_path():
        return __import__("pathlib").Path(__file__).resolve().parents[2]

    # ------------------------------------------------------------------
    # 解模板与结构转换
    # ------------------------------------------------------------------
    def _base_solution_template(self):
        return copy.deepcopy(self.env)

    def _determine_target_bays(self, template_solution):
        bay_array = np.asarray(getattr(template_solution.fbs_model, "bay", []), dtype=int).reshape(-1)
        current_bays = int(np.sum(bay_array == 1)) if bay_array.size > 0 else 0
        estimated_bays = FBSUtil.select_B(
            np.asarray(template_solution.areas, dtype=float),
            int(template_solution.n),
            float(template_solution.fac_limit_aspect),
            float(template_solution.W),
        )
        if estimated_bays is None:
            estimated_bays = current_bays
        if estimated_bays is None or int(estimated_bays) <= 0:
            estimated_bays = 2
        return int(max(1, min(int(template_solution.n), int(estimated_bays))))

    def _structure_to_encoding(self, structure):
        normalized = [list(map(int, bay)) for bay in structure if len(bay) > 0]
        if not normalized:
            raise ValueError("区带结构不能为空。")
        permutation, bay = FBSUtil.arrayToPermutation(normalized)
        bay = np.asarray(bay, dtype=int).reshape(-1)
        bay[-1] = 1
        return np.asarray(permutation, dtype=int).reshape(-1), bay

    def _solution_from_structure(self, template_solution, structure):
        permutation, bay = self._structure_to_encoding(structure)
        candidate = self._light_clone_solution(template_solution)
        candidate.fbs_model = FBSModel(
            permutation=np.asarray(permutation, dtype=int).tolist(),
            bay=np.asarray(bay, dtype=int).tolist(),
        )
        self._evaluate_solution(candidate)
        return candidate

    # ------------------------------------------------------------------
    # 构造阶段 — 代理评分（仅用于启发式引导，最终指标走 BiMO4 完整评价）
    # ------------------------------------------------------------------
    def _partial_structure_coordinates(self, structure):
        coordinates = {}
        for row_idx, bay in enumerate(structure):
            for col_idx, facility in enumerate(bay):
                coordinates[int(facility)] = (int(row_idx), int(col_idx))
        return coordinates

    def _partial_primary_score(self, structure, objective_name):
        """部分布局代理评分：MHC 用流量×网格曼哈顿距离，CR 用关系矩阵×邻接潜力。"""
        coordinates = self._partial_structure_coordinates(structure)
        facilities = sorted(coordinates.keys())
        if len(facilities) <= 1:
            return 0.0

        if objective_name == "mhc":
            matrix = np.asarray(getattr(self.env, "F", []), dtype=float)
            total_score = 0.0
            for left_idx, left_facility in enumerate(facilities[:-1]):
                left_row, left_col = coordinates[left_facility]
                for right_facility in facilities[left_idx + 1 :]:
                    right_row, right_col = coordinates[right_facility]
                    manhattan = abs(left_row - right_row) + abs(left_col - right_col)
                    total_score += float(matrix[left_facility - 1, right_facility - 1]) * float(manhattan)
            return float(total_score)

        # CR 导向：用关系矩阵 × 邻接潜力（网格距离=1 视为相邻）
        rel_matrix = np.asarray(getattr(self, "rel_matrix", []), dtype=float)
        if rel_matrix.size == 0:
            return 0.0

        total_score = 0.0
        for left_idx, left_facility in enumerate(facilities[:-1]):
            left_row, left_col = coordinates[left_facility]
            for right_facility in facilities[left_idx + 1 :]:
                right_row, right_col = coordinates[right_facility]
                manhattan = abs(left_row - right_row) + abs(left_col - right_col)
                adjacency = 1.0 if manhattan == 1 else 0.0
                total_score += float(rel_matrix[left_facility - 1, right_facility - 1]) * adjacency
        return float(total_score)

    def _facility_proxy_scores(self, remaining, placed, objective_name):
        """未放置设施的代理得分：与已放置设施之间的聚合流量/关系强度。"""
        remaining_idx = np.asarray(list(remaining), dtype=int) - 1
        if remaining_idx.size == 0:
            return np.asarray([], dtype=float)
        if not placed:
            return np.random.random(size=remaining_idx.size)

        placed_idx = np.asarray(list(placed), dtype=int) - 1
        if objective_name == "mhc":
            matrix = np.asarray(getattr(self.env, "F", []), dtype=float)
            proxy = np.sum(np.abs(matrix[np.ix_(remaining_idx, placed_idx)]), axis=1)
            return proxy.astype(float)

        rel_matrix = np.asarray(getattr(self, "rel_matrix", []), dtype=float)
        if rel_matrix.size == 0:
            return np.random.random(size=remaining_idx.size)
        proxy = np.sum(np.abs(rel_matrix[np.ix_(remaining_idx, placed_idx)]), axis=1)
        return proxy.astype(float)

    def _candidate_order_key(self, primary_value, objective_name):
        if objective_name == "mhc":
            return float(primary_value)
        return -float(primary_value)

    def _enumerate_constructive_candidates(self, template_solution, structure, remaining, objective_name):
        _ = template_solution
        placed = [facility for bay in structure for facility in bay]
        rows = len(structure)
        candidates = []

        for facility in remaining:
            for row_idx in range(rows):
                row_length = len(structure[row_idx])
                for position_idx in range(row_length + 1):
                    candidate_structure = [list(current_bay) for current_bay in structure]
                    candidate_structure[row_idx].insert(position_idx, int(facility))
                    primary_value = self._partial_primary_score(candidate_structure, objective_name)
                    candidates.append(
                        {
                            "facility": int(facility),
                            "row_idx": int(row_idx),
                            "position_idx": int(position_idx),
                            "primary": float(primary_value),
                            "order_key": self._candidate_order_key(primary_value, objective_name),
                        }
                    )
        candidates.sort(key=lambda item: item["order_key"])
        return candidates

    # ------------------------------------------------------------------
    # RCL 构造与选择
    # ------------------------------------------------------------------
    def _build_rcl(self, candidates, objective_name, variant_name):
        if not candidates:
            return []

        active_pool = sorted(candidates, key=lambda item: item["order_key"])

        if variant_name in {"C1_GR", "C3_GR"}:
            best_value = float(active_pool[0]["primary"])
            worst_value = float(active_pool[-1]["primary"])
            if objective_name == "mhc":
                threshold = best_value + (worst_value - best_value) * (1.0 - self.grasp_alpha)
                rcl = [item for item in active_pool if float(item["primary"]) <= threshold]
            else:
                threshold = best_value - (best_value - worst_value) * (1.0 - self.grasp_alpha)
                rcl = [item for item in active_pool if float(item["primary"]) >= threshold]
            return rcl if rcl else active_pool[:1]

        # RG 变体：随机子集，取最优
        subset_size = int(max(1, math.ceil(len(active_pool) * float(self.grasp_alpha))))
        chosen_indices = np.random.choice(len(active_pool), size=subset_size, replace=False)
        sampled = [active_pool[int(idx)] for idx in np.asarray(chosen_indices, dtype=int).tolist()]
        sampled.sort(key=lambda item: item["order_key"])
        return sampled

    def _select_constructive_choice(self, candidates, objective_name, variant_name):
        rcl = self._build_rcl(candidates, objective_name, variant_name)
        if not rcl:
            raise RuntimeError("GRASP 构造阶段生成了空 RCL。")
        if variant_name in {"C1_GR", "C3_GR"}:
            chosen = rcl[int(np.random.randint(0, len(rcl)))]
        else:
            chosen = min(rcl, key=lambda item: item["order_key"])
        return chosen

    # ------------------------------------------------------------------
    # 四个构造器
    # ------------------------------------------------------------------
    def _construct_solution(self, template_solution, variant_name):
        target_bays = self._determine_target_bays(template_solution)
        facilities = list(range(1, int(template_solution.n) + 1))
        np.random.shuffle(facilities)
        structure = [[] for _ in range(target_bays)]

        for bay_idx in range(target_bays):
            if facilities:
                structure[bay_idx].append(int(facilities.pop()))

        objective_name = "mhc" if variant_name in {"C1_GR", "C2_RG"} else "cr"
        while facilities:
            candidates = self._enumerate_constructive_candidates(
                template_solution=template_solution,
                structure=structure,
                remaining=facilities,
                objective_name=objective_name,
            )
            chosen = self._select_constructive_choice(candidates, objective_name, variant_name)
            structure[int(chosen["row_idx"])].insert(int(chosen["position_idx"]), int(chosen["facility"]))
            facilities.remove(int(chosen["facility"]))

        return self._solution_from_structure(template_solution, structure)

    def _sample_constructive_variant(self):
        variants = np.asarray(["C1_GR", "C2_RG", "C3_GR", "C4_RG"], dtype=object)
        chosen_index = int(
            np.random.choice(np.arange(variants.size), p=np.asarray(self.grasp_constructive_mix, dtype=float))
        )
        return str(variants[chosen_index])

    # ------------------------------------------------------------------
    # 局部搜索 — 后端 A: engineered（复用工程稳定邻域动作）
    # ------------------------------------------------------------------
    def _local_search_engineered(self, solution):
        """使用现有 action 邻域做 first-improvement 局部搜索。"""
        current = self._light_clone_solution(solution)
        current_score = float(getattr(current, "fitness", math.inf))

        for _ in range(self.grasp_local_search_passes):
            improved = True
            while improved:
                improved = False
                for action_idx in getattr(self, "valid_actions", [0, 2, 3, 6, 9, 10, 11]):
                    candidate = self.generate_candidate_by_action(current, action_idx)
                    if not bool(getattr(candidate, "current_is_feasible", False)):
                        continue
                    candidate_score = float(getattr(candidate, "fitness", math.inf))
                    if candidate_score + 1e-12 < current_score:
                        current = candidate
                        current_score = candidate_score
                        self._observe_feasible_state(current)
                        improved = True
                        break

        return current

    # ------------------------------------------------------------------
    # 局部搜索 — 后端 B: paper_adapted（论文风格 interchange + DBLS/AOLS）
    # ------------------------------------------------------------------
    @staticmethod
    def _enumerate_positions(solution):
        """返回当前解中所有设施位置列表 [(bay_idx, pos_idx, facility_id), ...]."""
        bay_structure = permutationToArray(
            np.asarray(solution.fbs_model.permutation, dtype=int),
            np.asarray(solution.fbs_model.bay, dtype=int),
        )
        positions = []
        for bay_idx, bay in enumerate(bay_structure):
            for pos_idx, facility in enumerate(bay):
                positions.append((int(bay_idx), int(pos_idx), int(facility)))
        return positions

    def _apply_interchange(self, solution, pos1, pos2):
        """交换两个设施位置，返回 BiMO4 完整评价后的候选解。"""
        bay_structure = permutationToArray(
            np.asarray(solution.fbs_model.permutation, dtype=int),
            np.asarray(solution.fbs_model.bay, dtype=int),
        )
        bi1, pi1, _ = pos1
        bi2, pi2, _ = pos2
        new_structure = [list(b) for b in bay_structure]
        new_structure[bi1][pi1], new_structure[bi2][pi2] = (
            new_structure[bi2][pi2],
            new_structure[bi1][pi1],
        )
        new_perm, new_bay = arrayToPermutation([np.array(b) for b in new_structure])
        candidate = self._light_clone_solution(solution)
        candidate.fbs_model = FBSModel(
            permutation=np.asarray(new_perm, dtype=int).tolist(),
            bay=np.asarray(new_bay, dtype=int).tolist(),
        )
        self._evaluate_solution(candidate)
        return candidate

    # ------------------------------------------------------------------
    # DBLS — 支配驱动局部搜索（对应论文 dominatedHybridLocalSearchInterchange）
    # ------------------------------------------------------------------
    def _paper_dbls_step(self, solution):
        """DBLS: 随机化 interchange 扫描，first-improvement 支配接受。

        若候选支配当前解 → 立即接受并重启扫描。
        若候选与当前解互不支配 → 加入 Pareto 档案但不替换当前解。
        若候选被当前解支配 → 拒绝。
        """
        current = solution
        positions = self._enumerate_positions(current)
        n = len(positions)
        if n < 2:
            return current

        improved = True
        while improved:
            improved = False
            # 随机打乱外层 (I) 和内层 (J) 扫描顺序，模拟论文随机扫描
            outer_order = np.random.permutation(n)
            for outer_idx in outer_order:
                inner_order = np.random.permutation(n)
                found_better = False
                for inner_idx in inner_order:
                    if int(outer_idx) >= int(inner_idx):
                        continue
                    candidate = self._apply_interchange(
                        current, positions[int(outer_idx)], positions[int(inner_idx)]
                    )
                    if not bool(getattr(candidate, "current_is_feasible", False)):
                        continue

                    comparison = MO_FBSUtil_BiMO4.compare_solution_quality(candidate, current)
                    if comparison < 0:
                        # 候选支配当前解 → 接受并重启扫描
                        self._observe_feasible_state(candidate)
                        current = candidate
                        positions = self._enumerate_positions(current)
                        n = len(positions)
                        improved = True
                        found_better = True
                        break
                    elif comparison == 0:
                        # 互不支配 → 仅归档
                        self._observe_feasible_state(candidate)

                if found_better:
                    break

        return current

    # ------------------------------------------------------------------
    # AOLS — 单目标交替优化（对应论文 hybridLocalSearchInterchange）
    # ------------------------------------------------------------------
    def _paper_aols_step(self, solution, factor):
        """AOLS: 随机化 interchange 扫描，按单目标改善接受。

        factor="mhc": 仅当候选 MHC 严格小于当前 MHC 时接受。
        factor="cr":  仅当候选 CR  严格大于当前 CR  时接受（BiMO4 中 CR 最大化）。

        找到改善立即接受并重启扫描（first-improvement）。
        """
        if factor not in {"mhc", "cr"}:
            raise ValueError(f"AOLS factor 必须为 'mhc' 或 'cr'，收到: {factor}")

        current = solution
        positions = self._enumerate_positions(current)
        n = len(positions)
        if n < 2:
            return current

        improved = True
        while improved:
            improved = False
            outer_order = np.random.permutation(n)
            for outer_idx in outer_order:
                inner_order = np.random.permutation(n)
                found_better = False
                for inner_idx in inner_order:
                    if int(outer_idx) >= int(inner_idx):
                        continue
                    candidate = self._apply_interchange(
                        current, positions[int(outer_idx)], positions[int(inner_idx)]
                    )
                    if not bool(getattr(candidate, "current_is_feasible", False)):
                        continue

                    if factor == "mhc":
                        current_mhc = float(getattr(current, "MHC", math.inf))
                        candidate_mhc = float(getattr(candidate, "MHC", math.inf))
                        accepts = candidate_mhc + 1e-12 < current_mhc
                    else:  # factor == "cr"
                        current_cr = float(getattr(current, "CR", 0.0))
                        candidate_cr = float(getattr(candidate, "CR", 0.0))
                        # BiMO4 中 CR 最大化，改善 = CR 更大
                        accepts = candidate_cr > current_cr + 1e-12

                    if accepts:
                        self._observe_feasible_state(candidate)
                        current = candidate
                        positions = self._enumerate_positions(current)
                        n = len(positions)
                        improved = True
                        found_better = True
                        break
                    else:
                        # 即使不满足单目标改善，互不支配的解仍可归档
                        comparison = MO_FBSUtil_BiMO4.compare_solution_quality(candidate, current)
                        if comparison == 0:
                            self._observe_feasible_state(candidate)

                if found_better:
                    break

        return current

    # ------------------------------------------------------------------
    # 论文风格局部搜索主流程
    # ------------------------------------------------------------------
    def _paper_local_search(self, solution):
        """论文风格局部搜索：DBLS → AOLS(mhc) → AOLS(cr)，多轮迭代。

        顺序对应论文 combiningDM / combiningMD 中的思路：
        先做支配驱动的 Pareto 扩张，再分别对 MHC 和 CR 做单目标交替优化。
        每轮三者任一有改进即继续，直至无改进或达到最大轮次。
        """
        current = solution
        total_passes = max(1, self.grasp_local_search_passes)

        for _ in range(total_passes):
            any_improved = False

            dbls_result = self._paper_dbls_step(current)
            if dbls_result is not current:
                any_improved = True
            current = dbls_result

            aols_mhc_result = self._paper_aols_step(current, "mhc")
            if aols_mhc_result is not current:
                any_improved = True
            current = aols_mhc_result

            aols_cr_result = self._paper_aols_step(current, "cr")
            if aols_cr_result is not current:
                any_improved = True
            current = aols_cr_result

            if not any_improved:
                break

        return current

    # ------------------------------------------------------------------
    # 局部搜索调度
    # ------------------------------------------------------------------
    def _apply_local_search(self, solution):
        if self.local_search_backend == "engineered":
            return self._local_search_engineered(solution)
        return self._paper_local_search(solution)

    # ------------------------------------------------------------------
    # 档案种子
    # ------------------------------------------------------------------
    def _seed_archive_if_needed(self, template_solution):
        if self.pareto_archive:
            return
        for _ in range(self.grasp_archive_seed_trials):
            variant_name = self._sample_constructive_variant()
            candidate = self._construct_solution(template_solution, variant_name)
            if bool(getattr(candidate, "current_is_feasible", False)):
                self._observe_feasible_state(candidate)
                self.s = self._light_clone_solution(candidate)
                break
        if self.pareto_archive:
            return
        fallback = self._light_clone_solution(template_solution)
        self._evaluate_solution(fallback)
        if bool(getattr(fallback, "current_is_feasible", False)):
            self._observe_feasible_state(fallback)
            self.s = self._light_clone_solution(fallback)

    # ------------------------------------------------------------------
    # 结果落盘
    # ------------------------------------------------------------------
    def _finalize_payload(self, start_time, end_time, fast_time):
        self._refresh_archive_state()
        archive_path = self._save_pareto_archive(start_time, algorithm_name=self.default_run_algorithm)
        reference_metrics = self._compute_reference_front_metrics()

        best_solution = self.representative_solution
        best_energy = (
            float(self.representative_decision_score)
            if self.representative_solution is not None
            else math.inf
        )
        if best_solution is None and self.best_feasible_solution is not None:
            best_solution = copy.deepcopy(self.best_feasible_solution)
            best_energy = float(
                getattr(best_solution, "decision_score", getattr(best_solution, "fitness", math.inf))
            )
        if best_solution is None:
            best_solution = copy.deepcopy(self.s)
            best_energy = float(
                getattr(best_solution, "decision_score", getattr(best_solution, "fitness", math.inf))
            )

        wall_time_seconds = float((end_time - start_time).total_seconds())

        self.last_run_payload = {
            "pareto_archive_path": archive_path,
            "pareto_size": len(self.pareto_archive),
            "rep_mhc": None if best_solution is None else self._safe_float(getattr(best_solution, "MHC", None)),
            "rep_cr": None if best_solution is None else self._safe_float(getattr(best_solution, "CR", None)),
            "rep_dr": None,
            "rep_ar": None,
            "decision_score": None if not np.isfinite(best_energy) else float(best_energy),
            "stable_decision_score": None if not np.isfinite(best_energy) else float(best_energy),
            "archive_hypervolume": reference_metrics.get("archive_hypervolume"),
            "archive_spacing": reference_metrics.get("archive_spacing"),
            "archive_igd": reference_metrics.get("archive_igd"),
            "reference_front_path": reference_metrics.get("reference_front_path"),
            "reference_front_size": reference_metrics.get("reference_front_size"),
            "reference_front_archive_count": reference_metrics.get("reference_front_archive_count"),
            "archive_hypervolume_mode": reference_metrics.get("archive_hypervolume_mode"),
            "archive_hypervolume_reference_point": reference_metrics.get("archive_hypervolume_reference_point"),
            "objective_definition_version": getattr(self, "OBJECTIVE_DEFINITION_VERSION", None),
            "grasp_alpha": float(self.grasp_alpha),
            "grasp_constructive_mix": list(self.grasp_constructive_mix),
            "grasp_local_search_passes": int(self.grasp_local_search_passes),
            "grasp_refine_steps": int(self.grasp_refine_steps),
            "local_search_backend": self.local_search_backend,
            "wall_time_limit_seconds": float(self.wall_time_limit) if self.wall_time_limit > 0 else None,
            "wall_time_terminated": bool(self._wall_time_terminated),
            "runtime_seconds": float(wall_time_seconds),
            "best_result_seconds": (
                None if fast_time is None else float((fast_time - start_time).total_seconds())
            ),
        }
        return best_solution, best_energy

    # ------------------------------------------------------------------
    # GRASP 主循环（wall-time 预算）
    # ------------------------------------------------------------------
    def _run_impl(self):
        start_time = datetime.datetime.now()
        wall_start = time.perf_counter()
        fast_time = start_time
        best_seen_score = math.inf

        self._reset_baseline_archive_state()
        self._wall_time_terminated = False

        template_solution = self._base_solution_template()
        self._evaluate_solution(template_solution)
        self.s = self._light_clone_solution(template_solution)
        self._seed_archive_if_needed(template_solution)

        iteration_idx = 0
        max_iterations = int(max(1, self.G))

        while iteration_idx < max_iterations:
            # Wall-time 检查
            if self.wall_time_limit > 0:
                elapsed = time.perf_counter() - wall_start
                if elapsed >= self.wall_time_limit:
                    self._wall_time_terminated = True
                    logger.info(
                        "GRASP wall-time budget exhausted: {:.2f}s >= {:.2f}s",
                        elapsed,
                        self.wall_time_limit,
                    )
                    break

            variant_name = self._sample_constructive_variant()
            candidate = self._construct_solution(template_solution, variant_name)
            if bool(getattr(candidate, "current_is_feasible", False)):
                self._observe_feasible_state(candidate)

            candidate = self._apply_local_search(candidate)
            if bool(getattr(candidate, "current_is_feasible", False)):
                self._observe_feasible_state(candidate)

            self.s = self._light_clone_solution(candidate)

            current_rep_score = float(getattr(self, "representative_decision_score", math.inf))
            if np.isfinite(current_rep_score) and current_rep_score + 1e-12 < best_seen_score:
                best_seen_score = current_rep_score
                fast_time = datetime.datetime.now()

            iteration_idx += 1
            logger.info(
                "GRASP iteration {} / {} | backend={} | variant={} | archive_size={} | rep_score={}",
                iteration_idx,
                max_iterations,
                self.local_search_backend,
                variant_name,
                len(self.pareto_archive),
                "NA" if not np.isfinite(current_rep_score) else f"{current_rep_score:.6f}",
            )

        end_time = datetime.datetime.now()
        best_solution, best_energy = self._finalize_payload(start_time, end_time, fast_time)
        is_valid = bool(getattr(best_solution, "current_is_feasible", False))
        total_iter = int(max(1, iteration_idx))
        return total_iter, is_valid, best_solution, best_energy, start_time, end_time, fast_time


# ------------------------------------------------------------------
# 格式化输出
# ------------------------------------------------------------------
def _format_summary_metrics(solver, best_energy):
    payload = getattr(solver, "last_run_payload", {}) or {}
    hv = payload.get("archive_hypervolume")
    spacing = payload.get("archive_spacing")
    rep_mhc = payload.get("rep_mhc")
    rep_cr = payload.get("rep_cr")
    backend = payload.get("local_search_backend", "NA")

    def _fmt(value):
        return "NA" if value is None else f"{float(value):.6f}"

    return (
        f"backend: {backend} | "
        f"representative decision score: {float(best_energy):.6f} | "
        f"rep_mhc: {_fmt(rep_mhc)} | rep_cr: {_fmt(rep_cr)} | "
        f"HV: {_fmt(hv)} | Spacing: {_fmt(spacing)}"
    )


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    exp_instance = os.getenv("ELP_EXP_INSTANCE", "Du62")
    backend_raw = os.getenv("ELP_GRASP_LOCAL_SEARCH_BACKEND", "paper_adapted").strip()
    if backend_raw not in {"paper_adapted", "engineered"}:
        backend_raw = "paper_adapted"
    backend_suffix = "PAPERLS" if backend_raw == "paper_adapted" else "ACTIONLS"
    exp_algorithm = os.getenv("ELP_EXP_ALGORITHM", f"ELP_DRL_BiMO4_GRASP_{backend_suffix}")
    _default_remark = (
        "Adapted GRASP baseline for BiMO4 with paper-style interchange local search"
        if backend_raw == "paper_adapted"
        else "Adapted GRASP baseline for BiMO4 with engineered action local search"
    )
    exp_remark = os.getenv("ELP_EXP_REMARK", _default_remark)
    exp_number = _parse_env_int("ELP_EXP_NUMBER", 1)
    is_exp = _parse_env_flag("ELP_IS_EXP", True)
    G = _parse_env_int("ELP_G", 80)
    t_max = _parse_env_int("ELP_T_MAX", 60)
    T_initial = _parse_env_float("ELP_T_INITIAL", 10000.0)
    k_hist = _parse_env_float("ELP_K_HIST", 10.0)
    base_seed = _parse_env_int("ELP_BASE_SEED", 20260520)

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
        return solver, solver.run()

    if is_exp:
        for i in range(exp_number):
            logger.info(f"Starting experiment {i + 1} for {exp_algorithm}")
            try:
                elp_solver, result_tuple = _run_once(i)
                total_iter, is_valid, best_sol, best_energy, start, end, fast = result_tuple
                logger.info(f"Experiment {i + 1} complete | {_format_summary_metrics(elp_solver, best_energy)}")
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

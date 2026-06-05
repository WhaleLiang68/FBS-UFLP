import math
import time

import numpy as np

from src.utils.FBSModel import FBSModel
from src.utils.FBSUtil import arrayToPermutation, permutationToArray
from src.utils.MO_FBSUtil_BiMO4 import MO_FBSUtil_BiMO4


class BiMO4PaperLocalSearch:
    """Paper-style interchange local search for BiMO4 archive intensification."""

    def __init__(self, solver, passes=2, time_limit_seconds=0.0, max_neighbor_evaluations=0):
        self.solver = solver
        self.passes = int(max(1, passes))
        self.time_limit_seconds = float(max(0.0, time_limit_seconds or 0.0))
        self.max_neighbor_evaluations = int(max(0, max_neighbor_evaluations or 0))
        self.started_at = time.perf_counter()
        self.neighbor_evaluations = 0
        self.accepted_moves = 0
        self.archive_insertions = 0
        self.stopped_by_time = False
        self.stopped_by_evaluation_limit = False

    def _can_continue(self):
        if self.time_limit_seconds > 0.0 and (time.perf_counter() - self.started_at) >= self.time_limit_seconds:
            self.stopped_by_time = True
            return False
        if self.max_neighbor_evaluations > 0 and self.neighbor_evaluations >= self.max_neighbor_evaluations:
            self.stopped_by_evaluation_limit = True
            return False
        return True

    @staticmethod
    def _enumerate_positions(solution):
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
        if not self._can_continue():
            return None
        bay_structure = permutationToArray(
            np.asarray(solution.fbs_model.permutation, dtype=int),
            np.asarray(solution.fbs_model.bay, dtype=int),
        )
        bi1, pi1, _ = pos1
        bi2, pi2, _ = pos2
        new_structure = [list(bay) for bay in bay_structure]
        new_structure[bi1][pi1], new_structure[bi2][pi2] = new_structure[bi2][pi2], new_structure[bi1][pi1]
        new_perm, new_bay = arrayToPermutation([np.array(bay) for bay in new_structure])
        candidate = self.solver._light_clone_solution(solution)
        candidate.fbs_model = FBSModel(
            permutation=np.asarray(new_perm, dtype=int).tolist(),
            bay=np.asarray(new_bay, dtype=int).tolist(),
        )
        self.solver._evaluate_solution(candidate)
        self.neighbor_evaluations += 1
        return candidate

    def _observe_candidate(self, candidate):
        before_count = int(getattr(self.solver, "archive_update_count", 0) or 0)
        changed = bool(self.solver._observe_feasible_state(candidate))
        after_count = int(getattr(self.solver, "archive_update_count", 0) or 0)
        if changed or after_count > before_count:
            self.archive_insertions += max(after_count - before_count, 1)
        return changed

    def _dbls_step(self, solution):
        current = solution
        positions = self._enumerate_positions(current)
        n = len(positions)
        if n < 2:
            return current

        improved = True
        while improved and self._can_continue():
            improved = False
            for outer_idx in np.random.permutation(n):
                found_better = False
                for inner_idx in np.random.permutation(n):
                    if int(outer_idx) >= int(inner_idx):
                        continue
                    candidate = self._apply_interchange(current, positions[int(outer_idx)], positions[int(inner_idx)])
                    if candidate is None:
                        return current
                    if not bool(getattr(candidate, "current_is_feasible", False)):
                        continue

                    comparison = MO_FBSUtil_BiMO4.compare_solution_quality(candidate, current)
                    if comparison < 0:
                        self._observe_candidate(candidate)
                        current = candidate
                        positions = self._enumerate_positions(current)
                        n = len(positions)
                        improved = True
                        found_better = True
                        self.accepted_moves += 1
                        break
                    if comparison == 0:
                        self._observe_candidate(candidate)
                if found_better or not self._can_continue():
                    break
        return current

    def _aols_step(self, solution, factor):
        if factor not in {"mhc", "cr"}:
            raise ValueError(f"AOLS factor 必须为 'mhc' 或 'cr'，收到: {factor}")

        current = solution
        positions = self._enumerate_positions(current)
        n = len(positions)
        if n < 2:
            return current

        improved = True
        while improved and self._can_continue():
            improved = False
            for outer_idx in np.random.permutation(n):
                found_better = False
                for inner_idx in np.random.permutation(n):
                    if int(outer_idx) >= int(inner_idx):
                        continue
                    candidate = self._apply_interchange(current, positions[int(outer_idx)], positions[int(inner_idx)])
                    if candidate is None:
                        return current
                    if not bool(getattr(candidate, "current_is_feasible", False)):
                        continue

                    if factor == "mhc":
                        accepts = float(getattr(candidate, "MHC", math.inf)) + 1e-12 < float(
                            getattr(current, "MHC", math.inf)
                        )
                    else:
                        accepts = float(getattr(candidate, "CR", 0.0)) > float(getattr(current, "CR", 0.0)) + 1e-12

                    if accepts:
                        self._observe_candidate(candidate)
                        current = candidate
                        positions = self._enumerate_positions(current)
                        n = len(positions)
                        improved = True
                        found_better = True
                        self.accepted_moves += 1
                        break

                    if MO_FBSUtil_BiMO4.compare_solution_quality(candidate, current) == 0:
                        self._observe_candidate(candidate)
                if found_better or not self._can_continue():
                    break
        return current

    def local_search(self, solution):
        current = solution
        for _ in range(self.passes):
            if not self._can_continue():
                break
            any_improved = False

            dbls_result = self._dbls_step(current)
            if dbls_result is not current:
                any_improved = True
            current = dbls_result

            mhc_result = self._aols_step(current, "mhc")
            if mhc_result is not current:
                any_improved = True
            current = mhc_result

            cr_result = self._aols_step(current, "cr")
            if cr_result is not current:
                any_improved = True
            current = cr_result

            if not any_improved:
                break
        return current

    def summary(self):
        return {
            "neighborEvaluations": int(self.neighbor_evaluations),
            "acceptedMoves": int(self.accepted_moves),
            "archiveInsertions": int(self.archive_insertions),
            "stoppedByTime": bool(self.stopped_by_time),
            "stoppedByEvaluationLimit": bool(self.stopped_by_evaluation_limit),
            "runtimeSeconds": float(time.perf_counter() - self.started_at),
        }

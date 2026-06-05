import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import src.utils.config as config
from src.utils.MO_ExperimentsUtil import repair_legacy_mo_result_csv


DEFAULT_RESULT_ROOT = Path(config.RESULT_PATH)
BENCHMARK_DIR_NAME = "benchmark_runs"
REMARK_KV_PATTERN = re.compile(r"\s*([^=;]+)\s*=\s*([^;]*)\s*")


def sanitize_filename(name):
    return re.sub(r'[\/*?:"<>|]', "", str(name or "UNKNOWN"))


def benchmark_root(result_root=None, benchmark_id=None):
    base = Path(result_root or DEFAULT_RESULT_ROOT)
    if benchmark_id is None:
        return base / BENCHMARK_DIR_NAME
    return base / BENCHMARK_DIR_NAME / sanitize_filename(benchmark_id)


def legacy_result_csv_path(instance, algorithm, result_root=None):
    base = Path(result_root or DEFAULT_RESULT_ROOT)
    return base / f"{sanitize_filename(instance)}-{sanitize_filename(algorithm)}.csv"


def parse_benchmark_remark(remark):
    payload = {}
    text = str(remark or "").strip()
    if not text:
        return payload
    for part in text.split(";"):
        match = REMARK_KV_PATTERN.fullmatch(part.strip())
        if not match:
            continue
        key = str(match.group(1) or "").strip()
        value = str(match.group(2) or "").strip()
        if key:
            payload[key] = value
    return payload


def coerce_int(value, default=None):
    if value is None or value == "":
        return default
    try:
        return int(str(value).strip())
    except Exception:
        return default


def coerce_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        numeric = float(value)
    except Exception:
        return default
    if not math.isfinite(numeric):
        return default
    return float(numeric)


def _weakly_dominates(left, right, atol=1e-9):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return bool(np.all(left <= right + atol) and np.any(left < right - atol))


def filter_nondominated(points, atol=1e-9):
    matrix = np.asarray(points, dtype=float)
    if matrix.size == 0:
        return np.empty((0, 2), dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] < 2:
        raise ValueError("至少需要两列双目标值。")
    matrix = matrix[:, :2]
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    kept = []
    for row in matrix.tolist():
        candidate = np.asarray(row, dtype=float)
        dominated = False
        remove_indices = []
        for idx, existing in enumerate(kept):
            if np.allclose(candidate, existing, atol=atol, rtol=1e-7):
                dominated = True
                break
            if _weakly_dominates(existing, candidate, atol=atol):
                dominated = True
                break
            if _weakly_dominates(candidate, existing, atol=atol):
                remove_indices.append(idx)
        if dominated:
            continue
        for idx in reversed(remove_indices):
            kept.pop(idx)
        kept.append(candidate)
    if not kept:
        return np.empty((0, 2), dtype=float)
    output = np.asarray(kept, dtype=float)
    order = np.lexsort((output[:, 1], output[:, 0]))
    return output[order]


def normalize_points(points, ideal, nadir):
    matrix = np.asarray(points, dtype=float)
    if matrix.size == 0:
        return np.empty((0, 2), dtype=float)
    span = np.maximum(np.asarray(nadir, dtype=float) - np.asarray(ideal, dtype=float), 1e-12)
    normalized = (matrix - np.asarray(ideal, dtype=float)) / span
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def compute_hypervolume_2d(points_norm, ref_point=None):
    matrix = filter_nondominated(points_norm)
    if matrix.size == 0:
        return 0.0
    reference = np.asarray([1.1, 1.1] if ref_point is None else ref_point, dtype=float)
    matrix = matrix[np.argsort(matrix[:, 0])]
    hv = 0.0
    best_y = float(reference[1])
    for row in matrix:
        x = float(row[0])
        y = float(row[1])
        if y < best_y - 1e-12:
            hv += max(float(reference[0]) - x, 0.0) * max(best_y - y, 0.0)
            best_y = y
    return float(hv)


def compute_igd(reference_norm, candidate_norm):
    reference = np.asarray(reference_norm, dtype=float)
    candidate = np.asarray(candidate_norm, dtype=float)
    if reference.size == 0 or candidate.size == 0:
        return None
    distances = []
    for ref_row in reference:
        norms = np.linalg.norm(candidate - ref_row, axis=1)
        if norms.size == 0:
            return None
        distances.append(float(np.min(norms)))
    return float(np.mean(distances))


def compute_spacing(candidate_norm):
    candidate = np.asarray(candidate_norm, dtype=float)
    count = int(candidate.shape[0]) if candidate.ndim == 2 else 0
    if count <= 1:
        return 0.0
    distances = []
    for idx in range(count):
        delta = candidate - candidate[idx]
        norms = np.linalg.norm(delta, axis=1)
        norms[idx] = np.inf
        nearest = float(np.min(norms))
        if np.isfinite(nearest):
            distances.append(nearest)
    if not distances:
        return 0.0
    distances = np.asarray(distances, dtype=float)
    mean_distance = float(np.mean(distances))
    if mean_distance <= 1e-12:
        return 0.0
    return float(np.sqrt(np.mean((distances - mean_distance) ** 2)))


def compute_generalized_spread(candidate_norm, reference_norm):
    candidate = np.asarray(candidate_norm, dtype=float)
    reference = np.asarray(reference_norm, dtype=float)
    if candidate.ndim != 2 or reference.ndim != 2 or candidate.shape[0] <= 1 or reference.shape[0] == 0:
        return 0.0
    reference_sorted = reference[np.lexsort((reference[:, 1], reference[:, 0]))]
    candidate_sorted = candidate[np.lexsort((candidate[:, 1], candidate[:, 0]))]
    extremes = np.vstack((reference_sorted[0], reference_sorted[-1]))
    d_extremes = []
    for extreme in extremes:
        distances = np.linalg.norm(candidate_sorted - extreme, axis=1)
        d_extremes.append(float(np.min(distances)))
    neighbor_distances = []
    for idx in range(candidate_sorted.shape[0]):
        distances = np.linalg.norm(candidate_sorted - candidate_sorted[idx], axis=1)
        distances[idx] = np.inf
        nearest = float(np.min(distances))
        if np.isfinite(nearest):
            neighbor_distances.append(nearest)
    if not neighbor_distances:
        return 0.0
    neighbor_distances = np.asarray(neighbor_distances, dtype=float)
    mean_distance = float(np.mean(neighbor_distances))
    numerator = float(np.sum(d_extremes)) + float(np.sum(np.abs(neighbor_distances - mean_distance)))
    denominator = float(np.sum(d_extremes)) + float(len(neighbor_distances) * mean_distance)
    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def compute_coverage(dominator_norm, dominated_norm):
    dominator = np.asarray(dominator_norm, dtype=float)
    dominated = np.asarray(dominated_norm, dtype=float)
    if dominator.size == 0 or dominated.size == 0:
        return 0.0
    dominated_count = 0
    for candidate in dominated:
        if np.any(np.all(dominator <= candidate + 1e-9, axis=1)):
            dominated_count += 1
    return float(dominated_count / dominated.shape[0])


def minimization_points_from_archive_payload(payload):
    rows = []
    for item in list(payload.get("items") or []):
        is_feasible = bool(item.get("isFeasible", item.get("current_is_feasible", False)))
        if not is_feasible:
            continue
        vector = item.get("moObjectivesMin") or item.get("mo_objectives_min")
        if isinstance(vector, (list, tuple)) and len(vector) >= 2:
            rows.append([float(vector[0]), float(vector[1])])
            continue
        mhc = coerce_float(item.get("mhc", item.get("MHC")), default=None)
        cr = coerce_float(item.get("cr", item.get("CR")), default=None)
        if mhc is None or cr is None:
            continue
        rows.append([float(mhc), float(-cr)])
    return filter_nondominated(rows)


def load_archive_payload_from_row(row, repo_root):
    archive_path = str(row.get("pareto_archive_path") or "").strip()
    if not archive_path:
        return None, None
    candidate_path = Path(archive_path)
    if not candidate_path.is_absolute():
        candidate_path = Path(repo_root) / archive_path
    candidate_path = candidate_path.resolve()
    if not candidate_path.exists():
        return None, None
    return json.loads(candidate_path.read_text(encoding="utf-8")), candidate_path


@dataclass
class ReferenceFrontBundle:
    instance: str
    budget_seconds: int
    benchmark_id: str
    ideal: list
    nadir: list
    points_min: np.ndarray
    algorithms: list
    source_rows: list

    def to_payload(self, output_path):
        points = np.asarray(self.points_min, dtype=float)
        payload = {
            "schemaVersion": "bimo4_benchmark_reference_front_v1",
            "benchmarkId": self.benchmark_id,
            "instance": self.instance,
            "budgetSeconds": int(self.budget_seconds),
            "pointCount": int(points.shape[0]),
            "ideal": list(self.ideal),
            "nadir": list(self.nadir),
            "sourceAlgorithms": list(self.algorithms),
            "sourceRowCount": int(len(self.source_rows)),
            "items": [
                {
                    "index": idx + 1,
                    "moObjectivesMin": [float(row[0]), float(row[1])],
                    "mhc": float(row[0]),
                    "cr": float(-row[1]),
                }
                for idx, row in enumerate(points.tolist())
            ],
            "outputPath": Path(output_path).resolve().as_posix(),
        }
        return payload


def load_benchmark_runs(
    benchmark_id,
    *,
    instances,
    algorithms,
    result_root=None,
):
    rows = []
    for instance in list(instances):
        for algorithm in list(algorithms):
            csv_path = legacy_result_csv_path(instance, algorithm, result_root=result_root)
            if not csv_path.exists():
                continue
            frame = repair_legacy_mo_result_csv(csv_path)
            if frame.empty:
                continue
            for _, record in frame.iterrows():
                remark_payload = parse_benchmark_remark(record.get("备注"))
                if str(remark_payload.get("benchmark_id") or "") != str(benchmark_id):
                    continue
                row = {
                    "instance": str(record.get("实例") or instance),
                    "algorithm": str(record.get("算法") or algorithm),
                    "benchmark_id": str(remark_payload.get("benchmark_id") or ""),
                    "phase": str(remark_payload.get("phase") or ""),
                    "seed": coerce_int(remark_payload.get("seed"), default=None),
                    "budget_seconds": coerce_int(
                        remark_payload.get("budget_seconds"),
                        default=coerce_int(record.get("wall_time_limit_seconds"), default=None),
                    ),
                    "remark": record.get("备注"),
                    "start_time": record.get("开始时间"),
                    "end_time": record.get("结束时间"),
                    "runtime_seconds": coerce_float(record.get("运行时间（秒）"), default=coerce_float(record.get("runtime_seconds"))),
                    "fast_seconds": coerce_float(record.get("最快最佳结果时间（秒）"), default=None),
                    "is_valid": record.get("宽高比是否满足"),
                    "decision_score": coerce_float(record.get("decision_score"), default=coerce_float(record.get("适应度值"))),
                    "rep_mhc": coerce_float(record.get("rep_mhc"), default=None),
                    "rep_cr": coerce_float(record.get("rep_cr"), default=None),
                    "archive_hypervolume_raw": coerce_float(record.get("HV"), default=None),
                    "archive_spacing_raw": coerce_float(record.get("Spread(Δ)"), default=None),
                    "archive_igd_raw": coerce_float(record.get("IGD"), default=None),
                    "archive_size": coerce_int(record.get("archiveSize"), default=coerce_int(record.get("pareto_size"), default=None)),
                    "pareto_archive_path": record.get("pareto_archive_path"),
                    "candidate_pool_size": coerce_int(
                        record.get("candidatePoolSize"),
                        default=coerce_int(record.get("candidate_pool_size"), default=None),
                    ),
                    "candidate_pool_limit": coerce_int(
                        record.get("candidatePoolLimit"),
                        default=coerce_int(record.get("candidate_pool_limit"), default=None),
                    ),
                    "candidate_pool_insertions": coerce_int(
                        record.get("candidatePoolInsertions"),
                        default=coerce_int(record.get("candidate_pool_insertions"), default=None),
                    ),
                    "candidate_pool_duplicates": coerce_int(
                        record.get("candidatePoolDuplicates"),
                        default=coerce_int(record.get("candidate_pool_duplicates"), default=None),
                    ),
                    "candidate_pool_bootstrap_target": coerce_int(
                        record.get("candidatePoolBootstrapTarget"),
                        default=coerce_int(record.get("candidate_pool_bootstrap_target"), default=None),
                    ),
                    "archive_paperls_enabled": record.get(
                        "archivePaperLsEnabled",
                        record.get("archive_paperls_enabled"),
                    ),
                    "archive_paperls_stats": record.get(
                        "archivePaperLsStats",
                        record.get("archive_paperls_stats"),
                    ),
                    "main_wall_time_limit_seconds": coerce_float(
                        record.get("mainWallTimeLimitSeconds"),
                        default=coerce_float(record.get("main_wall_time_limit_seconds"), default=None),
                    ),
                    "wall_time_limit_seconds": coerce_float(
                        record.get("wallTimeLimitSeconds"),
                        default=coerce_float(record.get("wall_time_limit_seconds"), default=None),
                    ),
                    "wall_time_terminated": record.get("wall_time_terminated"),
                    "baseline_seed": coerce_int(record.get("baseline_seed"), default=None),
                    "baseline_generations": coerce_int(record.get("baseline_generations"), default=None),
                    "baseline_generations_requested": coerce_int(record.get("baseline_generations_requested"), default=None),
                    "baseline_population": coerce_int(record.get("baseline_population"), default=None),
                    "baseline_termination_mode": record.get("baseline_termination_mode"),
                }
                if row["seed"] is None:
                    row["seed"] = row["baseline_seed"]
                rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["sort_start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
    frame = frame.sort_values(["instance", "algorithm", "budget_seconds", "seed", "sort_start_time"]).reset_index(drop=True)
    # 同一 seed 重跑时仅保留最新一条，避免旧结果混入统计。
    frame = (
        frame.groupby(["instance", "algorithm", "budget_seconds", "seed"], dropna=False, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return frame.drop(columns=["sort_start_time"])


def build_reference_fronts(run_frame, *, repo_root, output_dir, benchmark_id):
    reference_bundles = {}
    for (instance, budget_seconds), group in run_frame.groupby(["instance", "budget_seconds"], dropna=False):
        points = []
        source_rows = []
        algorithms = sorted(group["algorithm"].astype(str).unique().tolist())
        for row in group.to_dict("records"):
            payload, archive_path = load_archive_payload_from_row(row, repo_root=repo_root)
            if payload is None:
                continue
            point_matrix = minimization_points_from_archive_payload(payload)
            if point_matrix.size == 0:
                continue
            points.append(point_matrix)
            source_rows.append(
                {
                    "instance": str(instance),
                    "budgetSeconds": int(budget_seconds),
                    "algorithm": str(row["algorithm"]),
                    "seed": coerce_int(row.get("seed"), default=None),
                    "archivePath": archive_path.as_posix(),
                    "pointCount": int(point_matrix.shape[0]),
                }
            )
        if not points:
            continue
        union_points = np.vstack(points)
        reference_points = filter_nondominated(union_points)
        ideal = np.min(reference_points, axis=0).astype(float).tolist()
        nadir = np.max(reference_points, axis=0).astype(float).tolist()
        bundle = ReferenceFrontBundle(
            instance=str(instance),
            budget_seconds=int(budget_seconds),
            benchmark_id=str(benchmark_id),
            ideal=ideal,
            nadir=nadir,
            points_min=reference_points,
            algorithms=algorithms,
            source_rows=source_rows,
        )
        output_path = Path(output_dir) / f"{sanitize_filename(instance)}-budget{int(budget_seconds)}-reference-front.json"
        payload = bundle.to_payload(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        reference_bundles[(str(instance), int(budget_seconds))] = payload
    return reference_bundles


def compute_unified_metrics(run_frame, *, repo_root, reference_fronts):
    rows = []
    for row in run_frame.to_dict("records"):
        key = (str(row["instance"]), int(row["budget_seconds"]))
        reference_payload = reference_fronts.get(key)
        if reference_payload is None:
            continue
        payload, archive_path = load_archive_payload_from_row(row, repo_root=repo_root)
        candidate_points = minimization_points_from_archive_payload(payload) if payload is not None else np.empty((0, 2), dtype=float)
        reference_points = np.asarray(
            [item["moObjectivesMin"][:2] for item in list(reference_payload.get("items") or [])],
            dtype=float,
        )
        ideal = np.asarray(reference_payload.get("ideal") or [0.0, 0.0], dtype=float)
        nadir = np.asarray(reference_payload.get("nadir") or [1.0, 1.0], dtype=float)
        reference_norm = normalize_points(reference_points, ideal, nadir)
        candidate_norm = normalize_points(candidate_points, ideal, nadir)
        rows.append(
            {
                **row,
                "archive_path_abs": None if archive_path is None else archive_path.as_posix(),
                "candidate_point_count": int(candidate_points.shape[0]),
                "reference_point_count": int(reference_points.shape[0]),
                "reference_front_path": reference_payload.get("outputPath"),
                "hv_ref_front": compute_hypervolume_2d(candidate_norm),
                "igd_ref_front": compute_igd(reference_norm, candidate_norm),
                "spacing_ref_front": compute_spacing(candidate_norm),
                "spread_delta_ref_front": compute_generalized_spread(candidate_norm, reference_norm),
                "coverage_ref_to_s": compute_coverage(reference_norm, candidate_norm),
                "coverage_s_to_ref": compute_coverage(candidate_norm, reference_norm),
            }
        )
    return pd.DataFrame(rows)


def summarize_unified_metrics(unified_frame):
    if unified_frame.empty:
        return pd.DataFrame()
    metric_columns = [
        "rep_mhc",
        "rep_cr",
        "hv_ref_front",
        "igd_ref_front",
        "spread_delta_ref_front",
        "coverage_ref_to_s",
        "coverage_s_to_ref",
        "runtime_seconds",
    ]
    summary_rows = []
    group_columns = ["instance", "budget_seconds", "algorithm"]
    for keys, group in unified_frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["run_count"] = int(len(group))
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().astype(float)
            row[f"{metric}_mean"] = None if values.empty else float(values.mean())
            row[f"{metric}_std"] = None if values.empty else float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_median"] = None if values.empty else float(values.median())
            if metric == "rep_cr" or metric == "hv_ref_front" or metric == "coverage_s_to_ref":
                row[f"{metric}_best"] = None if values.empty else float(values.max())
            else:
                row[f"{metric}_best"] = None if values.empty else float(values.min())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)

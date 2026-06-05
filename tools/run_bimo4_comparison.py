import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.BiMO4BenchmarkUtil import benchmark_root
from src.utils.BiMO4BenchmarkUtil import sanitize_filename
import src.utils.config as config


RESULT_ROOT = Path(config.RESULT_PATH)
CR_MATRIX_DIR = REPO_ROOT / "data" / "cr_matrices"


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    algorithm_name: str
    module_name: str
    extra_env: dict


ALGORITHM_SPECS = {
    "bimo4": AlgorithmSpec(
        key="bimo4",
        algorithm_name="ELP_DRL_BiMO4",
        module_name="src.algorithms.ELP_DRL_BiMO4",
        extra_env={
            "ELP_BIMO_CR_PAIR_INSERT_ENABLE": "0",
            "ELP_BIMO_CR_BOUNDARY_REPARTITION_ENABLE": "1",
        },
    ),
    "bimo4_mechfix": AlgorithmSpec(
        key="bimo4_mechfix",
        algorithm_name="ELP_DRL_BiMO4_MECHFIX",
        module_name="src.algorithms.ELP_DRL_BiMO4",
        extra_env={
            "ELP_BIMO_CR_PAIR_INSERT_ENABLE": "0",
            "ELP_BIMO_CR_BOUNDARY_REPARTITION_ENABLE": "0",
        },
    ),
    "bimo4_paperls_intensify": AlgorithmSpec(
        key="bimo4_paperls_intensify",
        algorithm_name="ELP_DRL_BiMO4_PAPERLS_INTENSIFY",
        module_name="src.algorithms.ELP_DRL_BiMO4",
        extra_env={
            "ELP_BIMO_CR_PAIR_INSERT_ENABLE": "0",
            "ELP_BIMO_CR_BOUNDARY_REPARTITION_ENABLE": "1",
            "ELP_BIMO_ARCHIVE_PAPERLS_ENABLE": "1",
            "ELP_BIMO_ARCHIVE_PAPERLS_ANCHORS": "8",
            "ELP_BIMO_ARCHIVE_PAPERLS_PASSES": "2",
            "ELP_BIMO_ARCHIVE_PAPERLS_RESERVE_SECONDS": "80",
            "ELP_BIMO_ARCHIVE_PAPERLS_TIME_LIMIT_SECONDS": "80",
            "ELP_BIMO_ARCHIVE_PAPERLS_MAX_NEIGHBORS": "0",
        },
    ),
    "nsga2": AlgorithmSpec(
        key="nsga2",
        algorithm_name="MO_BASELINE_NSGA2",
        module_name="src.algorithms.ELP_DRL_BiMO4",
        extra_env={"ELP_MO_BASELINE_ALGO": "nsga2"},
    ),
    "moead": AlgorithmSpec(
        key="moead",
        algorithm_name="MO_BASELINE_MOEAD",
        module_name="src.algorithms.ELP_DRL_BiMO4",
        extra_env={"ELP_MO_BASELINE_ALGO": "moead"},
    ),
    "spea2": AlgorithmSpec(
        key="spea2",
        algorithm_name="MO_BASELINE_SPEA2",
        module_name="src.algorithms.ELP_DRL_BiMO4",
        extra_env={"ELP_MO_BASELINE_ALGO": "spea2"},
    ),
    "grasp_paperls": AlgorithmSpec(
        key="grasp_paperls",
        algorithm_name="ELP_DRL_BiMO4_GRASP_PAPERLS",
        module_name="src.algorithms.ELP_DRL_BiMO4_GRASP",
        extra_env={"ELP_GRASP_LOCAL_SEARCH_BACKEND": "paper_adapted"},
    ),
    "grasp_actionls": AlgorithmSpec(
        key="grasp_actionls",
        algorithm_name="ELP_DRL_BiMO4_GRASP_ACTIONLS",
        module_name="src.algorithms.ELP_DRL_BiMO4_GRASP",
        extra_env={"ELP_GRASP_LOCAL_SEARCH_BACKEND": "engineered"},
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="批量运行 BiMO4 双目标对比实验。")
    parser.add_argument("--benchmark-id", default="bimo4_compare_v1", help="实验批次 ID。")
    parser.add_argument("--instances", nargs="+", default=["Du62", "SC35", "AB20-ar3"], help="实例列表。")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(ALGORITHM_SPECS.keys()),
        choices=list(ALGORITHM_SPECS.keys()),
        help="算法列表。",
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=[1800, 3600], help="wall-time 预算（秒）。")
    parser.add_argument("--seed-base", type=int, default=20260526, help="默认种子起点。")
    parser.add_argument("--seed-count", type=int, default=10, help="默认种子数量。")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="显式种子列表，优先级高于 seed-base/seed-count。")
    parser.add_argument("--phase", default="main", help="实验阶段标签，例如 smoke/main/validation。")
    parser.add_argument("--run-tag", default="", help="同一 benchmark-id 下并行后台任务的计划和状态文件后缀。")
    parser.add_argument("--dry-run", action="store_true", help="仅生成计划，不实际执行。")
    parser.add_argument("--force-rerun", action="store_true", help="忽略计划状态文件中的完成记录。")
    parser.add_argument("--python", default=sys.executable, help="执行 Python 解释器路径。")
    parser.add_argument("--bimo4-g", type=int, default=1000000, help="BiMO4 主算法的 G 上限。")
    parser.add_argument("--bimo4-t-max", type=int, default=300, help="BiMO4 主算法的 t_max。")
    parser.add_argument("--grasp-g", type=int, default=1000000, help="GRASP 的 G 上限。")
    parser.add_argument("--grasp-t-max", type=int, default=60, help="GRASP 的 t_max。")
    parser.add_argument("--baseline-pop", type=int, default=64, help="MOEA 基线种群规模。")
    parser.add_argument("--baseline-gen", type=int, default=1000000, help="MOEA 基线最大代数上限。")
    parser.add_argument("--baseline-seq-len", type=int, default=300, help="MOEA 基线动作序列长度。")
    return parser.parse_args()


def build_seed_list(args):
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    return [int(args.seed_base + idx) for idx in range(int(max(args.seed_count, 1)))]


def build_remark(benchmark_id, budget_seconds, seed, phase):
    launched_at = datetime.now().isoformat(timespec="seconds")
    return (
        f"benchmark_id={benchmark_id};"
        f"budget_seconds={int(budget_seconds)};"
        f"seed={int(seed)};"
        f"phase={phase};"
        f"launcher=run_bimo4_comparison.py;"
        f"launched_at={launched_at}"
    )


INHERITED_EXPERIMENT_ENV_PREFIXES = (
    "ELP_BIMO_",
    "ELP_GRASP_",
    "ELP_MO_BASELINE_",
)
INHERITED_EXPERIMENT_ENV_NAMES = {
    "ELP_FIXED_SEEDS",
    "ELP_MO_BASELINE_ALGO",
}


def clear_inherited_experiment_env(env):
    for name in list(env.keys()):
        if name in INHERITED_EXPERIMENT_ENV_NAMES or any(
            name.startswith(prefix) for prefix in INHERITED_EXPERIMENT_ENV_PREFIXES
        ):
            env.pop(name, None)


def build_run_matrix(args, seeds):
    runs = []
    for instance in list(args.instances):
        for budget_seconds in list(args.budgets):
            for algorithm_key in list(args.algorithms):
                spec = ALGORITHM_SPECS[algorithm_key]
                for seed in list(seeds):
                    runs.append(
                        {
                            "instance": str(instance),
                            "budget_seconds": int(budget_seconds),
                            "algorithm_key": spec.key,
                            "algorithm_name": spec.algorithm_name,
                            "seed": int(seed),
                            "phase": str(args.phase),
                        }
                    )
    return pd.DataFrame(runs)


def _required_cr_matrix_path(instance):
    return CR_MATRIX_DIR / f"{str(instance).strip()}_CR.pkl"


def preflight_check(args, run_frame):
    missing_paths = []
    for spec in ALGORITHM_SPECS.values():
        module_relpath = Path(*str(spec.module_name).split(".")).with_suffix(".py")
        module_path = REPO_ROOT / module_relpath
        if not module_path.exists():
            missing_paths.append(module_path)
    for instance in sorted(run_frame["instance"].astype(str).unique().tolist()):
        matrix_path = _required_cr_matrix_path(instance)
        if not matrix_path.exists():
            missing_paths.append(matrix_path)
    if missing_paths:
        raise FileNotFoundError(
            "预检查失败，缺少以下必需文件：\n" + "\n".join(str(path) for path in missing_paths)
        )


def build_run_env(args, row):
    spec = ALGORITHM_SPECS[str(row["algorithm_key"])]
    budget_seconds = int(row["budget_seconds"])
    seed = int(row["seed"])
    remark = build_remark(args.benchmark_id, budget_seconds, seed, row["phase"])
    env = os.environ.copy()
    # 算法行为必须由当前 spec 决定，不能继承上一次 PowerShell 会话中的实验开关。
    clear_inherited_experiment_env(env)
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "ELP_IS_EXP": "False",
            "ELP_EXP_NUMBER": "1",
            "ELP_EXP_INSTANCE": str(row["instance"]),
            "ELP_EXP_ALGORITHM": spec.algorithm_name,
            "ELP_EXP_REMARK": remark,
            "ELP_BASE_SEED": str(seed),
            "ELP_WALL_TIME_LIMIT_SECONDS": str(budget_seconds),
        }
    )
    env.update(spec.extra_env)
    if spec.key == "bimo4":
        env.update({"ELP_G": str(args.bimo4_g), "ELP_T_MAX": str(args.bimo4_t_max)})
    elif spec.key.startswith("grasp"):
        env.update({"ELP_G": str(args.grasp_g), "ELP_T_MAX": str(args.grasp_t_max)})
    else:
        env.update(
            {
                "ELP_G": str(args.bimo4_g),
                "ELP_T_MAX": str(args.bimo4_t_max),
                "ELP_MO_BASELINE_POP": str(args.baseline_pop),
                "ELP_MO_BASELINE_GEN": str(args.baseline_gen),
                "ELP_MO_BASELINE_SEQ_LEN": str(args.baseline_seq_len),
            }
        )
    return env


def execute_run(args, row, output_dir):
    spec = ALGORITHM_SPECS[str(row["algorithm_key"])]
    env = build_run_env(args, row)
    label = (
        f"{sanitize_filename(row['instance'])}-"
        f"{sanitize_filename(row['algorithm_name'])}-"
        f"budget{int(row['budget_seconds'])}-"
        f"seed{int(row['seed'])}"
    )
    log_path = Path(output_dir) / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [args.python, "-m", str(spec.module_name)]
    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        log_excerpt = ""
        try:
            log_excerpt = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
        except Exception:
            log_excerpt = "<无法读取日志内容>"
        raise RuntimeError(
            f"实验任务失败: instance={row['instance']} | algo={row['algorithm_name']} | "
            f"budget={row['budget_seconds']} | seed={row['seed']} | rc={completed.returncode}\n"
            f"log={log_path.resolve().as_posix()}\n"
            f"--- 日志尾部 ---\n{log_excerpt}"
        )
    return completed.returncode, log_path, command


def load_status_map(status_path):
    if not status_path.exists():
        return {}
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    return {str(item["run_key"]): item for item in list(payload.get("runs") or [])}


def dump_status(status_path, records):
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "runs": records,
    }
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_run_key(row):
    return (
        f"{row['instance']}|{row['budget_seconds']}|"
        f"{row['algorithm_name']}|{row['seed']}|{row['phase']}"
    )


def main():
    args = parse_args()
    seeds = build_seed_list(args)
    output_dir = benchmark_root(RESULT_ROOT, args.benchmark_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = sanitize_filename(str(args.run_tag).strip()) if str(args.run_tag).strip() else ""
    run_tag_suffix = f"-{run_tag}" if run_tag else ""

    run_frame = build_run_matrix(args, seeds)
    preflight_check(args, run_frame)
    plan_path = output_dir / f"run_plan{run_tag_suffix}.csv"
    manifest_path = output_dir / f"run_manifest{run_tag_suffix}.json"
    status_path = output_dir / f"run_status{run_tag_suffix}.json"

    status_map = load_status_map(status_path)
    plan_records = []
    status_records = []
    for row in run_frame.to_dict("records"):
        run_key = make_run_key(row)
        existing = status_map.get(run_key, {})
        row["run_key"] = run_key
        row["status"] = str(existing.get("status") or "pending")
        plan_records.append(row)
    pd.DataFrame(plan_records).to_csv(plan_path, index=False, encoding="utf-8-sig")
    manifest_path.write_text(
        json.dumps(
            {
                "benchmarkId": args.benchmark_id,
                "runTag": run_tag,
                "instances": list(args.instances),
                "algorithms": [ALGORITHM_SPECS[key].algorithm_name for key in args.algorithms],
                "budgets": [int(value) for value in args.budgets],
                "seeds": [int(seed) for seed in seeds],
                "phase": args.phase,
                "runPlanPath": plan_path.resolve().as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for row in plan_records:
        run_key = str(row["run_key"])
        existing = status_map.get(run_key, {})
        if existing.get("status") == "done" and not args.force_rerun:
            status_records.append(existing)
            continue
        if args.dry_run:
            status_records.append(
                {
                    **row,
                    "status": "planned",
                    "returncode": None,
                    "log_path": None,
                    "command": [args.python, "-m", str(ALGORITHM_SPECS[row["algorithm_key"]].module_name)],
                }
            )
            continue
        print(
            f"[run] instance={row['instance']} | algo={row['algorithm_name']} | "
            f"budget={row['budget_seconds']} | seed={row['seed']}",
            flush=True,
        )
        log_path = None
        command = [args.python, "-m", str(ALGORITHM_SPECS[row["algorithm_key"]].module_name)]
        try:
            returncode, log_path, command = execute_run(args, row, output_dir)
            status_records.append(
                {
                    **row,
                    "status": "done",
                    "returncode": int(returncode),
                    "log_path": log_path.resolve().as_posix(),
                    "command": command,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            dump_status(status_path, status_records)
            print(
                f"[done] instance={row['instance']} | algo={row['algorithm_name']} | "
                f"budget={row['budget_seconds']} | seed={row['seed']} | rc={returncode}",
                flush=True,
            )
        except Exception:
            status_records.append(
                {
                    **row,
                    "status": "failed",
                    "returncode": 1,
                    "log_path": None if log_path is None else log_path.resolve().as_posix(),
                    "command": command,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            dump_status(status_path, status_records)
            raise

    dump_status(status_path, status_records)


if __name__ == "__main__":
    main()

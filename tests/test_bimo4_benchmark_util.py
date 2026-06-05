import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools.run_bimo4_comparison import ALGORITHM_SPECS
from tools.run_bimo4_comparison import build_run_env
from src.utils.BiMO4BenchmarkUtil import build_reference_fronts
from src.utils.BiMO4BenchmarkUtil import compute_unified_metrics
from src.utils.BiMO4BenchmarkUtil import filter_nondominated
from src.utils.BiMO4BenchmarkUtil import load_benchmark_runs
from src.utils.BiMO4BenchmarkUtil import parse_benchmark_remark


class TestBiMO4BenchmarkUtil(unittest.TestCase):
    def test_parse_benchmark_remark(self):
        payload = parse_benchmark_remark(
            "benchmark_id=bimo4_compare_v1; budget_seconds=1800; seed=20260526; phase=main"
        )
        self.assertEqual(payload["benchmark_id"], "bimo4_compare_v1")
        self.assertEqual(payload["budget_seconds"], "1800")
        self.assertEqual(payload["seed"], "20260526")
        self.assertEqual(payload["phase"], "main")

    def test_runner_has_distinct_bimo4_variant_specs(self):
        bimo4 = ALGORITHM_SPECS["bimo4"]
        mechfix = ALGORITHM_SPECS["bimo4_mechfix"]
        paperls_intensify = ALGORITHM_SPECS["bimo4_paperls_intensify"]

        self.assertEqual(bimo4.algorithm_name, "ELP_DRL_BiMO4")
        self.assertEqual(bimo4.extra_env["ELP_BIMO_CR_PAIR_INSERT_ENABLE"], "0")
        self.assertEqual(bimo4.extra_env["ELP_BIMO_CR_BOUNDARY_REPARTITION_ENABLE"], "1")
        self.assertEqual(mechfix.algorithm_name, "ELP_DRL_BiMO4_MECHFIX")
        self.assertEqual(mechfix.extra_env["ELP_BIMO_CR_PAIR_INSERT_ENABLE"], "0")
        self.assertEqual(mechfix.extra_env["ELP_BIMO_CR_BOUNDARY_REPARTITION_ENABLE"], "0")
        self.assertEqual(paperls_intensify.algorithm_name, "ELP_DRL_BiMO4_PAPERLS_INTENSIFY")
        self.assertEqual(paperls_intensify.extra_env["ELP_BIMO_CR_PAIR_INSERT_ENABLE"], "0")
        self.assertEqual(paperls_intensify.extra_env["ELP_BIMO_CR_BOUNDARY_REPARTITION_ENABLE"], "1")
        self.assertEqual(paperls_intensify.extra_env["ELP_BIMO_ARCHIVE_PAPERLS_ENABLE"], "1")
        self.assertEqual(paperls_intensify.extra_env["ELP_BIMO_ARCHIVE_PAPERLS_RESERVE_SECONDS"], "80")
        self.assertEqual(paperls_intensify.extra_env["ELP_BIMO_ARCHIVE_PAPERLS_TIME_LIMIT_SECONDS"], "80")
        self.assertNotIn("bimo4_cr17only_b24e3", ALGORITHM_SPECS)
        self.assertNotIn("bimo4_cr17nosplit_b24e3", ALGORITHM_SPECS)
        self.assertNotIn("bimo4_cr17adaptive_b24e3", ALGORITHM_SPECS)
        self.assertNotIn("bimo4_paperls_2only", ALGORITHM_SPECS)
        self.assertNotIn("bimo4_paperls_3only", ALGORITHM_SPECS)
        self.assertNotIn("bimo4_paperls_2plus3", ALGORITHM_SPECS)
        self.assertNotIn("bimo4_paperls_current80", ALGORITHM_SPECS)
        self.assertNotIn("bimo4_paperls_oldpool", ALGORITHM_SPECS)

    def test_runner_env_drops_inherited_fixed_seeds(self):
        args = SimpleNamespace(
            benchmark_id="seed_guard",
            bimo4_g=1000,
            bimo4_t_max=300,
            grasp_g=1000,
            grasp_t_max=300,
            baseline_pop=64,
            baseline_gen=80,
            baseline_seq_len=300,
        )
        row = {
            "algorithm_key": "bimo4_paperls_intensify",
            "budget_seconds": 600,
            "seed": 20260601,
            "phase": "validation",
            "instance": "AB20-ar3",
        }

        with patch.dict(os.environ, {"ELP_FIXED_SEEDS": "20260328", "ELP_BASE_SEED": "1"}, clear=False):
            env = build_run_env(args, row)

        self.assertNotIn("ELP_FIXED_SEEDS", env)
        self.assertEqual(env["ELP_BASE_SEED"], "20260601")
        self.assertIn("seed=20260601", env["ELP_EXP_REMARK"])

    def test_runner_env_isolates_inherited_algorithm_switches(self):
        args = SimpleNamespace(
            benchmark_id="env_guard",
            bimo4_g=1000,
            bimo4_t_max=300,
            grasp_g=1000,
            grasp_t_max=300,
            baseline_pop=64,
            baseline_gen=80,
            baseline_seq_len=300,
        )
        row = {
            "algorithm_key": "bimo4_paperls_intensify",
            "budget_seconds": 600,
            "seed": 20260601,
            "phase": "validation",
            "instance": "AB20-ar3",
        }
        inherited = {
            "ELP_BIMO_CR_PAIR_INSERT_ENABLE": "1",
            "ELP_BIMO_CR_BOUNDARY_REPARTITION_ENABLE": "0",
            "ELP_BIMO_CR_BOUNDARY_REPARTITION_BUDGET": "999",
            "ELP_GRASP_LOCAL_SEARCH_BACKEND": "engineered",
            "ELP_MO_BASELINE_ALGO": "nsga2",
            "ELP_MO_BASELINE_POP": "999",
        }

        with patch.dict(os.environ, inherited, clear=False):
            env = build_run_env(args, row)

        self.assertEqual(env["ELP_BIMO_CR_PAIR_INSERT_ENABLE"], "0")
        self.assertEqual(env["ELP_BIMO_CR_BOUNDARY_REPARTITION_ENABLE"], "1")
        self.assertNotIn("ELP_BIMO_CR_BOUNDARY_REPARTITION_BUDGET", env)
        self.assertNotIn("ELP_GRASP_LOCAL_SEARCH_BACKEND", env)
        self.assertNotIn("ELP_MO_BASELINE_ALGO", env)
        self.assertEqual(env["ELP_MO_BASELINE_POP"], "64")

    def test_runner_env_disables_action16_for_default_bimo4_after_isolation(self):
        args = SimpleNamespace(
            benchmark_id="env_guard",
            bimo4_g=1000,
            bimo4_t_max=300,
            grasp_g=1000,
            grasp_t_max=300,
            baseline_pop=64,
            baseline_gen=80,
            baseline_seq_len=300,
        )
        row = {
            "algorithm_key": "bimo4",
            "budget_seconds": 600,
            "seed": 20260601,
            "phase": "validation",
            "instance": "AB20-ar3",
        }

        with patch.dict(os.environ, {"ELP_BIMO_CR_PAIR_INSERT_ENABLE": "1"}, clear=False):
            env = build_run_env(args, row)

        self.assertEqual(env["ELP_BIMO_CR_PAIR_INSERT_ENABLE"], "0")
        self.assertEqual(env["ELP_BIMO_CR_BOUNDARY_REPARTITION_ENABLE"], "1")
        self.assertNotIn("ELP_BIMO_CR_BOUNDARY_REPARTITION_BUDGET", env)

    def test_filter_nondominated_2d(self):
        points = np.asarray(
            [
                [1.0, 3.0],
                [2.0, 2.0],
                [3.0, 1.0],
                [3.5, 3.5],
            ],
            dtype=float,
        )
        filtered = filter_nondominated(points)
        self.assertEqual(filtered.shape[0], 3)
        self.assertFalse(np.any(np.all(np.isclose(filtered, [3.5, 3.5]), axis=1)))

    def test_load_benchmark_runs_reads_snake_case_paperls_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_root = Path(tmp_dir)
            csv_path = result_root / "Du62-ELP_DRL_BiMO4_PAPERLS_INTENSIFY.csv"
            row = {
                "实例": "Du62",
                "算法": "ELP_DRL_BiMO4_PAPERLS_INTENSIFY",
                "日期": "2026-06-03",
                "迭代次数": 1,
                "解": "[]",
                "适应度值": 0.1,
                "开始时间": "2026-06-03 10:00:00",
                "最快时间": "2026-06-03 10:00:00",
                "结束时间": "2026-06-03 10:10:00",
                "运行时间（秒）": 600.0,
                "最快最佳结果时间（秒）": 600.0,
                "宽高比是否满足": True,
                "gbest更新次数": 0,
                "备注": "benchmark_id=bench;budget_seconds=600;seed=20260601;phase=validation",
            }
            row.update(
                {
                    "rep_mhc": 1.0,
                    "rep_cr": 2.0,
                    "archive_paperls_enabled": True,
                    "archive_paperls_stats": "{'anchorsUsed': 8, 'anchorDiagnostics': []}",
                    "main_wall_time_limit_seconds": 520.0,
                    "wall_time_limit_seconds": 600.0,
                }
            )
            pd.DataFrame(
                [row]
            ).to_csv(csv_path, index=False, encoding="utf-8-sig")

            runs = load_benchmark_runs(
                "bench",
                instances=["Du62"],
                algorithms=["ELP_DRL_BiMO4_PAPERLS_INTENSIFY"],
                result_root=result_root,
            )

        self.assertEqual(len(runs), 1)
        row = runs.iloc[0]
        self.assertIn(str(row["archive_paperls_enabled"]).lower(), {"true", "1"})
        self.assertIn("anchorsUsed", row["archive_paperls_stats"])
        self.assertEqual(row["main_wall_time_limit_seconds"], 520.0)

    def test_build_reference_fronts_and_unified_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            archive_dir = repo_root / "files" / "expresults" / "pareto_archives"
            archive_dir.mkdir(parents=True, exist_ok=True)

            archive_a = archive_dir / "Du62-ELP_DRL_BiMO4-a.json"
            archive_a.write_text(
                json.dumps(
                    {
                        "instance": "Du62",
                        "algorithm": "ELP_DRL_BiMO4",
                        "items": [
                            {"isFeasible": True, "moObjectivesMin": [1.0, -4.0], "mhc": 1.0, "cr": 4.0},
                            {"isFeasible": True, "moObjectivesMin": [2.0, -5.0], "mhc": 2.0, "cr": 5.0},
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            archive_b = archive_dir / "Du62-MO_BASELINE_NSGA2-b.json"
            archive_b.write_text(
                json.dumps(
                    {
                        "instance": "Du62",
                        "algorithm": "MO_BASELINE_NSGA2",
                        "items": [
                            {"isFeasible": True, "moObjectivesMin": [1.5, -4.5], "mhc": 1.5, "cr": 4.5},
                            {"isFeasible": True, "moObjectivesMin": [3.0, -3.0], "mhc": 3.0, "cr": 3.0},
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            run_frame = pd.DataFrame(
                [
                    {
                        "instance": "Du62",
                        "budget_seconds": 1800,
                        "algorithm": "ELP_DRL_BiMO4",
                        "seed": 1,
                        "phase": "main",
                        "benchmark_id": "bench",
                        "pareto_archive_path": archive_a.relative_to(repo_root).as_posix(),
                        "rep_mhc": 1.0,
                        "rep_cr": 4.0,
                    },
                    {
                        "instance": "Du62",
                        "budget_seconds": 1800,
                        "algorithm": "MO_BASELINE_NSGA2",
                        "seed": 1,
                        "phase": "main",
                        "benchmark_id": "bench",
                        "pareto_archive_path": archive_b.relative_to(repo_root).as_posix(),
                        "rep_mhc": 1.5,
                        "rep_cr": 4.5,
                    },
                ]
            )

            reference_fronts = build_reference_fronts(
                run_frame,
                repo_root=repo_root,
                output_dir=repo_root / "reference_fronts",
                benchmark_id="bench",
            )
            payload = reference_fronts[("Du62", 1800)]
            self.assertEqual(payload["pointCount"], 3)

            unified = compute_unified_metrics(run_frame, repo_root=repo_root, reference_fronts=reference_fronts)
            self.assertEqual(len(unified), 2)
            self.assertIn("hv_ref_front", unified.columns)
            self.assertTrue((pd.to_numeric(unified["coverage_ref_to_s"], errors="coerce") >= 0.0).all())


if __name__ == "__main__":
    unittest.main()

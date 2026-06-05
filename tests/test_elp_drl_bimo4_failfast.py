from types import SimpleNamespace

import numpy as np
import pytest

import src.algorithms.ELP_DRL_BiMO4 as bimo4
from src.utils.FBSModel import FBSModel


def test_parse_env_int_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("ELP_TEST_BAD_INT", "abc")

    with pytest.raises(ValueError, match="ELP_TEST_BAD_INT"):
        bimo4._parse_env_int("ELP_TEST_BAD_INT", 1)


def test_parse_env_int_list_rejects_invalid_token(monkeypatch):
    monkeypatch.setenv("ELP_TEST_BAD_LIST", "1, bad, 3")

    with pytest.raises(ValueError, match="bad"):
        bimo4._parse_env_int_list("ELP_TEST_BAD_LIST")


def test_parse_cr_boundary_operation_filter_rejects_invalid_operation(monkeypatch):
    monkeypatch.setenv("ELP_BIMO_CR_BOUNDARY_REPARTITION_OPERATIONS", "align_top,bad_op")

    with pytest.raises(ValueError, match="bad_op"):
        bimo4.ELP._parse_cr_boundary_operation_filter()


def test_parse_cr_boundary_operation_filter_preserves_valid_order(monkeypatch):
    monkeypatch.setenv("ELP_BIMO_CR_BOUNDARY_REPARTITION_OPERATIONS", "align_top, split, align_top")

    assert bimo4.ELP._parse_cr_boundary_operation_filter() == ("align_top", "split")


def test_preflight_required_files_fails_fast(monkeypatch, tmp_path):
    missing_file = tmp_path / "missing_instances.pkl"
    monkeypatch.setattr(bimo4.config, "FILE_PATH", str(missing_file))

    with pytest.raises(FileNotFoundError, match="预检查失败"):
        bimo4._preflight_required_files()


def test_ensure_cr_matrix_available_generates_missing_matrix(monkeypatch, tmp_path):
    target_dir = tmp_path / "cr_matrices"
    monkeypatch.setattr(bimo4.CRMatrixStore, "default_data_dir", staticmethod(lambda: target_dir))

    output_path = bimo4.ELP._ensure_cr_matrix_available("TEST", facility_count=4)

    assert output_path == target_dir / "TEST_CR.pkl"
    assert output_path.exists()
    matrix, payload, loaded_path = bimo4.CRMatrixStore.load_matrix("TEST", expected_facility_count=4)
    assert loaded_path == output_path
    assert payload["facility_count"] == 4
    assert matrix.shape == (4, 4)


def test_ensure_cr_matrix_available_does_not_overwrite_existing(monkeypatch, tmp_path):
    target_dir = tmp_path / "cr_matrices"
    monkeypatch.setattr(bimo4.CRMatrixStore, "default_data_dir", staticmethod(lambda: target_dir))
    existing_path = bimo4.ELP._ensure_cr_matrix_available("TEST", facility_count=3)
    original_mtime = existing_path.stat().st_mtime_ns

    returned_path = bimo4.ELP._ensure_cr_matrix_available("TEST", facility_count=5)

    matrix, payload, _loaded_path = bimo4.CRMatrixStore.load_matrix("TEST", expected_facility_count=3)
    assert returned_path == existing_path
    assert existing_path.stat().st_mtime_ns == original_mtime
    assert payload["facility_count"] == 3
    assert matrix.shape == (3, 3)


def test_refresh_dynamic_weights_skips_target_before_interval():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.mo_adaptive_weights_enabled = True
    solver.mo_weights = np.asarray([0.5, 0.5], dtype=float)
    solver.mo_base_weights = np.asarray([0.5, 0.5], dtype=float)
    solver.mo_weight_update_count = 1
    solver.mo_last_weight_update_step = 10
    solver.mo_adaptive_weight_refresh_interval_steps = 250
    solver._trace_global_step = 20

    def fail_if_called():
        raise AssertionError("target should not be recomputed before refresh interval")

    solver._compute_adaptive_weight_target = fail_if_called

    assert solver._refresh_dynamic_weights() == pytest.approx([0.5, 0.5])


def test_observe_feasible_state_skips_archive_refresh_when_archive_unchanged(monkeypatch):
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.pareto_archive = [
        SimpleNamespace(current_is_feasible=True, mo_objectives_min=np.asarray([1.0, -1.0], dtype=float))
    ]
    solver.archive_limit = 64
    solver.archive_update_count = 0
    solver.feasible_solution_count = 0
    solver.mo_worst_feasible_mhc = None
    solver.representative_solution = SimpleNamespace(
        current_is_feasible=True,
        mo_objectives_min=np.asarray([1.0, -1.0], dtype=float),
    )
    solver.representative_decision_score = 0.5
    solver.best_feasible_solution = SimpleNamespace()
    solver.best_feasible_cost = 0.5
    solver.best_energy = 0.5

    calls = {"archive_refresh": 0, "solution_refresh": 0}

    def fake_update_pareto_archive(candidates, candidate, **_kwargs):
        return list(candidates), False, 0

    monkeypatch.setattr(bimo4.MO_FBSUtil_BiMO4, "update_pareto_archive", fake_update_pareto_archive)

    def fake_refresh_archive_state():
        calls["archive_refresh"] += 1

    def fake_refresh_solution_search_metrics(solution):
        calls["solution_refresh"] += 1
        solution.decision_score = 0.6
        return 0.6

    solver._refresh_archive_state = fake_refresh_archive_state
    solver._refresh_solution_search_metrics = fake_refresh_solution_search_metrics

    candidate = SimpleNamespace(
        current_is_feasible=True,
        MHC=10.0,
        mo_objectives_min=np.asarray([2.0, -0.5], dtype=float),
        decision_score=0.6,
    )

    assert solver._observe_feasible_state(candidate) is False
    assert calls["archive_refresh"] == 0
    assert calls["solution_refresh"] == 1
    assert solver.archive_update_count == 0
    assert solver._last_archive_observation == {
        "archive_changed": False,
        "rep_changed": False,
        "removed_count": 0,
    }


def test_archive_anchor_selection_uses_stale_weight(monkeypatch):
    solver = bimo4.ELP.__new__(bimo4.ELP)
    first = SimpleNamespace(
        current_is_feasible=True,
        mo_objectives_min=np.asarray([0.0, -1.0], dtype=float),
        decision_score=0.1,
    )
    second = SimpleNamespace(
        current_is_feasible=True,
        mo_objectives_min=np.asarray([1.0, -2.0], dtype=float),
        decision_score=0.9,
    )
    solver.pareto_archive = [first, second]
    solver.mo_ideal = np.asarray([0.0, -2.0], dtype=float)
    solver.mo_nadir = np.asarray([1.0, -1.0], dtype=float)
    solver.mo_weights = np.asarray([0.5, 0.5], dtype=float)
    solver.mo_running_min = solver.mo_ideal.copy()
    solver.mo_running_max = solver.mo_nadir.copy()
    solver.bimo_archive_anchor_min_size = 2
    solver.bimo_anchor_quality_weight = 0.0
    solver.bimo_anchor_sparse_weight = 0.0
    solver.bimo_anchor_extreme_weight = 0.0
    solver.bimo_anchor_stale_weight = 1.0
    solver.bimo_anchor_min_probability_weight = 0.0
    solver._bimo_anchor_visit_counts = {
        solver._bimo_solution_key(first): 5,
        solver._bimo_solution_key(second): 0,
    }

    captured = {}

    def choose_highest_probability(_count, p):
        captured["p"] = np.asarray(p, dtype=float)
        return int(np.argmax(captured["p"]))

    monkeypatch.setattr(bimo4.np.random, "choice", choose_highest_probability)

    selected, details = solver._select_bimo_archive_anchor()

    assert selected is second
    assert details["visit_count"] == 0
    assert captured["p"][1] > captured["p"][0]


def test_prepare_episode_start_switches_to_archive_anchor(monkeypatch):
    solver = bimo4.ELP.__new__(bimo4.ELP)
    selected = SimpleNamespace(
        current_is_feasible=True,
        mo_objectives_min=np.asarray([0.5, -2.0], dtype=float),
        fitness=0.3,
        decision_score=0.3,
        MHC=10.0,
        CR=2.0,
    )
    solver.s = SimpleNamespace(fitness=1.0)
    solver.pareto_archive = [selected]
    solver.bimo_archive_anchor_selection_enabled = True
    solver.bimo_archive_anchor_switch_interval = 1
    solver._bimo_last_anchor_episode = -10**9
    solver._bimo_anchor_visit_counts = {}
    solver._safe_float = lambda value: None if value is None else float(value)
    solver._record_mo_event = lambda *_args, **_kwargs: None

    monkeypatch.setattr(bimo4.MO4ELP, "_prepare_episode_start", lambda _self, _episode_idx: None)
    monkeypatch.setattr(
        solver,
        "_select_bimo_archive_anchor",
        lambda: (selected, {"candidate_count": 1, "selected_index": 0, "probability": 1.0}),
    )
    solver._evaluate_solution = lambda solution: setattr(solution, "fitness", 0.3)

    solver._prepare_episode_start(0)

    assert solver.s is not selected
    assert solver.s.fitness == pytest.approx(0.3)
    assert solver.current_energy == pytest.approx(0.3)
    assert solver._bimo_anchor_visit_counts[solver._bimo_solution_key(selected)] == 1


def test_bimo_bootstrap_initial_archive_collects_multiple_candidates():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.s = SimpleNamespace(name="seed", decision_score=0.2, fitness=0.2)
    solver.env = SimpleNamespace()
    solver.pareto_archive = []
    solver.bimo_archive_bootstrap_size = 2
    solver.bimo_archive_bootstrap_attempt_factor = 1
    solver.bootstrap_recipes = [[0]]
    solver.mo_base_weights = np.asarray([0.5, 0.5], dtype=float)
    solver.mo_weights = np.asarray([0.5, 0.5], dtype=float)
    solver.mo_adaptive_weight_min_component = 0.2
    solver.mo_last_weight_target = solver.mo_base_weights.copy()
    solver.mo_weight_update_count = 0
    solver.mo_last_weight_update_step = -10**9
    solver.best_feasible_solution = None
    solver.best_feasible_cost = np.inf
    solver.best_energy = np.inf
    solver.worst_feasible_cost = None
    solver.feasible_solution_count = 0
    solver.mo_worst_feasible_mhc = None
    solver.gbest_update_count = 0
    solver._bimo_anchor_visit_counts = {}
    solver._record_mo_event = lambda *_args, **_kwargs: None
    solver._light_clone_solution = lambda solution: SimpleNamespace(**solution.__dict__)

    generated = [
        SimpleNamespace(
            name="candidate",
            decision_score=0.1,
            fitness=0.1,
            current_is_feasible=True,
            mo_objectives_min=np.asarray([0.5, -2.0], dtype=float),
            MHC=5.0,
            CR=2.0,
        )
    ]

    def generate_candidate(_base, _recipe):
        return generated.pop(0)

    def evaluate(solution):
        solution.current_is_feasible = True
        if getattr(solution, "name", "") == "candidate":
            solution.mo_objectives_min = np.asarray([0.5, -2.0], dtype=float)
            solution.MHC = 5.0
            solution.CR = 2.0
        else:
            solution.mo_objectives_min = np.asarray([1.0, -1.0], dtype=float)
            solution.MHC = 10.0
            solution.CR = 1.0
        return {}

    def observe(solution):
        solver.pareto_archive.append(SimpleNamespace(**solution.__dict__))
        solver.best_feasible_solution = solver.pareto_archive[-1]
        solver.best_feasible_cost = float(solution.decision_score)
        solver.best_energy = float(solution.decision_score)
        solver.worst_feasible_cost = float(solution.fitness)
        return True

    def refresh_archive_state():
        solver.representative_solution = solver.pareto_archive[-1]
        solver.representative_decision_score = float(solver.representative_solution.decision_score)
        solver.representative_archive_index = len(solver.pareto_archive) - 1

    solver._generate_candidate_by_recipe = generate_candidate
    solver._evaluate_solution = evaluate
    solver._observe_feasible_state = observe
    solver._refresh_archive_state = refresh_archive_state

    assert solver._bootstrap_bimo_initial_archive(max_attempts=2) is True
    assert len(solver.pareto_archive) == 2
    assert solver.representative_solution.name == "candidate"


def test_agent_state_context_exposes_bimo_objective_features():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.bimo_dqn_context_enabled = True
    solver.rl_context_dim = 10
    solver.archive_limit = 64
    solver.mo_weights = np.asarray([0.4, 0.6], dtype=float)
    solver.mo_ideal = np.asarray([0.0, -10.0], dtype=float)
    solver.mo_nadir = np.asarray([10.0, 0.0], dtype=float)
    solver.mo_running_min = solver.mo_ideal.copy()
    solver.mo_running_max = solver.mo_nadir.copy()
    solver.representative_solution = SimpleNamespace(
        current_is_feasible=True,
        mo_objectives_min=np.asarray([4.0, -6.0], dtype=float),
    )
    solution = SimpleNamespace(
        current_is_feasible=True,
        mo_objectives_min=np.asarray([2.0, -8.0], dtype=float),
    )
    solver.pareto_archive = [
        solver.representative_solution,
        SimpleNamespace(current_is_feasible=True, mo_objectives_min=np.asarray([8.0, -2.0], dtype=float)),
    ]

    context = solver._agent_state_context(solution)

    assert context.shape == (10,)
    assert context.dtype == np.float32
    assert np.all(context >= 0.0)
    assert np.all(context <= 1.0)
    assert context[8] == pytest.approx(0.4)
    assert context[9] == pytest.approx(0.6)


def test_bimo_reward_shaping_rewards_unaccepted_archive_candidate(monkeypatch):
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.bimo_reward_archive_bonus = 0.45
    solver.bimo_reward_unaccepted_archive_bonus = 0.25
    solver.bimo_reward_cr_gain_weight = 0.40
    solver.bimo_reward_cr_loss_weight = 0.08
    solver.bimo_reward_extreme_gain_weight = 0.35
    solver.bimo_reward_sparse_gain_weight = 0.25
    solver.bimo_reward_hv_proxy_weight = 0.50
    solver.bimo_reward_clip = 4.0
    solver._last_transition_meta = {
        "archive_would_change": True,
        "cr_relative_gain": 0.5,
        "archive_extreme_gain": 0.4,
        "archive_sparse_distance": 0.6,
        "archive_hv_gain_proxy": 0.3,
    }

    monkeypatch.setattr(bimo4.MO4ELP, "_compute_transition_reward", lambda *_args, **_kwargs: 0.0)

    reward = solver._compute_transition_reward(
        previous_cost=1.0,
        next_cost=1.0,
        previous_d_inf=0,
        next_d_inf=0,
        previous_best_feasible=1.0,
        accept=False,
    )

    assert reward > 1.0


def test_candidate_archive_quality_features_reports_hv_gain_proxy():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.pareto_archive = [
        SimpleNamespace(current_is_feasible=True, mo_objectives_min=np.asarray([0.0, -1.0], dtype=float)),
        SimpleNamespace(current_is_feasible=True, mo_objectives_min=np.asarray([1.0, -2.0], dtype=float)),
    ]
    candidate = SimpleNamespace(
        current_is_feasible=True,
        mo_objectives_min=np.asarray([0.5, -1.6], dtype=float),
    )

    features = solver._candidate_archive_quality_features(candidate)

    assert features["hv_gain_proxy"] > 0.0
    assert 0.0 <= features["hv_gain_proxy"] <= 1.0


def test_register_bimo_cr_boundary_repartition_action_adds_action_and_elite():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.valid_actions = [0, 2, 3]
    solver.action_labels = {0: "facility_swap", 2: "bay_swap", 3: "repair"}
    solver.action_telemetry = {}
    solver.action_base_explore_weights = {}
    solver.action_recent_selected = {}
    solver.action_recent_accepted = {}
    solver.action_recent_gbest = {}
    solver.elite_actions = [0]
    solver.elite_action_trials = {0: 1}
    solver.bimo_cr_boundary_repartition_enabled = True
    solver.bimo_cr_boundary_elite_enabled = True
    solver.bimo_cr_boundary_elite_trials = 3
    solver.bimo_cr_boundary_explore_weight = 1.20

    solver._register_bimo_cr_boundary_repartition_action()

    assert bimo4.ELP.CR_BOUNDARY_REPARTITION_ACTION_ID in solver.valid_actions
    assert solver.action_labels[bimo4.ELP.CR_BOUNDARY_REPARTITION_ACTION_ID] == "cr_boundary_repartition"
    assert solver.action_telemetry[bimo4.ELP.CR_BOUNDARY_REPARTITION_ACTION_ID]["name"] == "cr_boundary_repartition"
    assert solver.elite_actions[0] == bimo4.ELP.CR_BOUNDARY_REPARTITION_ACTION_ID
    assert solver.elite_action_trials[bimo4.ELP.CR_BOUNDARY_REPARTITION_ACTION_ID] == 3
    assert solver.action_base_explore_weights[bimo4.ELP.CR_BOUNDARY_REPARTITION_ACTION_ID] == pytest.approx(1.20)


def test_cr_boundary_operation_telemetry_records_selected_operation():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver._reset_bimo_cr_boundary_operation_telemetry()
    solver._record_bimo_cr_boundary_generated_encodings(
        [
            {"operation": "split"},
            {"operation": "align_top"},
        ]
    )
    solver._last_generated_cr_boundary_info = {
        "operation": "split",
        "archive_would_change": True,
        "relation_score": 10.0,
        "cr_gain": 4.0,
        "mhc_loss": 2.0,
        "hv_gain_proxy": 0.5,
        "extreme_gain": 0.25,
        "sparse_distance": 0.75,
    }

    solver._record_bimo_cr_boundary_selection(
        bimo4.ELP.CR_BOUNDARY_REPARTITION_ACTION_ID,
        previous_cost=100.0,
        next_cost=98.0,
        phase="main",
    )
    solver._record_bimo_cr_boundary_acceptance(
        bimo4.ELP.CR_BOUNDARY_REPARTITION_ACTION_ID,
        previous_cost=100.0,
        next_cost=98.0,
        improved=True,
        phase="main",
    )
    solver._last_transition_meta = {
        "cr_boundary_repartition_used": True,
        "cr_boundary_repartition_operation": "split",
    }
    solver._record_bimo_cr_boundary_global_best(
        bimo4.ELP.CR_BOUNDARY_REPARTITION_ACTION_ID,
        phase="main",
    )

    telemetry = solver.get_bimo_cr_boundary_operation_telemetry()
    split_stats = telemetry["split"]
    assert split_stats["generated"] == 1
    assert split_stats["selected"] == 1
    assert split_stats["accepted"] == 1
    assert split_stats["improved"] == 1
    assert split_stats["global_best_hits"] == 1
    assert split_stats["archive_would_change"] == 1
    assert split_stats["archive_changed"] == 1
    assert split_stats["avg_cr_gain"] == pytest.approx(4.0)
    assert split_stats["avg_mhc_loss"] == pytest.approx(2.0)
    assert any("Action 17 op=split" in line for line in solver.format_bimo_cr_boundary_operation_telemetry())


def test_cr_boundary_repartition_generates_candidate_with_cross_boundary_alignment():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.rel_matrix = np.asarray(
        [
            [0.0, 1.0, 10.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [10.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    solver.bimo_cr_boundary_repartition_budget = 8
    solver.bimo_cr_boundary_top_bay_pairs = 2
    solver.bimo_cr_boundary_block_limit = 2
    solver.pareto_archive = []
    solver.archive_limit = 64
    solver.mo_ideal = None
    solver.mo_nadir = None
    solver.mo_weights = np.asarray([0.5, 0.5], dtype=float)
    solver.mo_running_min = np.asarray([0.0, -10.0], dtype=float)
    solver.mo_running_max = np.asarray([20.0, 0.0], dtype=float)

    base = SimpleNamespace(
        fbs_model=FBSModel([1, 2, 4, 3], [0, 1, 0, 1]),
        current_is_feasible=True,
        current_d_inf=0,
        constraint_violation=0.0,
        MHC=10.0,
        CR=0.0,
        mo_objectives_min=np.asarray([10.0, 0.0], dtype=float),
        fitness=0.5,
        decision_score=0.5,
    )

    def evaluate(candidate):
        layout = candidate.fbs_model.array_2d
        target_aligned = False
        if len(layout) >= 2:
            left_bay = list(layout[0])
            right_bay = list(layout[1])
            if 1 in left_bay and 3 in right_bay:
                target_aligned = left_bay.index(1) == right_bay.index(3)
        candidate.current_is_feasible = True
        candidate.current_d_inf = 0
        candidate.constraint_violation = 0.0
        candidate.MHC = 11.0 if target_aligned else 10.0
        candidate.CR = 10.0 if target_aligned else 0.0
        candidate.mo_objectives_min = np.asarray([candidate.MHC, -candidate.CR], dtype=float)
        candidate.fitness = 0.2 if target_aligned else 0.5
        candidate.decision_score = candidate.fitness
        return {}

    solver._evaluate_solution = evaluate

    candidate = solver._generate_bimo_cr_boundary_repartition_candidate(base)

    assert candidate.CR == pytest.approx(10.0)
    assert candidate.bimo_cr_boundary_repartition_info["cr_gain"] == pytest.approx(10.0)
    assert candidate.bimo_cr_boundary_repartition_info["archive_would_change"] is True
    assert candidate.bimo_cr_boundary_repartition_info["operation"] in {"align_top", "align_bottom", "split", "block_swap"}


def test_cr_boundary_operation_filter_excludes_split_candidates():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.rel_matrix = np.asarray(
        [
            [0.0, 1.0, 10.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [10.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    solver.bimo_cr_boundary_repartition_budget = 32
    solver.bimo_cr_boundary_top_bay_pairs = 2
    solver.bimo_cr_boundary_block_limit = 2
    solver.bimo_cr_boundary_enabled_operations = {"align_top", "align_bottom", "block_swap"}
    base = SimpleNamespace(
        fbs_model=FBSModel([1, 2, 4, 3], [0, 1, 0, 1]),
    )

    encodings = solver._bimo_cr_boundary_repartition_encodings(base)
    operations = {encoding["operation"] for encoding in encodings}

    assert encodings
    assert "split" not in operations
    assert operations <= {"align_top", "align_bottom", "block_swap"}


def test_cr_boundary_adaptive_weight_penalizes_unproductive_split():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.bimo_cr_boundary_enabled_operations = set(bimo4.ELP.CR_BOUNDARY_REPARTITION_OPERATIONS)
    solver.bimo_cr_boundary_adaptive_operation_enabled = True
    solver.bimo_cr_boundary_adaptive_min_selected = 10
    solver.bimo_cr_boundary_adaptive_min_weight = 0.15
    solver.bimo_cr_boundary_adaptive_max_weight = 2.5
    solver.bimo_cr_boundary_adaptive_strength = 1.0
    solver.bimo_cr_boundary_adaptive_loss_weight = 1.0
    solver.bimo_cr_boundary_adaptive_accept_weight = 0.25
    solver._reset_bimo_cr_boundary_operation_telemetry()

    telemetry = solver.bimo_cr_boundary_operation_telemetry
    telemetry["split"].update(
        {
            "selected": 120,
            "accepted": 2,
            "global_best_hits": 0,
            "archive_changed": 0,
            "mhc_loss_norm_sum": 18.0,
            "cr_gain_norm_sum": 0.2,
        }
    )
    telemetry["block_swap"].update(
        {
            "selected": 60,
            "accepted": 20,
            "global_best_hits": 8,
            "archive_changed": 8,
            "mhc_loss_norm_sum": 3.0,
            "cr_gain_norm_sum": 2.0,
        }
    )
    telemetry["align_top"].update({"selected": 60, "accepted": 8, "mhc_loss_norm_sum": 4.0})
    telemetry["align_bottom"].update({"selected": 60, "accepted": 8, "mhc_loss_norm_sum": 4.0})

    split_weight = solver._bimo_cr_boundary_operation_adaptive_weight("split")
    block_weight = solver._bimo_cr_boundary_operation_adaptive_weight("block_swap")

    assert split_weight < 1.0
    assert block_weight > 1.0
    assert split_weight < block_weight


def test_cr_boundary_adaptive_budget_caps_follow_operation_weights():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.bimo_cr_boundary_enabled_operations = set(bimo4.ELP.CR_BOUNDARY_REPARTITION_OPERATIONS)
    solver.bimo_cr_boundary_adaptive_operation_enabled = True
    solver.bimo_cr_boundary_adaptive_min_selected = 10
    solver.bimo_cr_boundary_adaptive_min_weight = 0.15
    solver.bimo_cr_boundary_adaptive_max_weight = 2.5
    solver.bimo_cr_boundary_adaptive_strength = 1.0
    solver.bimo_cr_boundary_adaptive_loss_weight = 1.0
    solver.bimo_cr_boundary_adaptive_accept_weight = 0.25
    solver._reset_bimo_cr_boundary_operation_telemetry()
    solver.bimo_cr_boundary_operation_telemetry["split"].update(
        {"selected": 100, "accepted": 1, "mhc_loss_norm_sum": 20.0}
    )
    solver.bimo_cr_boundary_operation_telemetry["block_swap"].update(
        {"selected": 100, "accepted": 30, "archive_changed": 10, "global_best_hits": 10, "mhc_loss_norm_sum": 2.0}
    )

    caps = solver._bimo_cr_boundary_operation_budget_caps(
        bimo4.ELP.CR_BOUNDARY_REPARTITION_OPERATIONS,
        budget=24,
    )

    assert caps["block_swap"] > caps["split"]
    assert caps["split"] >= 1


def test_cr_boundary_candidate_selection_key_uses_adaptive_weight():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.bimo_cr_boundary_enabled_operations = set(bimo4.ELP.CR_BOUNDARY_REPARTITION_OPERATIONS)
    solver.bimo_cr_boundary_adaptive_operation_enabled = True
    solver.bimo_cr_boundary_adaptive_min_selected = 10
    solver.bimo_cr_boundary_adaptive_min_weight = 0.15
    solver.bimo_cr_boundary_adaptive_max_weight = 2.5
    solver.bimo_cr_boundary_adaptive_strength = 1.0
    solver.bimo_cr_boundary_adaptive_loss_weight = 1.0
    solver.bimo_cr_boundary_adaptive_accept_weight = 0.25
    solver._reset_bimo_cr_boundary_operation_telemetry()
    solver.bimo_cr_boundary_operation_telemetry["split"].update(
        {"selected": 100, "accepted": 1, "mhc_loss_norm_sum": 20.0}
    )
    solver.bimo_cr_boundary_operation_telemetry["block_swap"].update(
        {"selected": 100, "accepted": 30, "archive_changed": 10, "global_best_hits": 10, "mhc_loss_norm_sum": 2.0}
    )
    base_key = (1, 1, 0.1, 0.1, 0.1, 0.1, -0.01, -0.5, 1.0)
    candidate_info = {
        "archive_would_change": True,
        "cr_gain_norm": 0.1,
        "mhc_loss_norm": 0.01,
        "hv_gain_proxy": 0.1,
        "extreme_gain": 0.1,
        "sparse_distance": 0.1,
    }

    split_key, split_weight, _split_score = solver._bimo_cr_boundary_candidate_selection_key(
        base_key,
        candidate_info,
        "split",
    )
    block_key, block_weight, _block_score = solver._bimo_cr_boundary_candidate_selection_key(
        base_key,
        candidate_info,
        "block_swap",
    )

    assert block_weight > split_weight
    assert block_key > split_key


def _paperls_test_solution(permutation, bay, mhc, cr, score):
    return SimpleNamespace(
        fbs_model=FBSModel(permutation, bay),
        current_is_feasible=True,
        MHC=float(mhc),
        CR=float(cr),
        decision_score=float(score),
        fitness=float(score),
        mo_objectives_min=np.asarray([float(mhc), -float(cr)], dtype=float),
    )


def test_archive_paperls_effective_time_limit_uses_reserve_when_limit_missing():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.bimo_archive_paperls_time_limit_seconds = 0.0
    solver.bimo_archive_paperls_reserve_seconds = 7.5

    assert solver._bimo_archive_paperls_effective_time_limit() == pytest.approx(7.5)

    solver.bimo_archive_paperls_time_limit_seconds = 3.0
    assert solver._bimo_archive_paperls_effective_time_limit() == pytest.approx(3.0)


def test_archive_paperls_anchor_pool_uses_unique_representative_extreme_and_sparse():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.bimo_archive_paperls_anchor_count = 3
    solver.mo_ideal = np.asarray([10.0, -12.0], dtype=float)
    solver.mo_nadir = np.asarray([30.0, -1.0], dtype=float)
    solver.representative_solution = _paperls_test_solution([1, 2, 3], [0, 0, 1], 15.0, 5.0, 0.2)
    solver.best_feasible_solution = solver.representative_solution
    solver.pareto_archive = [
        solver.representative_solution,
        _paperls_test_solution([3, 2, 1], [0, 0, 1], 10.0, 1.0, 0.4),
        _paperls_test_solution([1, 3, 2], [0, 0, 1], 30.0, 12.0, 0.5),
        _paperls_test_solution([2, 1, 3], [0, 0, 1], 20.0, 6.0, 0.3),
    ]

    anchors = solver._bimo_archive_paperls_anchor_pool()
    keys = [solver._bimo_solution_key(anchor) for anchor in anchors]

    assert len(anchors) == 3
    assert len(set(keys)) == len(keys)
    assert solver._bimo_solution_key(solver.representative_solution) in set(keys)


def test_archive_paperls_anchor_pool_uses_candidate_pool_only_when_it_can_change_archive():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.bimo_archive_paperls_anchor_count = 3
    solver.mo_ideal = np.asarray([10.0, -20.0], dtype=float)
    solver.mo_nadir = np.asarray([90.0, -1.0], dtype=float)
    solver.archive_limit = 64
    solver.representative_solution = _paperls_test_solution([1, 2, 3], [0, 0, 1], 15.0, 5.0, 0.2)
    solver.best_feasible_solution = solver.representative_solution
    solver.pareto_archive = [
        solver.representative_solution,
    ]
    high_cr_pool = _paperls_test_solution([4, 5, 6], [0, 0, 1], 70.0, 20.0, 0.1)
    dominated_pool = _paperls_test_solution([6, 5, 4], [0, 0, 1], 80.0, 14.0, 0.15)
    solver.bimo_candidate_pool = [
        high_cr_pool,
        dominated_pool,
    ]

    anchors = solver._bimo_archive_paperls_anchor_pool()
    keys = {solver._bimo_solution_key(anchor) for anchor in anchors}

    assert solver._bimo_solution_key(high_cr_pool) in keys
    assert solver._bimo_solution_key(dominated_pool) not in keys


def test_archive_paperls_uses_per_anchor_budget_and_records_diagnostics(monkeypatch):
    solver = bimo4.ELP.__new__(bimo4.ELP)
    anchors = [
        _paperls_test_solution([1, 2, 3], [0, 0, 1], 10.0, 1.0, 0.1),
        _paperls_test_solution([3, 2, 1], [0, 0, 1], 20.0, 8.0, 0.2),
        _paperls_test_solution([1, 3, 2], [0, 0, 1], 30.0, 12.0, 0.3),
    ]
    solver.bimo_archive_paperls_enabled = True
    solver._bimo_archive_paperls_done = False
    solver.bimo_archive_paperls_anchor_count = 3
    solver.bimo_archive_paperls_passes = 2
    solver.bimo_archive_paperls_time_limit_seconds = 30.0
    solver.bimo_archive_paperls_reserve_seconds = 0.0
    solver.bimo_archive_paperls_max_neighbor_evaluations = 0
    solver.pareto_archive = [anchors[0]]
    solver.bimo_candidate_pool = anchors[1:]
    solver.representative_solution = None
    solver.best_feasible_solution = None
    solver.archive_update_count = 0
    solver._refresh_archive_state = lambda: None
    solver._bimo_archive_paperls_anchor_pool = lambda: anchors

    class FakePaperLocalSearch:
        instances = []

        def __init__(self, solver_arg, passes, time_limit_seconds, max_neighbor_evaluations):
            self.solver = solver_arg
            self.passes = passes
            self.time_limit_seconds = float(time_limit_seconds)
            self.max_neighbor_evaluations = int(max_neighbor_evaluations)
            self.archive_insertions = 0
            FakePaperLocalSearch.instances.append(self)

        def local_search(self, solution):
            return solution

        def _observe_candidate(self, _candidate):
            self.archive_insertions += 1
            self.solver.archive_update_count += 1
            return True

        def summary(self):
            return {
                "neighborEvaluations": 2,
                "acceptedMoves": 1,
                "archiveInsertions": self.archive_insertions,
                "stoppedByTime": False,
                "stoppedByEvaluationLimit": False,
                "runtimeSeconds": 0.01,
            }

    monkeypatch.setattr(bimo4, "BiMO4PaperLocalSearch", FakePaperLocalSearch)

    stats = solver._run_bimo_archive_paperls_intensification()

    assert stats["anchorsSelected"] == 3
    assert stats["anchorsUsed"] == 3
    assert stats["anchorsFromArchive"] == 1
    assert stats["anchorsFromCandidatePool"] == 2
    assert stats["perAnchorTimeLimitSeconds"] == pytest.approx(10.0)
    assert [item.time_limit_seconds for item in FakePaperLocalSearch.instances] == pytest.approx([10.0, 10.0, 10.0])
    assert stats["neighborEvaluations"] == 6
    assert stats["acceptedMoves"] == 3
    assert stats["archiveInsertions"] == 3
    assert len(stats["anchorDiagnostics"]) == 3
    assert [item["source"] for item in stats["anchorDiagnostics"]] == ["archive", "candidate_pool", "candidate_pool"]
    assert stats["anchorDiagnostics"][0]["anchor"]["mhc"] == pytest.approx(10.0)
    assert stats["anchorDiagnostics"][1]["refined"]["cr"] == pytest.approx(8.0)


def test_runtime_anchor_candidates_limit_candidate_pool_fraction():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.bimo_candidate_pool_runtime_max_fraction = 0.30
    solver.bimo_candidate_pool_min_anchor_size = 1
    solver.mo_ideal = np.asarray([10.0, -15.0], dtype=float)
    solver.mo_nadir = np.asarray([120.0, -1.0], dtype=float)
    solver.pareto_archive = [
        _paperls_test_solution([1, 2, 3], [0, 0, 1], 10.0, 1.0, 0.1),
        _paperls_test_solution([3, 2, 1], [0, 0, 1], 20.0, 5.0, 0.2),
        _paperls_test_solution([1, 3, 2], [0, 0, 1], 30.0, 9.0, 0.3),
        _paperls_test_solution([2, 1, 3], [0, 0, 1], 40.0, 12.0, 0.4),
    ]
    solver.bimo_candidate_pool = [
        _paperls_test_solution([4, 5, 6], [0, 0, 1], 60.0, 15.0, 0.1),
        _paperls_test_solution([6, 5, 4], [0, 0, 1], 70.0, 14.0, 0.2),
        _paperls_test_solution([4, 6, 5], [0, 0, 1], 80.0, 13.0, 0.3),
    ]

    candidates = solver._bimo_archive_anchor_candidates()
    archive_keys = {solver._bimo_solution_key(candidate) for candidate in solver.pareto_archive}
    pool_count = sum(1 for candidate in candidates if solver._bimo_solution_key(candidate) not in archive_keys)

    assert len(candidates) == 5
    assert pool_count == 1


def test_candidate_pool_prune_keeps_mhc_cr_and_balanced_slices():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    solver.bimo_candidate_pool_limit = 4
    solver.bimo_candidate_pool_mhc_quota_fraction = 0.25
    solver.bimo_candidate_pool_cr_quota_fraction = 0.25
    solver.bimo_candidate_pool_balanced_quota_fraction = 0.25
    solver.mo_ideal = np.asarray([1.0, -50.0], dtype=float)
    solver.mo_nadir = np.asarray([100.0, -1.0], dtype=float)
    solver._bimo_candidate_pool_visit_counts = {}
    low_mhc = _paperls_test_solution([1, 2, 3], [0, 0, 1], 1.0, 1.0, 0.9)
    high_cr = _paperls_test_solution([3, 2, 1], [0, 0, 1], 100.0, 50.0, 0.8)
    balanced = _paperls_test_solution([1, 3, 2], [0, 0, 1], 20.0, 10.0, 0.1)
    solver.bimo_candidate_pool = [
        low_mhc,
        high_cr,
        balanced,
        _paperls_test_solution([2, 1, 3], [0, 0, 1], 40.0, 20.0, 0.4),
        _paperls_test_solution([4, 5, 6], [0, 0, 1], 50.0, 30.0, 0.5),
    ]

    solver._prune_bimo_candidate_pool()
    kept_keys = {solver._bimo_solution_key(candidate) for candidate in solver.bimo_candidate_pool}

    assert len(solver.bimo_candidate_pool) == 4
    assert solver._bimo_solution_key(low_mhc) in kept_keys
    assert solver._bimo_solution_key(high_cr) in kept_keys
    assert solver._bimo_solution_key(balanced) in kept_keys


def test_archive_candidate_before_current_update_archives_unaccepted_candidate():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    candidate = SimpleNamespace(
        current_is_feasible=True,
        MHC=10.0,
        CR=3.0,
        decision_score=0.25,
    )
    calls = {"observed": 0, "events": 0}
    solver.pareto_archive = []
    solver._last_transition_meta = {"archive_would_change": True}
    solver._safe_float = lambda value: None if value is None else float(value)

    def observe(solution):
        calls["observed"] += 1
        solver.pareto_archive.append(solution)
        return True

    def record_event(*_args, **_kwargs):
        calls["events"] += 1

    solver._observe_feasible_state = observe
    solver._record_mo_event = record_event

    archived = solver._archive_candidate_before_current_update(candidate, accept=False)

    assert archived is True
    assert calls == {"observed": 1, "events": 1}
    assert solver.pareto_archive == [candidate]
    assert solver._last_transition_meta["archive_would_change"] is True
    assert solver._last_transition_meta["archive_inserted_before_accept"] is True


def test_archive_candidate_before_current_update_skips_non_archive_candidate():
    solver = bimo4.ELP.__new__(bimo4.ELP)
    candidate = SimpleNamespace(current_is_feasible=True)
    calls = {"observed": 0}
    solver._last_transition_meta = {"archive_would_change": False}
    solver._observe_feasible_state = lambda _solution: calls.__setitem__("observed", calls["observed"] + 1)

    archived = solver._archive_candidate_before_current_update(candidate, accept=False)

    assert archived is False
    assert calls["observed"] == 0
    assert solver._last_transition_meta["archive_inserted_before_accept"] is False

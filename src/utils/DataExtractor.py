import copy
import pickle
import sys
import uuid
from typing import Any, Dict, Optional

import gym
import numpy as np
from gym import spaces
from loguru import logger
from matplotlib import patches
from matplotlib import pyplot as plt

import src.utils.FBSUtil as FBSUtil
import src.utils.config as config
from src.utils.FBSModel import FBSModel
from src.utils.FlowMatrixUtil import FlowMatrixUtil

logger.remove()
logger.add(sys.stderr, level="INFO")
plt.rcParams["axes.unicode_minus"] = False

# === 新增：标准测试实例的长宽比（Aspect Ratio）官方映射库 ===
STANDARD_ASPECT_RATIOS = {
    # O 系列 (Meller et al.)
    'O7': 4.0, 'O8': 4.0, 'O9': 5.0,
    'O10': 5.0, 'O12': 5.0,

    # VC 系列 (Van Camp et al.)
    'VC10': 5.0,

    # FO / F / D 系列
    'FO7': 4.0, 'FO8': 4.0, 'FO9': 4.0,
    'FO10': 3.0, 'F10': 3.0, 'FO11': 4.0,
    'D6': 4.0, 'D8': 4.0, 'D10': 4.0, 'D12': 4.0,

    # BA / MB 系列 (Bazaraa / Bozer & Meller)
    # 注：BA原论文为最小边长约束，转化为长宽比时学界常放宽为 4.0 或 5.0
    'BA12': 5.0, 'BA14': 5.0, 'MB12': 4.0,

    # SC 系列 (Liu & Meller)
    'SC30': 5.0, 'SC35': 4.0,

    # Du 系列 (Dunker et al.)
    'Du62': 4.0,

    # TAM 系列 (Tam)
    'TAM20': 5.0, 'TAM30': 5.0,

    # 其他常见算例通用默认约束
    'BME15': 4.0, 'AEG20': 4.0, 'AML4': 4.0,
}


def patch_aspect_ratio(instance_name: str, original_limit: float) -> float:
    """
    智能修正实例的长宽比上限。
    处理逻辑：
    1. 自动解析名称后缀 (如 AB20-ar3 -> 3.0)
    2. 匹配官方字典库
    3. 去除自定义后缀 (如 -maoyan) 后再次匹配
    4. 若原数据缺失(如 99.0)，则提供通用备用值 5.0
    """
    import re
    # 1. 尝试从实例名称中自动提取 (匹配 -ar3, -ar50 等)
    ar_match = re.search(r'-ar(\d+)', instance_name.lower())
    if ar_match:
        return float(ar_match.group(1))

    # 去除可能存在的后缀，提取核心算例名
    base_name = instance_name.replace('-maoyan', '')

    # 2 & 3. 尝试从硬编码的标准字典中匹配
    if base_name in STANDARD_ASPECT_RATIOS:
        return STANDARD_ASPECT_RATIOS[base_name]
    if instance_name in STANDARD_ASPECT_RATIOS:
        return STANDARD_ASPECT_RATIOS[instance_name]

    # 4. 如果原数据是 99.0 且字典里找不到，赋予通用默认值 5.0
    if original_limit == 99.0 or original_limit > 50.0:
        logger.warning(f"实例 {instance_name} 未在 pkl 或字典中找到长宽比，采用默认值 5.0")
        return 5.0

    return original_limit


# ========================================================

class DataProcessingEnv(gym.Env):
    def __init__(self, instance=None, seed=None, options=None):
        super().__init__()
        with open(config.FILE_PATH, "rb") as file:
            (
                self.problems,
                self.FlowMatrices,
                self.sizes,
                self.LayoutWidths,
                self.LayoutLengths,
            ) = pickle.load(file)

        self.instance = instance
        if self.instance not in self.problems or self.instance not in self.FlowMatrices:
            valid_instances = list(self.FlowMatrices.keys())
            raise ValueError(f"Instance '{instance}' not found. Valid instances: {valid_instances}")

        self.uuid = uuid.uuid4()
        raw_F = FlowMatrixUtil.get_raw_flow_matrix(self.FlowMatrices, self.instance)
        # Du62 保持原始物流量矩阵，不做对称补全。
        if self.instance == "Du62":
            self.F = raw_F.copy()
        else:
            self.F = FlowMatrixUtil.symmetrize_if_upper_triangular(raw_F)
        self.n = self.problems[self.instance]
        # self.areas, self.aspect_limits = FBSUtil.getAreaData(self.sizes[self.instance])
        # self.fac_limit_aspect = FBSUtil.get_instance_aspect_limit(self.aspect_limits)
        # 1. 读取原始数据
        self.areas, self.aspect_limits = FBSUtil.getAreaData(self.sizes[self.instance])
        self.fac_limit_aspect = FBSUtil.get_instance_aspect_limit(self.aspect_limits)

        # 2. ==== 调用补丁，强行覆写缺失的长宽比 ====
        self.fac_limit_aspect = patch_aspect_ratio(self.instance, self.fac_limit_aspect)

        # 3. 同步修复数组形式的 aspect_limits（防止内部向量化计算时仍然使用 99.0）
        if isinstance(self.aspect_limits, np.ndarray):
            # 将数组内所有异常值(99.0)替换为正确的修正值
            self.aspect_limits[self.aspect_limits > 50.0] = self.fac_limit_aspect
        else:
            self.aspect_limits = np.full(self.n, self.fac_limit_aspect, dtype=float)

        self.H = self.LayoutWidths[self.instance]
        self.W = self.LayoutLengths[self.instance]

        self.actions = {
            0: "facility_swap",
            1: "bay_flip",
            2: "bay_swap",
            3: "repair",
            4: "idle",
            5: "facility_insert",
            6: "bay_shuffle",
            7: "facility_shuffle",
            8: "ga_action",
            9: "flow_guided_swap",
            10: "segment_insert",
            11: "cross_bay_relocate",
            12: "bay_split_by_flow",
            13: "bay_merge_by_flow",
            14: "adjacent_bay_repartition_by_flow",
            15: "adjacent_bay_block_repartition_by_flow",
        }
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.n * 3,),
            dtype=np.float64,
        )
        self.distance_metric = "manhattan"
        self.true_cost_penalty_scale = 1
        self._clear_runtime_tracking()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _clear_runtime_tracking(self):
        self.fbs_model = None
        self.fac_x = np.zeros(self.n, dtype=float)
        self.fac_y = np.zeros(self.n, dtype=float)
        self.fac_b = np.zeros(self.n, dtype=float)
        self.fac_h = np.zeros(self.n, dtype=float)
        self.fac_aspect_ratio = np.zeros(self.n, dtype=float)
        self.lower_bounds = np.zeros(self.n, dtype=float)
        self.upper_bounds = np.zeros(self.n, dtype=float)
        self.infeasible_mask = np.zeros(self.n, dtype=bool)
        self.D = np.zeros((self.n, self.n), dtype=float)
        self.TM = np.zeros((self.n, self.n), dtype=float)
        self.MHC = np.inf
        self.fitness = np.inf
        self.best_fitness = np.inf
        self.best_MHC = np.inf
        self.previous_fitness = np.inf
        self.previous_MHC = np.inf
        self.previous_d_inf = self.n
        self.current_d_inf = self.n
        self.current_is_feasible = False
        self.current_v_worst = None
        self.feasible_solution_count = 0
        self.best_feasible_cost = np.inf
        self.worst_feasible_cost = None
        self.best_feasible_solution = None
        self.state = np.zeros(self.n * 3, dtype=float)

    def _runtime_v_worst(self):
        if self.worst_feasible_cost is None or not np.isfinite(self.worst_feasible_cost):
            return None
        return float(self.worst_feasible_cost)

    def _register_feasible_solution(self, cost: float, snapshot: Optional[FBSModel] = None):
        if not np.isfinite(cost):
            return
        self.feasible_solution_count += 1
        if snapshot is not None and cost < self.best_feasible_cost:
            self.best_feasible_solution = copy.deepcopy(snapshot)
        if cost < self.best_feasible_cost:
            self.best_feasible_cost = float(cost)
        if self.worst_feasible_cost is None:
            self.worst_feasible_cost = float(cost)
        else:
            self.worst_feasible_cost = max(float(self.worst_feasible_cost), float(cost))
        self.best_fitness = self.best_feasible_cost
        self.current_v_worst = self._runtime_v_worst()

    def _sync_metrics(self, metrics: Dict[str, Any]):
        self.fac_x = metrics["fac_x"]
        self.fac_y = metrics["fac_y"]
        self.fac_b = metrics["fac_b"]
        self.fac_h = metrics["fac_h"]
        self.fac_aspect_ratio = metrics["fac_aspect_ratio"]
        self.lower_bounds = metrics["lower_bounds"]
        self.upper_bounds = metrics["upper_bounds"]
        self.infeasible_mask = metrics["infeasible_mask"]
        self.D = metrics["D"]
        self.TM = metrics["TM"]
        self.MHC = metrics["mhc"]
        self.fitness = metrics["cost"]
        self.current_d_inf = metrics["d_inf"]
        self.current_is_feasible = metrics["is_feasible"]
        self.current_v_worst = metrics.get("v_ref", metrics.get("v_worst"))

    def _evaluate_current_layout(self, snapshot_best: bool = True) -> Dict[str, Any]:
        metrics = FBSUtil.evaluate_layout_true_cost(
            self.fbs_model,
            self.areas,
            self.H,
            self.F,
            self.aspect_limits,
            v_worst=self._runtime_v_worst(),
            true_cost_penalty_scale=self.true_cost_penalty_scale,
            distance_metric=self.distance_metric,
        )
        self._sync_metrics(metrics)
        if metrics["is_feasible"]:
            snapshot = self.fbs_model if snapshot_best else None
            self._register_feasible_solution(metrics["cost"], snapshot=snapshot)
            self.fitness = metrics["cost"]
        else:
            self.current_v_worst = self._runtime_v_worst()
        self.best_MHC = min(self.best_MHC, self.MHC)
        return metrics

    def _make_initial_model(self) -> FBSModel:
        B = FBSUtil.select_B(self.areas, self.n, self.fac_limit_aspect, self.W)
        if B is None:
            B = 2
        genes, permutation = FBSUtil.ZGeneCoding.generate_genes(self.n, B)
        bay_list, bay = FBSUtil.ZGeneCoding.decode_genes(genes, permutation)
        permutation, bay = FBSUtil.arrayToPermutation(bay_list)
        bay[-1] = 1
        return FBSModel(
            permutation.astype(int).tolist(),
            bay.astype(int).tolist(),
            genes=genes.tolist() if isinstance(genes, np.ndarray) else genes,
        )

    def reset(self, seed=None, options=None, fbs_model: FBSModel = None):
        if seed is not None:
            np.random.seed(seed)
        if options is not None and "fbs_model" in options:
            fbs_model = options["fbs_model"]

        self._clear_runtime_tracking()
        self.fbs_model = copy.deepcopy(fbs_model) if fbs_model is not None else self._make_initial_model()
        self._evaluate_current_layout(snapshot_best=True)
        self.previous_fitness = self.fitness
        self.previous_MHC = self.MHC
        self.previous_d_inf = self.current_d_inf
        self.state = self.constructState()
        info = {
            "fitness": self.fitness,
            "mhc": self.MHC,
            "d_inf": self.current_d_inf,
            "is_feasible": self.current_is_feasible,
        }
        return self.state, info

    def _apply_action(self, action_name: str):
        if action_name == "facility_swap":
            new_perm, new_bay = FBSUtil.facility_swap(
                np.array(self.fbs_model.permutation), np.array(self.fbs_model.bay)
            )
        elif action_name == "bay_flip":
            new_perm, new_bay = FBSUtil.bay_flip(
                np.array(self.fbs_model.permutation), np.array(self.fbs_model.bay)
            )
        elif action_name == "bay_swap":
            new_perm, new_bay = FBSUtil.bay_swap(
                np.array(self.fbs_model.permutation), np.array(self.fbs_model.bay)
            )
        elif action_name == "repair":
            new_perm, new_bay = FBSUtil.repair(
                np.array(self.fbs_model.permutation),
                np.array(self.fbs_model.bay),
                self.fac_b,
                self.fac_h,
                self.fac_limit_aspect,
            )
        elif action_name == "facility_insert":
            new_perm, new_bay = FBSUtil.facility_insert(
                np.array(self.fbs_model.permutation), np.array(self.fbs_model.bay)
            )
        elif action_name == "bay_shuffle":
            new_perm, new_bay = FBSUtil.bay_shuffle(
                self.fbs_model.permutation, self.fbs_model.bay
            )
        elif action_name == "facility_shuffle":
            new_perm, new_bay = FBSUtil.facility_shuffle(
                self.fbs_model.permutation, self.fbs_model.bay
            )
        elif action_name == "flow_guided_swap":
            new_perm, new_bay = FBSUtil.flow_guided_swap(
                np.array(self.fbs_model.permutation),
                np.array(self.fbs_model.bay),
                self.areas,
                self.H,
                self.F,
                self.aspect_limits,
                v_worst=self._runtime_v_worst(),
                k_penalty=self.true_cost_penalty_scale,
                distance_metric=self.distance_metric,
            )
        elif action_name == "segment_insert":
            new_perm, new_bay = FBSUtil.segment_insert(
                np.array(self.fbs_model.permutation),
                np.array(self.fbs_model.bay),
                self.areas,
                self.H,
                self.F,
                self.aspect_limits,
                v_worst=self._runtime_v_worst(),
                k_penalty=self.true_cost_penalty_scale,
                distance_metric=self.distance_metric,
            )
        elif action_name == "cross_bay_relocate":
            new_perm, new_bay = FBSUtil.cross_bay_relocate(
                np.array(self.fbs_model.permutation),
                np.array(self.fbs_model.bay),
                self.areas,
                self.H,
                self.F,
                self.aspect_limits,
                v_worst=self._runtime_v_worst(),
                k_penalty=self.true_cost_penalty_scale,
                distance_metric=self.distance_metric,
            )
        elif action_name == "bay_split_by_flow":
            new_perm, new_bay = FBSUtil.bay_split_by_flow(
                np.array(self.fbs_model.permutation),
                np.array(self.fbs_model.bay),
                self.areas,
                self.H,
                self.F,
                self.aspect_limits,
                v_worst=self._runtime_v_worst(),
                k_penalty=self.true_cost_penalty_scale,
                distance_metric=self.distance_metric,
            )
        elif action_name == "bay_merge_by_flow":
            new_perm, new_bay = FBSUtil.bay_merge_by_flow(
                np.array(self.fbs_model.permutation),
                np.array(self.fbs_model.bay),
                self.areas,
                self.H,
                self.F,
                self.aspect_limits,
                v_worst=self._runtime_v_worst(),
                k_penalty=self.true_cost_penalty_scale,
                distance_metric=self.distance_metric,
            )
        elif action_name == "adjacent_bay_repartition_by_flow":
            new_perm, new_bay = FBSUtil.adjacent_bay_repartition_by_flow(
                np.array(self.fbs_model.permutation),
                np.array(self.fbs_model.bay),
                self.areas,
                self.H,
                self.F,
                self.aspect_limits,
                v_worst=self._runtime_v_worst(),
                k_penalty=self.true_cost_penalty_scale,
                distance_metric=self.distance_metric,
            )
        elif action_name == "adjacent_bay_block_repartition_by_flow":
            new_perm, new_bay = FBSUtil.adjacent_bay_block_repartition_by_flow(
                np.array(self.fbs_model.permutation),
                np.array(self.fbs_model.bay),
                self.areas,
                self.H,
                self.F,
                self.aspect_limits,
                v_worst=self._runtime_v_worst(),
                k_penalty=self.true_cost_penalty_scale,
                distance_metric=self.distance_metric,
            )
        elif action_name == "ga_action":
            new_perm, new_bay = FBSUtil.ga_population_action(
                self.fbs_model,
                self,
                pop_size=10,
                generations=30,
            )
        elif action_name == "idle":
            return
        else:
            raise ValueError(f"Invalid action: {action_name}")

        self.fbs_model.permutation = (
            new_perm.tolist() if isinstance(new_perm, np.ndarray) else list(new_perm)
        )
        self.fbs_model.bay = (
            new_bay.tolist() if isinstance(new_bay, np.ndarray) else list(new_bay)
        )

    def calculate_reward(self, previous_cost, previous_d_inf, previous_best_feasible):
        reward = 0.0
        scale = max(self._runtime_v_worst() or 0.0, 1.0)
        if np.isfinite(previous_cost) and np.isfinite(self.fitness):
            reward += (previous_cost - self.fitness) / scale
        reward += 0.25 * (previous_d_inf - self.current_d_inf)
        if previous_d_inf > 0 and self.current_d_inf == 0:
            reward += 0.5
        if self.current_is_feasible and self.fitness < previous_best_feasible:
            reward += 1.0
        return float(np.clip(reward, -2.5, 2.5))

    def _build_info(self, reward: float, action_name: str) -> Dict[str, Any]:
        return {
            "TimeLimit.truncated": False,
            "current_fitness": self.fitness,
            "previous_fitness": self.previous_fitness,
            "reward": reward,
            "facility_count": self.n,
            "action_taken": action_name,
            "layout_dimensions": (self.H, self.W),
            "instance": self.instance,
            "mhc": self.MHC,
            "d_inf": self.current_d_inf,
            "is_feasible": self.current_is_feasible,
            "v_worst": self._runtime_v_worst(),
            "best_feasible_cost": self.best_feasible_cost,
            "worst_feasible_cost": self.worst_feasible_cost,
        }

    def step(self, action):
        action_name = self.actions[int(action)]
        previous_cost = self.fitness
        previous_best_feasible = self.best_feasible_cost
        previous_d_inf = self.current_d_inf
        previous_mhc = self.MHC

        self._apply_action(action_name)
        self._evaluate_current_layout(snapshot_best=True)
        self.state = self.constructState()
        reward = self.calculate_reward(previous_cost, previous_d_inf, previous_best_feasible)
        self.previous_fitness = previous_cost
        self.previous_MHC = previous_mhc
        self.previous_d_inf = previous_d_inf
        info = self._build_info(reward=reward, action_name=action_name)
        return self.state, reward, False, False, info

    def step2(self, action):
        return self.step(action)

    def render(self):
        fig, ax = plt.subplots()
        ax.set_title("Facility layout")
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")
        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)
        plt.grid(False)
        plt.gca().set_aspect("equal", adjustable="box")

        state_reshaped = self.state.reshape(self.n, 3)
        for facility_label in self.fbs_model.permutation:
            facility_idx = int(facility_label) - 1
            x_from = self.fac_x[facility_idx] - self.fac_b[facility_idx] / 2
            y_from = self.fac_y[facility_idx] - self.fac_h[facility_idx] / 2
            line_color = (
                "red"
                if self.fac_aspect_ratio[facility_idx] > self.aspect_limits[facility_idx]
                else "green"
            )
            face_color = (
                state_reshaped[facility_idx, 0] / 255.0,
                state_reshaped[facility_idx, 1] / 255.0,
                state_reshaped[facility_idx, 2] / 255.0,
                0.7,
            )
            rect = patches.Rectangle(
                (x_from, y_from),
                width=self.fac_b[facility_idx],
                height=self.fac_h[facility_idx],
                edgecolor=line_color,
                facecolor=face_color,
                linewidth=1,
            )
            ax.add_patch(rect)
            ax.text(
                x_from + self.fac_b[facility_idx] / 2,
                y_from + self.fac_h[facility_idx] / 2,
                f"{int(facility_label)}",
                ha="center",
                va="center",
                color="white" if np.mean(face_color[:3]) < 0.5 else "black",
            )

        plt.figtext(0.5, 0.93, f"MHC: {self.MHC:.2f}", ha="center", fontsize=12)
        plt.figtext(0.5, 0.96, f"Cost: {self.fitness:.2f}", ha="center", fontsize=12)
        plt.show()

    def constructState(self):
        state = np.zeros((self.n, 3), dtype=np.float64)
        permutation = np.asarray(self.fbs_model.permutation, dtype=float)
        tm = np.asarray(self.TM, dtype=float)
        sources = np.sum(tm, axis=1)
        sinks = np.sum(tm, axis=0)

        def scale_to_255(values):
            values = np.asarray(values, dtype=float)
            if values.size == 0:
                return values
            value_min = np.min(values)
            value_max = np.max(values)
            if np.isclose(value_min, value_max):
                return np.zeros_like(values, dtype=np.float64)
            return ((values - value_min) / (value_max - value_min) * 255.0).astype(np.float64)

        state[:, 0] = scale_to_255(permutation)
        state[:, 1] = scale_to_255(sources)
        state[:, 2] = scale_to_255(sinks)
        return state.flatten()




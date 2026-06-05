import itertools
import math
import random
import gym
import numpy as np
import re
from itertools import permutations, product
import logging
import colorlog
from functools import wraps
import copy
from loguru import logger
from typing import Tuple, List
import pandas as pd
from src.utils.FBSModel import FBSModel


# 遗传算法中的变异和交叉操作
class FBSUtils:
    # 变异操作类
    class MutateActions:

        @staticmethod
        def facility_swap(fbs_model: FBSModel):
            logging.info("执行设施交换")
            ## 随机选择两个不同的位置进行交换
            # size = len(fbs_model.permutation)
            # pos1, pos2 = sorted(np.random.choice(size, 2, replace=False))
            # logging.info(f"facility_swap-->pos1: {pos1}, pos2: {pos2}")

            ## 复制一份当前排列
            # new_perm = fbs_model.permutation.copy() if isinstance(fbs_model.permutation, list) else fbs_model.permutation.tolist()

            ## 交换两个位置的设施
            # new_perm[pos1], new_perm[pos2] = new_perm[pos2], new_perm[pos1]

            # # 更新模型的排列
            # fbs_model.permutation = new_perm
            pass

        @staticmethod
        def bay_flip(fbs_model: FBSModel):
            logging.info("执行区带反转")
            pass

    class CrossoverActions:

        @staticmethod
        def order_crossover(
                parent1: FBSModel, parent2: FBSModel
        ) -> tuple[FBSModel, FBSModel]:
            parent1_perm = parent1.permutation
            parent2_perm = parent2.permutation
            parent1_bay = parent1.bay
            parent2_bay = parent2.bay
            # 类型转换
            if isinstance(parent1_perm, np.ndarray):
                parent1_perm = parent1_perm.tolist()
            if isinstance(parent2_perm, np.ndarray):
                parent2_perm = parent2_perm.tolist()
            if isinstance(parent1_bay, np.ndarray):
                parent1_bay = parent1_bay.tolist()
            if isinstance(parent2_bay, np.ndarray):
                parent2_bay = parent2_bay.tolist()
            size = len(parent1_perm)
            startPoint, endPoint = sorted(np.random.choice(size, 2, replace=False))
            logging.info(
                f"order_crossover-->startPoint: {startPoint}, endPoint: {endPoint}"
            )
            crossover_part_1 = parent1_perm[startPoint: endPoint + 1]
            crossover_part_2 = parent2_perm[startPoint: endPoint + 1]
            # 获取 parent1 中去除 crossover_part_2 的部分
            parent1_remaining = [
                elem for elem in parent1_perm if elem not in crossover_part_2
            ]
            # 获取 parent2 中去除 crossover_part_1 的部分
            parent2_remaining = [
                elem for elem in parent2_perm if elem not in crossover_part_1
            ]
            offspring_1_perm = (
                    parent1_remaining[:startPoint]
                    + crossover_part_2
                    + parent1_remaining[startPoint:]
            )
            offspring_2_perm = (
                    parent2_remaining[:startPoint]
                    + crossover_part_1
                    + parent2_remaining[startPoint:]
            )
            offspring_1_bay = (
                    parent1_bay[:startPoint]
                    + parent2_bay[startPoint: endPoint + 1]
                    + parent1_bay[endPoint + 1:]
            )
            offspring_2_bay = (
                    parent2_bay[:startPoint]
                    + parent1_bay[startPoint: endPoint + 1]
                    + parent2_bay[endPoint + 1:]
            )
            offspring_1 = FBSModel(offspring_1_perm, offspring_1_bay)
            offspring_2 = FBSModel(offspring_2_perm, offspring_2_bay)
            return offspring_1, offspring_2

# 物流强度矩阵转换
def transfer_matrix(matrix: np.ndarray):
    """
    转置矩阵
    :param matrix: 矩阵
    :return: 转置后的矩阵
    """
    print("转换前: ", matrix)
    LowerTriangular = np.tril(matrix, -1).T
    resultMatrix = LowerTriangular + matrix
    resultMatrix = np.triu(resultMatrix)
    print("转换后: ", resultMatrix)
    return resultMatrix

def getAreaData(df) -> Tuple[np.ndarray, float]:
    """
    从 DataFrame 中按列位置提取面积和长宽比数据（跳过标题行）。
    要求数据格式为两列：第一列面积，第二列长宽比。
    参数:
        df (pd.DataFrame): 输入的 DataFrame，需确保已跳过标题行。

    返回:
        Tuple[np.ndarray, float]: 面积数组和第一个长宽比值（若无数据则返回99）
    """

    # 获取包含特定关键词的列并转换为一维数组
    def get_column_data(df, pattern):
        cols = df.filter(regex=re.compile(pattern, re.IGNORECASE)).columns
        return df[cols].to_numpy().flatten() if not cols.empty else None

    areas = get_column_data(df, "Area")
    aspects = get_column_data(df, "Aspect")
    aspects = aspects[0] if aspects is not None else 99
    return areas, aspects


def check_constraints(facilities, W, H, ar_i=None, l_min=None):  # 未返check
    """
    检查所有设施是否满足UA-FLP的约束条件（公式2-8）

    参数：
    facilities: 设施列表，每个设施是包含'x', 'y', 'w', 'h'的字典
    W: 平面区域的宽度
    H: 平面区域的高度
    ar_i: 最大宽高比约束（可选，实例级别）
    l_min: 最小边长约束（可选，实例级别）

    返回：
    bool: 所有约束满足时返回True，否则返回False
    """

    # 公式2: 总面积约束 ΣA_i ≤ W*H
    total_area = sum(f['w'] * f['h'] for f in facilities)
    if total_area > W * H:
        return False

    # 遍历所有设施检查个体约束
    for f in facilities:
        w, h, x, y = f['w'], f['h'], f['x'], f['y']

        # 公式3: 宽高比约束（若ar_i激活）
        if ar_i is not None:
            if w == 0 or h == 0:
                return False  # 防止除零错误
            aspect_ratio = max(w / h, h / w)
            if aspect_ratio > ar_i:
                return False

        # 公式4: 最小边长约束（若l_min激活）
        if l_min is not None:
            if min(w, h) < l_min:
                return False

        # 公式8: 设施必须在平面区域内
        if (x < 0) or (x + w > W) or (y < 0) or (y + h > H):
            return False

    # 公式5-7: 设施间不重叠约束
    for i in range(len(facilities)):
        for j in range(i + 1, len(facilities)):
            f1, f2 = facilities[i], facilities[j]

            # 计算设施边界
            f1_left = f1['x']
            f1_right = f1['x'] + f1['w']
            f1_bottom = f1['y']
            f1_top = f1['y'] + f1['h']

            f2_left = f2['x']
            f2_right = f2['x'] + f2['w']
            f2_bottom = f2['y']
            f2_top = f2['y'] + f2['h']

            # 检查x和y方向是否同时重叠
            x_overlap = (f1_left < f2_right) and (f1_right > f2_left)
            y_overlap = (f1_bottom < f2_top) and (f1_top > f2_bottom)

            if x_overlap and y_overlap:
                return False  # 存在重叠

    return True  # 所有约束均满足


def select_B(area, n, beta, L):
    """
    根据宽高比约束选择合适的区带总数B

    参数:
        area: 设施面积数组
        n: 设施数量
        beta: 宽高比上限
        L: 厂房长度

    返回:
        选择的区带总数B，如果没有可行解则返回None
    """
    # 存储可行的B值
    feasible_Bs = []

    # B的初始值为2，最大值为n
    B = 2
    while B <= n:
        # 计算每个区带的理论长度
        zone_length = L / B

        # 计算每个设施的宽度和宽高比
        width = area / zone_length  # 宽度 = 面积 / 长度
        aspect_ratio = np.maximum(width, zone_length) / np.minimum(width, zone_length)

        # 验证当前B值是否可行
        if beta is not None:
            # 统计宽高比在[1, beta]范围内的设施数量
            qualified_count = np.sum((aspect_ratio >= 1) & (aspect_ratio <= beta))
        else:
            # 如果没有指定beta，验证宽度和长度是否都大于1
            qualified_count = np.sum((width > 1) & (zone_length > 1))

        # 可行性条件：合格设施数量不少于总设施数的3/4
        if qualified_count >= n * 3 / 4:
            feasible_Bs.append(B)

        # 尝试下一个可能的B值
        B += 1

    # 如果有可行的B值，从中随机选择一个
    if feasible_Bs:
        # 可以根据需要调整选择策略，这里随机选择
        selected_B = random.choice(feasible_Bs)
        # print(f"从可行的区带总数 {feasible_Bs} 中选择了 B = {selected_B}")
        return selected_B
    else:
        print("没有找到满足约束条件的区带总数B")
        return None

class ZGeneCoding:
    """新增：基于实数z_i的区带编码方式实现"""

    @staticmethod
    def generate_genes(n: int, B: int) -> tuple[np.ndarray, np.ndarray]:
        """
        生成n个基因的染色体，每个基因z_i ∈ (1, B+1)
        并确保区带编号连续
        :param n: 设施数量
        :param B: 区带总数
        :return:
            - 基因数组(z_1, z_2, ..., z_n)
            - 设施编号序列（元素为Python整数）
        """
        if B < 1:
            raise ValueError("区带总数B必须至少为1")

        # 生成(1, B+1)区间内的随机实数（基因数组）
        genes = np.random.uniform(low=1.0001, high=B + 0.9999, size=n)
        # print(genes)

        # 提取整数部分（区带编号）
        zone_ids = np.floor(genes).astype(int)

        # 获取唯一的区带编号并排序
        unique_zones = np.unique(zone_ids)
        sorted_zones = np.sort(unique_zones)

        # 创建区带编号映射：将实际出现的区带编号映射为连续编号
        zone_mapping = {zone: i + 1 for i, zone in enumerate(sorted_zones)}

        # 统计实际区带数量
        actual_B = len(unique_zones)

        # 应用映射：调整整数部分使其连续
        for i in range(len(genes)):
            original_zone = zone_ids[i]
            new_zone = zone_mapping[original_zone]
            # 保留小数部分，只替换整数部分
            genes[i] = new_zone + (genes[i] - original_zone)

        # 生成设施编号序列，确保元素为Python整数
        permutation = np.arange(1, n + 1)
        permutation = permutation.astype(object)
        permutation[:] = [int(x) for x in permutation]
        # 输出统计信息
        # print(f"生成的区带信息:")
        # print(f" - 理论最大区带数: {B}")
        # print(f" - 实际生成的区带数: {actual_B}")
        # print(f" - 区带编号映射: {zone_mapping}")

        return genes, permutation

    @staticmethod
    def decode_genes(genes: np.ndarray, permutation: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
        """
        将基因解码为区带结构和bay数组
        :param genes: 基因数组(z_1, z_2, ..., z_n)
        :param permutation: 设施编号序列(与基因顺序对应)
        :return:
            - 区带列表：每个子列表为对应区带的设施（按摆放顺序）
            - bay数组：指示区带分界的0-1数组
        """
        if isinstance(genes, list):
            genes = np.array(genes)
        if isinstance(permutation, list):
            permutation = np.array(permutation)
        if len(genes) != len(permutation):
            raise ValueError("基因长度与设施数量必须一致")

        # 1. 提取每个基因的整数部分(区带编号)和小数部分(摆放次序)
        zone_ids = np.floor(genes).astype(int)  # 区带编号(1-based)
        order_values = genes - zone_ids  # 小数部分(决定同一区带内的顺序)

        # 2. 按区带分组，并按小数部分排序(升序=放在下方)
        unique_zones = np.unique(zone_ids)
        zone_facilities = {}

        for zone in unique_zones:
            # 获取该区带内的设施索引
            mask = (zone_ids == zone)
            zone_perm = permutation[mask]
            zone_orders = order_values[mask]

            # 按小数部分升序排序(确保数字小的放下面)
            sorted_indices = np.argsort(zone_orders)

            # 关键修复：将numpy数组元素转换为Python整数
            sorted_facilities = zone_perm[sorted_indices].tolist()
            # 确保每个元素都是Python int类型，而非numpy数值类型
            sorted_facilities = [int(facility) for facility in sorted_facilities]

            zone_facilities[zone] = sorted_facilities

        # 3. 按区带编号升序排列，构建最终区带列表
        sorted_zones = sorted(unique_zones)
        bay_list = [zone_facilities[zone] for zone in sorted_zones]

        # 4. 生成bay数组(1表示区带结束位置)
        bay = np.zeros(len(permutation), dtype=int)
        current_idx = 0
        for zone in bay_list:
            current_idx += len(zone)
            if current_idx < len(permutation):
                bay[current_idx - 1] = 1  # 区带结束位置标记为1
        bay[-1] = 1  # 最后一个设施必为区带结束

        return bay_list, bay


# 辅助函数：将区带列表转换为排列和bay数组(兼容现有方法)
def arrayToPermutation(bay_list: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """将区带列表转换为设施排列和bay数组"""
    permutation = []
    bay = []
    for zone in bay_list:
        permutation.extend(zone)
        # 区带内除最后一个设施外，bay标记为0
        bay.extend([0] * (len(zone) - 1))
        # 区带最后一个设施标记为1(最后一个区带的1会被覆盖，最终统一处理)
        bay.append(1)
    # 确保最后一个元素为1
    if bay:
        bay[-1] = 1
    return np.array(permutation), np.array(bay)


# 随机解生成器
def random_solution_generator(n: int) -> tuple[list[int], list[int]]:
    """生成随机解"""
    # 生成随机排列
    permutation = np.arange(1, n + 1)
    np.random.shuffle(permutation)
    # 生成随机的0-1序列
    bay = np.random.randint(0, 2, n)
    # 确保最后一个元素为1
    bay[-1] = 1
    return (permutation, bay)


# k分初始解生成器(输入：面积数据a，设施数n，横纵比限制beta，厂房x轴长度L)
def binary_solution_generator(area, n, beta, L):
    # 存储可行的k分解
    bay_list = []
    # 分界参数
    k = 2
    # 计算面积之和
    total_area = np.sum(area)
    # print("总面积: ", total_area)
    # 生成一个设施默认的编号序列
    permutation = np.arange(1, n + 1)
    # 根据area对序列进行排序
    permutation = permutation[np.argsort(area[permutation - 1])]
    # 对a也进行排序
    area = np.sort(area)
    while k <= n:
        # 计算W的k分
        l = L / k
        w = area / l  # 每个设施的宽度
        aspect_ratio = np.maximum(w, l) / np.minimum(w, l)
        # 验证k分是否可行
        # print("a/l", a / l)
        # 合格个数
        if beta is not None:
            qualified_number = np.sum((aspect_ratio >= 1) & (aspect_ratio <= beta))
        else:
            qualified_number = np.sum((w > 1) & (l > 1))
        # 如果合格个数大于等于3/4*n，即此k值可行
        bay = np.zeros(n)
        if qualified_number >= n * 3 / 4:
            # print("可行的k: ", k)
            # print("符合的个数: ", qualified_number)
            # 根据面积和找到k分界点
            best_partition, partitions = _find_best_partition(area, k)
            # print("序列分界点: ", best_partition)
            # 将k分界点转换为bay
            for i, p in enumerate(best_partition):
                bay[p - 1] = 1
            # 将最后一个分界点设为1
            bay[n - 1] = 1
            bay_list.append(bay)
        k += 1
    print("可行的bay: ", bay_list)
    # 从可行的bay中随机选择一个
    if len(bay_list) > 0:
        bay = random.choice(bay_list)
    #  TODO 对permutation使用bay进行划分，并对每个bay中的设施进行随机排列
    j = 0
    for i in range(len(bay)):
        if bay[i] == 1:
            np.random.shuffle(permutation[j:i])
            j = i + 1
    return (permutation, bay)

# 动态划分bay的大小
# k分划分法的动态规划版（输入：排列序列a，划分数k）
def _find_best_partition(arr, k):
    # print(f"k分划分法-->k = {k}")
    n = len(arr)
    target_sum = np.sum(arr) // k

    # dp[i][j] 表示前i个设施被划分为j个组的最小差异和
    dp = np.full((n + 1, k + 1), float("inf"))
    dp[0][0] = 0

    # sum[i] 表示arr[0:i]的累积和
    cum_sum = np.cumsum(arr)

    partition_idx = [[[] for _ in range(k + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, k + 1):
            for m in range(i):
                current_sum = cum_sum[i - 1] - (cum_sum[m - 1] if m > 0 else 0)
                current_diff = abs(target_sum - current_sum)
                total_diff = dp[m][j - 1] + current_diff

                if total_diff < dp[i][j]:
                    dp[i][j] = total_diff
                    partition_idx[i][j] = partition_idx[m][j - 1] + [i]

    best_partition = partition_idx[-1][-1][:-1]  # 排除最后一个分界点
    return best_partition, np.split(arr, best_partition)

# 计算设施坐标和尺寸
def getCoordinates_mao(fbs_model: FBSModel, area, H):
    permutation = fbs_model.permutation
    bay = fbs_model.bay
    bays = permutationToArray(
        permutation, bay
    )  # 将排列按照划分点分割成多个子数组，每个子数组代表一个区段的排列
    # 初始化长度、宽度和坐标数组
    n = len(permutation)
    lengths = np.zeros(n)  # 每个设施的长度
    widths = np.zeros(n)  # 每个设施的宽度
    fac_x = np.zeros(n)  # 每个设施的x坐标
    fac_y = np.zeros(n)  # 每个设施的y坐标
    # 计算每个区带的坐标和尺寸
    x = 0
    start = 0  # 记录当前子数组的起始索引
    # 从上向下排列
    for b in bays:
        indices = np.array(b) - 1
        bay_areas = area[indices]
        # 计算每个设施的长度和宽度
        widths[start : start + len(bay_areas)] = np.sum(bay_areas) / H
        lengths[start : start + len(bay_areas)] = (
            bay_areas / widths[start : start + len(bay_areas)]
        )
        # 计算设施的x坐标
        fac_x[start : start + len(bay_areas)] = (
            widths[start : start + len(bay_areas)] * 0.5 + x
        )
        x += np.sum(bay_areas) / H
        # 计算设施的y坐标
        y = (
            np.cumsum(lengths[start : start + len(bay_areas)])
            - lengths[start : start + len(bay_areas)] * 0.5
        )
        fac_y[start : start + len(bay_areas)] = y
        start += len(bay_areas)
    # 顺序恢复
    order = np.argsort(permutation)
    fac_x = fac_x[order]
    fac_y = fac_y[order]
    lengths = lengths[order]
    widths = widths[order]
    return fac_x, fac_y, lengths, widths


def getCoordinates_mao_fast(fbs_model: FBSModel, area, H):
    permutation = np.asarray(fbs_model.permutation, dtype=int)
    bay = np.asarray(fbs_model.bay, dtype=int)
    n = permutation.size
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty
    if bay.size != n:
        zeros = np.zeros(n, dtype=float)
        return zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()

    bay_flags = bay.copy()
    bay_flags[-1] = 1
    bay_end_idx = np.flatnonzero(bay_flags == 1)
    bay_start_idx = np.concatenate(([0], bay_end_idx[:-1] + 1))
    bay_lengths = bay_end_idx - bay_start_idx + 1

    area_array = np.asarray(area, dtype=float)
    areas_in_perm_order = area_array[permutation - 1]
    bay_area_sums = np.add.reduceat(areas_in_perm_order, bay_start_idx)
    bay_widths = bay_area_sums / H

    widths_in_perm_order = np.repeat(bay_widths, bay_lengths)
    lengths_in_perm_order = areas_in_perm_order / widths_in_perm_order

    bay_x_offsets = np.concatenate(([0.0], np.cumsum(bay_widths[:-1], dtype=float)))
    fac_x_in_perm_order = np.repeat(bay_x_offsets + bay_widths * 0.5, bay_lengths)

    cumulative_lengths = np.cumsum(lengths_in_perm_order, dtype=float)
    bay_prefix_lengths = cumulative_lengths[bay_start_idx] - lengths_in_perm_order[bay_start_idx]
    fac_y_in_perm_order = (
        cumulative_lengths
        - lengths_in_perm_order * 0.5
        - np.repeat(bay_prefix_lengths, bay_lengths)
    )

    order = np.argsort(permutation)
    fac_x = fac_x_in_perm_order[order]
    fac_y = fac_y_in_perm_order[order]
    lengths = lengths_in_perm_order[order]
    widths = widths_in_perm_order[order]
    return fac_x, fac_y, lengths, widths


# 计算欧几里得距离矩阵
def getEuclideanDistances(x, y):
    """计算欧几里得距离矩阵
    Args:
        x (np.ndarray): 设施x坐标
        y (np.ndarray): 设施y坐标
    Returns:
        np.ndarray: 距离矩阵
    """
    return np.sqrt(
        np.array(
            [
                [(x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 for j in range(len(x))]
                for i in range(len(x))
            ]
        )
    )


def getManhattanDistances(x, y):
    """计算曼哈顿距离矩阵
    Args:
        x (np.ndarray): 设施x坐标
        y (np.ndarray): 设施y坐标
    Returns:
        np.ndarray: 曼哈顿距离矩阵
    Raises:
        ValueError: 如果输入不是数组或长度不匹配
    """
    # 输入验证
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("x 和 y 必须是 NumPy 数组")
    if len(x) != len(y):
        raise ValueError("x 和 y 的长度必须相同")
    if len(x) == 0:
        return np.array([], dtype=float)

    # 转换为 NumPy 数组（如果尚未是）
    x = np.asarray(x)
    y = np.asarray(y)

    # 使用向量化操作计算曼哈顿距离
    x_diff = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
    y_diff = np.abs(y[:, np.newaxis] - y[np.newaxis, :])
    return x_diff + y_diff


def permutationMatrix(a):
    P = np.zeros((len(a), len(a)))
    for idx, val in enumerate(a):
        logging.debug(f"idx: {idx}, val: {val}")
        P[idx][val - 1] = 1
    return P


def getTransportIntensity(D, F):
    # logger.info("计算物流强度矩阵")
    # logger.info(f"D: \n{D}")
    # logger.info(f"F: \n{F}")
    return D * F


# 计算MHC
def getMHC(D, F, fbs_model: FBSModel):
    # permutation = fbs_model.permutation
    # P = permutationMatrix(permutation)
    # logger.info(f"P: \n{P}")
    # MHC = np.sum(np.tril(np.dot(P.T, np.dot(D, P))) * (F.T))
    # MHC = np.sum(np.triu(D) * (F))
    MHC = np.sum(D * F)
    # transport_intensity = np.dot(np.dot(D, P), np.dot(F, P.T))
    # MHC = np.trace(transport_intensity)
    return MHC


# 计算适应度
import numpy as np

# V_worst=MHC
def getFitness(mhc, fac_b, fac_h, fac_limit_aspect=None, k=1): #k值在论文中是1
    """
    计算适应度。

    参数:
    mhc: float, MHC 的值
    fac_b: list or np.ndarray, 设施的宽度
    fac_h: list or np.ndarray, 设施的高度
    fac_limit_aspect: float or None, 宽高比的限制值，若为 None 则不限制宽高比
    k: int, 惩罚项的指数，默认为 3

    返回:
    fitness: float, 适应度值
    """
    # 将输入转换为 NumPy 数组
    fac_b = np.array(fac_b)
    fac_h = np.array(fac_h)
    MHC = mhc

    if fac_limit_aspect is None:
        # 检查宽度和高度是否都 >= 1
        non_feasible = (fac_b < 1) | (fac_h < 1)
    else:
        # 计算宽高比
        aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
        # 检查宽高比是否在 1 到 fac_limit_aspect 之间
        non_feasible = (aspect_ratio < 1) | (aspect_ratio > fac_limit_aspect)

    # 计算不可行设施的数量
    non_feasible_counter = np.sum(non_feasible)
    # 计算适应度
    fitness = MHC + MHC * (non_feasible_counter**k) 
    return fitness

# V_worst=当前最优解与已知最优解之间的差
def getFitness2(mhc, fac_b, fac_h, V_worst_multiplier, fac_limit_aspect=None, k=3):
    """
    计算适应度，使用 V_worst_multiplier作为惩罚基数。

    参数:
    mhc: float, MHC 的值
    fac_b: list or np.ndarray, 设施的宽度
    fac_h: list or np.ndarray, 设施的高度
    V_worst_multiplier: float, 惩罚乘数 (根据图像定义，这是 '当前最优解与已知最优解之间的差')
    fac_limit_aspect: float or None, 宽高比的限制值
    k: int, 惩罚项的指数 (根据图像设为 1-3, 此处设为 1)

    返回:
    fitness: float, 适应度值
    """
    # 将输入转换为 NumPy 数组
    fac_b = np.array(fac_b)
    fac_h = np.array(fac_h)
    MHC = mhc

    if fac_limit_aspect is None:
        non_feasible = (fac_b < 1) | (fac_h < 1)
    else:
        aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
        non_feasible = (aspect_ratio < 1) | (aspect_ratio > fac_limit_aspect)

    non_feasible_counter = np.sum(non_feasible) # 这对应图像中的 Dinf

    # 严格按照图像公式: Cost = MHC + (D_inf)^k * V_worst
    # V_worst_multiplier 在这里是 V_worst 的值 (即 '差值')

    # 如果“差值”为 0 或负数 (例如, 刚开始或没有BKV)，会导致惩罚失效或变为奖励。
    # 增加一个保护，确保惩罚基数至少是一个正值 (例如, 使用 MHC)。
    # 否则，一个不可行的解可能比一个可行的解得分还高。
    if V_worst_multiplier <= 0 or not np.isfinite(V_worst_multiplier):
        # 回退策略：使用 MHC 作为基数，避免负惩罚
        penalty_base = MHC
    else:
        # 确保这里使用的是 V_worst_multiplier
        penalty_base = V_worst_multiplier

    fitness = MHC + penalty_base * (non_feasible_counter**k)
    return fitness



def StatusUpdatingDevice(fbs_model: FBSModel, a, H, F, fac_limit_aspect_ratio):
    fac_x, fac_y, fac_b, fac_h = getCoordinates_mao(fbs_model, a, H)
    fac_aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
    D = getManhattanDistances(fac_x, fac_y)  # 曼哈顿距离
    # D = getEuclideanDistances(fac_x, fac_y) # 欧几里得距离
    TM = getTransportIntensity(D, F)
    mhc = getMHC(D, F, fbs_model)
    fitness = getFitness(mhc, fac_b, fac_h, fac_limit_aspect_ratio)
    return (fac_x, fac_y, fac_b, fac_h, fac_aspect_ratio, D, TM, mhc, fitness)

# getFitness2
def StatusUpdatingDevice2(
    fbs_model: FBSModel,
    a,
    H,
    F,
    fac_limit_aspect_ratio,
    g_best=0,
    penalty_k=3,
):
    fac_x, fac_y, fac_b, fac_h = getCoordinates_mao(fbs_model, a, H)
    g_best=5396.6
    # g_best=4720.35
    fac_aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
    D = getManhattanDistances(fac_x, fac_y)
    TM = getTransportIntensity(D, F)
    mhc = getMHC(D, F, fbs_model)
    # 将 V_worst (此时是 '差值') 传递给 getFitness
    V_worst=mhc-g_best
    # print(f"mhc: {mhc}")
    # print(f"g_best: {g_best}")
    # print(f"V_worst: {V_worst}")
    fitness = getFitness2(
        mhc, fac_b, fac_h, V_worst, fac_limit_aspect_ratio, k=penalty_k
    )
    
    return (fac_x, fac_y, fac_b, fac_h, fac_aspect_ratio, D, TM, mhc, fitness)

def encode_genes_from_solution(permutation, bay):
    """
    【【【 新版本：增加了随机性 】】】
    执行 ZGeneCoding 的逆向操作。
    读取一个 (permutation, bay) 结构，并生成一个
    保证可以解码回该结构的、全新的、且包含随机性的 genes 数组。
    """
    # 1. 将 (perm, bay) 转换为二维列表
    bay_list = permutationToArray(permutation, bay)
    
    n = len(permutation)
    genes = np.zeros(n)
    
    # 映射表 {设施标签: 基因索引}
    perm_map = {label: i for i, label in enumerate(permutation)}
    
    zone_id = 1
    for zone in bay_list:
        # 'zone' 是一个有序列表, 例如 [11, 15, 12]
        
        # --- 【【【 新增随机性 】】】 ---
        
        # 1. 获取当前区带的设施数量
        zone_size = len(zone)
        
        # 2. 生成 'zone_size' 个 (0.01, 0.99) 之间的随机小数
        #    (我们避免 0.0 和 1.0 以防浮点数边界问题)
        random_keys = np.random.uniform(low=0.01, high=0.99, size=zone_size)
        
        # 3. 【关键】对小数进行排序，以确保它们是升序的
        #    这 100% 满足 decode_genes 的升序解码约束
        sorted_random_keys = np.sort(random_keys)
        
        # --- 【【【 修改结束 】】】 ---

        # 4. 将这些升序的随机小数分配给区带中的设施
        for i, facility_label in enumerate(zone):
            
            # 5. 找到该设施在原始 permutation 数组中的索引
            gene_index = perm_map[facility_label]
            
            # 6. 创建新的 gene
            #    整数部分 = 区带ID
            #    小数部分 = 升序的随机键
            genes[gene_index] = float(zone_id) + sorted_random_keys[i]
            
        zone_id += 1 # 增加下一个区带的 ID
        
    return genes

# ---------------------------------------------------FBS动作空间开始---------------------------------------------------
# 返回的类型为：(np.ndarray, np.ndarray)
# 动作装饰器
def log_action(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not logging.getLogger().isEnabledFor(logging.DEBUG):
            return func(*args, **kwargs)
        # 输出方法名
        logging.debug(f"方法名：{func.__name__}")
        logging.debug(
            f"变换前的排列：{args[0]}，变换前的区带：{args[1]}, 设施布局为：{permutationToArray(args[0], args[1])}"
        )
        result = func(*args, **kwargs)
        logging.debug(
            f"变换后的排列：{result[0]}，变换后的区带：{result[1]}, 设施布局为：{permutationToArray(result[0], result[1])}"
        )
        return result

    return wrapper


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


def _build_model_from_encoding(permutation, bay):
    return FBSModel(
        permutation.tolist() if isinstance(permutation, np.ndarray) else list(permutation),
        bay.tolist() if isinstance(bay, np.ndarray) else list(bay),
    )


def _evaluate_candidate_encoding(
    permutation,
    bay,
    area,
    H,
    F,
    fac_limit_aspect,
    v_worst=None,
    k_penalty=1,
    distance_metric="manhattan",
):
    model = _build_model_from_encoding(permutation, bay)
    return evaluate_layout_fast(
        model,
        area,
        H,
        F,
        fac_limit_aspect,
        v_worst=v_worst,
        k_penalty=k_penalty,
        distance_metric=distance_metric,
    )


def _evaluate_candidate_encoding_fast(
    permutation,
    bay,
    area,
    H,
    F,
    fac_limit_aspect,
    v_worst=None,
    k_penalty=1,
    distance_metric="manhattan",
):
    model = _build_model_from_encoding(permutation, bay)
    return evaluate_layout_fast(
        model,
        area,
        H,
        F,
        fac_limit_aspect,
        v_worst=v_worst,
        k_penalty=k_penalty,
        distance_metric=distance_metric,
    )


def _weighted_choice(values, weights):
    values = np.asarray(values, dtype=int).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    positive_mask = weights > 0
    if values.size == 0 or weights.size == 0 or not np.any(positive_mask):
        return None
    values = values[positive_mask]
    weights = weights[positive_mask]
    probabilities = weights / np.sum(weights)
    return int(np.random.choice(values, p=probabilities))


def flow_guided_swap(
    permutation,
    bay,
    area,
    H,
    F=None,
    fac_limit_aspect=None,
    v_worst=None,
    k_penalty=1,
    distance_metric="manhattan",
):
    permutation = np.asarray(permutation, dtype=int).copy()
    bay = np.asarray(bay, dtype=int).copy()
    n = len(permutation)
    if n < 2:
        return permutation, bay

    metrics = _evaluate_candidate_encoding(
        permutation,
        bay,
        area,
        H,
        F,
        fac_limit_aspect,
        v_worst=v_worst,
        k_penalty=k_penalty,
        distance_metric=distance_metric,
    )
    pair_scores = np.triu(metrics["TM"], 1)
    pair_indices = np.argwhere(pair_scores > 0)
    if pair_indices.size == 0:
        return permutation, bay

    top_k = min(max(3, math.ceil(n / 4)), len(pair_indices))
    score_values = pair_scores[pair_indices[:, 0], pair_indices[:, 1]]
    top_positions = np.argpartition(score_values, -top_k)[-top_k:]
    selected_pair = pair_indices[int(np.random.choice(top_positions))]

    facility_a = int(selected_pair[0] + 1)
    facility_b = int(selected_pair[1] + 1)
    pos_a = int(np.where(permutation == facility_a)[0][0])
    pos_b = int(np.where(permutation == facility_b)[0][0])

    new_perm = permutation.copy()
    new_perm[pos_a], new_perm[pos_b] = new_perm[pos_b], new_perm[pos_a]
    return new_perm, bay


def segment_insert(
    permutation,
    bay,
    area,
    H,
    F=None,
    fac_limit_aspect=None,
    v_worst=None,
    k_penalty=1,
    distance_metric="manhattan",
):
    permutation = np.asarray(permutation, dtype=int).copy()
    bay = np.asarray(bay, dtype=int).copy()
    bay_structure = _copy_bay_structure(permutationToArray(permutation, bay))
    if not bay_structure:
        return permutation, bay

    candidate_pool = []
    seen = set()
    top_k = max(3, math.ceil(len(permutation) / 5))

    for bay_idx, current_bay in enumerate(bay_structure):
        bay_length = len(current_bay)
        for segment_length in (2, 3):
            if bay_length <= segment_length:
                continue
            for start_idx in range(bay_length - segment_length + 1):
                segment = current_bay[start_idx : start_idx + segment_length]
                remaining = (
                    current_bay[:start_idx] + current_bay[start_idx + segment_length :]
                )
                for insert_idx in range(len(remaining) + 1):
                    candidate_bay = (
                        remaining[:insert_idx] + segment + remaining[insert_idx:]
                    )
                    if candidate_bay == current_bay:
                        continue
                    candidate_structure = _copy_bay_structure(bay_structure)
                    candidate_structure[bay_idx] = candidate_bay
                    candidate_perm, candidate_bay_flags = arrayToPermutation(
                        candidate_structure
                    )
                    candidate_key = (
                        tuple(candidate_perm.tolist()),
                        tuple(candidate_bay_flags.tolist()),
                    )
                    if candidate_key in seen:
                        continue
                    seen.add(candidate_key)
                    metrics = _evaluate_candidate_encoding(
                        candidate_perm,
                        candidate_bay_flags,
                        area,
                        H,
                        F,
                        fac_limit_aspect,
                        v_worst=v_worst,
                        k_penalty=k_penalty,
                        distance_metric=distance_metric,
                    )
                    if not np.isfinite(metrics["cost"]):
                        continue
                    candidate_pool.append(
                        (
                            float(metrics["cost"]),
                            float(metrics["mhc"]),
                            candidate_perm,
                            candidate_bay_flags,
                        )
                    )

    if not candidate_pool:
        return permutation, bay

    candidate_pool.sort(key=lambda item: (item[0], item[1]))
    chosen_idx = int(np.random.randint(0, min(top_k, len(candidate_pool))))
    _, _, best_perm, best_bay = candidate_pool[chosen_idx]
    return best_perm, best_bay


def cross_bay_relocate(
    permutation,
    bay,
    area,
    H,
    F=None,
    fac_limit_aspect=None,
    v_worst=None,
    k_penalty=1,
    distance_metric="manhattan",
):
    permutation = np.asarray(permutation, dtype=int).copy()
    bay = np.asarray(bay, dtype=int).copy()
    bay_structure = _copy_bay_structure(permutationToArray(permutation, bay))
    if len(bay_structure) < 2:
        return permutation, bay

    facility_to_bay = {}
    for bay_idx, current_bay in enumerate(bay_structure):
        for facility in current_bay:
            facility_to_bay[int(facility)] = bay_idx

    candidate_facilities = []
    facility_weights = []
    for facility, source_bay_idx in facility_to_bay.items():
        other_facilities = [
            other_facility
            for bay_idx, current_bay in enumerate(bay_structure)
            if bay_idx != source_bay_idx
            for other_facility in current_bay
        ]
        if not other_facilities:
            continue
        cross_flow = float(
            np.sum(
                F[
                    facility - 1,
                    np.asarray(other_facilities, dtype=int) - 1,
                ]
            )
        )
        if cross_flow <= 0:
            continue
        candidate_facilities.append(int(facility))
        facility_weights.append(cross_flow)

    selected_facility = _weighted_choice(candidate_facilities, facility_weights)
    if selected_facility is None:
        return permutation, bay

    source_bay_idx = facility_to_bay[selected_facility]

    target_bay_ids = []
    target_bay_weights = []
    for bay_idx, current_bay in enumerate(bay_structure):
        if bay_idx == source_bay_idx or len(current_bay) == 0:
            continue
        target_flow = float(
            np.sum(
                F[
                    selected_facility - 1,
                    np.asarray(current_bay, dtype=int) - 1,
                ]
            )
        )
        if target_flow <= 0:
            continue
        target_bay_ids.append(bay_idx)
        target_bay_weights.append(target_flow)

    target_bay_idx = _weighted_choice(target_bay_ids, target_bay_weights)
    if target_bay_idx is None:
        return permutation, bay

    target_bay = bay_structure[target_bay_idx]
    anchor_flows = np.asarray(
        [F[selected_facility - 1, facility - 1] for facility in target_bay],
        dtype=float,
    )
    if anchor_flows.size == 0 or np.max(anchor_flows) <= 0:
        return permutation, bay
    anchor_candidates = np.asarray(target_bay, dtype=int)[
        np.isclose(anchor_flows, np.max(anchor_flows))
    ]
    anchor_facility = int(np.random.choice(anchor_candidates))

    reduced_structure = []
    target_new_idx = None
    for bay_idx, current_bay in enumerate(bay_structure):
        if bay_idx == source_bay_idx:
            updated_bay = [
                facility
                for facility in current_bay
                if facility != selected_facility
            ]
        else:
            updated_bay = list(current_bay)

        if not updated_bay:
            continue
        if bay_idx == target_bay_idx:
            target_new_idx = len(reduced_structure)
        reduced_structure.append(updated_bay)

    if target_new_idx is None:
        return permutation, bay

    candidate_pool = []
    anchor_position = reduced_structure[target_new_idx].index(anchor_facility)
    for insert_offset in (0, 1):
        candidate_structure = _copy_bay_structure(reduced_structure)
        target_list = list(candidate_structure[target_new_idx])
        target_list.insert(anchor_position + insert_offset, selected_facility)
        candidate_structure[target_new_idx] = target_list

        candidate_perm, candidate_bay_flags = arrayToPermutation(candidate_structure)
        metrics = _evaluate_candidate_encoding(
            candidate_perm,
            candidate_bay_flags,
            area,
            H,
            F,
            fac_limit_aspect,
            v_worst=v_worst,
            k_penalty=k_penalty,
            distance_metric=distance_metric,
        )
        if not np.isfinite(metrics["cost"]):
            continue
        candidate_pool.append(
            (
                float(metrics["cost"]),
                float(metrics["mhc"]),
                candidate_perm,
                candidate_bay_flags,
            )
        )

    if not candidate_pool:
        return permutation, bay

    candidate_pool.sort(key=lambda item: (item[0], item[1]))
    _, _, best_perm, best_bay = candidate_pool[0]
    return best_perm, best_bay


def bay_split_by_flow(
    permutation,
    bay,
    area,
    H,
    F=None,
    fac_limit_aspect=None,
    v_worst=None,
    k_penalty=1,
    distance_metric="manhattan",
):
    """Split the bay whose weakest cut has the smallest cross-flow."""
    permutation = np.asarray(permutation, dtype=int).copy()
    bay = np.asarray(bay, dtype=int).copy()
    flow = np.asarray(F, dtype=float)
    bay_structure = _copy_bay_structure(permutationToArray(permutation, bay))

    best_bay_idx = None
    best_split_idx = None
    min_cross_flow = float("inf")

    for bay_idx, current_bay in enumerate(bay_structure):
        if len(current_bay) < 2:
            continue
        for split_idx in range(1, len(current_bay)):
            left = current_bay[:split_idx]
            right = current_bay[split_idx:]
            cross_flow = float(
                sum(
                    flow[int(left_facility) - 1, int(right_facility) - 1]
                    + flow[int(right_facility) - 1, int(left_facility) - 1]
                    for left_facility in left
                    for right_facility in right
                )
            )
            if cross_flow < min_cross_flow:
                min_cross_flow = cross_flow
                best_bay_idx = bay_idx
                best_split_idx = split_idx

    if best_bay_idx is None:
        return permutation, bay

    selected_bay = list(bay_structure[best_bay_idx])
    bay_structure[best_bay_idx] = selected_bay[:best_split_idx]
    bay_structure.insert(best_bay_idx + 1, selected_bay[best_split_idx:])
    return arrayToPermutation(bay_structure)


def bay_merge_by_flow(
    permutation,
    bay,
    area,
    H,
    F=None,
    fac_limit_aspect=None,
    v_worst=None,
    k_penalty=1,
    distance_metric="manhattan",
):
    """Merge the adjacent bay pair with the strongest cross-flow."""
    permutation = np.asarray(permutation, dtype=int).copy()
    bay = np.asarray(bay, dtype=int).copy()
    flow = np.asarray(F, dtype=float)
    bay_structure = _copy_bay_structure(permutationToArray(permutation, bay))
    if len(bay_structure) < 2:
        return permutation, bay

    best_bay_idx = None
    max_cross_flow = -1.0
    for bay_idx in range(len(bay_structure) - 1):
        left = bay_structure[bay_idx]
        right = bay_structure[bay_idx + 1]
        cross_flow = float(
            sum(
                flow[int(left_facility) - 1, int(right_facility) - 1]
                + flow[int(right_facility) - 1, int(left_facility) - 1]
                for left_facility in left
                for right_facility in right
            )
        )
        if cross_flow > max_cross_flow:
            max_cross_flow = cross_flow
            best_bay_idx = bay_idx

    if best_bay_idx is None:
        return permutation, bay

    merged_bay = list(bay_structure[best_bay_idx]) + list(
        bay_structure[best_bay_idx + 1]
    )
    bay_structure[best_bay_idx] = merged_bay
    bay_structure.pop(best_bay_idx + 1)
    return arrayToPermutation(bay_structure)


def adjacent_bay_repartition_by_flow(
    permutation,
    bay,
    area,
    H,
    F=None,
    fac_limit_aspect=None,
    v_worst=None,
    k_penalty=1,
    distance_metric="manhattan",
):
    """Re-cut the boundary of a high-flow adjacent bay pair."""
    permutation = np.asarray(permutation, dtype=int).copy()
    bay = np.asarray(bay, dtype=int).copy()
    flow = np.asarray(F, dtype=float)
    bay_structure = _copy_bay_structure(permutationToArray(permutation, bay))
    if len(bay_structure) < 2:
        return permutation, bay

    pair_candidates = []
    for bay_idx in range(len(bay_structure) - 1):
        left = bay_structure[bay_idx]
        right = bay_structure[bay_idx + 1]
        if len(left) + len(right) < 2:
            continue
        cross_flow = float(
            sum(
                flow[int(left_facility) - 1, int(right_facility) - 1]
                + flow[int(right_facility) - 1, int(left_facility) - 1]
                for left_facility in left
                for right_facility in right
            )
        )
        pair_candidates.append((bay_idx, cross_flow))

    if not pair_candidates:
        return permutation, bay

    pair_candidates.sort(key=lambda item: item[1], reverse=True)
    top_pair_candidates = pair_candidates[: min(3, len(pair_candidates))]
    pair_indices = [item[0] for item in top_pair_candidates]
    pair_weights = [max(float(item[1]), 0.0) for item in top_pair_candidates]

    selected_pair_idx = _weighted_choice(pair_indices, pair_weights)
    if selected_pair_idx is None:
        selected_pair_idx = int(
            np.random.choice(np.asarray(pair_indices, dtype=int))
        )

    merged_sequence = list(bay_structure[selected_pair_idx]) + list(
        bay_structure[selected_pair_idx + 1]
    )
    if len(merged_sequence) < 2:
        return permutation, bay

    candidate_pool = []
    for split_idx in range(1, len(merged_sequence)):
        candidate_structure = _copy_bay_structure(bay_structure)
        candidate_structure[selected_pair_idx] = merged_sequence[:split_idx]
        candidate_structure[selected_pair_idx + 1] = merged_sequence[split_idx:]

        candidate_perm, candidate_bay_flags = arrayToPermutation(candidate_structure)
        metrics = _evaluate_candidate_encoding(
            candidate_perm,
            candidate_bay_flags,
            area,
            H,
            F,
            fac_limit_aspect,
            v_worst=v_worst,
            k_penalty=k_penalty,
            distance_metric=distance_metric,
        )
        if not np.isfinite(metrics["cost"]):
            continue
        candidate_pool.append(
            (
                float(metrics["cost"]),
                float(metrics["mhc"]),
                candidate_perm,
                candidate_bay_flags,
            )
        )

    if not candidate_pool:
        return permutation, bay

    candidate_pool.sort(key=lambda item: (item[0], item[1]))
    chosen_idx = int(np.random.randint(0, min(3, len(candidate_pool))))
    _, _, best_perm, best_bay = candidate_pool[chosen_idx]
    return best_perm, best_bay


def adjacent_bay_block_repartition_by_flow(
    permutation,
    bay,
    area,
    H,
    F=None,
    fac_limit_aspect=None,
    v_worst=None,
    k_penalty=1,
    distance_metric="manhattan",
):
    """Repartition a high-flow adjacent bay pair with small boundary block swaps."""
    permutation = np.asarray(permutation, dtype=int).copy()
    bay = np.asarray(bay, dtype=int).copy()
    flow = np.asarray(F, dtype=float)
    bay_structure = _copy_bay_structure(permutationToArray(permutation, bay))
    if len(bay_structure) < 2:
        return permutation, bay

    pair_candidates = []
    for bay_idx in range(len(bay_structure) - 1):
        left = bay_structure[bay_idx]
        right = bay_structure[bay_idx + 1]
        if len(left) + len(right) < 2:
            continue
        cross_flow = float(
            sum(
                flow[int(left_facility) - 1, int(right_facility) - 1]
                + flow[int(right_facility) - 1, int(left_facility) - 1]
                for left_facility in left
                for right_facility in right
            )
        )
        pair_candidates.append((bay_idx, cross_flow))

    if not pair_candidates:
        return permutation, bay

    pair_candidates.sort(key=lambda item: item[1], reverse=True)
    top_pair_candidates = pair_candidates[: min(3, len(pair_candidates))]
    pair_indices = [item[0] for item in top_pair_candidates]
    pair_weights = [max(float(item[1]), 0.0) for item in top_pair_candidates]

    selected_pair_idx = _weighted_choice(pair_indices, pair_weights)
    if selected_pair_idx is None:
        selected_pair_idx = int(
            np.random.choice(np.asarray(pair_indices, dtype=int))
        )

    left = list(bay_structure[selected_pair_idx])
    right = list(bay_structure[selected_pair_idx + 1])
    merged_sequence = left + right
    if len(merged_sequence) < 2:
        return permutation, bay

    candidate_pool = []
    seen_encodings = set()

    def _append_candidate(candidate_structure):
        candidate_perm, candidate_bay_flags = arrayToPermutation(candidate_structure)
        key = (
            tuple(int(v) for v in np.asarray(candidate_perm, dtype=int).tolist()),
            tuple(int(v) for v in np.asarray(candidate_bay_flags, dtype=int).tolist()),
        )
        if key in seen_encodings:
            return
        seen_encodings.add(key)
        metrics = _evaluate_candidate_encoding(
            candidate_perm,
            candidate_bay_flags,
            area,
            H,
            F,
            fac_limit_aspect,
            v_worst=v_worst,
            k_penalty=k_penalty,
            distance_metric=distance_metric,
        )
        if not np.isfinite(metrics["cost"]):
            return
        candidate_pool.append(
            (
                float(metrics["cost"]),
                float(metrics["mhc"]),
                candidate_perm,
                candidate_bay_flags,
            )
        )

    for split_idx in range(1, len(merged_sequence)):
        candidate_structure = _copy_bay_structure(bay_structure)
        candidate_structure[selected_pair_idx] = merged_sequence[:split_idx]
        candidate_structure[selected_pair_idx + 1] = merged_sequence[split_idx:]
        _append_candidate(candidate_structure)

    max_left_block = min(3, len(left))
    max_right_block = min(3, len(right))
    for left_block_size in range(1, max_left_block + 1):
        left_prefix = left[:-left_block_size]
        left_suffix = left[-left_block_size:]
        for right_block_size in range(1, max_right_block + 1):
            right_prefix = right[:right_block_size]
            right_suffix = right[right_block_size:]
            new_left = left_prefix + right_prefix
            new_right = left_suffix + right_suffix
            if not new_left or not new_right:
                continue
            candidate_structure = _copy_bay_structure(bay_structure)
            candidate_structure[selected_pair_idx] = new_left
            candidate_structure[selected_pair_idx + 1] = new_right
            _append_candidate(candidate_structure)

    if not candidate_pool:
        return permutation, bay

    candidate_pool.sort(key=lambda item: (item[0], item[1]))
    chosen_idx = int(np.random.randint(0, min(3, len(candidate_pool))))
    _, _, best_perm, best_bay = candidate_pool[chosen_idx]
    return best_perm, best_bay


def is_feasible_eq34(fac_b, fac_h, area, fac_limit_aspect):
    d_inf, _, _, _, _ = calculate_d_inf(fac_b, fac_h, area, fac_limit_aspect)
    return d_inf == 0


@log_action
def facility_swap(permutation: np.ndarray, bay: np.ndarray):
    """交换两个设施"""
    i, j = np.random.choice(len(permutation), 2, replace=False)  # 随机选择两个设施
    permutation[i], permutation[j] = permutation[j], permutation[i]  # 交换设施
    return permutation, bay


# 将bay的值转换
@log_action
# def bay_flip(permutation: np.ndarray, bay: np.ndarray):
#     """将bay的值转换"""
#     index = np.random.choice(len(bay))
#     bay[index] = 1 - bay[index]
#     return permutation, bay
# def bay_flip(permutation: np.ndarray, bay: np.ndarray):
#     """
#     论文版 Bay Flip：主要用于拆分和细化区带
#     参考文献逻辑：Section 4.2, Pages 5673-5674
#     """
#     n = len(bay)
#     # 随机选择一个设施索引 (范围 0 到 n-1)
#     index = np.random.choice(n)
    
#     # 情况 1: 如果该设施是区带的最后一个 (bay[index] == 1)
#     # 论文描述: "detached from the current bay ... becoming the sole item" [cite: 257]
#     # 实现逻辑: 要让它成为单独的区带，必须确保它的前一个设施也是一个区带的结束。
#     if bay[index] == 1:
#         # 只有当它不是第一个设施时，才能修改前一个设施
#         if index > 0:
#             bay[index - 1] = 1
#             # logger.debug(f"Bay Flip (Last): 设施 {index} 被独立，设施 {index-1} 变为区带结束")
            
#     # 情况 2: 如果该设施在区带中间 (bay[index] == 0)
#     # 论文描述: "all facilities following it are separated ... assigned a new bay" [cite: 258]
#     # 实现逻辑: 在该设施处截断，将其设为区带结束。
#     else:
#         bay[index] = 1
#         # logger.debug(f"Bay Flip (Middle): 设施 {index} 处截断，产生新区带")

#     # 强制约束: 最后一个设施必须始终是区带结束 [cite: 215]
#     bay[-1] = 1
    
#     return permutation, bay
@log_action
def bay_flip(permutation: np.ndarray, bay: np.ndarray):
    """
    修正版 bay_flip：执行标准的 0/1 翻转，允许拆分和合并区带。
    同时加入了重试机制，确保产生有效变异。
    """
    n = len(bay)
    if n <= 1:
        return permutation, bay

    # 尝试最多 10 次，找到一个能产生变化的翻转
    # (因为 bay[-1] 锁定为 1，如果随机一直选到 n-1，就会一直无效)
    for _ in range(10):
        # 随机选择索引，范围 [0, n-2] (避开最后一个，因为最后一个必须是1)
        index = np.random.randint(0, n - 1)
        
        # 执行翻转 (0->1 或 1->0)
        bay[index] = 1 - bay[index]
        
        # 翻转成功，跳出循环
        break
        
    # 强制约束：最后一个位置必须是1
    bay[-1] = 1
    
    return permutation, bay

# 交换两个bay
@log_action
# def bay_swap(permutation: np.ndarray, bay: np.ndarray):
#     """交换两个bay"""
#     # 转换为二维数组
#     array = permutationToArray(permutation, bay)
#     if len(array) < 2:
#         return permutation, bay  # 如果bay的数量小于2，则直接返回
#     # 随机选择两个bay
#     i, j = np.random.choice(len(array), 2, replace=False)
#     # 交换两个bay
#     array[i], array[j] = array[j], array[i]
#     # 转换为排列和bay
#     permutation, bay = arrayToPermutation(array)
#     return permutation, bay
# 修改 FBSUtil.py 中的 bay_swap
@log_action
def bay_swap(permutation: np.ndarray, bay: np.ndarray):
    """
    智能版 bay_swap：
    如果区带数量 >= 2，执行区带交换。
    如果区带数量 < 2，自动降级为 '设施交换' (facility_swap)，确保变异有效。
    """
    array = permutationToArray(permutation, bay)
    
    # --- 修复逻辑开始 ---
    if len(array) < 2:
        # 只有一个区带，没法换区带，改换设施！
        # 直接调用 facility_swap (注意：不要加 @log_action 否则会打印两次日志)
        # 这里把 facility_swap 的逻辑内联进来，或者去掉 facility_swap 的装饰器再调用
        # 为简单起见，直接写交换逻辑：
        if len(permutation) >= 2:
             i, j = np.random.choice(len(permutation), 2, replace=False)
             permutation[i], permutation[j] = permutation[j], permutation[i]
        return permutation, bay
    # --- 修复逻辑结束 ---

    # 正常的区带交换
    i, j = np.random.choice(len(array), 2, replace=False)
    array[i], array[j] = array[j], array[i]
    permutation, bay = arrayToPermutation(array)
    return permutation, bay

# 对区带shuffle
@log_action
def bay_shuffle(permutation: np.ndarray, bay: np.ndarray):
    """对区带shuffle"""
    fac_list = permutationToArray(permutation, bay)
    np.random.shuffle(fac_list)
    permutation, bay = arrayToPermutation(fac_list)
    return permutation, bay


# 对设施排列shuffle
@log_action
def facility_shuffle(permutation: np.ndarray, bay: np.ndarray):
    """对设施排列shuffle"""
    fac_list = permutationToArray(permutation, bay)
    for i in range(len(fac_list)):
        np.random.shuffle(fac_list[i])
    permutation, bay = arrayToPermutation(fac_list)
    return permutation, bay


# 对排列shuffle
@log_action
def permutation_shuffle(permutation: np.ndarray, bay: np.ndarray):
    """对排列shuffle"""
    np.random.shuffle(permutation)
    return permutation, bay

@log_action
def facility_insert(permutation: np.ndarray, bay: np.ndarray):
    """
    设施插入操作：随机选择一个设施，将其移动到序列中的另一个位置。
    """
    n = len(permutation)
    if n < 2:
        return permutation, bay
        
    # 随机选择源位置和目标位置
    i, j = np.random.choice(n, 2, replace=False)
    
    # 使用 numpy 操作进行插入
    # 注意：permutation 是 ndarray，操作不如 list 方便，需要切片拼接
    val = permutation[i]
    
    # 移除 i
    new_perm = np.delete(permutation, i)
    
    # 插入到 j (注意：如果 i < j，删除 i 后 j 的索引实际前移了一位，但 insert 会处理)
    new_perm = np.insert(new_perm, j, val)
    
    return new_perm, bay


# 修复bay
# def repair(
#     permutation: np.ndarray,
#     bay: np.ndarray,
#     fac_b: np.ndarray,
#     fac_h: np.ndarray,
#     fac_limit_aspect: float,
# ):
#     """修复bay"""
#     # logger.info(f"{permutation}，{bay}")
#     # 转换为二维数组
#     array = permutationToArray(permutation, bay)
#     if not array or len(array) == 0:
#         # 如果转换失败，返回原始值
#         return permutation, bay
#     # 遍历每个bay（注意：循环变量名改为 current_bay 避免与参数 bay 冲突）
#     for i, current_bay in enumerate(array):
#         # logger.info(f"当前第{i}个区带：{current_bay}")
#         tmp_array = array[:]
#         # 计算所有的设施的横纵比
#         fac_aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
#         current_bay_fac_aspect_ratio = np.array([fac_aspect_ratio[b - 1] for b in current_bay])
#         current_bay_fac_hv_ratio = np.array([fac_b[b - 1] / fac_h[b - 1] for b in current_bay])
#         # 如果当前bay的设施的横纵比不满足条件，则进行修复
#         if np.any(
#             (current_bay_fac_aspect_ratio < 1)
#             | (current_bay_fac_aspect_ratio > fac_limit_aspect)
#         ):
#             # logger.info(f"区带{i}不满足条件")
#             # 如果太宽了，说明这个bay中的设施过多，则将其对半分（太宽：横坐标长度/纵坐标长度 > 横纵比）这里使用bay的平均值
#             if np.any(current_bay_fac_hv_ratio > fac_limit_aspect):
#                 # print(f"区带{i}有设施太宽了")
#                 # 将当前bay的设施随机对半分
#                 np.random.shuffle(tmp_array[i])
#                 split_array = np.array_split(tmp_array[i], 2)
#                 tmp_array[i] = split_array[0]
#                 tmp_array.insert(i + 1, split_array[1])
#             # 如果太窄了，说明这个bay中的设施过少，则将当前bay与相邻的bay进行合并（太窄：纵坐标长度/横坐标长度 > 横纵比）
#             else:
#                 # print(f"区带{i}有设施太窄了")
#                 # 将当前bay的设施与相邻的bay进行合并
#                 if i + 1 < len(tmp_array):
#                     tmp_array[i] = np.concatenate((tmp_array[i], tmp_array[i + 1]))
#                     tmp_array.pop(i + 1)
#                 else:
#                     tmp_array[i] = np.concatenate((tmp_array[i], tmp_array[i - 1]))
#                     tmp_array.pop(i - 1)
#             array = tmp_array
#             break
#     # logger.info(f"修复后的bay：{array}")
#     # 转换为排列和bay
#     permutation, bay = arrayToPermutation(array)
#     return permutation, bay


# def repair(
# # def smart_geometric_repair(
#     permutation: np.ndarray,
#     bay: np.ndarray,
#     fac_b: np.ndarray,
#     fac_h: np.ndarray,
#     fac_limit_aspect: float,
# ):
#     """
#     智能几何修复：
#     1. 不打乱原有序列（保留优良基因）。
#     2. 根据宽高比的具体情况，定向选择拆分或合并。
#     3. 使得布局向形状更合理的“正方形”趋势进化。
#     """
#     # 1. 转换为二维列表结构
#     bay_structure = permutationToArray(permutation, bay)
#     if not bay_structure or len(bay_structure) == 0:
#         return permutation, bay

#     changed = False
#     new_structure = []
    
#     # 记录当前处理到的索引，用于跳过已被合并的 Bay
#     skip_next = False

#     for i in range(len(bay_structure)):
#         if skip_next:
#             skip_next = False
#             continue

#         current_bay = bay_structure[i]
        
#         # 获取当前 Bay 内所有设施的尺寸
#         # 注意：这里的 fac_b, fac_h 应该是对应于 permutation 顺序的
#         # 我们需要根据设施编号(假设是从1开始的整数)找到对应的宽高
#         bay_indices = [idx - 1 for idx in current_bay] # 假设设施ID是1-based
        
#         curr_w = fac_b[bay_indices]
#         curr_h = fac_h[bay_indices]
        
#         # 计算纵横比 (Aspect Ratio)
#         # 宽/高
#         ratios_w_h = curr_w / curr_h 
#         # 高/宽
#         ratios_h_w = curr_h / curr_w
        
#         # 检查违规情况
#         # 情况 A: 设施太扁 (Too Flat, Width >> Height)
#         # 意味着 Bay 的宽度过大，或者这一行塞了太多东西（如果横向排列）
#         # 在 FBS 中，通常意味着 Bay 需要 Split
#         is_too_flat = np.any(ratios_w_h > fac_limit_aspect)
        
#         # 情况 B: 设施太瘦 (Too Tall, Height >> Width)
#         # 意味着 Bay 的宽度太小
#         # 在 FBS 中，意味着需要 Merge (变宽)
#         is_too_tall = np.any(ratios_h_w > fac_limit_aspect)

#         if is_too_flat:
#             # --- 策略：拆分 (Split) ---
#             # 不打乱顺序，直接从中间切开
#             # 这样能让每个新 Bay 分到的总高度变小（或宽度变小），从而改善比例
#             if len(current_bay) > 1:
#                 mid = len(current_bay) // 2
#                 new_structure.append(current_bay[:mid])
#                 new_structure.append(current_bay[mid:])
#                 changed = True
#             else:
#                 # 只有一个设施还太扁，说明它自己形状就有问题，或者受限于总面积
#                 # 这种情况下无法通过 FBS 拆分解决，保持原样
#                 new_structure.append(current_bay)

#         elif is_too_tall:
#             # --- 策略：合并 (Merge) ---
#             # 需要变宽，所以要吃掉邻居
#             changed = True
            
#             # 决策：跟左边合还是跟右边合？
#             # 贪婪规则：谁比较短（设施少），就跟谁合，这样不容易造成新的违规
#             target_neighbor_idx = -1
            
#             # 检查右邻居 (i+1)
#             can_merge_right = (i + 1 < len(bay_structure))
#             # 检查左邻居 (new_structure 的最后一个)
#             can_merge_left = (len(new_structure) > 0)

#             if can_merge_right and not can_merge_left:
#                 # 只能合右边
#                 merged_bay = np.concatenate((current_bay, bay_structure[i+1]))
#                 new_structure.append(merged_bay)
#                 skip_next = True # 下次循环跳过 i+1
                
#             elif can_merge_left and not can_merge_right:
#                 # 只能合左边
#                 # 取出刚刚加进去的左邻居
#                 last_bay = new_structure.pop()
#                 merged_bay = np.concatenate((last_bay, current_bay))
#                 new_structure.append(merged_bay)
                
#             elif can_merge_left and can_merge_right:
#                 # 左右都能合，选个更合适的
#                 # 简单启发式：选元素更少的那个邻居合并，减小对整体结构的冲击
#                 len_left = len(new_structure[-1])
#                 len_right = len(bay_structure[i+1])
                
#                 if len_left <= len_right:
#                     last_bay = new_structure.pop()
#                     merged_bay = np.concatenate((last_bay, current_bay))
#                     new_structure.append(merged_bay)
#                 else:
#                     merged_bay = np.concatenate((current_bay, bay_structure[i+1]))
#                     new_structure.append(merged_bay)
#                     skip_next = True
#             else:
#                 # 无法合并（也就是只有一个 Bay），无法修复
#                 new_structure.append(current_bay)
                
#         else:
#             # 没有违规，保持原样
#             new_structure.append(current_bay)

#     # 只有发生改变时才更新，避免无效计算
#     if changed:
#         p, b = arrayToPermutation(new_structure)
#         return p, b
    
#     return permutation, bay

def repair(
    permutation: np.ndarray,
    bay: np.ndarray,
    fac_b: np.ndarray,
    fac_h: np.ndarray,
    fac_limit_aspect: float,
):
    """
    智能几何修复（修正版）：
    适应 StatusUpdatingDevice 传入的 (Height, Width) 顺序，
    正确识别“太瘦”和“太扁”并执行对应修复。
    """
    # 1. 转换为二维列表结构
    bay_structure = permutationToArray(permutation, bay)
    if not bay_structure or len(bay_structure) == 0:
        return permutation, bay

    changed = False
    new_structure = []
    
    # 记录当前处理到的索引，用于跳过已被合并的 Bay
    skip_next = False

    for i in range(len(bay_structure)):
        if skip_next:
            skip_next = False
            continue

        current_bay = bay_structure[i]
        
        # 获取当前 Bay 内所有设施的尺寸
        # 注意：这里需要传入正确的索引来获取宽高
        bay_indices = [idx - 1 for idx in current_bay] 
        
        # 【关键修正】：
        # 由于 StatusUpdatingDevice 返回顺序是 (..., Height, Width, ...)
        # 所以传入 repair 的 fac_b 实际上是 Height，fac_h 实际上是 Width
        # 我们在这里进行“纠正命名”，以保证逻辑正确
        curr_h = fac_b[bay_indices]  # 传入的 fac_b 是高度
        curr_w = fac_h[bay_indices]  # 传入的 fac_h 是宽度
        
        # 计算有方向的纵横比
        # 太扁 (Too Flat/Wide): 宽度 >> 高度 -> 需要拆分 (Split)
        ratios_w_h = curr_w / curr_h 
        # 太瘦 (Too Tall/Thin): 高度 >> 宽度 -> 需要合并 (Merge)
        ratios_h_w = curr_h / curr_w
        
        # 检查违规情况
        is_too_flat = np.any(ratios_w_h > fac_limit_aspect)
        is_too_tall = np.any(ratios_h_w > fac_limit_aspect)

        if is_too_flat:
            # --- 策略：拆分 (Split) ---
            # 设施太宽/太扁，说明区带太宽了，需要切开
            if len(current_bay) > 1:
                mid = len(current_bay) // 2
                new_structure.append(current_bay[:mid])
                new_structure.append(current_bay[mid:])
                changed = True
            else:
                new_structure.append(current_bay)

        elif is_too_tall:
            # --- 策略：合并 (Merge) ---
            # 设施太瘦/太高，说明区带太窄了，需要变宽（合并）
            changed = True
            
            # 决策：跟左边合还是跟右边合？
            can_merge_right = (i + 1 < len(bay_structure))
            can_merge_left = (len(new_structure) > 0)

            if can_merge_right and not can_merge_left:
                # 只能合右边
                merged_bay = np.concatenate((current_bay, bay_structure[i+1]))
                new_structure.append(merged_bay)
                skip_next = True 
                
            elif can_merge_left and not can_merge_right:
                # 只能合左边
                last_bay = new_structure.pop()
                merged_bay = np.concatenate((last_bay, current_bay))
                new_structure.append(merged_bay)
                
            elif can_merge_left and can_merge_right:
                # 左右都能合，选元素更少的邻居合并
                len_left = len(new_structure[-1])
                len_right = len(bay_structure[i+1])
                
                if len_left <= len_right:
                    last_bay = new_structure.pop()
                    merged_bay = np.concatenate((last_bay, current_bay))
                    new_structure.append(merged_bay)
                else:
                    merged_bay = np.concatenate((current_bay, bay_structure[i+1]))
                    new_structure.append(merged_bay)
                    skip_next = True
            else:
                new_structure.append(current_bay)
                
        else:
            # 没有违规，保持原样
            new_structure.append(current_bay)

    # 只有发生改变时才更新，避免无效计算
    if changed:
        p, b = arrayToPermutation(new_structure)
        return p, b
    
    return permutation, bay

# -------------------------------------------------个体动作结束-------------------------------------------------
# =============================================================================
# 模块: 纯净的种群优化器 (GA)
# (基于 PopulationOptimizer.py，作为 Action 5 的逻辑)
# =============================================================================
class PopulationOptimizer:
    """
    遗传算法(GA)优化器。
    该类使用实数编码(genes)进行交叉和变异。
    """
    def __init__(self, env, population_size, 
                 crossover_rate, mutation_rate, elite_rate):
        
        self.env = env
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
        
        self.population = [] # 种群 (FBSModel 列表)
        self.fitnesses = []  # 适应度列表
        self.best_solution = None # 最优 FBSModel
        self.best_fitness = float('inf')

    def _selection(self, k=2):
        # 锦标赛选择
        selected_indices = np.random.choice(len(self.population), k, replace=False)
        selected_fitnesses = [self.fitnesses[i] for i in selected_indices]
        best_index = selected_indices[np.argmin(selected_fitnesses)]
        return self.population[best_index]

    def _crossover(self, parent1, parent2):
        # 实数交叉
        parent1_genes = parent1.genes
        parent2_genes = parent2.genes
        n_genes = len(parent1_genes)
        offspring1_genes = np.zeros(n_genes)
        offspring2_genes = np.zeros(n_genes)

        for i in range(n_genes):
            beta = (np.random.rand() * 2.0) - 0.5 # 随机数 [-0.5, 1.5]
            offspring1_genes[i] = beta * parent1_genes[i] + (1.0 - beta) * parent2_genes[i]
            offspring2_genes[i] = (1.0 - beta) * parent1_genes[i] + beta * parent2_genes[i]

        # 1. 解码子代 1
        bay_list1, bay1 = ZGeneCoding.decode_genes(offspring1_genes, parent1.permutation)
        perm1, bay1 = arrayToPermutation(bay_list1)
        
        # 2. 解码子代 2
        bay_list2, bay2 = ZGeneCoding.decode_genes(offspring2_genes, parent2.permutation)
        perm2, bay2 = arrayToPermutation(bay_list2)
        
        # 3. 在所有数据都准备好之后，再创建实例
        offspring1 = FBSModel(
            permutation=perm1.tolist(), 
            bay=bay1.tolist(), 
            genes=offspring1_genes.tolist()
        )
        offspring2 = FBSModel(
            permutation=perm2.tolist(), 
            bay=bay2.tolist(), 
            genes=offspring2_genes.tolist()
        )
        # --- 【【【 修复结束 】】】 ---
        
        return offspring1, offspring2

    # def _mutation(self, individual):
    #     # 实数变异
    #     genes = individual.genes
    #     n_genes = len(genes)
    #     eta_m = 10.0 # 变异分布指数 (来自)
        
    #     for i in range(n_genes):
    #         if np.random.rand() < (1.0 / n_genes):
    #             delta = (2.0 * np.random.rand()) ** (1.0 / (eta_m + 1.0)) - 1.0
    #             genes[i] += delta
        
    #     individual.genes = genes
        
    #     # (重要) 解码以更新 perm 和 bay
    #     bay_list_m, bay_m = ZGeneCoding.decode_genes(individual.genes, individual.permutation)
    #     perm_m, bay_m = arrayToPermutation(bay_list_m)
    #     individual.permutation = perm_m.tolist()
    #     individual.bay = bay_m.tolist()

    #     return individual


    def _mutation(self, individual):
        genes = individual.genes
        n_genes = len(genes)
        
        # --- 策略 A: 基因位交换 (模拟局部搜索 Swap) ---
        # 以较高概率执行交换，这是最直接的局部优化手段
        if np.random.rand() < 0.7:  # 70% 概率做交换
            idx1, idx2 = np.random.choice(n_genes, 2, replace=False)
            # 直接交换两个实数值
            genes[idx1], genes[idx2] = genes[idx2], genes[idx1]
        
        # --- 策略 B: 传统的微小扰动 (保留一点点以维持多样性) ---
        else:
            # 仅对 1-2 个基因做微调，不要全员变异
            idx = np.random.randint(0, n_genes)
            # 施加一个极小的扰动，尝试微调排序临界点
            genes[idx] += np.random.normal(0, 0.1) 

        individual.genes = genes
        
        # 解码并更新 (保持原样)
        bay_list_m, bay_m = ZGeneCoding.decode_genes(individual.genes, individual.permutation)
        perm_m, bay_m = arrayToPermutation(bay_list_m)
        individual.permutation = perm_m.tolist()
        individual.bay = bay_m.tolist()

        return individual


    def _evaluate(self, individual):
        # 评估个体适应度
        _, info = self.env.reset(fbs_model=individual)
        return info['fitness']

    def set_population(self, population_list):
        """用于从外部接收临时种群"""
        self.population = population_list
        self.fitnesses = [self._evaluate(ind) for ind in self.population]
        if not self.fitnesses:
            self.best_fitness = float('inf')
            self.best_solution = None
        else:
            self.best_fitness = np.min(self.fitnesses)
            self.best_solution = self.population[np.argmin(self.fitnesses)]

    def run_one_generation(self):
        """仅运行 GA 一代"""
        sorted_indices = np.argsort(self.fitnesses)
        num_elites = int(self.population_size * self.elite_rate)
        elites = [self.population[i] for i in sorted_indices[:num_elites]]

        new_population = elites
        num_offspring = self.population_size - num_elites
        
        for _ in range(num_offspring // 2):
            parent1 = self._selection(k=3)
            parent2 = self._selection(k=3)
            
            if np.random.rand() < self.crossover_rate:
                offspring1, offspring2 = self._crossover(parent1, parent2)
            else:
                offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            if np.random.rand() < self.mutation_rate:
                offspring1 = self._mutation(offspring1)
            if np.random.rand() < self.mutation_rate:
                offspring2 = self._mutation(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        self.population = new_population[:self.population_size]
        self.fitnesses = [self._evaluate(ind) for ind in self.population]
        
        current_best_fitness = np.min(self.fitnesses)
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_solution = self.population[np.argmin(self.fitnesses)]
        
        return self.best_solution, self.best_fitness
# =============================================================================
# (GA 类结束)
# =============================================================================

# =============================================================================
# 动作 5: GA 种群优化动作
# =============================================================================
def re_encode_genes_perfectly(permutation):
    """
    根据排列顺序，生成一组完美的线性基因。
    例如 permutation = [2, 0, 1] (假设 N=3)
    我们希望基因能反映这个顺序，比如 genes[2]=0.0, genes[0]=0.33, genes[1]=0.66
    这样能最大程度稳固当前的排序。
    """
    n = len(permutation)
    new_genes = np.zeros(n)
    # 归一化区间 [0, 1]
    step = 1.0 / n
    for rank, fac_idx in enumerate(permutation):
        # 赋予 fac_idx 一个与其排名 (rank) 对应的值
        new_genes[fac_idx] = rank * step
    return new_genes

def ga_population_action(current_model, env_instance, pop_size=10, generations=5):
    """
    执行一次 GA '大跳' 动作。
    由 DataExtractor.py 调用。
    """
    # logger.info(">>> (Action 5) 触发 GA (genes-based) 动作...")
    
    # 1. 实例化 GA 求解器 (它现在定义在 FBSUtil.py 内部)
    ga_solver = PopulationOptimizer(
        env=env_instance,
        population_size=pop_size,
        crossover_rate=0.8, # (来自)
        mutation_rate=0.3,  # (来自)
        elite_rate=0.1      # (来自)
    )
    
    # 2. 【即时重新编码 (方案 B)】
    #    读取 base_model 中 *最新* 的 (perm, bay)
    #    并生成一个 *新鲜* 的、与之同步的 genes 数组
    base_genes = encode_genes_from_solution(
        current_model.permutation, 
        current_model.bay
    )
    # 创建 "新鲜" 的基础解
    base_model_fresh = FBSModel(
        current_model.permutation, 
        current_model.bay, 
        base_genes.tolist()
    )
    
    # 3. 围绕 "新鲜" 解创建临时种群
    population = [base_model_fresh]
    for _ in range(pop_size - 1):
        mutated_model = copy.deepcopy(base_model_fresh)
        
        # (调用 'genes'-based 变异)
        mutated_model = ga_solver._mutation(mutated_model)
        
        # (解码以保持 perm/bay 同步)
        bay_list_m, bay_m = ZGeneCoding.decode_genes(mutated_model.genes, mutated_model.permutation)
        perm_m, bay_m = arrayToPermutation(bay_list_m)
        mutated_model.permutation = perm_m.tolist()
        mutated_model.bay = bay_m.tolist()

        population.append(mutated_model)
    
    # 4. 设置种群并运行 GA
    ga_solver.set_population(population)
    best_ga_model = current_model
    for _ in range(generations):
        best_ga_model, best_fitness = ga_solver.run_one_generation()

    # logger.info(f"...GA 动作完成。找到 Fitness: {best_fitness}")
    perfect_genes = encode_genes_from_solution(
        best_ga_model.permutation,
        best_ga_model.bay
    )
    # 更新模型的基因 (虽然 ELP 主循环暂时不用基因，但这保证了数据一致性)
    best_ga_model.genes = perfect_genes.tolist()
    # 5. 返回 GA 找到的最优解 (perm 和 bay)
    #    ELP 主循环只关心 perm 和 bay
    return best_ga_model.permutation, best_ga_model.bay

# 贪婪搜索
def greedy_local_search_action(current_model, env):
    """
    贪婪局部搜索：尝试微调当前解，如果变好就保留，直到无法变好或达到步数限制。
    替代原来的 GA 动作。
    """
    best_sol = copy.deepcopy(current_model)
    best_energy = env.get_fitness(best_sol) # 假设有这个接口
    
    improved = True
    max_steps = 50 # 限制步数，防止死循环
    step = 0
    
    while improved and step < max_steps:
        improved = False
        step += 1
        
        # 尝试邻域动作：例如随机交换两个元素
        neighbor = copy.deepcopy(best_sol)
        # 随机交换
        idx1, idx2 = np.random.choice(len(neighbor.permutation), 2, replace=False)
        neighbor.permutation[idx1], neighbor.permutation[idx2] = neighbor.permutation[idx2], neighbor.permutation[idx1]
        
        neighbor_energy = env.get_fitness(neighbor)
        
        # 贪婪准则：只接受更好的
        if neighbor_energy < best_energy:
            best_sol = neighbor
            best_energy = neighbor_energy
            improved = True # 继续搜索
            
    return best_sol.permutation, best_sol.bay
# -------------------------------------------------群体动作结束-------------------------------------------------

# ---------------------------------------------------FBS动作空间结束---------------------------------------------------


def permutationToArray(permutation, bay):
    """将排列转换为二维数组"""
    if len(permutation) == 0 or len(bay) == 0:
        return []
    if len(permutation) != len(bay):
        return []
    bay_copy = bay.copy() if hasattr(bay, 'copy') else list(bay)
    bay_copy[-1] = 1  # 将bay的最后一个元素设置为1
    array = []
    start = 0
    for i, val in enumerate(bay_copy):
        if val == 1:
            array.append(permutation[start : i + 1])
            start = i + 1
    return array


# 将二维数组转换为排列和bay
def arrayToPermutation(array):
    if not array or len(array) == 0:
        return np.array([]), np.array([])
    permutation = []
    bay = []
    for sub_array in array:
        if len(sub_array) > 0:
            permutation.extend(sub_array)
            bay.extend([0] * (len(sub_array) - 1) + [1])
    if len(permutation) == 0 or len(bay) == 0:
        return np.array([]), np.array([])
    permutation = np.array(permutation)
    bay = np.array(bay)
    return permutation, bay


def sayHello():
    logging.info("Hello World")


def constructState(fac_x, fac_y, fac_b, fac_h, W, L, fbsModel, TM):
    n = len(fbsModel.permutation)
    permutation = fbsModel.permutation
    bay = fbsModel.bay
    state_prelim = np.zeros((4 * n,), dtype=float)
    state_prelim[0::4] = fac_x  # 0，4，8
    state_prelim[1::4] = fac_y  # 1，5，9
    state_prelim[2::4] = fac_b  # 2，6，10
    state_prelim[3::4] = fac_h  # 3，7，11
    data = np.zeros((int(W), int(L), 3), dtype=np.uint8)
    sources = np.sum(TM, axis=1)
    sinks = np.sum(TM, axis=0)
    R = np.array(
        ((permutation - np.min(permutation))/ (np.max(permutation) - np.min(permutation)))* 255
    ).astype(np.uint8)
    G = np.array(
        ((sources - np.min(sources)) / (np.max(sources) - np.min(sources))) * 255
    ).astype(np.uint8)
    B = np.array(
        ((sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks))) * 255
    ).astype(np.uint8)
        # 将坐标和颜色结合到图像中
    for i in range(n):
        # 计算设施的像素范围
        x_start = max(0, int(np.floor(fac_x[i])))              # 左边界
        x_end = min(L, int(np.ceil(fac_x[i] + fac_b[i])))      # 右边界
        y_start = max(0, int(np.floor(fac_y[i])))              # 上边界
        y_end = min(W, int(np.ceil(fac_y[i] + fac_h[i])))      # 下边界

        # 填充颜色到对应区域
        data[y_start:y_end, x_start:x_end, :] = [R[i], G[i], B[i]]
    return data


# Modern fitness helpers used by ELP/DataExtractor.
def get_instance_aspect_limit(fac_limit_aspect) -> float:
    aspect_limits = np.asarray(fac_limit_aspect, dtype=float).reshape(-1)
    if aspect_limits.size == 0:
        return 99.0
    finite_limits = aspect_limits[np.isfinite(aspect_limits)]
    if finite_limits.size == 0:
        return 99.0
    return float(np.max(finite_limits))


def _normalize_aspect_limits(fac_limit_aspect, facility_count: int) -> np.ndarray:
    aspect_limits = np.asarray(fac_limit_aspect, dtype=float).reshape(-1)
    if aspect_limits.size == 0:
        aspect_limits = np.full(facility_count, 99.0, dtype=float)
    elif aspect_limits.size == 1:
        aspect_limits = np.full(facility_count, float(aspect_limits[0]), dtype=float)
    elif aspect_limits.size != facility_count:
        raise ValueError(
            f"Aspect limit count {aspect_limits.size} does not match facility count {facility_count}."
        )
    return np.clip(aspect_limits, 1.0 + 1e-12, None)


def calculate_eq34_bounds(area, fac_limit_aspect):
    areas = np.asarray(area, dtype=float).reshape(-1)
    aspect_limits = _normalize_aspect_limits(fac_limit_aspect, len(areas))
    lower_bounds = np.sqrt(areas / aspect_limits)
    upper_bounds = np.sqrt(areas * aspect_limits)
    return lower_bounds, upper_bounds, aspect_limits


def calculate_d_inf(fac_b, fac_h, area, fac_limit_aspect):
    widths = np.asarray(fac_b, dtype=float).reshape(-1)
    heights = np.asarray(fac_h, dtype=float).reshape(-1)
    short_side = np.minimum(widths, heights)
    long_side = np.maximum(widths, heights)
    lower_bounds, upper_bounds, aspect_limits = calculate_eq34_bounds(area, fac_limit_aspect)
    infeasible_mask = (short_side < lower_bounds) | (long_side > upper_bounds)
    d_inf = int(np.sum(infeasible_mask))
    return d_inf, infeasible_mask, lower_bounds, upper_bounds, aspect_limits


def calculate_true_cost(
    mhc,
    fac_b,
    fac_h,
    area,
    fac_limit_aspect,
    v_ref=None,
    v_worst=None,
    true_cost_penalty_scale=0.35,
    tau=0.2,
    alpha=0.7,
    beta=10.0,
):
    if v_ref is None and v_worst is not None:
        v_ref = v_worst
    _d_inf_unused, _infeasible_mask_unused, lower_bounds, upper_bounds, aspect_limits = calculate_d_inf(
        fac_b, fac_h, area, fac_limit_aspect
    )
    widths = np.asarray(fac_b, dtype=float).reshape(-1)
    heights = np.asarray(fac_h, dtype=float).reshape(-1)
    short_side = np.minimum(widths, heights)
    long_side = np.maximum(widths, heights)
    aspect_ratio = np.divide(
        long_side,
        np.maximum(short_side, 1e-12),
        out=np.full_like(long_side, np.inf, dtype=float),
        where=short_side > 0,
    )
    aspect_limit_arr = np.asarray(aspect_limits, dtype=float).reshape(-1)
    if aspect_limit_arr.size != aspect_ratio.size:
        aspect_limit_arr = np.full(aspect_ratio.shape, np.inf, dtype=float)
    valid_limits = np.isfinite(aspect_limit_arr) & (aspect_limit_arr > 0.0)
    delta = np.zeros_like(aspect_ratio, dtype=float)
    delta[valid_limits] = np.maximum(
        0.0,
        (aspect_ratio[valid_limits] - aspect_limit_arr[valid_limits]) / aspect_limit_arr[valid_limits],
    )
    infeasible_mask = delta > 0.0
    d_inf = int(np.sum(infeasible_mask))
    tau_value = max(float(tau), 0.0)
    alpha_value = max(float(alpha), 0.0)
    beta_value = max(float(beta), 0.0)
    phi = np.zeros_like(delta, dtype=float)
    linear_mask = (delta > 0.0) & (delta <= tau_value)
    phi[linear_mask] = alpha_value * delta[linear_mask]
    quadratic_mask = delta > tau_value
    phi[quadratic_mask] = (
        alpha_value * delta[quadratic_mask]
        + beta_value * np.square(delta[quadratic_mask] - tau_value)
    )
    violation_sum = float(np.sum(phi))
    mhc_value = float(mhc)
    if d_inf == 0:
        cost = mhc_value
    else:
        if v_ref is None or not np.isfinite(v_ref) or float(v_ref) <= 0.0:
            ref_cost = mhc_value
        else:
            ref_cost = float(v_ref)
        cost = mhc_value + float(true_cost_penalty_scale) * ref_cost * violation_sum
    return {
        "cost": cost,
        "d_inf": d_inf,
        "infeasible_mask": infeasible_mask,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "aspect_limits": aspect_limits,
        "violation_sum": violation_sum,
        "v_ref": None if v_ref is None else float(v_ref),
        "v_worst": None if v_ref is None else float(v_ref),
    }


def evaluate_layout_true_cost(
    fbs_model: FBSModel,
    area,
    H,
    F,
    fac_limit_aspect,
    v_ref=None,
    v_worst=None,
    true_cost_penalty_scale=0.35,
    tau=0.2,
    alpha=0.7,
    beta=10.0,
    distance_metric="manhattan",
):
    fac_x, fac_y, fac_b, fac_h = getCoordinates_mao(fbs_model, area, H)
    min_side = np.minimum(fac_b, fac_h)
    fac_aspect_ratio = np.divide(
        np.maximum(fac_b, fac_h),
        min_side,
        out=np.full_like(np.asarray(fac_b, dtype=float), np.inf, dtype=float),
        where=min_side > 0,
    )
    if distance_metric == "euclidean":
        D = getEuclideanDistances(fac_x, fac_y)
    else:
        D = getManhattanDistances(fac_x, fac_y)
    TM = getTransportIntensity(D, F)
    mhc = float(np.sum(TM))
    cost_data = calculate_true_cost(
        mhc,
        fac_b,
        fac_h,
        area,
        fac_limit_aspect,
        v_ref=v_ref,
        v_worst=v_worst,
        true_cost_penalty_scale=true_cost_penalty_scale,
        tau=tau,
        alpha=alpha,
        beta=beta,
    )
    return {
        "fac_x": fac_x,
        "fac_y": fac_y,
        "fac_b": fac_b,
        "fac_h": fac_h,
        "fac_aspect_ratio": fac_aspect_ratio,
        "D": D,
        "TM": TM,
        "mhc": mhc,
        "cost": cost_data["cost"],
        "d_inf": cost_data["d_inf"],
        "infeasible_mask": cost_data["infeasible_mask"],
        "lower_bounds": cost_data["lower_bounds"],
        "upper_bounds": cost_data["upper_bounds"],
        "aspect_limits": cost_data["aspect_limits"],
        "violation_sum": cost_data["violation_sum"],
        "is_feasible": cost_data["d_inf"] == 0,
        "v_ref": None if (v_ref is None and v_worst is None) else float(v_ref if v_ref is not None else v_worst),
        "v_worst": None if (v_ref is None and v_worst is None) else float(v_ref if v_ref is not None else v_worst),
    }


def evaluate_layout_fast_true_cost(
    fbs_model: FBSModel,
    area,
    H,
    F,
    fac_limit_aspect,
    v_ref=None,
    v_worst=None,
    true_cost_penalty_scale=0.35,
    tau=0.2,
    alpha=0.7,
    beta=10.0,
    distance_metric="manhattan",
):
    fac_x, fac_y, fac_b, fac_h = getCoordinates_mao_fast(fbs_model, area, H)
    min_side = np.minimum(fac_b, fac_h)
    fac_aspect_ratio = np.divide(
        np.maximum(fac_b, fac_h),
        min_side,
        out=np.full_like(np.asarray(fac_b, dtype=float), np.inf, dtype=float),
        where=min_side > 0,
    )
    if distance_metric == "euclidean":
        D = getEuclideanDistances(fac_x, fac_y)
    else:
        D = getManhattanDistances(fac_x, fac_y)
    TM = getTransportIntensity(D, F)
    mhc = float(np.sum(TM))
    cost_data = calculate_true_cost(
        mhc,
        fac_b,
        fac_h,
        area,
        fac_limit_aspect,
        v_ref=v_ref,
        v_worst=v_worst,
        true_cost_penalty_scale=true_cost_penalty_scale,
        tau=tau,
        alpha=alpha,
        beta=beta,
    )
    return {
        "fac_x": fac_x,
        "fac_y": fac_y,
        "fac_b": fac_b,
        "fac_h": fac_h,
        "fac_aspect_ratio": fac_aspect_ratio,
        "D": D,
        "TM": TM,
        "mhc": mhc,
        "cost": cost_data["cost"],
        "d_inf": cost_data["d_inf"],
        "infeasible_mask": cost_data["infeasible_mask"],
        "lower_bounds": cost_data["lower_bounds"],
        "upper_bounds": cost_data["upper_bounds"],
        "aspect_limits": cost_data["aspect_limits"],
        "violation_sum": cost_data["violation_sum"],
        "is_feasible": cost_data["d_inf"] == 0,
        "v_ref": None if (v_ref is None and v_worst is None) else float(v_ref if v_ref is not None else v_worst),
        "v_worst": None if (v_ref is None and v_worst is None) else float(v_ref if v_ref is not None else v_worst),
    }


def calculate_cost(
    mhc,
    fac_b,
    fac_h,
    area,
    fac_limit_aspect,
    v_ref=None,
    v_worst=None,
    k_penalty=0.35,
    tau=0.2,
    alpha=0.7,
    beta=10.0,
):
    return calculate_true_cost(
        mhc,
        fac_b,
        fac_h,
        area,
        fac_limit_aspect,
        v_ref=v_ref,
        v_worst=v_worst,
        true_cost_penalty_scale=k_penalty,
        tau=tau,
        alpha=alpha,
        beta=beta,
    )


def evaluate_layout(
    fbs_model: FBSModel,
    area,
    H,
    F,
    fac_limit_aspect,
    v_ref=None,
    v_worst=None,
    k_penalty=0.35,
    tau=0.2,
    alpha=0.7,
    beta=10.0,
    distance_metric="manhattan",
):
    return evaluate_layout_true_cost(
        fbs_model,
        area,
        H,
        F,
        fac_limit_aspect,
        v_ref=v_ref,
        v_worst=v_worst,
        true_cost_penalty_scale=k_penalty,
        tau=tau,
        alpha=alpha,
        beta=beta,
        distance_metric=distance_metric,
    )


def evaluate_layout_fast(
    fbs_model: FBSModel,
    area,
    H,
    F,
    fac_limit_aspect,
    v_ref=None,
    v_worst=None,
    k_penalty=0.35,
    tau=0.2,
    alpha=0.7,
    beta=10.0,
    distance_metric="manhattan",
):
    return evaluate_layout_fast_true_cost(
        fbs_model,
        area,
        H,
        F,
        fac_limit_aspect,
        v_ref=v_ref,
        v_worst=v_worst,
        true_cost_penalty_scale=k_penalty,
        tau=tau,
        alpha=alpha,
        beta=beta,
        distance_metric=distance_metric,
    )

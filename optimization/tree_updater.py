# 文件路径: optimization/tree_updater.py

import pandas as pd
import numpy as np # 需要 numpy

# --- 导入重构后的模块 ---
from core.plan_node import PlanNode
from core.onnx_manager import ONNXModelManager
from .threading_utils import get_root_nodes
from config.structure_config import dop_operators_exec # 需要判断哪些算子需要更新时间
# --- 结束导入 ---

# --- 新增一个递归函数来更新估计输入 ---
def update_tree_estimated_inputs(node: PlanNode):
    """
    自底向上递归地更新树中每个节点的估计输入行数。
    """
    # 先递归到最深的子节点
    for child in node.child_plans:
        update_tree_estimated_inputs(child)
    
    # 从子节点返回后，更新当前节点的估计输入
    node.update_estimated_inputs()
    
    # Preserve plan fields such as join type, filters and aggregation metadata.
    node.get_feature_data(node.plan_data)
    if node.onnx_manager is not None:
        node.infer_exec_with_onnx()
        node.infer_mem_with_onnx()

def build_query_trees(df_plans_base_dop, onnx_manager, use_estimates=False): # <--- 在这里添加参数
    """
    根据基准 DOP (例如 query_dop=8) 的 DataFrame 构建查询计划树。
    (逻辑基本来自原 dop_choosen.py 的 build_query_trees)
    """
    query_groups = df_plans_base_dop.groupby(['query_id', 'query_dop']) # 按基准DOP分组
    query_trees = {} # 存储基准树 {query_id: [PlanNode list]}

    for (query_id, query_dop), group in query_groups:
        key = query_id # 使用 query_id 作为键
        query_trees[key] = []
        nodes_in_query = {} # 用于快速查找节点

        # 1. 创建所有节点 (传递 use_estimates 开关)
        for _, row in group.iterrows():
            # Estimated inputs are complete only after the children are attached.
            initial_manager = None if use_estimates else onnx_manager
            plan_node = PlanNode(row.to_dict(), initial_manager, use_estimates=use_estimates)
            plan_node.onnx_manager = onnx_manager
            query_trees[key].append(plan_node)
            nodes_in_query[plan_node.plan_id] = plan_node

        # 按 plan_id 排序
        query_trees[key] = sorted(query_trees[key], key=lambda x: x.plan_id)

        # 2. 处理父子关系
        for _, row in group.iterrows():
            plan_id = row.get('plan_id')
            if plan_id not in nodes_in_query: continue

            parent_plan = nodes_in_query[plan_id]
            child_plan_str = row['child_plan']
            if pd.isna(child_plan_str) or not str(child_plan_str).strip(): continue

            try:
                child_plan_ids = [int(pid) for pid in str(child_plan_str).split(',')]
                for child_id in child_plan_ids:
                    if child_id in nodes_in_query:
                        child_node = nodes_in_query[child_id]
                        parent_plan.add_child(child_node)
                    # else: 原始代码可能没处理找不到子节点的情况
            except (ValueError, TypeError):
                 print(f"构建基准树时处理 child_plan '{child_plan_str}' 出错 (Query: {query_id}, Plan: {plan_id})")
            except Exception as e:
                 print(f"构建基准树时处理父子关系未知错误 (Parent: {plan_id}): {e}")

     # --- 如果是模拟模式，增加一个更新步骤 ---
    if use_estimates:
        print("模拟模式：正在自底向上更新节点的估计输入行数...")
        for qid, nodes_list in query_trees.items():
            root_nodes = get_root_nodes(nodes_list)
            for root in root_nodes:
                update_tree_estimated_inputs(root)

    return query_trees


def update_dop_information(df_plans_all_dops, query_trees, onnx_manager, base_dop=8):
    """
    使用所有 DOP 的运行数据更新基准查询树中节点的信息。
    (严格按照原始 dop_choosen.py 的 update_dop_plan_nodes 逻辑适配)
    """
    print("开始更新节点 DOP 信息...")

    # 1. 筛选出非基准 DOP 的数据
    df_other_dops = df_plans_all_dops[df_plans_all_dops['query_dop'] != base_dop].copy()
    # --- 修改：存储原始行数据和 PlanNode ---
    # other_dop_nodes_map = {} # {(query_id, query_dop): [PlanNode list]}
    # 改为存储原始 row 数据，键是 (query_id, query_dop)
    other_dop_data_map = {} # {(query_id, query_dop): [pd.Series list]}
    print(f"  处理 {len(df_other_dops)} 条非基准 DOP 数据...")

    for index, row in df_other_dops.iterrows(): # 获取 row Series
        key = (row['query_id'], row['query_dop']) # key 包含 query_dop
        if key not in other_dop_data_map:
            other_dop_data_map[key] = []
        # --- 只存储原始行数据 ---
        other_dop_data_map[key].append(row)
        # --- 不再需要创建临时的 PlanNode ---
        # try:
        #     temp_node = PlanNode(row, onnx_manager) # 不在这里创建
        #     other_dop_nodes_map[key].append(temp_node)
        # except Exception as e:
        #     print(f"创建临时 PlanNode 出错...")
        #     continue
    # --- 结束修改 ---


    # 2. 遍历基准查询树，用其他 DOP 的数据更新节点的执行时间 map
    updated_queries = 0
    # --- 修改：遍历 other_dop_data_map 的键来获取 query_id ---
    all_query_ids = set(k[0] for k in other_dop_data_map.keys())
    for query_id in all_query_ids:
    # --- 结束修改 ---
        base_nodes = query_trees.get(query_id) # 从基准树字典获取节点列表
        if not base_nodes: continue

        # 获取该 query_id 下所有其他 DOP 的原始数据行
        relevant_other_rows = []
        for (qid, qdop), rows in other_dop_data_map.items():
            if qid == query_id:
                relevant_other_rows.extend(rows)

        if not relevant_other_rows: continue

        other_estimates = {
            (row['query_dop'], row['plan_id']): row['estimate_rows']
            for row in relevant_other_rows
        } if base_nodes[0].use_estimates else {}
        used_other_row_indices = set() # 使用索引来跟踪用过的行

        num_base_nodes = len(base_nodes)
        for base_node in base_nodes:
            if base_node.operator_type not in dop_operators_exec: continue # 保持不变

            match_found_for_base_node = False
            # --- 修改：遍历原始数据行 relevant_other_rows ---
            for other_row in relevant_other_rows:
                other_row_idx = other_row.name # 获取行的索引
                if other_row_idx in used_other_row_indices: continue

                # --- 从 row 中直接获取信息 ---
                other_operator_type = other_row['operator_type']
                other_plan_id = other_row['plan_id']
                # --- 结束 ---

                if other_operator_type != base_node.operator_type: continue

                matched = False
                # --- 修改：从 key 中获取 query_dop ---
                other_query_id = other_row['query_id']
                other_query_dop = other_row['query_dop'] # 直接从行获取
                num_other_nodes_for_dop = len(other_dop_data_map.get((other_query_id, other_query_dop), []))
                # --- 结束修改 ---

                # 方式1 & 2: Plan ID 相同且节点数量匹配
                if other_plan_id == base_node.plan_id and num_base_nodes == num_other_nodes_for_dop:
                    matched = True

                # 方式3: 特征匹配 (需要创建临时的 PlanNode 或直接比较 row 数据)
                # --- 修改：如果需要特征匹配，需要创建临时 other_node ---
                if not matched:
                     # Build comparable features without running a temporary ONNX inference.
                     temp_other_node = None
                     try:
                         temp_other_node = PlanNode(other_row, None, use_estimates=base_node.use_estimates)
                         child_plan = other_row['child_plan']
                         if base_node.use_estimates and pd.notna(child_plan) and str(child_plan).strip():
                             child_estimates = [
                                 other_estimates.get((other_query_dop, int(child_id)))
                                 for child_id in str(child_plan).split(',')
                             ]
                             temp_other_node.update_estimated_inputs(child_estimates)
                             temp_other_node.get_feature_data(temp_other_node.plan_data)
                     except Exception as e_tmp:
                         print(f"特征匹配时创建临时节点出错: {e_tmp}")

                     if temp_other_node and base_node.exec_feature_data is not None and temp_other_node.exec_feature_data is not None:
                        key_features_indices = [0, 1, 2] # 假设索引
                        try:
                            base_vals = np.array(base_node.exec_feature_data)[key_features_indices]
                            other_vals = np.array(temp_other_node.exec_feature_data)[key_features_indices]
                            if np.array_equal(base_vals, other_vals):
                                matched = True
                        except IndexError: pass
                        except Exception as e_comp: print(f"比较特征时出错: {e_comp}")
                # --- 结束修改 ---


                if matched:
                    # --- 从 row 获取 dop, updop, execution_time ---
                    other_dop = other_row['dop']
                    other_updop = other_row['up_dop']
                    other_execution_time = other_row['execution_time']
                    other_send_time = other_row['stream_data_send_time']
                    other_build_time = other_row['build_time']
                    # --- 结束 ---

                    actual_dop = other_dop
                    # 原始逻辑考虑 updop
                    if pd.notna(other_updop) and other_updop > actual_dop:
                        actual_dop = other_updop

                    base_node.update_real_exec_time(actual_dop, other_execution_time)
                    base_node.update_send_time(actual_dop, other_send_time, other_build_time)
                    used_other_row_indices.add(other_row_idx) # 使用行索引标记
                    match_found_for_base_node = True
                    # break # 原始逻辑可能是找到一个就 break

        updated_queries += 1

    print(f"... {updated_queries} 个查询的节点信息已更新。")

    # 3. 对每个基准查询树，调用 compute_parallel_dop_predictions (保持不变)
    print("开始计算并行预测和插值...")
    calculated_queries = 0
    for query_id, base_nodes in query_trees.items():
        if not base_nodes: continue
        root_nodes = get_root_nodes(base_nodes)
        if not root_nodes: continue
        try:
            for root in root_nodes:
                 if isinstance(root, PlanNode) and hasattr(root, 'compute_parallel_dop_predictions'):
                     root.compute_parallel_dop_predictions()
        except Exception as e:
            print(f"  Query {query_id} 计算并行预测时出错: {e}")
            import traceback
            traceback.print_exc()
        calculated_queries += 1
    print(f"... {calculated_queries} 个查询的并行预测和插值计算完成。")

    return query_trees

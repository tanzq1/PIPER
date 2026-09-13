# 文件路径: scripts/run_non_dop_aware_training.py

import sys
import os
import pandas as pd

# --- 步骤 1: 将项目根目录添加到 sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"将项目根目录添加到 sys.path: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print("项目根目录已添加到 sys.path。")
else:
    print("项目根目录已在 sys.path 中。")

# --- 步骤 2: 导入需要的训练函数 ---
try:
    # 从 training 包的 non_dop_aware 子包中导入主训练函数
    from training.operator_non_dop_aware.train import train_all_operators
    # 如果需要 utils 或 config 中的其他功能，也可以在这里导入
    # 例如: from utils.helpers import some_function
except ImportError as e:
    print(f"导入错误: {e}")
    print("\n请检查:")
    print(f"1. 项目结构是否正确，并且 '{PROJECT_ROOT}' 确实是你的项目根目录。")
    print(f"2. training/operator_non_dop_aware/train.py 文件是否存在且包含 train_all_operators 函数。")
    print(f"3. 相关的 __init__.py 文件是否存在于各级包目录中。")
    sys.exit(1)

# --- 步骤 3: 定义输入数据路径 ---
# (请根据你的实际数据存放位置修改这些示例路径)
DATA_DIR = os.path.join(PROJECT_ROOT, "data_kunpeng")
# 假设训练和测试数据来自不同的文件集
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "tpcds_100g_output") # 示例训练数据目录
TEST_DATA_DIR = os.path.join(DATA_DIR, "tpcds_100g_output")   # 示例测试数据目录

train_plan_info_csv = os.path.join(TRAIN_DATA_DIR, "plan_info.csv")
test_plan_info_csv = os.path.join(TEST_DATA_DIR, "plan_info.csv")

# 检查输入文件是否存在
if not os.path.exists(train_plan_info_csv):
    print(f"错误: 训练数据文件未找到: {train_plan_info_csv}")
    sys.exit(1)
if not os.path.exists(test_plan_info_csv):
    print(f"错误: 测试数据文件未找到: {test_plan_info_csv}")
    sys.exit(1)

# --- 步骤 4: 定义训练参数 (如果需要的话) ---
# 这些参数可能需要传递给 train_all_operators，或者由 train_all_operators 内部处理
total_queries = 500 # 示例：如果需要指定查询总数
train_ratio = 1   # 示例：如果需要指定训练比例（但看起来你的训练函数内部处理了测试集）

# --- 步骤 5: 主逻辑函数 ---
def main():
    """执行非 DOP 感知算子训练"""
    print(f"开始加载训练数据: {train_plan_info_csv}")
    train_data = pd.read_csv(train_plan_info_csv, delimiter=';', encoding='utf-8')
    print("训练数据加载完成。")

    print(f"开始加载测试数据: {test_plan_info_csv}")
    test_data = pd.read_csv(test_plan_info_csv, delimiter=';', encoding='utf-8')
    print("测试数据加载完成。")

    print("\n开始训练所有非 DOP 感知算子模型...")
    # 注意：这里的参数需要与你实际的 train_all_operators 函数签名匹配
    # 看起来原函数需要 train_data, test_data, total_queries, train_ratio
    # 如果你的 train_all_operators 现在只需要 train_data 和 test_data，则相应修改调用
    train_all_operators(
        train_data=train_data,
        test_data=test_data,
        total_queries=total_queries, # 如果需要传递
        train_ratio=train_ratio     # 如果需要传递
    )
    print("\n所有非 DOP 感知算子模型训练完成。")
    print(f"检查输出目录 'output/models/operator_non_dop_aware/' 和 'output/predictions/', 'output/evaluations/' 查看结果。")

# --- 步骤 6: 脚本入口点 ---
if __name__ == "__main__":
    main()
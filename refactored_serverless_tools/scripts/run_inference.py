# 文件路径: scripts/run_inference.py

import sys
import os
import pandas as pd # 保留导入，即使在此脚本中可能不直接使用

# --- 步骤 1: 将项目根目录添加到 sys.path ---
# 获取当前脚本文件(__file__)所在的目录 (即 scripts 目录)
scripts_dir = os.path.dirname(os.path.abspath(__file__)) # 使用 abspath 确保路径完整
# 获取 scripts 目录的上级目录，即项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(scripts_dir, '..'))
print(f"将项目根目录添加到 sys.path: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) # 插入到开头，优先搜索项目内模块
    print("项目根目录已添加到 sys.path。")
else:
    print("项目根目录已在 sys.path 中。")

# --- 步骤 2: 导入需要的推理函数 ---
try:
    # 从 inference 包中导入主推理函数
    # 假设你将 'predict_non_dop_queries.py' 文件名改为了 'predict_queries.py'
    from inference.predict_queries import run_inference
except ImportError as e:
    print(f"导入错误: {e}")
    print("\n请检查:")
    print(f"1. 项目结构是否正确，并且 '{PROJECT_ROOT}' 确实是你的项目根目录。")
    print(f"2. inference/predict_queries.py 文件是否存在且包含 run_inference 函数。")
    print(f"3. 相关的 __init__.py 文件是否存在于各级包目录中 (inference/ 等)。")
    sys.exit(1)
except FileNotFoundError: # 如果 inference/predict_queries.py 文件本身不存在
    print(f"错误: 推理文件 'inference/predict_queries.py' 未找到。")
    sys.exit(1)


# --- 步骤 3: 定义输入数据路径 ---
# (请根据你的实际数据存放位置修改这些示例路径)
DATA_DIR = os.path.join(PROJECT_ROOT, "data_kunpeng")
# 假设使用 tpch_output_22 作为测试集 (与原始 no_dop_inference.py 一致)
TEST_DATA_DIR = os.path.join(DATA_DIR, "tpcds_100g_output")

plan_csv_path = os.path.join(TEST_DATA_DIR, "plan_info.csv")
query_csv_path = os.path.join(TEST_DATA_DIR, "query_info.csv")

# 检查输入文件是否存在
if not os.path.exists(plan_csv_path):
    print(f"错误: 输入 plan 文件未找到: {plan_csv_path}")
    sys.exit(1)
if not os.path.exists(query_csv_path):
    print(f"错误: 输入 query 文件未找到: {query_csv_path}")
    sys.exit(1)

# --- 步骤 4: 定义输出文件路径 ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
# 将结果放在 predictions 目录下
PREDICTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "predictions")
os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True) # 确保目录存在

# 定义输出 CSV 文件名 (沿用原始文件名)
output_csv_filename = "test_queries_results.csv"
output_csv_path = os.path.join(PREDICTION_OUTPUT_DIR, output_csv_filename)

# --- 步骤 5: 定义模型目录路径 ---
# 因为 ONNXModelManager 的 __init__ 现在需要这些路径
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
no_dop_model_dir_path = os.path.join(MODEL_DIR, "operator_non_dop_aware")
dop_model_dir_path = os.path.join(MODEL_DIR, "operator_dop_aware")

# 可选：检查模型目录是否存在，给出警告
if not os.path.exists(no_dop_model_dir_path):
     print(f"警告: 非 DOP 感知模型目录未找到: {no_dop_model_dir_path}")
if not os.path.exists(dop_model_dir_path):
     print(f"警告: DOP 感知模型目录未找到: {dop_model_dir_path}")
     print("推理可能无法加载所有需要的模型。")


# --- 步骤 6: 主逻辑函数 ---
def main():
    """执行查询推理"""
    print("\n开始执行查询推理...")

    # 调用 run_inference 函数，并传入所有必需的路径参数
    try:
        run_inference(
            plan_csv_path=plan_csv_path,
            query_csv_path=query_csv_path,
            output_csv_path=output_csv_path,
            no_dop_model_dir=no_dop_model_dir_path, # 传递模型路径
            dop_model_dir=dop_model_dir_path      # 传递模型路径
        )
        print("\n查询推理完成。")
        print(f"结果已保存到: {output_csv_path}")
    except Exception as e:
        print(f"\n运行 run_inference 时发生错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细的回溯信息


# --- 步骤 7: 脚本入口点 ---
if __name__ == "__main__":
    main()
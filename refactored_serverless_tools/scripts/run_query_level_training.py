# 文件路径: scripts/run_query_level_training.py

import sys
import os

# --- 添加这里：将项目根目录添加到 sys.path ---
# __file__ 是当前脚本(run_query_level_training.py)的路径
# os.path.dirname(__file__) 是 scripts 目录
# os.path.join(..., '..') 是 scripts 的上级目录，也就是项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"将项目根目录添加到 sys.path: {PROJECT_ROOT}") # 打印出来方便调试
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) # 使用 insert(0, ...) 优先搜索项目内模块
    print("项目根目录已添加到 sys.path。")
else:
    print("项目根目录已在 sys.path 中。")
# --- 添加结束 ---

# --- 现在应该可以安全地使用绝对导入了 ---
try:
    # 从 training 包中导入需要的函数 (这些导入本身已经是绝对路径)
    from training.query_level.train import (
        train_and_save_xgboost_onnx,
        test_onnx_xgboost,
        compute_qerror_by_bins
    )
    # 导入其他需要的模块
    # from config.structure import some_config_variable # 示例
    # from utils.helpers import some_helper_function   # 示例

except ImportError as e:
    print(f"导入错误: {e}")
    print("\n请检查:")
    print(f"1. 项目结构是否正确，并且 '{PROJECT_ROOT}' 确实是你的项目根目录。")
    print(f"2. 目标模块（例如 utils/feature_engineering.py）是否存在。")
    print(f"3. 相关的 __init__.py 文件是否存在于各级包目录中（config/, utils/, training/, training/query_level/ 等）。")
    sys.exit(1) # 导入失败则退出程序

# --- 定义文件路径 (这部分保持不变) ---
# 获取当前脚本文件(__file__)所在的目录 (即 scripts 目录)
# scripts_dir = os.path.dirname(__file__) # 不再需要，因为上面已经用了
# 获取 scripts 目录的上级目录，即项目根目录
# PROJECT_ROOT = os.path.abspath(os.path.join(scripts_dir, '..')) # 不再需要

# 定义数据和输出目录路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data_kunpeng")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models", "query_level")
PREDICTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "predictions")
EVALUATION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "evaluations")

# 确保输出目录存在
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)

# 定义具体的输入输出文件路径 (示例，请根据实际情况修改)
feature_csv = os.path.join(DATA_DIR, "tpch_output_500", "plan_info.csv")
true_val_csv = os.path.join(DATA_DIR, "tpch_output_500", "query_info.csv")
test_feature_csv = os.path.join(DATA_DIR, "tpch_output_22", "plan_info.csv")
test_execution_csv = os.path.join(DATA_DIR, "tpch_output_22", "query_info.csv")

execution_onnx_path = os.path.join(MODEL_OUTPUT_DIR, "execution_time_model.onnx")
memory_onnx_path = os.path.join(MODEL_OUTPUT_DIR, "memory_usage_model.onnx")
prediction_output_file = os.path.join(PREDICTION_OUTPUT_DIR, "query_level_test_predictions.csv")
qerror_output_file = os.path.join(EVALUATION_OUTPUT_DIR, "query_level_qerror_stats.csv")

target_dop = 96

# --- 主逻辑函数 (这部分保持不变) ---
def main():
    """封装主要的执行流程"""
    print("开始训练 Query-Level 模型...")
    # train_and_save_xgboost_onnx(
    #     feature_csv=feature_csv,
    #     true_val_csv=true_val_csv,
    #     execution_onnx_path=execution_onnx_path,
    #     memory_onnx_path=memory_onnx_path,
    #     n_trials=30
    # )
    # print(f"训练完成。模型已保存到: {MODEL_OUTPUT_DIR}")

    print("开始测试和评估...")
    results_df = test_onnx_xgboost(
        execution_onnx_path=execution_onnx_path,
        memory_onnx_path=memory_onnx_path,
        feature_csv=test_feature_csv,
        true_val_csv=test_execution_csv,
        output_file=prediction_output_file
    )
    print(f"测试完成。预测结果已保存到: {prediction_output_file}")

    print("开始计算 Q-error 统计...")
    compute_qerror_by_bins(results_df, qerror_output_file, target_dop)
    print(f"Q-error 计算完成。统计结果已保存到: {qerror_output_file}")

# --- 脚本入口点 (这部分保持不变) ---
if __name__ == "__main__":
    main()
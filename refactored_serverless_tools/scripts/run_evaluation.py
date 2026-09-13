# 文件路径: scripts/run_evaluation.py

import sys
import os
import pandas as pd # 保留导入，可能用于后续检查

# --- 步骤 1: 将项目根目录添加到 sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"将项目根目录添加到 sys.path: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print("项目根目录已添加到 sys.path。")
else:
    print("项目根目录已在 sys.path 中。")

# --- 步骤 2: 导入需要的评估函数 ---
try:
    # 从 evaluation 包中导入主评估函数
    from evaluation.evaluate_predictions import generate_qerror_report
except ImportError as e:
    print(f"导入错误: {e}")
    print("\n请检查:")
    print(f"1. 项目结构是否正确，并且 '{PROJECT_ROOT}' 确实是你的项目根目录。")
    print(f"2. evaluation/evaluate_predictions.py 文件是否存在且包含 generate_qerror_report 函数。")
    print(f"3. 相关的 __init__.py 文件是否存在于各级包目录中 (evaluation/ 等)。")
    sys.exit(1)
except FileNotFoundError:
     print(f"错误: 评估文件 'evaluation/evaluate_predictions.py' 未找到。")
     sys.exit(1)

# --- 步骤 3: 定义输入预测结果文件路径 ---
# 该文件应该是由 run_inference.py 或其他预测脚本生成的
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")

# 指定要评估的预测结果文件名
# 假设我们评估由 run_inference.py 生成的文件
predictions_csv_filename = "tpcds_pre.csv" # 或者 "non_dop_inference_results.csv"
predictions_csv_path = os.path.join(PREDICTION_DIR, predictions_csv_filename)

# 检查输入文件是否存在
if not os.path.exists(predictions_csv_path):
    print(f"错误: 输入的预测结果文件未找到: {predictions_csv_path}")
    print("请先运行相应的推理脚本 (例如: scripts/run_inference.py) 来生成此文件。")
    sys.exit(1)

# --- 步骤 4: 定义输出评估报告文件路径 ---
EVALUATION_DIR = os.path.join(OUTPUT_DIR, "evaluations")
os.makedirs(EVALUATION_DIR, exist_ok=True) # 确保目录存在

# 定义输出文件名
output_stats_filename = "qerror_report.csv" # 或者更具体，如 non_dop_qerror_report.csv
output_stats_csv_path = os.path.join(EVALUATION_DIR, output_stats_filename)

# --- 步骤 5: 定义评估参数 ---
# 指定要针对哪个 Query DOP 进行评估 (与 tmp.py 保持一致)
target_dop_for_evaluation = 96

# --- 步骤 6: 主逻辑函数 ---
def main():
    """执行 Q-error 评估"""
    print(f"\n开始执行 Q-error 评估...")
    print(f"  输入预测文件: {predictions_csv_path}")
    print(f"  输出报告文件: {output_stats_csv_path}")
    print(f"  评估目标 DOP: {target_dop_for_evaluation}")

    try:
        # 调用评估函数
        generate_qerror_report(
            predictions_csv_path=predictions_csv_path,
            output_stats_csv_path=output_stats_csv_path,
            target_dop=target_dop_for_evaluation
        )
        print(f"\nQ-error 评估完成。")
        print(f"评估报告已保存到: {output_stats_csv_path}")
    except Exception as e:
        print(f"\n运行 generate_qerror_report 时发生错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细的回溯信息

# --- 步骤 7: 脚本入口点 ---
if __name__ == "__main__":
    main()
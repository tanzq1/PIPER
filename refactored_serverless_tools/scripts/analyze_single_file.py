#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def analyze_single_file():
    """分析单个文件的训练时间"""
    file_path = r'D:\研究生文件\项目\python\refactored_serverless_tools_cht\output\tpch\evaluations\estimated_dop_aware\all_operators_performance_exec.csv'
    
    try:
        # 读取文件
        df = pd.read_csv(file_path)
        print(f"文件: {file_path}")
        print("=" * 60)
        
        # 检查是否有训练时间列
        if 'Training Time (s)' in df.columns:
            training_times = df['Training Time (s)'].dropna()
            
            print(f"操作符数量: {len(training_times)}")
            print(f"总训练时间: {training_times.sum():.2f} 秒")
            print(f"平均训练时间: {training_times.mean():.2f} 秒")
            print(f"中位数训练时间: {training_times.median():.2f} 秒")
            print(f"最小训练时间: {training_times.min():.2f} 秒")
            print(f"最大训练时间: {training_times.max():.2f} 秒")
            print(f"标准差: {training_times.std():.2f} 秒")
            
            print("\n各操作符训练时间详情:")
            print("-" * 50)
            for _, row in df.iterrows():
                operator = row['Operator']
                training_time = row['Training Time (s)']
                print(f"{operator:35} {training_time:8.2f} 秒")
            
            print(f"\n总训练时间: {training_times.sum():.2f} 秒")
            print(f"总训练时间: {training_times.sum()/60:.2f} 分钟")
            print(f"总训练时间: {training_times.sum()/3600:.2f} 小时")
            
        else:
            print("文件中没有找到 'Training Time (s)' 列")
            print("可用的列名:")
            for col in df.columns:
                print(f"  - {col}")
                
    except Exception as e:
        print(f"读取文件失败: {e}")

if __name__ == "__main__":
    analyze_single_file() 
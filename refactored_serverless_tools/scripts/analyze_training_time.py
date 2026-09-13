#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import glob

def load_training_costs(file_path):
    """加载训练成本数据"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Failed to load data: {file_path}, Error: {e}")
        return None

def analyze_operator_training_times(base_dir):
    """分析操作符级别的训练时间"""
    operator_files = []
    
    # 查找所有操作符性能文件
    patterns = [
        '*/all_operators_performance_exec.csv',
        '*/all_operators_performance_mem.csv'
    ]
    
    for pattern in patterns:
        files = glob.glob(os.path.join(base_dir, pattern))
        operator_files.extend(files)
    
    all_training_times = []
    
    for file_path in operator_files:
        try:
            df = pd.read_csv(file_path)
            if 'Training Time (s)' in df.columns:
                # 提取训练时间数据
                training_times = df['Training Time (s)'].dropna()
                all_training_times.extend(training_times.tolist())
                
                print(f"File: {os.path.basename(file_path)}")
                print(f"  Operators: {len(training_times)}")
                print(f"  Total training time: {training_times.sum():.2f} seconds")
                print(f"  Average training time: {training_times.mean():.2f} seconds")
                print(f"  Min training time: {training_times.min():.2f} seconds")
                print(f"  Max training time: {training_times.max():.2f} seconds")
                print()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return all_training_times

def analyze_model_training_times(tpch_file, tpcds_file):
    """分析模型级别的训练时间"""
    print("=" * 60)
    print("MODEL-LEVEL TRAINING TIME ANALYSIS")
    print("=" * 60)
    
    # 加载数据
    tpch_data = load_training_costs(tpch_file)
    tpcds_data = load_training_costs(tpcds_file)
    
    if tpch_data is not None:
        print("\nTPC-H Model Training Times:")
        print("-" * 40)
        for _, row in tpch_data.iterrows():
            model_name = row['model_name']
            training_time = row['training_time_seconds']
            print(f"{model_name:35} {training_time:8.2f} seconds")
        
        # 计算TPC-H平均训练时间
        tpch_avg = tpch_data['training_time_seconds'].mean()
        print(f"\nTPC-H Average Training Time: {tpch_avg:.2f} seconds")
    
    if tpcds_data is not None:
        print("\nTPC-DS Model Training Times:")
        print("-" * 40)
        for _, row in tpcds_data.iterrows():
            model_name = row['model_name']
            training_time = row['training_time_seconds']
            print(f"{model_name:35} {training_time:8.2f} seconds")
        
        # 计算TPC-DS平均训练时间
        tpcds_avg = tpcds_data['training_time_seconds'].mean()
        print(f"\nTPC-DS Average Training Time: {tpcds_avg:.2f} seconds")
    
    # 计算总体平均
    if tpch_data is not None and tpcds_data is not None:
        all_times = pd.concat([tpch_data['training_time_seconds'], 
                              tpcds_data['training_time_seconds']])
        overall_avg = all_times.mean()
        print(f"\nOverall Average Training Time: {overall_avg:.2f} seconds")
        
        # 按模型类型分组统计
        print("\n" + "=" * 60)
        print("TRAINING TIME BY MODEL TYPE")
        print("=" * 60)
        
        # 提取模型类型
        def extract_model_type(model_name):
            if 'query_level' in model_name:
                return 'Query-Level'
            elif 'PPM-GNN' in model_name:
                return 'PPM-GNN'
            elif 'PPM-NN' in model_name:
                return 'PPM-NN'
            else:
                return 'Other'
        
        # 合并数据并添加模型类型
        combined_data = pd.concat([
            tpch_data.assign(dataset='TPC-H'),
            tpcds_data.assign(dataset='TPC-DS')
        ])
        combined_data['model_type'] = combined_data['model_name'].apply(extract_model_type)
        
        # 按模型类型统计
        model_type_stats = combined_data.groupby('model_type')['training_time_seconds'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        
        print("\nTraining Time Statistics by Model Type:")
        print(model_type_stats)
        
        # 按数据集和模型类型统计
        dataset_model_stats = combined_data.groupby(['dataset', 'model_type'])['training_time_seconds'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        
        print("\nTraining Time Statistics by Dataset and Model Type:")
        print(dataset_model_stats)

def main():
    """主函数"""
    print("TRAINING TIME ANALYSIS")
    print("=" * 60)
    
    # 文件路径
    tpch_training_file = 'output/tpch/evaluations/training_costs_summary.csv'
    tpcds_training_file = 'output/tpcds/evaluations/training_costs_summary.csv'
    
    # 分析模型级别训练时间
    analyze_model_training_times(tpch_training_file, tpcds_training_file)
    
    # 分析操作符级别训练时间
    print("\n" + "=" * 60)
    print("OPERATOR-LEVEL TRAINING TIME ANALYSIS")
    print("=" * 60)
    
    print("\nTPC-H Operator Training Times:")
    print("-" * 40)
    tpch_operator_times = analyze_operator_training_times('output/tpch/evaluations')
    
    print("\nTPC-DS Operator Training Times:")
    print("-" * 40)
    tpcds_operator_times = analyze_operator_training_times('output/tpcds/evaluations')
    
    # 计算操作符级别总体统计
    all_operator_times = tpch_operator_times + tpcds_operator_times
    if all_operator_times:
        print("\n" + "=" * 60)
        print("OVERALL OPERATOR TRAINING TIME STATISTICS")
        print("=" * 60)
        print(f"Total operators: {len(all_operator_times)}")
        print(f"Total training time: {sum(all_operator_times):.2f} seconds")
        print(f"Average training time per operator: {np.mean(all_operator_times):.2f} seconds")
        print(f"Median training time per operator: {np.median(all_operator_times):.2f} seconds")
        print(f"Min training time: {min(all_operator_times):.2f} seconds")
        print(f"Max training time: {max(all_operator_times):.2f} seconds")
        print(f"Standard deviation: {np.std(all_operator_times):.2f} seconds")
        
        # 计算分位数
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            value = np.percentile(all_operator_times, p)
            print(f"  {p}th percentile: {value:.2f} seconds")

if __name__ == "__main__":
    main() 
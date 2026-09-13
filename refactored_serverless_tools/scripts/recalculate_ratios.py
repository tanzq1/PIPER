#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def recalculate_cost_and_ratios():
    """重新计算cost值（执行时间*线程数）和所有比率"""
    file_path = 'output/evaluations/tpcds_graph.csv'
    
    try:
        # 读取文件
        df = pd.read_csv(file_path, sep=';')
        print(f"原始文件: {file_path}")
        print(f"数据行数: {len(df)}")
        print("=" * 60)
        
        # 创建新的DataFrame来存储修正后的数据
        df_corrected = df.copy()
        
        # 首先重新计算base_cost
        print("重新计算base_cost:")
        print("-" * 30)
        df_corrected['base_cost'] = df_corrected['base_execution_time'] * df_corrected['base_thread_count']
        print("  base_cost: 已重新计算 (base_execution_time * base_thread_count)")
        
        # 显示一些base_cost的示例对比
        print(f"\n  base_cost示例对比 (前5行):")
        for i in range(min(5, len(df))):
            original_base_cost = df.iloc[i]['base_cost']
            corrected_base_cost = df_corrected.iloc[i]['base_cost']
            print(f"    行 {i+1}: base_cost {original_base_cost:.2f} -> {corrected_base_cost:.2f}")
        
        # 定义需要计算的方法
        methods = ['Auto_DOP', 'PPM', 'PIO', 'POO']
        
        # 重新计算每个方法的cost和比率
        for method in methods:
            time_col = f'{method}_execution_time'
            thread_col = f'{method}_thread_count'
            cost_col = f'{method}_cost'
            time_ratio_col = f'{method}_time_ratio'
            cost_ratio_col = f'{method}_cost_ratio'
            
            print(f"\n重新计算 {method} 的cost和比率:")
            print("-" * 50)
            
            # 检查列是否存在
            if time_col in df.columns and thread_col in df.columns:
                # 重新计算cost: execution_time * thread_count
                df_corrected[cost_col] = df_corrected[time_col] * df_corrected[thread_col]
                
                # 计算time_ratio: method_time / base_time
                df_corrected[time_ratio_col] = df_corrected[time_col] / df_corrected['base_execution_time']
                
                # 计算cost_ratio: method_cost / base_cost
                df_corrected[cost_ratio_col] = df_corrected[cost_col] / df_corrected['base_cost']
                
                # 处理缺失值（当method数据为空时，设置为1）
                df_corrected[time_ratio_col] = df_corrected[time_ratio_col].fillna(1.0)
                df_corrected[cost_ratio_col] = df_corrected[cost_ratio_col].fillna(1.0)
                
                print(f"  {cost_col}: 已重新计算 (执行时间 * 线程数)")
                print(f"  {time_ratio_col}: 已重新计算")
                print(f"  {cost_ratio_col}: 已重新计算")
                
                # 显示一些示例对比
                print(f"\n  示例对比 (前5行):")
                for i in range(min(5, len(df))):
                    original_cost = df.iloc[i][cost_col] if pd.notna(df.iloc[i][cost_col]) else 'N/A'
                    corrected_cost = df_corrected.iloc[i][cost_col]
                    original_cost_ratio = df.iloc[i][cost_ratio_col] if pd.notna(df.iloc[i][cost_ratio_col]) else 'N/A'
                    corrected_cost_ratio = df_corrected.iloc[i][cost_ratio_col]
                    
                    print(f"    行 {i+1}: {cost_col} {original_cost} -> {corrected_cost:.2f}")
                    print(f"           {cost_ratio_col} {original_cost_ratio} -> {corrected_cost_ratio:.6f}")
            else:
                print(f"  警告: 找不到 {time_col} 或 {thread_col} 列")
        
        # 保存修正后的文件
        output_path = 'output/evaluations/tpcds_graph_cost_corrected.csv'
        df_corrected.to_csv(output_path, sep=';', index=False)
        print(f"\n修正后的文件已保存至: {output_path}")
        
        # 显示一些统计信息
        print("\n" + "=" * 60)
        print("修正后的比率统计:")
        print("=" * 60)
        
        for method in methods:
            time_ratio_col = f'{method}_time_ratio'
            cost_ratio_col = f'{method}_cost_ratio'
            
            if time_ratio_col in df_corrected.columns:
                time_ratios = df_corrected[time_ratio_col].dropna()
                cost_ratios = df_corrected[cost_ratio_col].dropna()
                
                print(f"\n{method} 方法:")
                print(f"  Time Ratio - 平均: {time_ratios.mean():.4f}, 中位数: {time_ratios.median():.4f}")
                print(f"  Cost Ratio - 平均: {cost_ratios.mean():.4f}, 中位数: {cost_ratios.median():.4f}")
        
        return df_corrected
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None

def verify_cost_calculations(df_corrected):
    """验证cost计算是否正确"""
    print("\n" + "=" * 60)
    print("验证cost计算:")
    print("=" * 60)
    
    # 选择几个示例进行验证
    sample_queries = [3, 7, 13, 22, 25]
    
    for query_id in sample_queries:
        row = df_corrected[df_corrected['query_id'] == query_id]
        if len(row) > 0:
            row = row.iloc[0]
            expected_base_cost = row['base_execution_time'] * row['base_thread_count']
            actual_base_cost = row['base_cost']
            
            print(f"\nQuery {query_id}:")
            print(f"  基线时间: {row['base_execution_time']:.2f}, 基线线程: {row['base_thread_count']}")
            print(f"  基线成本: {actual_base_cost:.2f} (期望: {expected_base_cost:.2f})")
            
            methods = ['Auto_DOP', 'PPM', 'PIO', 'POO']
            for method in methods:
                time_col = f'{method}_execution_time'
                thread_col = f'{method}_thread_count'
                cost_col = f'{method}_cost'
                time_ratio_col = f'{method}_time_ratio'
                cost_ratio_col = f'{method}_cost_ratio'
                
                if pd.notna(row[time_col]) and pd.notna(row[thread_col]):
                    expected_cost = row[time_col] * row[thread_col]
                    actual_cost = row[cost_col]
                    expected_time_ratio = row[time_col] / row['base_execution_time']
                    expected_cost_ratio = actual_cost / actual_base_cost
                    actual_time_ratio = row[time_ratio_col]
                    actual_cost_ratio = row[cost_ratio_col]
                    
                    print(f"  {method}:")
                    print(f"    时间: {row[time_col]:.2f}, 线程: {row[thread_col]}")
                    print(f"    成本: {actual_cost:.2f} (期望: {expected_cost:.2f})")
                    print(f"    时间比率: {actual_time_ratio:.6f} (期望: {expected_time_ratio:.6f})")
                    print(f"    成本比率: {actual_cost_ratio:.6f} (期望: {expected_cost_ratio:.6f})")

def compare_original_vs_corrected():
    """比较原始数据和修正后的数据"""
    print("\n" + "=" * 60)
    print("原始数据 vs 修正后数据对比:")
    print("=" * 60)
    
    # 读取原始文件
    original_df = pd.read_csv('output/evaluations/tpcds_graph.csv', sep=';')
    corrected_df = pd.read_csv('output/evaluations/tpcds_graph_cost_corrected.csv', sep=';')
    
    methods = ['Auto_DOP', 'PPM', 'PIO', 'POO']
    
    for method in methods:
        cost_col = f'{method}_cost'
        cost_ratio_col = f'{method}_cost_ratio'
        
        print(f"\n{method} 方法:")
        print("-" * 30)
        
        # 计算差异
        cost_diff = corrected_df[cost_col] - original_df[cost_col]
        cost_ratio_diff = corrected_df[cost_ratio_col] - original_df[cost_ratio_col]
        
        print(f"  Cost差异统计:")
        print(f"    平均差异: {cost_diff.mean():.2f}")
        print(f"    最大差异: {cost_diff.max():.2f}")
        print(f"    最小差异: {cost_diff.min():.2f}")
        
        print(f"  Cost Ratio差异统计:")
        print(f"    平均差异: {cost_ratio_diff.mean():.6f}")
        print(f"    最大差异: {cost_ratio_diff.max():.6f}")
        print(f"    最小差异: {cost_ratio_diff.min():.6f}")

if __name__ == "__main__":
    # 重新计算cost和比率
    df_corrected = recalculate_cost_and_ratios()
    
    if df_corrected is not None:
        # 验证计算
        verify_cost_calculations(df_corrected)
        
        # 比较原始数据和修正后的数据
        compare_original_vs_corrected()
        
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60) 
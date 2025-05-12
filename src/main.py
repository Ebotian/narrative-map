#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
from narrativeProcessor import NarrativeProcessor
from networkGeneratorWithPyVis import SemanticNetworkVisualizer
from utils import simplifyNarrativeTuples, filterSpecificEntityTpyes, filterNarrativeByTupleFrequency, filterNarrativeByNodeFrequency, mapEntityNames, filterExcludeSpecificEntities
import json
from datetime import datetime


def main():
    """
    主函数，处理命令行参数并执行叙事网络分析流程
    """
    parser = argparse.ArgumentParser(description='叙事网络分析工具')

    # 基本参数
    parser.add_argument('--input', '-i', required=True,
                        help='输入文件路径 (.txt, .csv, .xlsx)')
    parser.add_argument(
        '--output',
        '-o',
        default='../data/narrative_network.json',
        help='输出JSON文件路径')
    parser.add_argument('--text-column', default='text', help='文本列名')
    parser.add_argument('--time-column', default='time', help='时间列名')

    # 时间筛选参数
    parser.add_argument('--start-time', help='开始时间 (格式: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end-time', help='结束时间 (格式: YYYY-MM-DD HH:MM:SS)')

    # 分句和聚类参数
    parser.add_argument('--no-slice', action='store_true', help='不分句')
    parser.add_argument(
        '--cluster-algo',
        choices=[
            'HDBSCAN',
            'KMeans'],
        default='HDBSCAN',
        help='聚类算法 (默认: HDBSCAN)')
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=10,
        help='HDBSCAN最小聚类大小')
    parser.add_argument(
        '--min-samples',
        type=int,
        default=5,
        help='HDBSCAN最小样本数')
    parser.add_argument('--min-k', type=int, default=5, help='KMeans最小K值')
    parser.add_argument(
        '--no-auto-k',
        action='store_true',
        help='不自动选择K值 (KMeans)')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"开始处理文件: {args.input}")

    # 初始化NarrativeProcessor
    processor = NarrativeProcessor(
        if_silice=(not args.no_slice),
        cluster_algorithm=args.cluster_algo,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        min_k=args.min_k,
        if_auto_k=(not args.no_auto_k)
    )

    # 处理数据
    print("开始提取叙事三元组...")
    start_time = datetime.now()
    result = processor.process(
        file_path=args.input,
        text_column=args.text_column,
        time_column=args.time_column
    )
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    print(f"叙事三元组提取完成，共计{len(result)}条，耗时{processing_time:.2f}秒")

    # 获取时间范围
    min_time, max_time = processor.get_time_range(result)
    if min_time and max_time:
        print(f"数据时间范围: {min_time} 至 {max_time}")

    # 根据参数过滤时间范围
    filter_start_time = args.start_time
    filter_end_time = args.end_time

    # 转换为JSON并保存
    print("开始生成JSON文件...")
    json_data = processor.to_json(
        result,
        start_time=filter_start_time,
        end_time=filter_end_time,
        output_path=args.output
    )

    print(f"JSON数据已保存到: {args.output}")
    print("处理完成！请使用narrativeNetwork.html查看可视化结果")

    return 0


if __name__ == "__main__":
    sys.exit(main())

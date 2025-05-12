#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import jieba
import gensim
import numpy as np
import os
import sys
import json
from prettytable import PrettyTable
from nonNEsKMeansMapper import nonNEsKMeansMapper


def compare_label_extraction(test_data, word_vectors=None):
    """
    比较不同标签提取方法的效果

    参数:
        test_data: 测试数据，字典形式，键为簇ID，值为该簇中的短语列表
        word_vectors: 词向量字典，可以为None
    """
    # 创建原始标签提取器
    original_mapper = nonNEsKMeansMapper({})

    # 分别使用不同的标签提取方法
    original_labels = original_mapper._extract_original_labels(test_data)
    tfidf_labels = original_mapper.extract_tfidf_labels(test_data)

    # 如果有词向量
    if word_vectors:
        repr_labels = original_mapper.select_representative_label(
            test_data, word_vectors)
        hybrid_labels = original_mapper.extract_hybrid_labels(
            test_data, word_vectors)
        optimized_labels = original_mapper.optimize_cluster_labels(
            test_data, word_vectors)
    else:
        # 模拟简单的词向量（实际使用中应该加载真实的词向量）
        simple_word_vectors = {}
        print("警告: 未提供词向量，使用模拟词向量进行测试，结果可能不准确")
        repr_labels = {"模拟": "模拟数据 - 需要提供真实词向量"}
        hybrid_labels = {"模拟": "模拟数据 - 需要提供真实词向量"}
        optimized_labels = {"模拟": "模拟数据 - 需要提供真实词向量"}

    # 创建漂亮的表格输出
    table = PrettyTable()
    table.field_names = [
        "簇ID",
        "包含短语",
        "原始标签",
        "TF-IDF标签",
        "代表性标签",
        "混合标签",
        "优化标签"]

    # 添加数据行
    for cluster_id in test_data.keys():
        phrases_str = "\n".join(test_data[cluster_id][:3])
        if len(test_data[cluster_id]) > 3:
            phrases_str += f"\n...等{len(test_data[cluster_id])}个"

        row = [
            cluster_id,
            phrases_str,
            original_labels.get(
                cluster_id,
                "N/A"),
            tfidf_labels.get(
                cluster_id,
                "N/A"),
            repr_labels.get(
                cluster_id,
                "N/A") if word_vectors else "需要词向量",
            hybrid_labels.get(
                cluster_id,
                "N/A") if word_vectors else "需要词向量",
            optimized_labels.get(
                cluster_id,
                "N/A") if word_vectors else "需要词向量"]
        table.add_row(row)

    # 设置表格样式
    table.align = "l"
    table.max_width = 30

    return table


def main():
    """主函数"""
    # 示例测试数据
    test_data = {
        0: ["美国媒体", "人们", "清醒头脑", "战争不可避免", "事实并非如此"],
        1: ["加剧乌克兰紧张局势", "乌克兰边境局势", "两国边境地区", "乌东部边境地区"],
        2: ["双方", "部署大量军事人员和武器装备", "集结重兵", "入侵之势"],
        3: ["1月31日", "1月28日晚", "近几个月", "近几天来", "当天", "同时"]
    }

    print("=== 簇标签提取方法对比测试 ===")

    # 加载词向量模型
    try:
        import gensim
        print("正在加载词向量模型，这可能需要一些时间...")
        model_path = "/home/ebit/LMZ/src/model/sgns.wiki.word"
        word_vector_model = gensim.models.KeyedVectors.load_word2vec_format(
            model_path, binary=False, encoding='utf-8')

        # 创建词向量字典
        word_vectors = {}
        # 为测试数据中的所有词提取向量
        all_words = set()
        for phrases in test_data.values():
            for phrase in phrases:
                words = jieba.lcut(phrase)
                all_words.update(words)

        # 填充词向量字典
        for word in all_words:
            if word in word_vector_model:
                word_vectors[word] = word_vector_model[word]

        print(f"成功加载了 {len(word_vectors)} 个词的向量")
        table = compare_label_extraction(test_data, word_vectors)
    except Exception as e:
        print(f"加载词向量模型失败: {e}")
        print("将使用无词向量模式进行测试...")
        table = compare_label_extraction(test_data)

    print(table)

    print("\n提示：使用真实词向量可以获得更准确的标签提取效果。")
    print("在完整系统中，可以这样调用优化方法：")
    print("mapper = nonNEsKMeansMapper(input_data=data_dict, label_method='optimized')")
    print("result = mapper.process()")


if __name__ == "__main__":
    main()

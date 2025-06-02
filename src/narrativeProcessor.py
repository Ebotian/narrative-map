# -*- coding: utf-8 -*-
import pandas as pd
import json
import os
from datetime import datetime
from SRL import ChineseSRL
from NER import ChineseNER
from sliceSentences import ChineseSentenceSegmenter
from nonNEsKMeansMapper import nonNEsKMeansMapper
from nonNEsHDBSCANMapper import nonNEsHDBSCANMapper
from tqdm import tqdm
from utils import get_most_important_word


class NarrativeProcessor:
    def __init__(
        self,
        if_silice=True,
        cluster_algorithm='HDBSCAN',
        min_cluster_size=10,
        min_samples=5,
        min_k=5,
        if_auto_k=True,
    ):
        self.if_silice = if_silice
        self.cluster_algorithm = nonNEsHDBSCANMapper if cluster_algorithm == 'HDBSCAN' else 'K-Means'
        # HDBSCAN参数
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        # K-Means参数
        self.min_k = min_k
        self.if_auto_k = if_auto_k

    # 检查命名实体是否在参数中
    @staticmethod
    def isEntityInArgument(entity_text, argument_text):
        return entity_text in argument_text

    # 将命名实体映射到SRL的参数中
    def mappingNEs(self, ner_data, srl_data):
        from tqdm import tqdm
        results = {}

        for idx in tqdm(ner_data, desc="实体词映射中", total=len(ner_data)):
            sentence_ner = ner_data[idx]
            sentence_srl = srl_data.get(idx, [])
            original_text = sentence_ner['original_text']
            entities = sentence_ner['entities']
            timeline = sentence_ner['time']

            # 初始化结果
            sentence_result = {
                'original_text': original_text,
                'entities': entities,
                'time': timeline,
                'srl_mappings': []
            }

            # 处理每个SRL结果
            for srl_parse in sentence_srl:
                mapped_srl = []

                for item in srl_parse:
                    text, role, start, end = item

                    # 只映射ARG0和ARG1
                    if role in ['ARG0', 'ARG1']:
                        # 检查是否有命名实体在这个参数中
                        for entity in entities:
                            if self.isEntityInArgument(entity['text'], text):
                                # 替换为命名实体
                                mapped_srl.append([entity['text'], role, {
                                    'original': text,
                                    'entity_type': entity['type']
                                }])
                                break
                        else:
                            # 如果没有找到匹配的命名实体，保持原样
                            mapped_srl.append([text, role, None])
                    else:
                        # 非ARG0/ARG1的角色保持不变
                        mapped_srl.append([text, role, None])

                sentence_result['srl_mappings'].append(mapped_srl)

            results[idx] = sentence_result

        return results

    # 提取三元组，过滤只包含ARG0, ARG1和PRED的结果
    def filterMappingResults(self, mapResult):
        from tqdm import tqdm
        filtered_results = {}

        for idx, sentence_data in tqdm(mapResult.items(), desc="三元组过滤中", total=len(mapResult)):
            filtered_srl_mappings = []

            for mapping in sentence_data['srl_mappings']:
                # 检查当前映射是否包含ARG0, ARG1和PRED
                roles = [item[1] for item in mapping]
                if 'ARG0' in roles and 'ARG1' in roles and 'PRED' in roles:
                    filtered_srl_mappings.append(mapping)

            # 只有在过滤后仍有映射的情况下才添加到结果中
            if filtered_srl_mappings:
                filtered_result = sentence_data.copy()
                filtered_result['srl_mappings'] = filtered_srl_mappings
                filtered_results[idx] = filtered_result

        return filtered_results

    # 主流程
    def process(self, file_path=None, text_column='text', time_column='time'):
        # 读取不同输入格式的数据
        if isinstance(file_path, pd.DataFrame):
            data_df = file_path
        else:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_lines = file.readlines()
                    data_df = pd.DataFrame(text_lines, columns=['text'])
            elif file_path.endswith('.csv'):
                data_df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                data_df = pd.read_excel(file_path)
            elif isinstance(file_path, pd.DataFrame):
                data_df = file_path
            else:
                print("文件格式不支持！(仅支持txt、csv、xlsx)")
                return None
        print(f"读取数据成功，数据框大小: {data_df.shape}")
        if self.if_silice:
            # 检查是否有时间列
            if time_column not in data_df.columns:
                print(f"数据框中没有找到时间列: {time_column}")
                return None
            # 创建分句器
            segmenter = ChineseSentenceSegmenter()

            # 用于存储分句结果的列表
            all_sentences = []

            # 对数据框中的每一行进行处理
            for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="分句中"):
                text = row[text_column]
                time_info = row[time_column] if time_column in data_df.columns else None

                # 对当前文档进行分句
                sentences = segmenter.segment_sentences(text)

                # 将每个分句与原始时间信息一起存储
                for sent in sentences:
                    all_sentences.append({
                        'original_index': idx,
                        'sentence': sent,
                        'time': time_info
                    })

            # 将分句结果转换为DataFrame
            sentences_df = pd.DataFrame(all_sentences)
            print(f"分句成功，分句数量: {len(sentences_df)}")
        else:
            all_sentences = []
            # 如果不需要分句，直接使用原始数据框
            for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="收集句子"):
                all_sentences.append({
                    'original_index': idx,
                    'sentence': row[text_column],
                    'time': row[time_column] if time_column in data_df.columns else None
                })
            sentences_df = pd.DataFrame(all_sentences)
            print('共计{}条句子'.format(len(sentences_df)))

        # 分别进行NER和SRL步骤
        print("命名实体识别进程开始！")
        # 如需进度条，请在 NER.py 的 analyze 方法内部实现 tqdm
        ner_data = ChineseNER().analyze(
            sentences_df,
            text_column='sentence',
            time_column='time')
        print(f"命名实体识别进程完成！")
        print("语义角色标注进程开始！")
        # 如需进度条，请在 SRL.py 的 analyze 方法内部实现 tqdm
        srl_data = ChineseSRL().analyze(sentences_df, text_column='sentence')
        print(f"语义角色标注进程完成！")

        # 执行映射
        print("实体词映射进程开始！")
        mapResult = self.mappingNEs(ner_data, srl_data)  # 先映射实体词
        print(f"实体词映射完成！")
        print("三元组过滤进程开始！")
        res = self.filterMappingResults(mapResult)  # 过滤出同时包含ARG0, ARG1和PRED的结果
        print(f"过滤出三元组{len(res)}条！")
        if self.cluster_algorithm == 'K-Means':
            # 使用K-Means聚类算法
            mapResult = nonNEsKMeansMapper(
                input_data=res, min_k=self.min_k, if_auto_k=self.if_auto_k)
        else:
            # 使用HDBSCAN聚类算法
            mapResult = nonNEsHDBSCANMapper(
                input_data=res,
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples)
        print("非实体词降维与聚类进程开始！")
        processedData = mapResult.process()  # 非实体词降维，映射未映射到命名实体的ARG0和ARG1参数
        print(f"非实体词映射且降维完成！")

        # 提取叙事三元组：ARG0, PRED, ARG1, timeline
        narrative_triples = {}
        i = 0
        print("三元组提取进程开始！")
        for idx, sentence_data in tqdm(processedData.items(), desc="三元组提取", total=len(processedData)):
            timeline = sentence_data['time']
            for srl_mapping in sentence_data['srl_mappings']:
                role_dict = {}
                role_dict['timeline'] = timeline
                for item in srl_mapping:
                    text, role, ner_info = item
                    # 优先用实体，否则分词选关键词
                    if role in ['ARG0', 'PRED', 'ARG1']:
                        if ner_info and 'entity_type' in ner_info and ner_info['entity_type'] != 'non-NE':
                            node_text = text
                            node_info = ner_info
                        else:
                            # 分词选关键词
                            node_text = get_most_important_word(text)
                            node_info = {'entity_type': 'non-NE'}
                        role_dict[role] = (node_text, node_info)
                narrative_triples[i] = role_dict
                i += 1
        print("三元组提取完成！")
        return narrative_triples

    # 将narrative_triples转换为JSON格式，支持时间筛选
    def to_json(
            self,
            narrative_triples,
            start_time=None,
            end_time=None,
            output_path=None):
        """
        将narrative_triples转换为JSON格式，可选择按时间范围筛选

        参数:
            narrative_triples: 处理后的叙事三元组
            start_time: 开始时间，格式为'YYYY-MM-DD HH:MM:SS'，为None则不限制开始时间
            end_time: 结束时间，格式为'YYYY-MM-DD HH:MM:SS'，为None则不限制结束时间
            output_path: 输出JSON文件的路径，为None则不保存文件

        返回:
            JSON格式的字符串
        """
        # 处理时间筛选
        filtered_triples = {}

        if start_time or end_time:
            start_dt = datetime.strptime(
                start_time, '%Y-%m-%d %H:%M:%S') if start_time else None
            end_dt = datetime.strptime(
                end_time, '%Y-%m-%d %H:%M:%S') if end_time else None

            for idx, triple in narrative_triples.items():
                if 'timeline' in triple:
                    try:
                        triple_time = datetime.strptime(
                            triple['timeline'], '%Y-%m-%d %H:%M:%S')

                        if start_dt and triple_time < start_dt:
                            continue
                        if end_dt and triple_time > end_dt:
                            continue

                        filtered_triples[idx] = triple
                    except (ValueError, TypeError):
                        # 如果时间解析失败，保留该条目
                        filtered_triples[idx] = triple
                else:
                    # 如果没有时间信息，保留该条目
                    filtered_triples[idx] = triple
        else:
            filtered_triples = narrative_triples

        # 生成节点和边
        nodes = {}
        edges = {}  # 改为字典，以便跟踪同一条边的多个出现
        edge_counts = {}  # 用于统计边的频次

        # 记录节点和边的时间信息
        node_timestamps = {}  # 存储每个节点的所有出现时间
        edge_timestamps = {}  # 存储每条边的所有出现时间

        for idx, triple in filtered_triples.items():
            # 获取当前三元组的时间
            timeline = triple.get('timeline')

            # 处理ARG0节点
            if 'ARG0' in triple:
                arg0_text, arg0_info = triple['ARG0']
                node_type = arg0_info['entity_type'] if arg0_info and 'entity_type' in arg0_info else 'non-NE'

                # 记录节点的所有时间戳
                if arg0_text not in node_timestamps:
                    node_timestamps[arg0_text] = []

                if timeline and timeline not in node_timestamps[arg0_text]:
                    node_timestamps[arg0_text].append(timeline)

                if arg0_text not in nodes:
                    nodes[arg0_text] = {
                        'id': arg0_text,
                        'label': arg0_text,
                        'value': 1,
                        'type': node_type,
                        # 使用列表存储多个时间戳
                        'timestamps': [timeline] if timeline else []
                    }
                else:
                    nodes[arg0_text]['value'] += 1
                    if timeline and timeline not in nodes[arg0_text]['timestamps']:
                        nodes[arg0_text]['timestamps'].append(timeline)

            # 处理ARG1节点
            if 'ARG1' in triple:
                arg1_text, arg1_info = triple['ARG1']
                node_type = arg1_info['entity_type'] if arg1_info and 'entity_type' in arg1_info else 'non-NE'

                # 记录节点的所有时间戳
                if arg1_text not in node_timestamps:
                    node_timestamps[arg1_text] = []

                if timeline and timeline not in node_timestamps[arg1_text]:
                    node_timestamps[arg1_text].append(timeline)

                if arg1_text not in nodes:
                    nodes[arg1_text] = {
                        'id': arg1_text,
                        'label': arg1_text,
                        'value': 1,
                        'type': node_type,
                        # 使用列表存储多个时间戳
                        'timestamps': [timeline] if timeline else []
                    }
                else:
                    nodes[arg1_text]['value'] += 1
                    if timeline and timeline not in nodes[arg1_text]['timestamps']:
                        nodes[arg1_text]['timestamps'].append(timeline)

            # 处理边
            if 'ARG0' in triple and 'ARG1' in triple and 'PRED' in triple:
                arg0_text = triple['ARG0'][0]
                arg1_text = triple['ARG1'][0]
                pred_text = triple['PRED'][0]

                edge_key = f"{arg0_text}_{arg1_text}_{pred_text}"

                # 记录边的所有时间戳
                if edge_key not in edge_timestamps:
                    edge_timestamps[edge_key] = []

                if timeline and timeline not in edge_timestamps[edge_key]:
                    edge_timestamps[edge_key].append(timeline)

                # 统计边的出现次数
                if edge_key in edge_counts:
                    edge_counts[edge_key] += 1
                else:
                    edge_counts[edge_key] = 1

                # 记录边的信息，包括所有时间戳
                if edge_key not in edges:
                    edges[edge_key] = {
                        'from': arg0_text,
                        'to': arg1_text,
                        'label': pred_text,
                        # 使用列表存储多个时间戳
                        'timestamps': [timeline] if timeline else []
                    }
                else:
                    if timeline and timeline not in edges[edge_key]['timestamps']:
                        edges[edge_key]['timestamps'].append(timeline)

        # 更新节点标签，添加频次
        for node_id, node_data in nodes.items():
            node_data['label'] = f"{node_id}({node_data['value']})"

        # 根据边的频次构建边列表
        edges_list = []
        for edge_key, edge_data in edges.items():
            count = edge_counts[edge_key]
            edge_data['label'] = f"{edge_data['label']}({count})"
            # 确保边的时间戳是按时间顺序排序的
            edge_data['timestamps'] = sorted(
                edge_data['timestamps']) if edge_data['timestamps'] else []
            edges_list.append(edge_data)

        # 获取数据中的时间范围，用于前端自动设置时间轴端点
        all_times = []

        for node_id, timestamps in node_timestamps.items():
            for ts in timestamps:
                if ts:
                    try:
                        all_times.append(
                            datetime.strptime(
                                ts, '%Y-%m-%d %H:%M:%S'))
                    except (ValueError, TypeError):
                        pass

        for edge_key, timestamps in edge_timestamps.items():
            for ts in timestamps:
                if ts:
                    try:
                        all_times.append(
                            datetime.strptime(
                                ts, '%Y-%m-%d %H:%M:%S'))
                    except (ValueError, TypeError):
                        pass

        data_start_time = min(all_times).strftime(
            '%Y-%m-%d %H:%M:%S') if all_times else None
        data_end_time = max(all_times).strftime(
            '%Y-%m-%d %H:%M:%S') if all_times else None

        # 构建最终的JSON对象
        result = {
            'nodes': list(nodes.values()),
            'edges': edges_list,
            'timeRange': {
                'start': data_start_time,
                'end': data_end_time
            }
        }

        # 保存到文件
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"JSON数据已保存到: {output_path}")

        return json.dumps(result, ensure_ascii=False, indent=2)

    # 获取数据的时间范围
    def get_time_range(self, narrative_triples):
        """
        获取narrative_triples数据中的时间范围

        参数:
            narrative_triples: 处理后的叙事三元组

        返回:
            包含最早和最晚时间的元组 (min_time, max_time)
        """
        times = []
        for idx, triple in narrative_triples.items():
            if 'timeline' in triple and triple['timeline']:
                try:
                    triple_time = datetime.strptime(
                        triple['timeline'], '%Y-%m-%d %H:%M:%S')
                    times.append(triple_time)
                except (ValueError, TypeError):
                    pass

        if times:
            return min(times).strftime(
                '%Y-%m-%d %H:%M:%S'), max(times).strftime('%Y-%m-%d %H:%M:%S')
        return None, None


def main():
    processor = NarrativeProcessor(
        if_silice=False,
        cluster_algorithm='K-Means',
        min_cluster_size=5,
        min_samples=2)
    result = processor.process(file_path=r'/home/ebit/narrative-map/data/test.xlsx')

    # 获取时间范围
    start_time, end_time = processor.get_time_range(result)
    print(f"数据时间范围: {start_time} 至 {end_time}")


    # 将结果转换为JSON并保存
    json_data = processor.to_json(
        result,
        # start_time="2022-03-01 00:00:00",  # 可以设置时间范围
        # end_time="2022-03-31 23:59:59",    # 可以设置时间范围
        output_path="./data/narrative_network.json"
    )

    print("JSON数据生成完成，可供前端解析与绘图")


if __name__ == "__main__":
    main()

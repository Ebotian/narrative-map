# -*- coding: utf-8 -*-
from narrativeProcessor import NarrativeProcessor
from networkGeneratorWithPyVis import SemanticNetworkVisualizer
from utils import simplifyNarrativeTuples, filterSpecificEntityTpyes, filterNarrativeByTupleFrequency, filterNarrativeByNodeFrequency, mapEntityNames, filterExcludeSpecificEntities
import json

tuples_path = ''
output_file = "narrativeNetwork.html"


def main():
    # Pt.1 加载数据、处理数据
    # 这一部分目前的主要问题在于SRL质量不高，后续考虑自己训练一个基于BERT的SRL模型（25.4.18 by MoonRiver）
    if not tuples_path:
        # 统一了接口（推荐使用UMAP+HDBSCAN，不需要手动设定簇数量）
        processor = NarrativeProcessor(
            if_silice=False,
            cluster_algorithm='HDBSCAN',
            min_cluster_size=5,
            min_samples=2)
        # 从Excel文件读取数据并处理（也支持csv、txt）
        narrative_triples = processor.process(file_path='./data/test.xlsx')
        # 因计算开销非常大，故增加了防中断机制
        with open('./temp.json', 'w', encoding='utf-8') as f:
            json.dump(narrative_triples, f, ensure_ascii=False, indent=4)
        print('已将narrative_triples临时保存到当前路径')
    else:
        # 从指定路径读取保存的三元组数据
        with open(tuples_path, 'r', encoding='utf-8') as file:
            # 将JSON数据加载为Python字典
            narrative_triples = json.load(file)
        print(f'从{tuples_path}加载了narrative_triples数据')

    # Pt.2 用户自定义部分
    # 支持过滤特定实体、无实义词、频率过低的三元组
    # 这部分后续考虑支持时间序列前台展示（25.4.18 by MoonRiver）
    filteredList = filterSpecificEntityTpyes(
        narrative_triples, entity_types=[
            "ORGANIZATION", "PERSON", "LOCATION"])  # 过滤特定类型实体词
    # 将复杂的三元组结构简化为渲染页面所需要的简化格式
    simpliedNarratives = simplifyNarrativeTuples(filteredList)
    print('叙事三元组:', simpliedNarratives)
    simpliedNarratives = filterNarrativeByTupleFrequency(
        simpliedNarratives, min_frequency=2)  # 根据三元组频率过滤
    simpliedNarratives = mapEntityNames(
        simpliedNarratives, entity_mapping={})  # 映射自定义实体名称
    simpliedNarratives = filterExcludeSpecificEntities(
        simpliedNarratives, exclude_entities=[])  # 剔除部分实体词
    print('过滤后的叙事三元组:', simpliedNarratives)

    # Pt.3 生成叙事网络
    # 支持给不同类型实体词不同的颜色等属性
    # 这部分主要问题在于交互性不够、前台展示效果单一，后续考虑加入时序过滤（即鼠标悬浮到时间序列轴上查看不同事件阶段的叙事结构），同时支持Gephi导出（25.4.18
    # by MoonRiver）
    builder = SemanticNetworkVisualizer()
    builder.generate_network(
        input_data=simpliedNarratives,
        output_file=output_file)
    print(f'构建网络成功！文件名是：{output_file}')


if __name__ == '__main__':
    main()

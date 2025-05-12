import pandas as pd

# 将嵌套的三元组列表转换为简化格式的字典


def simplifyNarrativeTuples(narrative_tuples):
    simplified_tuples = {}
    index_counter = 0

    for key, tuple in narrative_tuples.items():
        simple_tuple = {
            'ARG0': tuple['ARG0'][0],
            'ARG0_type': tuple['ARG0'][1]['entity_type'] if tuple['ARG0'][1] else 'non-NE',
            'PRED': tuple['PRED'][0],
            'ARG1': tuple['ARG1'][0],
            'ARG1_type': tuple['ARG1'][1]['entity_type'] if tuple['ARG1'][1] else 'non-NE',
            'timeline': tuple['timeline']}
        simplified_tuples[index_counter] = simple_tuple
        index_counter += 1

    return simplified_tuples

# 筛选出特定类型实体的叙事列表


def filterSpecificEntityTpyes(narrativeList, entity_types=[]):
    filtered_tuples = {}

    if not entity_types:
        return narrativeList

    for key, lst in narrativeList.items():
        try:
            if lst['ARG0'][1]['entity_type'] in entity_types or lst['ARG1'][1]['entity_type'] in entity_types:
                filtered_tuples[key] = lst
        except BaseException:
            continue
    return filtered_tuples

# 按三元组频次过滤


def filterNarrativeByTupleFrequency(narrativeList, min_frequency=2):
    tuple_frequency = {}
    for key, tuple_data in narrativeList.items():
        tuple_key = (
            tuple_data['ARG0'],
            tuple_data['PRED'],
            tuple_data['ARG1'])
        if tuple_key in tuple_frequency:
            tuple_frequency[tuple_key] += 1
        else:
            tuple_frequency[tuple_key] = 1

    df = pd.DataFrame([
        {'ARG0': k[0], 'PRED': k[1], 'ARG1': k[2], 'Frequency': v}
        for k, v in tuple_frequency.items()
    ])
    df.to_excel('tuple_frequency.xlsx', index=False)

    # 根据频次筛选三元组
    filtered_tuples = {}
    index_counter = 0

    for key, tuple_data in narrativeList.items():
        tuple_key = (
            tuple_data['ARG0'],
            tuple_data['PRED'],
            tuple_data['ARG1'])
        if tuple_frequency[tuple_key] >= min_frequency:
            filtered_tuples[index_counter] = tuple_data
            index_counter += 1

    return filtered_tuples

# 按节点(ARG0、ARG1)频次过滤


def filterNarrativeByNodeFrequency(narrativeList, min_frequency=2):
    node_frequency = {}

    # 统计每个节点出现的频率
    for key, tuple_data in narrativeList.items():
        # 统计ARG0节点
        arg0 = tuple_data['ARG0']
        if arg0 in node_frequency:
            node_frequency[arg0] += 1
        else:
            node_frequency[arg0] = 1

        # 统计ARG1节点
        arg1 = tuple_data['ARG1']
        if arg1 in node_frequency:
            node_frequency[arg1] += 1
        else:
            node_frequency[arg1] = 1

    # 保存节点频率到Excel
    node_df = pd.DataFrame([
        {'Node': k, 'Frequency': v}
        for k, v in node_frequency.items()
    ])
    node_df.to_excel('node_frequency.xlsx', index=False)

    # 根据节点频率筛选三元组
    filtered_tuples = {}
    index_counter = 0

    for key, tuple_data in narrativeList.items():
        arg0 = tuple_data['ARG0']
        arg1 = tuple_data['ARG1']

        # 如果ARG0或ARG1的频率大于等于阈值，保留该三元组
        if node_frequency[arg0] >= min_frequency or node_frequency[arg1] >= min_frequency:
            filtered_tuples[index_counter] = tuple_data
            index_counter += 1

    return filtered_tuples

# 过滤掉包含指定ARG0或ARG1的三元组


def filterExcludeSpecificEntities(narrativeList, exclude_entities=[]):
    if not exclude_entities:
        return narrativeList

    filtered_tuples = {}
    index_counter = 0

    for key, tuple_data in narrativeList.items():
        arg0 = tuple_data['ARG0']
        arg1 = tuple_data['ARG1']

        # 如果ARG0和ARG1都不在排除列表中，则保留该三元组
        if arg0 not in exclude_entities and arg1 not in exclude_entities:
            filtered_tuples[index_counter] = tuple_data
            index_counter += 1

    return filtered_tuples

# 修改叙事列表中的实体名称，通过映射字典实现


def mapEntityNames(narrativeList, entity_mapping={}):
    if not entity_mapping:
        return narrativeList

    mapped_tuples = {}

    for key, tuple_data in narrativeList.items():
        # 创建元组数据的深拷贝，避免修改原始数据
        new_tuple = tuple_data.copy()

        # 映射ARG0
        arg0 = tuple_data['ARG0']
        if arg0 in entity_mapping:
            new_tuple['ARG0'] = entity_mapping[arg0]

        # 映射ARG1
        arg1 = tuple_data['ARG1']
        if arg1 in entity_mapping:
            new_tuple['ARG1'] = entity_mapping[arg1]

        # 保存修改后的三元组
        mapped_tuples[key] = new_tuple

    return mapped_tuples

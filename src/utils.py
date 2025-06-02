import pandas as pd
import hanlp
import os

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

# 加载停用词表
STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), '../hit_stopwords.txt')
if os.path.exists(STOPWORDS_PATH):
    with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        STOPWORDS = set([line.strip() for line in f if line.strip()])
else:
    STOPWORDS = set()

# HanLP分词和词性标注
tokenizer = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ERNIE_GRAM_ZH)

def get_most_important_word(text):
    """
    对文本分词，优先返回第一个非停用词的名词/动词，否则返回第一个非停用词，否则返回最长的单个词（不返回原文整句）。
    """
    if not isinstance(text, str) or not text.strip():
        return text
    try:
        result = tokenizer(text)
        words = result['tok/fine'] if 'tok/fine' in result else result['tok/coarse']
        pos = result['pos'] if 'pos' in result else None
        # 优先名词/动词且非停用词
        if pos:
            for w, p in zip(words, pos):
                if (p.startswith('N') or p.startswith('V')) and w not in STOPWORDS and len(w) <= 8:
                    return w
        # 其次非停用词且长度合理
        for w in words:
            if w not in STOPWORDS and len(w) <= 8:
                return w
        # 否则返回所有分词中最长且不超过12字的词
        candidates = [w for w in words if len(w) <= 12]
        if candidates:
            return max(candidates, key=len)
        # 若全为超长词，返回第一个词的前12字
        if words:
            return words[0][:12]
    except Exception as e:
        # 分词失败则返回原文前12字
        return text[:12]
    return text[:12]

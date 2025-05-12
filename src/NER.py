import hanlp
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Union


class ChineseNER:
    def __init__(self, top_n: int = 1000):
        # 或者CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH
        self.recognizer = hanlp.load(
            hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ERNIE_GRAM_ZH)

        self.top_n = top_n
        self.entity_counter = Counter()

    def analyze(self, data: pd.DataFrame, text_column: str = 'text',
                time_column: str = 'time') -> Dict[int, Dict[str, Any]]:
        """
        从DataFrame中提取实体

        Args:
            data: 包含文本数据的DataFrame
            text_column: 要处理的文本列名，默认为'text'

        Returns:
            包含实体提取结果的字典
        """
        results = {}
        all_entities = []

        # 确保text_column在DataFrame中
        if text_column not in data.columns:
            raise ValueError(f"文本列 '{text_column}' 在DataFrame中不存在")

        # 处理DataFrame中的文本并提取实体
        for idx, row in data.iterrows():
            text = row[text_column]
            timeline = row[time_column] if time_column in data.columns else None
            # 确保timeline是字符串类型
            if timeline is not None and not isinstance(timeline, str):
                timeline = str(timeline)
            if idx not in results:
                results[idx] = {}
            ner_result = self.recognizer(
                str(text))['ner/msra']  # 使用MSRA数据集进行NER
            entities = []

            for entity in ner_result:
                entity_text = entity[0]
                entity_type = entity[1]
                self.entity_counter[entity_text] += 1
                entities.append({
                    "text": entity_text,
                    "type": entity_type,
                })
                all_entities.append(entity_text)

            results[idx].update({
                "original_text": text,
                "time": timeline,
                "entities": entities
            })

        # 用频率筛选前N个实体
        if self.top_n > 0:
            top_entities = set(
                entity for entity,
                _ in self.entity_counter.most_common(
                    self.top_n))

            # 只保留前N个实体
            for idx in results:
                results[idx]["entities"] = [
                    entity for entity in results[idx]["entities"]
                    if entity["text"] in top_entities
                ]

        return results

    def get_entity_frequencies(self) -> Dict[str, int]:
        return dict(self.entity_counter)


if __name__ == "__main__":
    df = pd.read_excel('test.xlsx')

    ner = ChineseNER(top_n=1000)
    results = ner.analyze(df, text_column='text')
    print("NER结果:", results)
    print("实体词词频:", ner.get_entity_frequencies())

# -*- coding: utf-8 -*-
import hanlp
import pandas as pd
from typing import Dict


class ChineseSRL:
    def __init__(self):
        # 或者CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH
        self.analyzer = hanlp.load(
            hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ERNIE_GRAM_ZH)

    def analyze(self, df: pd.DataFrame, text_column: str = 'text') -> Dict:
        # 检查输入的DataFrame是否包含指定的列
        if text_column not in df.columns:
            raise ValueError(f"输入的DataFrame中不包含'{text_column}'列")

        # 存储所有句子的结果
        all_results = {}

        # 遍历DataFrame的每一行
        for idx, row in df.iterrows():
            sentence = row[text_column]
            # 使用HanLP进行分析，保留SRL任务
            result = self.analyzer(str(sentence), tasks=['srl'])
            srl_result = result['srl']
            # 存储结果的字典，使用原始DataFrame的索引作为键
            all_results[idx] = srl_result

        return all_results


# 使用示例
if __name__ == "__main__":
    srl = ChineseSRL()
    df = pd.read_excel('./data/test.xlsx')
    result = srl.analyze(df)
    print(result)

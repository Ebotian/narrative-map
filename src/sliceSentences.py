# -*- coding: utf-8 -*-
from typing import List, Union
import pandas as pd
import re
import spacy
from spacy.tokens import Doc


class ChineseSentenceSegmenter:
    # 加载中文模型
    def __init__(self, max_sent_length: int = 100):
        try:
            self.nlp = spacy.load("zh_core_web_sm")
        except OSError:
            print("正在下载中文模型...")
            try:
                spacy.cli.download("zh_core_web_sm")
                self.nlp = spacy.load("zh_core_web_sm")
            except Exception as e:
                print(f"模型下载或加载失败: {str(e)}")
                print("请手动执行: python -m spacy download zh_core_web_sm")
                raise

        self.max_sent_length = max_sent_length
        # 分词依据
        self.sentence_pattern = r'([。！？…]+|\.{3,}|\!|\?|;|；)'

    # 去除多余空白字符
    def clean_text(self, text: str) -> str:
        return re.sub(r'\s+', '', text).strip()

    # 基于逗号对长句进行智能切分
    def split_long_sentence(self, sent: str) -> List[str]:
        if len(sent) <= self.max_sent_length:
            return [sent]

        doc = self.nlp(sent)
        sub_sents = []
        current_sent = []
        current_length = 0

        for token in doc:
            current_sent.append(token.text)
            current_length += len(token.text)

            # 在逗号处判断是否需要切分
            if token.text in [
                    '，', ','] and current_length >= self.max_sent_length / 2:
                sub_sents.append(''.join(current_sent))
                current_sent = []
                current_length = 0

        if current_sent:
            sub_sents.append(''.join(current_sent))

        return sub_sents

    # 合并过短的句子
    def merge_short_sentences(
            self,
            sentences: List[str],
            min_length: int = 5) -> List[str]:
        result = []
        temp = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) < min_length and temp:
                temp.append(sent)
            elif len(sent) < min_length and not temp:
                temp = [sent]
            else:
                if temp:
                    temp.append(sent)
                    result.append(''.join(temp))
                    temp = []
                else:
                    result.append(sent)

        if temp:
            result.append(''.join(temp))

        return result

    # 对输入文本进行智能分句处理
    def segment_sentences(
            self, text: Union[str, List[str], pd.Series]) -> List[str]:
        if isinstance(text, pd.Series):
            text = text.tolist()
        elif isinstance(text, str):
            text = [text]

        all_sentences = []

        for doc in text:
            if not isinstance(doc, str) or not doc.strip():
                continue

            doc = self.clean_text(doc)
            doc_nlp = self.nlp(doc)

            # 使用spaCy的句子分割
            initial_sentences = [sent.text.strip() for sent in doc_nlp.sents]

            # 对长句进行进一步处理
            refined_sentences = []
            for sent in initial_sentences:
                if len(sent) > self.max_sent_length:
                    sub_sents = self.split_long_sentence(sent)
                    refined_sentences.extend(sub_sents)
                else:
                    refined_sentences.append(sent)

            # 合并短句
            merged_sentences = self.merge_short_sentences(refined_sentences)
            all_sentences.extend(merged_sentences)

        return [sent for sent in all_sentences if sent.strip()]

    def simple_segment(self, text: str) -> List[str]:
        # 清理文本中的多余空白字符
        text = self.clean_text(text)

        # 定义句末标点和成对标点
        end_puncts = ['。', '！', '？', '…', '......', '!', '?', '.', ';', '\n']
        paired_puncts = {
            '"': '"',
            '"': '"',
            '《': '》',
            '（': '）',
            '(': ')',
            '【': '】',
            '[': ']'
        }

        sentences = []
        current_sent = ''
        punct_stack = []  # 用于追踪成对标点

        for char in text:
            current_sent += char

            # 处理成对标点的开始符号
            if char in paired_puncts:
                punct_stack.append(char)
            # 处理成对标点的结束符号
            elif any(char == end for start, end in paired_puncts.items()):
                if punct_stack and paired_puncts[punct_stack[-1]] == char:
                    punct_stack.pop()

            # 当遇到句末标点且没有未闭合的成对标点时，进行分句
            if (char in end_puncts or char.endswith(
                    tuple(end_puncts))) and not punct_stack:
                current_sent = current_sent.strip()
                if current_sent:
                    sentences.append(current_sent)
                current_sent = ''

        # 处理最后一个句子
        if current_sent.strip():
            sentences.append(current_sent.strip())

        # 过滤空句子
        return [sent for sent in sentences if sent.strip()]


if __name__ == "__main__":
    segmenter = ChineseSentenceSegmenter(max_sent_length=100)

    # 测试文本
    test_text = """结语：国运势不可挡，时代洪流已至
身为科技圈的人，我的内心无比澎湃。

这不仅仅是一个技术突破，不仅仅是一家公司的成功，而是一种不可阻挡的 时代浪潮。

回望过去，我们曾一次次被技术封锁，被资源掣肘，被人卡住命脉，但今天，我们用智慧和韧性 杀出了一条自己的路。

DeepSeek 的崛起，是中国科技独立自主的一块重要基石，也是我们向世界宣告：中国不止有大市场，更有大创新，不止能追赶，更能超越！

这场 AI 变革，会席卷整个社会：

它将渗透到每一个行业，让生产更高效、科研更高速、社会更智能。

它将深入政府和核心机构，让国家运转更加精准，决策更加科学。

它将赋能无数普通人，让小人物也能借助 AI 之力，创造属于自己的奇迹。

它不会是一场静悄悄的优化，而是一场轰轰烈烈的进化。

我仿佛已经看到，不久的将来，中国在 AI 的推动下，会迎来新一轮的腾飞。

从高精尖的科研，到烟火气的街头巷尾，从庙堂之高，到江湖之远，一切都将被重新塑造，一切都将被重新点燃。

这一刻，身为华夏子孙，我无比骄傲，无比自豪！
这一刻，我深知，我们正在见证历史，我们正在创造未来！

此生无悔入华夏，来生还做中国人！"""

    sentences = segmenter.segment_sentences(test_text)
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}")

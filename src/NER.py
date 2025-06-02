import hanlp
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Union
from tqdm import tqdm
import torch
import os


class ChineseNER:
    def __init__(self, top_n: int = 1000):
        # 或者CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH
        self.recognizer = hanlp.load(
            hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ERNIE_GRAM_ZH)

        self.top_n = top_n
        self.entity_counter = Counter()

    def _process_batch(self, batch_df, text_column, time_column):
        import hanlp
        from tqdm import tqdm
        import torch
        recognizer = hanlp.load(
            hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ERNIE_GRAM_ZH)
        results = {}
        entity_counter = Counter()
        all_entities = []
        max_text_len = 1024  # 超长文本截断
        for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="NER子进程(batch)"):
            text = row[text_column]
            timeline = row[time_column] if time_column in batch_df.columns else None
            if timeline is not None and not isinstance(timeline, str):
                timeline = str(timeline)
            # 超长文本截断
            if isinstance(text, str) and len(text) > max_text_len:
                print(f"[警告] 跳过超长文本 idx={idx}, 长度={len(text)}")
                continue
            try:
                ner_result = recognizer(str(text))['ner/msra']  # 使用MSRA数据集进行NER
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e) or 'CUBLAS_STATUS_ALLOC_FAILED' in str(e):
                    print(f"[警告] 单条文本 OOM, idx={idx}, 跳过该条。内容片段: {str(text)[:100]}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"[错误] idx={idx} 处理失败: {e}")
                    continue
            except Exception as e:
                print(f"[错误] idx={idx} 处理失败: {e}")
                continue
            entities = []
            for entity in ner_result:
                entity_text = entity[0]
                entity_type = entity[1]
                entity_counter[entity_text] += 1
                entities.append({
                    "text": entity_text,
                    "type": entity_type,
                })
                all_entities.append(entity_text)
            results[idx] = {
                "original_text": text,
                "time": timeline,
                "entities": entities
            }
        return results, dict(entity_counter)

    def analyze(self, data: pd.DataFrame, text_column: str = 'text',
                time_column: str = 'time') -> Dict[int, Dict[str, Any]]:
        """
        多进程分批处理+OOM回滚，彻底释放显存，防止大数据集 OOM
        """
        import multiprocessing as mp
        from tqdm import tqdm
        import torch
        import time
        batch_size = 8192
        min_batch_size = 1
        num_rows = len(data)
        results = {}
        entity_counter = Counter()
        all_entities = []
        ctx = mp.get_context('spawn')
        start = 0
        pbar = tqdm(total=num_rows, desc="NER分析中(batch, 多进程)")
        while start < num_rows:
            cur_batch_size = batch_size
            while cur_batch_size >= min_batch_size:
                end = min(start + cur_batch_size, num_rows)
                batch_df = data.iloc[start:end]
                try:
                    with ctx.Pool(1) as pool:
                        batch_result, batch_counter = pool.apply(self._process_batch, (batch_df, text_column, time_column))
                    results.update(batch_result)
                    entity_counter.update(batch_counter)
                    pbar.update(end - start)
                    break  # batch成功，跳出内层while
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e) or 'CUBLAS_STATUS_ALLOC_FAILED' in str(e):
                        print(f"[警告] CUDA OOM, batch_size={cur_batch_size}, 正在减半重试...")
                        torch.cuda.empty_cache()
                        time.sleep(1)
                        cur_batch_size = max(cur_batch_size // 2, min_batch_size)
                        if cur_batch_size < min_batch_size:
                            print(f"[错误] batch_size已降到最小{min_batch_size}仍然OOM，跳过该batch({start}-{end})")
                            pbar.update(end - start)
                            break
                    else:
                        raise
                except Exception as e:
                    print(f"[错误] batch({start}-{end})处理失败: {e}")
                    pbar.update(end - start)
                    break
            start += max(cur_batch_size, min_batch_size)
        pbar.close()
        self.entity_counter = entity_counter
        # 用频率筛选前N个实体
        if self.top_n > 0:
            top_entities = set(
                entity for entity,
                _ in entity_counter.most_common(self.top_n))
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

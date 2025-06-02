# -*- coding: utf-8 -*-
import hanlp
import pandas as pd
from typing import Dict
from tqdm import tqdm


class ChineseSRL:
    def __init__(self):
        # 或者CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH
        self.analyzer = hanlp.load(
            hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ERNIE_GRAM_ZH)

    def _process_batch(self, batch_df, text_column):
        import hanlp
        from tqdm import tqdm
        import torch
        recognizer = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ERNIE_GRAM_ZH)
        batch_results = {}
        max_text_len = 1024  # 超长文本截断
        for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="SRL子进程(batch)"):
            sentence = row[text_column]
            # 超长文本截断
            if isinstance(sentence, str) and len(sentence) > max_text_len:
                print(f"[警告] 跳过超长文本 idx={idx}, 长度={len(sentence)}")
                continue
            try:
                result = recognizer(str(sentence), tasks=['srl'])
                srl_result = result['srl']
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e) or 'CUBLAS_STATUS_ALLOC_FAILED' in str(e):
                    print(f"[警告] 单条文本 OOM, idx={idx}, 跳过该条。内容片段: {str(sentence)[:100]}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"[错误] idx={idx} 处理失败: {e}")
                    continue
            except Exception as e:
                print(f"[错误] idx={idx} 处理失败: {e}")
                continue
            batch_results[idx] = srl_result
        return batch_results

    def analyze(self, df: pd.DataFrame, text_column: str = 'text') -> Dict:
        import multiprocessing as mp
        from tqdm import tqdm
        import torch
        import time
        batch_size = 8192
        min_batch_size = 8
        num_rows = len(df)
        all_results = {}
        ctx = mp.get_context('spawn')
        start = 0
        pbar = tqdm(total=num_rows, desc="SRL分析中(batch, 多进程)")
        while start < num_rows:
            cur_batch_size = batch_size
            while cur_batch_size >= min_batch_size:
                end = min(start + cur_batch_size, num_rows)
                batch_df = df.iloc[start:end]
                try:
                    with ctx.Pool(1) as pool:
                        batch_results = pool.apply(self._process_batch, (batch_df, text_column))
                    all_results.update(batch_results)
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
        return all_results

    def analyze_batch(self, df: pd.DataFrame, text_column: str = 'text') -> Dict:
        # 检查输入的DataFrame是否包含指定的列
        if text_column not in df.columns:
            raise ValueError(f"输入的DataFrame中不包含'{text_column}'列")

        # 存储所有句子的结果
        all_results = {}

        batch_size = 128  # 可根据显存调整
        min_batch_size = 8
        num_rows = len(df)
        start = 0
        pbar = tqdm(total=num_rows, desc="SRL分析中(batch)")
        import torch
        while start < num_rows:
            cur_batch_size = batch_size
            success = False
            while not success and cur_batch_size >= min_batch_size:
                try:
                    end = min(start + cur_batch_size, num_rows)
                    batch = df.iloc[start:end]
                    for idx, row in batch.iterrows():
                        sentence = row[text_column]
                        result = self.analyzer(str(sentence), tasks=['srl'])
                        srl_result = result['srl']
                        all_results[idx] = srl_result
                        pbar.update(1)
                    torch.cuda.empty_cache()
                    # os.system('nvidia-smi')  # 如需监控可取消注释
                    success = True
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f"[警告] CUDA OOM, batch_size {cur_batch_size} -> {cur_batch_size//2}")
                        torch.cuda.empty_cache()
                        import time; time.sleep(2)
                        cur_batch_size = cur_batch_size // 2
                        if cur_batch_size < min_batch_size:
                            pbar.close()
                            raise RuntimeError("batch_size 已降到最小仍然 OOM，请减少显卡负载或分批处理数据！")
                    else:
                        pbar.close()
                        raise e
            start += cur_batch_size
        pbar.close()
        return all_results


# 使用示例
if __name__ == "__main__":
    srl = ChineseSRL()
    df = pd.read_excel('./data/test.xlsx')
    result = srl.analyze(df)
    print(result)

import hanlp
import gensim
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import time

# 添加全局缓存
_GLOBAL_TOKENIZER = None
_GLOBAL_WORD_VECTORS = None


class nonNEsKMeansMapper:
    def __init__(self, input_data={}, if_auto_k=True, min_k=2):
        # 设置标签提取方法
        self.label_method = "optimized"  # 可选: "original", "tfidf", "hybrid", "optimized"

        self.input_data = input_data
        self.if_auto_k = if_auto_k
        self.min_k = min_k
        self.non_NE_dict = {}
        self.weight_dict = {}
        self.vector_dict = {}

        # 使用全局缓存的分词器
        global _GLOBAL_TOKENIZER
        if _GLOBAL_TOKENIZER is None:
            print("首次加载HanLP分词器...")
            start_time = time.time()
            _GLOBAL_TOKENIZER = hanlp.load(
                hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
            print(f"加载HanLP分词器完成，耗时: {time.time() - start_time:.2f}秒")
        self.tokenizer = _GLOBAL_TOKENIZER

        self.output_data = input_data.copy()
        self._stop_words = None  # 缓存停用词

    # 新增：加载停用词方法
    def _load_stopwords(self):
        """加载停用词列表"""
        if self._stop_words is not None:
            return self._stop_words

        self._stop_words = set()
        stop_words_path = os.environ.get(
            "STOP_WORDS_PATH",
            "/home/ebit/LMZ/src/model/hit_stopwords.txt")
        try:
            with open(stop_words_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self._stop_words.add(line.strip())
            print(f"成功加载 {len(self._stop_words)} 个停用词")
        except Exception as e:
            print(f"加载停用词失败: {e}")

        return self._stop_words

    # 清洗出实体词
    def _filterNonNEs(self, tokens_dict={}):
        tokens_dict = tokens_dict if tokens_dict else self.input_data
        for idx, item in tokens_dict.items():
            srl_mappings = item.get('srl_mappings', [])
            non_ents_in_item = []

            for position, mapping_list in enumerate(srl_mappings):
                for token_tuple in mapping_list:
                    if (token_tuple[1] == 'ARG0' or token_tuple[1]
                            == 'ARG1') and token_tuple[2] is None:
                        non_ne_tuple = [
                            token_tuple[0],
                            token_tuple[1],
                            token_tuple[2],
                            position]
                        non_ents_in_item.append(non_ne_tuple)

            if non_ents_in_item:
                self.non_NE_dict[idx] = non_ents_in_item

        return self.non_NE_dict

    # 计算词频
    def _generateWeightDict(self):
        # 用于存储原始计数
        raw_counts = {}
        total_tokens = 0

        for idx, item_list in self.non_NE_dict.items():
            for i, token_list in enumerate(item_list):
                phrase = token_list[0]
                # 使用HanLP对短语进行分词
                result = self.tokenizer(phrase)
                # 获取分词结果
                tokens = result['tok/fine']

                # 将分词信息保存到non_NE_dict中
                # 创建一个新的元组，包含原始信息和分词结果
                updated_list = [
                    token_list[0],
                    token_list[1],
                    token_list[2],
                    token_list[3],
                    tokens]
                self.non_NE_dict[idx][i] = updated_list

                # 统计每个分词的出现次数
                for token in tokens:
                    if token not in raw_counts:
                        raw_counts[token] = 1
                    else:
                        raw_counts[token] += 1
                    total_tokens += 1

        # 计算每个词的逆权重
        alpha = 0.01  # 平滑参数
        if total_tokens > 0:
            for token, count in raw_counts.items():
                freq = count / total_tokens
                self.weight_dict[token] = 1.0 / (freq + alpha)

        return self.weight_dict

    # 生成逆词频加权平均词向量
    def _generateWeightedAverageVector(self, input_data=None):
        load_dotenv()
        # 加载预训练中文词向量模型
        global _GLOBAL_WORD_VECTORS
        if _GLOBAL_WORD_VECTORS is None:
            try:
                print("首次加载词向量模型...")
                start_time = time.time()
                model_path = os.environ.get(
                    "WORD_VECTOR_PATH",
                    r"/home/ebit/LMZ/src/model/sgns.wiki.word")
                _GLOBAL_WORD_VECTORS = gensim.models.KeyedVectors.load_word2vec_format(
                    model_path, binary=False, encoding='utf-8')
                print(f"加载词向量模型完成，耗时: {time.time() - start_time:.2f}秒")
            except Exception as e:
                print(f"加载词向量模型失败: {e}")
                return {}

        word_vector_model = _GLOBAL_WORD_VECTORS

        # 将每个词输入到预训练词向量模型中计算向量
        for word, freq in self.weight_dict.items():
            try:
                if word in word_vector_model.key_to_index:
                    self.vector_dict[word] = word_vector_model[word]
            except BaseException:
                continue

        # 为每个短语计算加权平均向量
        weighted_average_vectors = {}

        for idx, item_list in self.non_NE_dict.items():
            for i, phrase_list in enumerate(item_list):
                phrase = phrase_list[0]
                tokens = phrase_list[4] if len(phrase_list) > 3 else []
                position = phrase_list[3] if len(phrase_list) > 3 else None

                # 计算该短语的加权平均向量
                if tokens:
                    valid_vectors = []
                    total_weight = 0
                    weighted_sum = None

                    for token in tokens:
                        if token in self.vector_dict:
                            weight = self.weight_dict.get(token, 1.0)
                            vector = self.vector_dict.get(token, 1.0)

                            if weighted_sum is None:
                                weighted_sum = vector * weight
                            else:
                                weighted_sum += vector * weight

                            total_weight += weight
                            valid_vectors.append(True)
                        else:
                            valid_vectors.append(False)

                    # 如果有有效向量，计算加权平均
                    if total_weight > 0:
                        phrase_vector = weighted_sum / total_weight
                        weighted_average_vectors[(idx, position, i)] = {
                            'vector': phrase_vector,
                            'phrase': phrase,
                        }

        # 将加权平均向量覆盖到self.vector_dict中以减少内存开销
        self.vector_dict = weighted_average_vectors
        return self.vector_dict

    # 对词向量进行K-means聚类（优化版）
    def nonNEsCluster(self):
        # 检查词向量数量是否足够进行聚类
        if not self.vector_dict or len(self.vector_dict) < self.min_k:
            print(f"词向量数量不足: {len(self.vector_dict)}")
            return {}

        # 准备数据
        keys = list(self.vector_dict.keys())
        vectors = np.array([self.vector_dict[k]['vector'] for k in keys])
        phrases = [self.vector_dict[k]['phrase'] for k in keys]

        # 确定最佳K值 (优化版)
        if self.if_auto_k:
            # 限制样本数量，以减少计算量
            max_samples = 1000
            if len(vectors) > max_samples:
                # 随机抽样进行K值评估
                indices = np.random.choice(
                    len(vectors), max_samples, replace=False)
                sample_vectors = vectors[indices]
            else:
                sample_vectors = vectors

            # 限制K的搜索范围，避免不必要的计算
            # 使用平方根法则限制K，最大不超过20
            max_k = min(max(self.min_k, int(np.sqrt(len(sample_vectors)))), 20)
            best_k = self.min_k
            best_score = -1

            # 使用二分搜索而不是线性搜索
            k_values = list(range(self.min_k, max_k + 1,
                            max(1, (max_k - self.min_k) // 5)))
            if max_k not in k_values:
                k_values.append(max_k)

            for k in k_values:
                try:
                    # 减少KMeans的迭代次数和初始化次数
                    kmeans = KMeans(
                        n_clusters=k,
                        random_state=42,
                        max_iter=100,
                        n_init=3)
                    clusters = kmeans.fit_predict(sample_vectors)

                    # 对于大样本，只使用部分样本计算轮廓系数
                    if len(sample_vectors) > 500:
                        subsample_indices = np.random.choice(
                            len(sample_vectors), 500, replace=False)
                        score = silhouette_score(
                            sample_vectors[subsample_indices],
                            clusters[subsample_indices])
                    else:
                        score = silhouette_score(sample_vectors, clusters)

                    print(f"K={k}的轮廓系数: {score}")

                    if score > best_score:
                        best_score = score
                        best_k = k
                except Exception as e:
                    print(f"计算K={k}时出错: {e}")

            print(f"最佳K值: {best_k}")
        else:
            best_k = self.min_k
            print(f"使用指定的K值: {best_k}")

        # 使用最佳K值进行聚类 (减少迭代次数)
        kmeans = KMeans(
            n_clusters=best_k,
            random_state=42,
            max_iter=100,
            n_init=2)
        clusters = kmeans.fit_predict(vectors)
        print(f"聚类结果: {clusters}")

        # 为每个簇提取标签
        cluster_phrases = {}
        for i in range(best_k):
            cluster_idx = np.where(clusters == i)[0]
            cluster_phrases[i] = [phrases[idx] for idx in cluster_idx]

        # 根据选择的方法提取标签
        if hasattr(self, 'label_method') and self.label_method in [
                "optimized", "hybrid", "tfidf"]:
            # 使用对应的标签提取方法
            if self.label_method == "optimized":
                word_vectors = {}
                for word, vec in self.vector_dict.items():
                    if isinstance(vec, dict) and 'vector' in vec:
                        continue  # 跳过短语向量
                    word_vectors[word] = vec
                cluster_labels = self.optimize_cluster_labels(
                    cluster_phrases, word_vectors)
            elif self.label_method == "tfidf":
                cluster_labels = self.extract_tfidf_labels(cluster_phrases)
            else:  # hybrid
                word_vectors = {}
                for word, vec in self.vector_dict.items():
                    if isinstance(vec, dict) and 'vector' in vec:
                        continue  # 跳过短语向量
                    word_vectors[word] = vec
                cluster_labels = self.extract_hybrid_labels(
                    cluster_phrases, word_vectors)
        else:
            # 使用原始标签提取方法
            cluster_labels = self._extract_original_labels(cluster_phrases)

        print(f"聚类标签: {cluster_labels}")

        # 使用dict.get而不是嵌套循环来提高映射效率
        phrase_to_cluster = {}
        for cluster_id, phrases_list in cluster_phrases.items():
            for phrase in phrases_list:
                phrase_to_cluster[phrase] = cluster_id

        for idx, item_list in self.output_data.items():
            for i, token_list in enumerate(item_list['srl_mappings']):
                for j, phrase_list in enumerate(token_list):
                    phrase = phrase_list[0]
                    role = phrase_list[1]
                    entity = phrase_list[2]
                    if role in ['ARG0', 'ARG1'] and entity is None:
                        # 直接查找短语所在簇
                        cluster_id = phrase_to_cluster.get(phrase)
                        if cluster_id is not None:
                            # 替换成非实体词的映射
                            self.output_data[idx]['srl_mappings'][i][j][2] = {
                                'original': self.output_data[idx]['srl_mappings'][i][j][0], 'entity_type': 'non-NE'}
                            self.output_data[idx]['srl_mappings'][i][j][0] = cluster_labels[cluster_id]

        return self.output_data

    # 新增：原始的簇标签提取逻辑，从nonNEsCluster方法中提取出来
    def _extract_original_labels(self, cluster_phrases):
        """
        原始的簇标签提取方法

        参数:
            cluster_phrases: 字典，键为簇ID，值为该簇中所有短语的列表

        返回:
            字典，键为簇ID，值为簇标签
        """
        cluster_labels = {}
        for cluster_id, phrase_list in cluster_phrases.items():
            # 将所有短语分词并统计
            all_words = []
            for phrase in phrase_list:
                words = self.tokenizer(phrase)['tok/fine']
                all_words.extend(list(words))

            # 加载停用词
            stop_words = self._load_stopwords()

            # 过滤掉停用词和单个字符
            filtered_words = [w for w in all_words if len(
                w) > 1 and w not in stop_words]

            # 获取最高频词作为标签
            if filtered_words:
                word_counts = Counter(filtered_words)
                cluster_labels[cluster_id] = word_counts.most_common(1)[0][0]
            else:
                # 如果没有合适的词，则使用最长的短语
                cluster_labels[cluster_id] = max(
                    phrase_list, key=len) if phrase_list else f"簇_{cluster_id}"

        return cluster_labels

    # 新增：使用TF-IDF提取标签方法
    def extract_tfidf_labels(self, cluster_phrases):
        """
        使用TF-IDF方法为每个簇提取标签

        参数:
            cluster_phrases: 字典，键为簇ID，值为该簇中所有短语的列表

        返回:
            字典，键为簇ID，值为簇标签
        """
        # 准备簇文本数据
        cluster_texts = {}
        for cluster_id, phrases in cluster_phrases.items():
            cluster_texts[cluster_id] = " ".join(phrases)

        # 将停用词集合转换为列表，因为TfidfVectorizer不接受set类型的停用词
        stop_words_list = list(
            self._load_stopwords()) if self._load_stopwords() else None

        # 创建TF-IDF向量化器
        tfidf_vectorizer = TfidfVectorizer(
            tokenizer=jieba.lcut,  # 使用jieba分词
            stop_words=stop_words_list,  # 加载停用词列表
            min_df=1,  # 最小文档频率
            max_features=1000  # 最大特征数
        )

        # 转换文本数据
        all_texts = list(cluster_texts.values())
        if not all_texts:
            return {}

        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

            # 获取特征名（词汇）
            feature_names = tfidf_vectorizer.get_feature_names_out()

            # 为每个簇选择最重要的词作为标签
            cluster_labels = {}
            for i, cluster_id in enumerate(cluster_texts.keys()):
                # 获取该簇的TF-IDF分数
                cluster_tfidf = tfidf_matrix[i].toarray()[0]
                # 获取最高TF-IDF分数的词索引
                sorted_indices = cluster_tfidf.argsort()[::-1]

                # 选择前三个最重要的词组合为标签（如果存在）
                important_words = []
                for idx in sorted_indices[:3]:
                    if cluster_tfidf[idx] > 0 and len(
                            feature_names[idx]) > 1:  # 过滤单字词
                        important_words.append(feature_names[idx])

                if important_words:
                    # 使用前两个词作为标签
                    cluster_labels[cluster_id] = "+".join(important_words[:2])
                else:
                    # 回退到使用最长短语
                    cluster_labels[cluster_id] = max(
                        cluster_phrases[cluster_id],
                        key=len) if cluster_phrases[cluster_id] else f"簇_{cluster_id}"
        except Exception as e:
            print(f"TF-IDF标签提取失败: {e}")
            # 回退到原始方法
            cluster_labels = self._extract_original_labels(cluster_phrases)

        return cluster_labels

    # 新增：选择代表性标签方法
    def select_representative_label(self, cluster_phrases, word_vectors):
        """
        选择与簇中心最接近的短语作为标签

        参数:
            cluster_phrases: 字典，键为簇ID，值为该簇中所有短语的列表
            word_vectors: 字典，键为词，值为词向量

        返回:
            字典，键为簇ID，值为簇标签
        """
        import numpy as np

        cluster_labels = {}

        for cluster_id, phrases in cluster_phrases.items():
            if not phrases:
                cluster_labels[cluster_id] = f"簇_{cluster_id}"
                continue

            # 获取每个短语的词向量
            phrase_vectors = []
            valid_phrases = []

            for phrase in phrases:
                # 分词并获取词向量
                tokens = jieba.lcut(phrase)
                token_vectors = []

                for token in tokens:
                    if token in word_vectors:
                        token_vectors.append(word_vectors[token])

                if token_vectors:
                    # 使用平均向量表示短语
                    phrase_vector = np.mean(token_vectors, axis=0)
                    phrase_vectors.append(phrase_vector)
                    valid_phrases.append(phrase)

            if not valid_phrases:
                cluster_labels[cluster_id] = max(
                    phrases, key=len) if phrases else f"簇_{cluster_id}"
                continue

            # 计算簇中心向量
            center_vector = np.mean(phrase_vectors, axis=0)

            # 找出与中心最接近的短语
            similarities = []
            for vec in phrase_vectors:
                similarity = np.dot(
                    vec, center_vector) / (np.linalg.norm(vec) * np.linalg.norm(center_vector))
                similarities.append(similarity)

            # 找出最相似短语的索引
            most_representative_idx = np.argmax(similarities)
            cluster_labels[cluster_id] = valid_phrases[most_representative_idx]

        return cluster_labels

    # 新增：评估簇内相似度方法
    def evaluate_cluster_cohesion(self, cluster_phrases, word_vectors):
        """
        评估簇内相似度，为低相似度的簇使用更具体的标签

        参数:
            cluster_phrases: 字典，键为簇ID，值为该簇中所有短语的列表
            word_vectors: 字典，键为词，值为词向量

        返回:
            字典，键为簇ID，值为簇内相似度分数(0-1之间)
        """
        import numpy as np

        cluster_cohesion = {}

        for cluster_id, phrases in cluster_phrases.items():
            if len(phrases) <= 1:
                cluster_cohesion[cluster_id] = 1.0
                continue

            # 获取所有有效短语向量
            valid_vectors = []
            for phrase in phrases:
                tokens = jieba.lcut(phrase)
                phrase_vectors = []

                for token in tokens:
                    if token in word_vectors:
                        phrase_vectors.append(word_vectors[token])

                if phrase_vectors:
                    phrase_vector = np.mean(phrase_vectors, axis=0)
                    valid_vectors.append(phrase_vector)

            if len(valid_vectors) <= 1:
                cluster_cohesion[cluster_id] = 1.0
                continue

            # 计算所有向量对之间的相似度
            similarities = []
            for i in range(len(valid_vectors)):
                for j in range(i + 1, len(valid_vectors)):
                    vec1 = valid_vectors[i]
                    vec2 = valid_vectors[j]
                    similarity = np.dot(
                        vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    similarities.append(similarity)

            # 使用平均相似度作为簇内凝聚力
            cluster_cohesion[cluster_id] = np.mean(
                similarities) if similarities else 0.5

        return cluster_cohesion

    # 新增：混合标签提取方法
    def extract_hybrid_labels(self, cluster_phrases, word_vectors):
        """
        结合多种方法的混合标签提取策略

        参数:
            cluster_phrases: 字典，键为簇ID，值为该簇中所有短语的列表
            word_vectors: 字典，键为词，值为词向量

        返回:
            字典，键为簇ID，值为簇标签
        """
        # 1. 提取TF-IDF重要词
        tfidf_labels = self.extract_tfidf_labels(cluster_phrases)

        # 2. 提取代表性短语
        try:
            repr_labels = self.select_representative_label(
                cluster_phrases, word_vectors)
        except Exception as e:
            print(f"代表性标签提取失败: {e}")
            repr_labels = {
                cluster_id: max(
                    phrases,
                    key=len) if phrases else f"簇_{cluster_id}" for cluster_id,
                phrases in cluster_phrases.items()}

        # 3. 结合两种标签
        hybrid_labels = {}
        for cluster_id in cluster_phrases.keys():
            tfidf_label = tfidf_labels.get(cluster_id, "")
            repr_label = repr_labels.get(cluster_id, "")

            # 如果TF-IDF标签是repr_label的一部分，只使用repr_label
            if tfidf_label in repr_label:
                hybrid_labels[cluster_id] = repr_label
            # 否则，组合两种标签（避免过长）
            elif len(tfidf_label) + len(repr_label) < 20:
                hybrid_labels[cluster_id] = f"{tfidf_label}_{repr_label}"
            # 太长则选择较短的那个
            else:
                hybrid_labels[cluster_id] = tfidf_label if len(
                    tfidf_label) < len(repr_label) else repr_label

        return hybrid_labels

    # 新增：整合多种标签提取方法的优化方法
    def optimize_cluster_labels(self, cluster_phrases, word_vectors):
        """
        优化的簇标签提取逻辑，根据簇特征选择不同的标签提取策略

        参数:
            cluster_phrases: 字典，键为簇ID，值为该簇中所有短语的列表
            word_vectors: 字典，键为词，值为词向量

        返回:
            字典，键为簇ID，值为优化后的簇标签
        """
        # 评估簇内凝聚力
        cohesion_scores = self.evaluate_cluster_cohesion(
            cluster_phrases, word_vectors)

        # 根据凝聚力选择不同的标签提取策略
        optimized_labels = {}

        for cluster_id, phrases in cluster_phrases.items():
            cohesion = cohesion_scores.get(cluster_id, 0.5)

            # 高凝聚力簇：使用TF-IDF方法提取概括性标签
            if cohesion > 0.7:
                try:
                    labels = self.extract_tfidf_labels({cluster_id: phrases})
                    optimized_labels[cluster_id] = labels[cluster_id]
                except Exception as e:
                    print(f"高凝聚力簇标签提取失败: {e}")
                    optimized_labels[cluster_id] = self._extract_original_labels(
                        {cluster_id: phrases})[cluster_id]

            # 中等凝聚力簇：使用混合方法
            elif cohesion > 0.4:
                try:
                    labels = self.extract_hybrid_labels(
                        {cluster_id: phrases}, word_vectors)
                    optimized_labels[cluster_id] = labels[cluster_id]
                except Exception as e:
                    print(f"中等凝聚力簇标签提取失败: {e}")
                    optimized_labels[cluster_id] = self._extract_original_labels(
                        {cluster_id: phrases})[cluster_id]

            # 低凝聚力簇：使用最有代表性的短语
            else:
                try:
                    # 对于低凝聚力的簇，包含多个主题，使用最具代表性的短语更好
                    labels = self.select_representative_label(
                        {cluster_id: phrases}, word_vectors)
                    optimized_labels[cluster_id] = labels[cluster_id]
                except Exception as e:
                    print(f"低凝聚力簇标签提取失败: {e}")
                    optimized_labels[cluster_id] = self._extract_original_labels(
                        {cluster_id: phrases})[cluster_id]

        return optimized_labels

    # 简化的接口函数，直接调用处理流程
    def process(self, tokens_dict=None):
        if tokens_dict is not None:
            self.input_data = tokens_dict

        # 处理流程
        self._filterNonNEs()
        self._generateWeightDict()
        self._generateWeightedAverageVector()
        result2 = self.nonNEsCluster()

        return result2


def main():
    tokens_dict = {
        1: {'original_text': '俄罗斯1月31日敦促美国及其欧洲盟友放弃加剧乌克兰紧张局势的路线，采取建设性立场。', 'entities': [{'text': '俄罗斯', 'type': 'LOCATION'}, {'text': '1月', 'type': 'DATE'}, {'text': '31日', 'type': 'DATE'}, {'text': '美国', 'type': 'LOCATION'}, {'text': '欧洲', 'type': 'LOCATION'}, {'text': '乌克兰', 'type': 'LOCATION'}], 'time': '2022-02-03 09:18:15', 'srl_mappings': [[['俄罗斯', 'ARG0', {'original': '俄罗斯', 'entity_type': 'LOCATION'}], ['1月31日', 'ARGM-TMP', None], ['敦 促', 'PRED', None], ['美国', 'ARG1', {'original': '美国及其欧洲盟友', 'entity_type': 'LOCATION'}], ['放弃加剧乌克兰紧张局势的路线，采取建设性立场', 'ARG2', None]], [['加剧', 'PRED', None], ['乌克兰', 'ARG1', {'original': '乌克兰紧张局势', 'entity_type': 'LOCATION'}], ['路线', 'ARG0', None]]]},
        3: {'original_text': ';俄总统新闻秘书佩斯科夫当天在莫斯科对俄媒体说，美国媒体近几个月围绕乌 克兰边境局势公布大量“未经证实的、歪曲的和挑衅性的信息”。', 'entities': [{'text': '俄', 'type': 'LOCATION'}, {'text': '佩斯科夫', 'type': 'PERSON'}, {'text': '当天', 'type': 'DATE'}, {'text': '莫斯科', 'type': 'LOCATION'}, {'text': '俄', 'type': 'LOCATION'}, {'text': '美国', 'type': 'LOCATION'}, {'text': '乌克兰', 'type': 'LOCATION'}], 'time': '2022-02-03 09:18:15', 'srl_mappings': [[['俄', 'ARG0', {'original': '俄总统新闻秘书佩斯科夫', 'entity_type': 'LOCATION'}], ['当天', 'ARGM-TMP', None], ['在莫斯科', 'ARGM-LOC', None], ['对俄媒体', 'ARGM-DIR', None], ['说', 'PRED', None], ['美国', 'ARG1', {'original': '美国媒体近几个月围绕乌克兰边境 局势公布大量“未经证实的、歪曲的和挑衅性的信息”', 'entity_type': 'LOCATION'}]], [['美国', 'ARG0', {'original': '美国媒体', 'entity_type': 'LOCATION'}], ['近几个月', 'ARGM-TMP', None], ['围绕', 'PRED', None], ['乌克兰', 'ARG1', {'original': '乌克兰边境局势', 'entity_type': 'LOCATION'}]], [['美国', 'ARG0', {'original': '美国媒体', 'entity_type': 'LOCATION'}], ['近几个月', 'ARGM-TMP', None], ['公布', 'PRED', None], ['大量“未经证实的、歪曲的和挑衅性的信息”', 'ARG1', None]]]},
        4: {'original_text': '他建议人们对此保持清醒头脑。', 'entities': [], 'time': '2022-02-03 09:18:15', 'srl_mappings': [[['他', 'ARG0', None], ['建议', 'PRED', None], ['人 们', 'ARG2', None], ['对此保持清醒头脑', 'ARG1', None]], [['人们', 'ARG0', None], ['对', 'ARGM-ADV', None], ['保持', 'PRED', None], ['清醒头脑', 'ARG1', None]]]},
        5: {'original_text': '据塔斯社报道，乌克兰总统泽连斯基1月28日晚在为外国媒体举行的记者会上，对西方鼓噪战争不可避免表示不满，认为事实并非如此。', 'entities': [{'text': '塔斯社', 'type': 'ORGANIZATION'}, {'text': '乌 克兰', 'type': 'LOCATION'}, {'text': '泽连斯基', 'type': 'PERSON'}, {'text': '1月', 'type': 'DATE'}, {'text': '28日', 'type': 'DATE'}, {'text': '晚', 'type': 'TIME'}], 'time': '2022-02-03 09:18:15', 'srl_mappings': [[['西方', 'ARG0', None], ['鼓噪', 'PRED', None], ['战争', 'ARG1', None]], [['西方', 'ARG0', None], ['鼓噪战争', 'ARG1', None], ['不', 'ARGM-ADV', None], ['避免', 'PRED', None]], [['据塔斯社报道', 'ARGM-MNR', None], ['乌克兰', 'ARG0', {'original': '乌克兰总统泽连斯基', 'entity_type': 'LOCATION'}], ['1月28日晚', 'ARGM-TMP', None], ['在为外国媒体举行的记者会上', 'ARGM-LOC', None], ['对西方鼓噪战争不可避免', 'ARG3', None], ['表示', 'PRED', None], ['不满', 'ARG1', None]], [['据塔斯社报道', 'ARGM-MNR', None], ['乌克兰', 'ARG0', {'original': '乌克兰总统泽连斯基', 'entity_type': 'LOCATION'}], ['1月28日晚', 'ARGM-TMP', None], ['在为外国媒体举行的记者会上', 'ARGM-LOC', None], ['认为', 'PRED', None], ['事实并非如此', 'ARG1', None]], [['事实', 'ARG0', None], ['并', 'ARGM-ADV', None], ['非', 'PRED', None], ['如此', 'ARG1', None]]]},
        6: {'original_text': '他同时强调，加剧乌克兰边境局势恐慌使乌付出巨大代价。', 'entities': [{'text': '乌克兰', 'type': 'LOCATION'}, {'text': '乌', 'type': 'LOCATION'}], 'time': '2022-02-03 09:18:15', 'srl_mappings': [[['他', 'ARG0', None], ['同时', 'ARGM-ADV', None], ['强调', 'PRED', None], ['乌克兰', 'ARG1', {'original': '加剧乌克兰边境局势恐慌使乌付出巨大代价', 'entity_type': 'LOCATION'}]], [['乌克兰', 'ARG0', {'original': '加剧乌克兰边境局势恐慌', 'entity_type': 'LOCATION'}], ['使', 'PRED', None], ['乌', 'ARG1', {'original': '乌', 'entity_type': 'LOCATION'}], ['付出巨大代价', 'ARG2', None]]]},
        7: {'original_text': '近期，乌俄关系加速恶化，双方在两国边境地区部署了大量军事人员和武器装备。',
            'entities': [{'text': '乌', 'type': 'LOCATION'}, {'text': '俄', 'type': 'LOCATION'}, {'text': '两', 'type': 'INTEGER'}],
            'time': '2022-02-03 09:18:15',
            'srl_mappings': [
                [['双方', 'ARG0', None], ['在两国边境地区', 'ARG2', None], ['部署', 'PRED', None], ['大量军事人员和武器装备', 'ARG1', None]]
            ]},
        8: {'original_text': '美国、乌克兰和北约声称，俄罗斯在靠近乌东部边境地区集结重兵，有“入侵”之势。',
            'entities': [{'text': '美国', 'type': 'LOCATION'}, {'text': '乌克兰', 'type': 'LOCATION'}, {'text': '北约', 'type': 'ORGANIZATION'}, {'text': '俄罗斯', 'type': 'LOCATION'}, {'text': '乌', 'type': 'LOCATION'}],
            'time': '2022-02-03 09:18:15',
            'srl_mappings': [
                [['美国', 'ARG0', {'original': '美国、乌克兰和北约', 'entity_type': 'LOCATION'}], ['声称', 'PRED', None], ['俄罗斯', 'ARG1', {'original': '俄罗斯在靠近乌东部边境地区集结重兵，有“入侵”之势', 'entity_type': 'LOCATION'}]],
                [['俄罗斯', 'ARG0', {'original': '俄罗斯', 'entity_type': 'LOCATION'}], ['在靠近乌东部边境地区', 'ARGM-LOC', None], ['集结', 'PRED', None], ['重兵', 'ARG1', None]],
                [['俄罗斯', 'ARG0', {'original': '俄罗斯', 'entity_type': 'LOCATION'}], ['有', 'PRED', None], ['“入侵”之势', 'ARG1', None]]
            ]},
        9: {'original_text': '俄方予以否认，强调北约活动威胁俄边境安全，俄方有权在境内调动部队以保卫领土。', 'entities': [{'text': '北约', 'type': 'ORGANIZATION'}, {'text': '俄', 'type': 'LOCATION'}], 'time': '2022-02-03 09:18:15', 'srl_mappings': [[['俄', 'ARG0', {'original': '俄方', 'entity_type': 'LOCATION'}], ['予以', 'PRED', None], ['否认', 'ARG1', None]], [['俄', 'ARG0', {'original': '俄方', 'entity_type': 'LOCATION'}], ['强调', 'PRED', None], ['北约', 'ARG1', {'original': '北约活动威胁俄边境安全，俄方有权在境内调动部队以保卫领土', 'entity_type': 'ORGANIZATION'}]], [['北约', 'ARG0', {'original': '北约活动', 'entity_type': 'ORGANIZATION'}], ['威胁', 'PRED', None], ['俄', 'ARG1', {'original': '俄边境安全', 'entity_type': 'LOCATION'}]], [['俄', 'ARG0', {'original': '俄方', 'entity_type': 'LOCATION'}], ['在境内', 'ARGM-LOC', None], ['调动', 'PRED', None], ['部队', 'ARG1', None]], [['俄', 'ARG0', {'original': '俄方', 'entity_type': 'LOCATION'}], ['在境内', 'ARGM-LOC', None], ['保卫', 'PRED', None], ['领土', 'ARG1', None]]]},
        10: {'original_text': '近几天来，俄官员连 续表示俄不希望战争。', 'entities': [{'text': '俄', 'type': 'LOCATION'}, {'text': '俄', 'type': 'LOCATION'}], 'time': '2022-02-03 09:18:15', 'srl_mappings': [[['近几天来', 'ARGM-TMP', None], ['俄', 'ARG0', {'original': '俄官员', 'entity_type': 'LOCATION'}], ['连续', 'ARGM-ADV', None], ['表示', 'PRED', None], ['俄', 'ARG1', {'original': '俄不希望战争', 'entity_type': 'LOCATION'}]], [[' 俄', 'ARG0', {'original': '俄', 'entity_type': 'LOCATION'}], ['不', 'ARGM-ADV', None], ['希望', 'PRED', None], ['战争', 'ARG1', None]]]}
    }
    mapper = nonNEsKMeansMapper(input_data=tokens_dict)
    result = mapper.process()
    print('输出数据:')
    print(result)


if __name__ == "__main__":
    main()

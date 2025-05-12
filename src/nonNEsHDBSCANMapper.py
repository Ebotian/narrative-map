import gensim
from dotenv import load_dotenv
import os
import numpy as np
from hdbscan import HDBSCAN
from collections import Counter
import umap
import jieba
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from scipy.spatial import distance


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')


class nonNEsHDBSCANMapper:
    def __init__(
            self,
            input_data={},
            min_cluster_size=5,
            min_samples=3,
            adaptive_clustering=True):
        self.input_data = input_data
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.adaptive_clustering = adaptive_clustering
        self.non_NE_dict = {}
        self.weight_dict = {}
        self.vector_dict = {}
        self.output_data = input_data.copy()

        # 内置基本停用词列表，作为备份
        self.default_stopwords = set([
            "的", "了", "和", "是", "就", "都", "而", "及", "与", "着",
            "或", "其", "把", "由", "于", "中", "却", "为", "及", "了",
            "对", "么", "之", "还", "从", "到", "给", "又", "等", "在",
            "此", "每", "并", "很", "但", "还", "个", "其", "些", "该",
            "以", "能", "你", "我", "他", "她", "它", "这", "那", "有"
        ])

        # 加载外部停用词列表
        self.stopwords = self._load_stopwords()

        # 日志
        logging.info("初始化nonNEsHDBSCANMapper")

    def _load_stopwords(self):
        """加载停用词列表，优先使用外部文件，如果加载失败则使用内置列表"""
        stopwords = self.default_stopwords.copy()

        # 指定停用词文件路径
        stopwords_path = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "model",
            "hit_stopwords.txt")

        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                external_stopwords = set()
                for line in f:
                    word = line.strip()
                    if word:  # 跳过空行
                        external_stopwords.add(word)

                if external_stopwords:
                    logging.info(
                        f"成功从{stopwords_path}加载了{len(external_stopwords)}个停用词")
                    # 合并外部停用词和内置停用词
                    stopwords.update(external_stopwords)
                else:
                    logging.warning(f"外部停用词文件{stopwords_path}为空，使用内置停用词")
        except Exception as e:
            logging.warning(f"加载外部停用词文件失败: {e}，使用内置停用词")

        return stopwords

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

        return self

    # 计算词频
    def _generateWeightDict(self):
        raw_counts = {}
        total_tokens = 0

        # 统计每个分词的出现次数
        for idx, item_list in self.non_NE_dict.items():
            for i, token_list in enumerate(item_list):
                phrase = token_list[0]
                tokens = jieba.lcut(phrase)
                updated_list = [
                    token_list[0],
                    token_list[1],
                    token_list[2],
                    token_list[3],
                    tokens]
                self.non_NE_dict[idx][i] = updated_list
                for token in tokens:
                    if token not in raw_counts:
                        raw_counts[token] = 1
                    else:
                        raw_counts[token] += 1
                    total_tokens += 1

        # 计算每个词的逆权重
        alpha = 0.1  # 平滑参数
        if total_tokens > 0:
            for token, count in raw_counts.items():
                freq = count / total_tokens
                self.weight_dict[token] = 1.0 / (freq + alpha)
        return self

    # 生成逆词频加权平均词向量
    def _generateWeightedAverageVector(self, input_data=None):
        load_dotenv()
        try:
            model_path = os.environ.get(
                "WORD_VECTOR_PATH",
                "/home/ebit/LMZ/src/model/sgns.wiki.word")  # 用的wiki词向量模型
            word_vector_model = gensim.models.KeyedVectors.load_word2vec_format(
                model_path, binary=False, encoding='utf-8')
        except Exception as e:
            print(f"加载词向量模型失败: {e}")
            return {}

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
        return self

    # 使用TF-IDF提取关键词
    def _extract_keywords_with_tfidf(self, texts, n_keywords=3):
        """使用TF-IDF算法从文本集合中提取关键词

        Args:
            texts: 文本列表
            n_keywords: 每个簇提取的关键词数量

        Returns:
            dict: 每个文本的关键词列表
        """
        if not texts:
            return {}

        # 对每个文本进行分词
        tokenized_texts = []
        for text in texts:
            words = jieba.lcut(text)
            # 过滤停用词和单字词
            filtered_words = [w for w in words if len(
                w) > 1 and w not in self.stopwords]
            if not filtered_words:  # 如果过滤后没有词，则保留原词
                filtered_words = words
            tokenized_texts.append(" ".join(filtered_words))

        # 构建TF-IDF矩阵
        vectorizer = TfidfVectorizer(
            analyzer='word',
            max_features=500,  # 限制特征数量
            max_df=0.95,       # 忽略出现在95%以上文档中的词
            min_df=0.05,       # 忽略出现在5%以下文档中的词
            use_idf=True
        )

        try:
            # 如果文本太少，可能会出错
            if len(tokenized_texts) < 2:
                return {0: texts[0].split()}

            tfidf_matrix = vectorizer.fit_transform(tokenized_texts)
            feature_names = vectorizer.get_feature_names_out()

            # 为每个文本提取关键词
            keywords = {}
            for i, text_vector in enumerate(tfidf_matrix):
                # 获取词的TF-IDF分数
                tfidf_scores = zip(
                    feature_names, text_vector.toarray().flatten())
                # 按分数排序
                sorted_scores = sorted(
                    tfidf_scores, key=lambda x: x[1], reverse=True)
                # 提取前n_keywords个关键词
                top_keywords = [word for word,
                                score in sorted_scores[:n_keywords] if score > 0]
                keywords[i] = top_keywords

            return keywords
        except Exception as e:
            logging.error(f"TF-IDF关键词提取失败: {e}")
            # 返回简单分词结果作为备选
            return {i: jieba.lcut(text)[:n_keywords]
                    for i, text in enumerate(texts)}

    # 获取簇的代表性标签
    def _get_representative_label(self, phrases, cluster_id):
        """为簇选择最具代表性的标签

        Args:
            phrases: 簇中的短语列表
            cluster_id: 簇ID

        Returns:
            str: 代表性标签
        """
        if not phrases:
            return f"簇_{cluster_id}"

        # 使用TF-IDF提取关键词
        keywords_dict = self._extract_keywords_with_tfidf(phrases)

        # 合并所有关键词
        all_keywords = []
        for _, keywords in keywords_dict.items():
            all_keywords.extend(keywords)

        # 如果有关键词，选择频率最高的
        if all_keywords:
            keyword_counts = Counter(all_keywords)
            # 选择最频繁的关键词作为标签
            most_common = keyword_counts.most_common(1)
            if most_common:
                return most_common[0][0]

        # 备选方案：选择最长的短语
        return max(phrases, key=len)

    # 自适应调整聚类参数
    def _get_adaptive_clustering_params(self, vectors):
        """根据数据特点自适应调整聚类参数

        Args:
            vectors: 数据向量

        Returns:
            tuple: (min_cluster_size, min_samples, cluster_selection_epsilon)
        """
        n_samples = vectors.shape[0]

        # 根据样本数量动态调整min_cluster_size
        if self.adaptive_clustering:
            # 样本数量越多，聚类尺寸可以适当增大
            if n_samples < 10:
                min_cluster_size = max(2, self.min_cluster_size)
                min_samples = max(1, self.min_samples)
                cluster_selection_epsilon = 0.5
            elif n_samples < 50:
                min_cluster_size = max(3, self.min_cluster_size)
                min_samples = max(2, self.min_samples)
                cluster_selection_epsilon = 0.6
            elif n_samples < 100:
                min_cluster_size = max(5, self.min_cluster_size)
                min_samples = max(3, self.min_samples)
                cluster_selection_epsilon = 0.7
            else:
                min_cluster_size = max(8, self.min_cluster_size)
                min_samples = max(4, self.min_samples)
                cluster_selection_epsilon = 0.8

            logging.info(
                f"自适应聚类参数: min_cluster_size={min_cluster_size}, "
                f"min_samples={min_samples}, "
                f"cluster_selection_epsilon={cluster_selection_epsilon}")
        else:
            # 使用用户提供的固定参数
            min_cluster_size = self.min_cluster_size
            min_samples = self.min_samples
            cluster_selection_epsilon = 0.7

        return min_cluster_size, min_samples, cluster_selection_epsilon

    # 将噪声点分配到最相似的簇
    def _reassign_noise_points(self, vectors, clusters, cluster_centers):
        """将噪声点分配到最相似的簇

        Args:
            vectors: 向量数据
            clusters: 聚类结果
            cluster_centers: 簇中心

        Returns:
            numpy.ndarray: 重新分配后的聚类结果
        """
        # 创建结果的副本
        new_clusters = clusters.copy()

        # 找出噪声点
        noise_indices = np.where(clusters == -1)[0]

        if len(noise_indices) == 0 or len(cluster_centers) == 0:
            return new_clusters

        logging.info(f"重新分配{len(noise_indices)}个噪声点...")

        # 计算每个噪声点与每个簇中心的距离
        for idx in noise_indices:
            min_dist = float('inf')
            closest_cluster = -1

            # 找到最近的簇
            for cluster_id, center in cluster_centers.items():
                dist = distance.cosine(vectors[idx], center)  # 使用余弦距离
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = cluster_id

            # 如果找到距离较近的簇且距离小于阈值，将噪声点归入该簇
            if closest_cluster != -1 and min_dist < 0.4:  # 阈值可调整
                new_clusters[idx] = closest_cluster
                logging.info(
                    f"噪声点 {idx} 重新分配到簇 {closest_cluster}, 距离: {min_dist:.4f}")
            else:
                # 否则保留为噪声点
                pass

        return new_clusters

    # 评估聚类质量
    def _evaluate_clustering(self, vectors, clusters):
        """评估聚类质量

        Args:
            vectors: 向量数据
            clusters: 聚类结果

        Returns:
            dict: 包含各种聚类质量指标
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        import numpy as np

        # 去除噪声点
        non_noise_indices = np.where(clusters != -1)[0]
        if len(non_noise_indices) < 2 or len(
                set(clusters[non_noise_indices])) < 2:
            logging.warning("聚类评估失败: 聚类结果中没有足够多的非噪声样本或簇")
            return {
                'silhouette': -1,
                'calinski_harabasz': -1,
                'davies_bouldin': -1,
                'noise_ratio': sum(
                    clusters == -1) / len(clusters) if len(clusters) > 0 else 1}

        try:
            # 提取非噪声点的向量和簇标签
            non_noise_vectors = vectors[non_noise_indices]
            non_noise_clusters = clusters[non_noise_indices]

            # 计算轮廓系数 (越接近1越好)
            silhouette = silhouette_score(
                non_noise_vectors,
                non_noise_clusters) if len(non_noise_indices) > 1 else -1

            # 计算Calinski-Harabasz指数 (值越大越好)
            calinski_harabasz = calinski_harabasz_score(
                non_noise_vectors, non_noise_clusters)

            # 计算Davies-Bouldin指数 (值越小越好)
            davies_bouldin = davies_bouldin_score(
                non_noise_vectors, non_noise_clusters)

            # 噪声点比例
            noise_ratio = sum(clusters == -1) / \
                len(clusters) if len(clusters) > 0 else 1

            metrics = {
                'silhouette': silhouette,
                'calinski_harabasz': calinski_harabasz,
                'davies_bouldin': davies_bouldin,
                'noise_ratio': noise_ratio
            }

            logging.info(f"聚类质量评估: {metrics}")
            return metrics
        except Exception as e:
            logging.error(f"聚类评估出错: {e}")
            return {
                'silhouette': -1,
                'calinski_harabasz': -1,
                'davies_bouldin': -1,
                'noise_ratio': sum(
                    clusters == -1) / len(clusters) if len(clusters) > 0 else 1}

    # 优化聚类参数
    def _optimize_clustering_params(self, vectors):
        """尝试不同的聚类参数，选择最优的一组

        Args:
            vectors: 向量数据

        Returns:
            tuple: 最优的(min_cluster_size, min_samples, cluster_selection_epsilon)
        """
        if len(vectors) < 10:
            logging.info("样本量太小，使用默认参数")
            return self._get_adaptive_clustering_params(vectors)

        # 参数网格
        min_cluster_sizes = [3, 5, 8]
        min_samples_list = [1, 3, 5]
        epsilons = [0.5, 0.7, 0.9]

        best_score = -float('inf')
        best_params = None

        # 只测试部分参数组合，避免耗时过长
        for mcs in min_cluster_sizes:
            for ms in min_samples_list:
                if ms > mcs:
                    continue  # min_samples不应大于min_cluster_size

                for eps in epsilons:
                    try:
                        clusterer = HDBSCAN(
                            min_cluster_size=mcs,
                            min_samples=ms,
                            metric='euclidean',
                            cluster_selection_epsilon=eps,
                            alpha=0.8
                        )

                        clusters = clusterer.fit_predict(vectors)

                        # 如果只有噪声点或只有一个簇，则跳过
                        if len(set(clusters) - set([-1])) < 2:
                            continue

                        metrics = self._evaluate_clustering(vectors, clusters)

                        # 计算综合分数 (需要综合考虑多个指标)
                        # 轮廓系数越高越好，Davies-Bouldin越低越好
                        # 噪声比例适中为佳（既不要太高，也不要太低）
                        score = metrics['silhouette'] - 0.5 * metrics['davies_bouldin'] - \
                            2.0 * abs(metrics['noise_ratio'] - 0.2)

                        logging.info(
                            f"参数 (mcs={mcs}, ms={ms}, eps={eps}) 得分: {score}")

                        if score > best_score:
                            best_score = score
                            best_params = (mcs, ms, eps)
                    except Exception as e:
                        logging.error(
                            f"参数 (mcs={mcs}, ms={ms}, eps={eps}) 测试失败: {e}")

        if best_params:
            logging.info(f"找到最优参数: {best_params}, 得分: {best_score}")
            return best_params
        else:
            logging.warning("参数优化失败，使用自适应参数")
            return self._get_adaptive_clustering_params(vectors)

    # 使用HDBSCAN对词向量进行聚类
    def _nonNEsCluster(self):
        # 检查词向量数量是否足够进行聚类
        if not self.vector_dict or len(
                self.vector_dict) < self.min_cluster_size:
            logging.warning(f"词向量数量不足: {len(self.vector_dict)}, 无法进行有效聚类")
            return self.output_data

        # 准备数据
        keys = list(self.vector_dict.keys())
        vectors = np.array([self.vector_dict[k]['vector'] for k in keys])
        phrases = [self.vector_dict[k]['phrase'] for k in keys]

        logging.info(f"准备聚类数据: {len(vectors)}个向量")

        # UMAP 降维 - 降低维度以提高聚类效果
        n_components = min(50, max(2, vectors.shape[0] // 3))  # 根据数据量动态调整维度
        n_neighbors = min(10, max(2, vectors.shape[0] // 5))  # 根据数据量动态调整邻居数
        min_dist = 0.1

        logging.info(
            f"UMAP配置: n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}")

        try:
            # 确保向量数量足够进行降维
            if vectors.shape[0] > n_components + 2:
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=min_dist,
                    metric='cosine',
                    random_state=42  # 为了可复现性
                )
                reduced_vectors = reducer.fit_transform(vectors)
                logging.info(
                    f"UMAP降维完成: {vectors.shape} -> {reduced_vectors.shape}")
            else:
                logging.info(f"向量数量({vectors.shape[0]})不足以进行有效降维，跳过UMAP")
                reduced_vectors = vectors  # 如果向量数量不足，则不进行降维

        except Exception as e:
            logging.error(f"UMAP降维失败: {e}, 使用原始向量")
            reduced_vectors = vectors

        # 优化聚类参数
        if len(vectors) > 10:
            try:
                logging.info("尝试优化聚类参数...")
                min_cluster_size, min_samples, cluster_selection_epsilon = self._optimize_clustering_params(
                    reduced_vectors)
            except Exception as e:
                logging.error(f"参数优化失败: {e}, 使用自适应参数")
                min_cluster_size, min_samples, cluster_selection_epsilon = self._get_adaptive_clustering_params(
                    reduced_vectors)
        else:
            # 样本量小时使用自适应参数
            min_cluster_size, min_samples, cluster_selection_epsilon = self._get_adaptive_clustering_params(
                reduced_vectors)

        # 使用HDBSCAN进行聚类
        try:
            logging.info(
                f"使用HDBSCAN聚类, 参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',  # 在降维后的空间通常使用欧氏距离
                cluster_selection_epsilon=cluster_selection_epsilon,
                alpha=0.8,
                cluster_selection_method='eom'  # 使用密度局部极大值方法
            )

            clusters = clusterer.fit_predict(reduced_vectors)

            # 计算聚类质量指标
            metrics = self._evaluate_clustering(reduced_vectors, clusters)

            distinct_clusters = list(set(clusters) - set([-1]))
            logging.info(
                f"聚类结果: {len(distinct_clusters)}个簇, {sum(clusters == -1)}个噪声点")

            # 如果聚类质量不佳或只得到1个簇，尝试调整参数重新聚类
            if (metrics['silhouette'] < 0.1 or len(
                    distinct_clusters) < 2) and len(vectors) > 10:
                logging.warning("聚类质量不佳，尝试调整参数...")
                # 使用更宽松的参数
                min_cluster_size = max(2, min_cluster_size - 1)
                min_samples = max(1, min_samples - 1)
                cluster_selection_epsilon = min(
                    0.9, cluster_selection_epsilon + 0.1)

                clusterer = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    alpha=0.8,
                    cluster_selection_method='eom'
                )

                clusters = clusterer.fit_predict(reduced_vectors)
                logging.info(
                    f"重新聚类: {len(set(clusters) - set([-1]))}个簇, {sum(clusters == -1)}个噪声点")

        except Exception as e:
            logging.error(f"HDBSCAN聚类失败: {e}")
            # 如果HDBSCAN失败，使用简单的K-means聚类作为备选
            from sklearn.cluster import KMeans

            try:
                # 估计合适的簇数量
                k = min(5, max(2, len(vectors) // 3))
                kmeans = KMeans(n_clusters=k, random_state=42)
                clusters = kmeans.fit_predict(reduced_vectors)
                logging.info(f"使用K-means聚类作为备选, k={k}")
            except BaseException:
                # 如果所有聚类方法都失败，将所有点分配到一个簇
                clusters = np.zeros(len(vectors), dtype=int)
                logging.warning("所有聚类方法都失败，将所有点分配到一个簇")

        # 计算簇中心
        cluster_centers = {}
        for cluster_id in set(clusters) - set([-1]):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_vectors = reduced_vectors[cluster_indices]
            cluster_centers[cluster_id] = np.mean(cluster_vectors, axis=0)

        # 尝试重新分配噪声点
        if -1 in clusters and len(set(clusters) - set([-1])) > 0:
            original_noise_count = sum(clusters == -1)
            clusters = self._reassign_noise_points(
                reduced_vectors, clusters, cluster_centers)
            new_noise_count = sum(clusters == -1)
            logging.info(f"噪声点处理: {original_noise_count} -> {new_noise_count}")

        # 为每个簇提取标签
        cluster_phrases = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id == -1:  # 处理噪声点
                if 'noise' not in cluster_phrases:
                    cluster_phrases['noise'] = []
                cluster_phrases['noise'].append(phrases[i])
            else:
                if cluster_id not in cluster_phrases:
                    cluster_phrases[cluster_id] = []
                cluster_phrases[cluster_id].append(phrases[i])

        # 使用改进的方法提取簇标签
        cluster_labels = {}
        for cluster_id, phrase_list in cluster_phrases.items():
            if cluster_id == 'noise':
                continue

            # 使用TF-IDF提取代表性标签
            cluster_labels[cluster_id] = self._get_representative_label(
                phrase_list, cluster_id)
            logging.info(
                f"簇 {cluster_id} 标签: {cluster_labels[cluster_id]}, 包含 {len(phrase_list)} 个短语")

        # 单独处理噪声点
        if 'noise' in cluster_phrases:
            noise_phrases = cluster_phrases['noise']

            # 对每个噪声点，尝试使用短语本身作为标签
            for i, phrase in enumerate(noise_phrases):
                noise_key = f"noise_{i}_{phrase}"
                cluster_phrases[noise_key] = [phrase]
                # 尝试提取更有意义的标签，如果失败就使用原短语
                word_list = jieba.lcut(phrase)
                filtered_words = [w for w in word_list if len(
                    w) > 1 and w not in self.stopwords]
                if filtered_words:
                    cluster_labels[noise_key] = filtered_words[0]  # 使用第一个有意义的词
                else:
                    cluster_labels[noise_key] = phrase

            # 删除噪声集合
            del cluster_phrases['noise']

        # 应用簇标签到原始数据
        for idx, item_list in self.output_data.items():
            for i, token_list in enumerate(item_list.get('srl_mappings', [])):
                for j, phrase_list in enumerate(token_list):
                    phrase = phrase_list[0]
                    role = phrase_list[1]
                    entity = phrase_list[2]
                    if role in ['ARG0', 'ARG1'] and entity is None:
                        # 如果没有映射到实体，则尝试使用非实体词的映射
                        matched = False
                        for cluster_id, cluster_phrase in cluster_phrases.items():
                            if phrase in cluster_phrase:
                                # 替换成非实体词的映射
                                self.output_data[idx]['srl_mappings'][i][j][2] = {
                                    'original': self.output_data[idx]['srl_mappings'][i][j][0],
                                    'entity_type': 'non-NE',
                                    # 保存簇ID以便于后续分析
                                    'cluster_id': str(cluster_id)
                                }
                                self.output_data[idx]['srl_mappings'][i][j][0] = cluster_labels[cluster_id]
                                matched = True
                                break

        logging.info("HDBSCAN聚类和标签应用完成")
        return self.output_data

    # 简化的接口函数，直接调用处理流程
    def process(self, tokens_dict=None):
        return self._filterNonNEs()._generateWeightDict(
        )._generateWeightedAverageVector()._nonNEsCluster()


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
    mapper = nonNEsHDBSCANMapper(input_data=tokens_dict)
    result = mapper.process()
    print('输出数据:')
    print(result)


if __name__ == "__main__":
    main()

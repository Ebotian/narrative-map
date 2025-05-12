import { ref } from "vue";

/**
 * 网络数据处理的可复用逻辑
 * @returns {Object} 包含网络数据相关状态和方法
 */
export function useNetworkData() {
	// 原始数据
	const originalData = ref(null);

	// 处理后的数据（用于显示）
	const processedData = ref(null);

	// 加载状态
	const loading = ref(false);

	/**
	 * 加载网络数据
	 * @param {Object} data - JSON格式的网络数据
	 */
	const loadNetworkData = (data) => {
		loading.value = true;

		try {
			// 保存原始数据
			originalData.value = data;

			// 设置处理后的数据（当前实现直接使用原始数据）
			processedData.value = data;
		} catch (error) {
			console.error("处理网络数据时出错:", error);
		} finally {
			loading.value = false;
		}
	};

	/**
	 * 加载默认的JSON数据
	 */
	const loadDefaultData = async () => {
		loading.value = true;

		try {
			// 从默认路径加载数据
			const response = await fetch("/data/narrative_network.json");

			if (!response.ok) {
				throw new Error(`HTTP错误: ${response.status}`);
			}

			const data = await response.json();
			loadNetworkData(data);
		} catch (error) {
			console.error("加载默认数据时出错:", error);
			alert("加载默认数据失败: " + error.message);
		} finally {
			loading.value = false;
		}
	};

	/**
	 * 根据时间范围过滤数据
	 * @param {Object} filterParams - 过滤参数
	 */
	const filterByTimeRange = (filterParams = {}) => {
		if (!originalData.value) return;

		const { mode = 'percentage', centerTime, windowPercentage = 5, startTime, endTime } = filterParams;

		// 全部模式直接返回原始数据
		if (mode === 'all') {
			processedData.value = originalData.value;
			return;
		}

		loading.value = true;

		try {
			const timeRange = originalData.value.timeRange;
			if (!timeRange || !timeRange.start || !timeRange.end) {
				// 如果原始数据没有时间范围信息，直接使用原始数据
				processedData.value = originalData.value;
				return;
			}

			// 计算时间范围
			const minTime = new Date(timeRange.start).getTime();
			const maxTime = new Date(timeRange.end).getTime();
			const totalTimeRange = maxTime - minTime;

			let startTimestamp, endTimestamp;

			if (mode === 'percentage') {
				// 百分比模式
				if (!centerTime) {
					processedData.value = originalData.value;
					return;
				}

				const windowSize = totalTimeRange * (windowPercentage / 100);
				const centerTimestamp =
					typeof centerTime === "number"
						? centerTime
						: new Date(centerTime).getTime();
				startTimestamp = Math.max(minTime, centerTimestamp - windowSize / 2);
				endTimestamp = Math.min(maxTime, centerTimestamp + windowSize / 2);
			} else if (mode === 'range') {
				// 区间模式
				if (!startTime || !endTime) {
					processedData.value = originalData.value;
					return;
				}

				startTimestamp = typeof startTime === "number"
					? startTime
					: new Date(startTime).getTime();
				endTimestamp = typeof endTime === "number"
					? endTime
					: new Date(endTime).getTime();
			}

			// 筛选节点
			const filteredNodes = originalData.value.nodes.filter((node) => {
				// 检查节点是否有timestamps数组
				const timestamps =
					node.timestamps || (node.timestamp ? [node.timestamp] : []);
				if (timestamps.length === 0) return true; // 保留没有时间戳的节点

				// 多时间戳逻辑：如果节点至少有一个时间戳在时间窗口内，则保留该节点
				return timestamps.some((timestamp) => {
					const nodeTime = new Date(timestamp).getTime();
					return nodeTime >= startTimestamp && nodeTime <= endTimestamp;
				});
			});

			// 提取过滤后的节点ID，用于边的筛选
			const filteredNodeIds = new Set(filteredNodes.map((node) => node.id));

			// 筛选边
			const filteredEdges = originalData.value.edges.filter((edge) => {
				// 首先检查边的两端节点是否都存在于过滤后的节点中
				if (!filteredNodeIds.has(edge.from) || !filteredNodeIds.has(edge.to)) {
					return false;
				}

				// 检查边是否有timestamps数组
				const timestamps =
					edge.timestamps || (edge.timestamp ? [edge.timestamp] : []);
				if (timestamps.length === 0) return true; // 保留没有时间戳的边

				// 多时间戳逻辑：如果边至少有一个时间戳在时间窗口内，则保留该边
				return timestamps.some((timestamp) => {
					const edgeTime = new Date(timestamp).getTime();
					return edgeTime >= startTimestamp && edgeTime <= endTimestamp;
				});
			});

			// 更新处理后的数据
			processedData.value = {
				...originalData.value,
				nodes: filteredNodes,
				edges: filteredEdges,
				timeRange: {
					...timeRange,
					...(mode === 'percentage' ? {
						center: new Date(centerTimestamp).toISOString(),
						windowPercentage
					} : {
						center: new Date((startTimestamp + endTimestamp) / 2).toISOString()
					}),
					filterStart: new Date(startTimestamp).toISOString(),
					filterEnd: new Date(endTimestamp).toISOString(),
				},
			};

			// 添加过滤模式信息
			processedData.value.timeRange.mode = mode;
		} catch (error) {
			console.error("过滤数据时出错:", error);
			// 发生错误时使用原始数据
			processedData.value = originalData.value;
		} finally {
			loading.value = false;
		}
	};

	return {
		originalData,
		processedData,
		loading,
		loadNetworkData,
		loadDefaultData,
		filterByTimeRange,
	};
}

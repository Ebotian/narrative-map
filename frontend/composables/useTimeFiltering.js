import { ref, watch, computed } from "vue";

/**
 * 时间过滤相关的可复用逻辑
 * @param {Ref} originalData - 原始网络数据的响应式引用
 * @param {Function} filterCallback - 应用时间过滤的回调函数
 * @returns {Object} 包含时间过滤相关状态和方法
 */
export function useTimeFiltering(originalData, filterCallback) {
	// 时间范围
	const timeRange = computed(() => {
		if (!originalData.value || !originalData.value.timeRange) {
			return { start: null, end: null };
		}
		return {
			start: originalData.value.timeRange.start,
			end: originalData.value.timeRange.end,
		};
	});

	// 当前选择的时间点
	const currentTimePoint = ref(null);

	// 时间窗口百分比
	const timeWindowPercentage = ref(5);

	// 区间模式下的开始和结束时间
	const startTime = ref(null);
	const endTime = ref(null);

	// 观察原始数据变化，重置当前时间点
	watch(originalData, (newData) => {
		if (
			newData &&
			newData.timeRange &&
			newData.timeRange.start &&
			newData.timeRange.end
		) {
			// 计算中间时间点
			const startTime = new Date(newData.timeRange.start).getTime();
			const endTime = new Date(newData.timeRange.end).getTime();
			currentTimePoint.value = startTime + (endTime - startTime) / 2;
		} else {
			currentTimePoint.value = null;
		}
	});

	/**
	 * 应用时间过滤
	 * @param {Object} filterParams - 过滤参数
	 */
	const applyTimeFilter = (filterParams = {}) => {
		const {
			mode = 'percentage',
			centerTime = currentTimePoint.value,
			windowPercentage = timeWindowPercentage.value,
			startTime: newStartTime = startTime.value,
			endTime: newEndTime = endTime.value,
		} = filterParams;

		// 更新状态
		currentTimePoint.value = centerTime;
		timeWindowPercentage.value = windowPercentage;
		startTime.value = newStartTime;
		endTime.value = newEndTime;

		// 调用过滤回调
		if (typeof filterCallback === "function") {
			if (mode === 'range') {
				filterCallback({
					mode: 'range',
					startTime: newStartTime,
					endTime: newEndTime
				});
			} else {
				// 将百分比模式转换为时间区间
				if (!timeRange.value || !timeRange.value.start || !timeRange.value.end) {
					console.warn('时间范围未初始化，跳过时间过滤');
					return;
				}

				const { start, end } = timeRange.value;
				const startTime = new Date(start).getTime();
				const endTime = new Date(end).getTime();
				const windowSize = (endTime - startTime) * (windowPercentage / 100);
				const windowStart = centerTime - windowSize / 2;
				const windowEnd = centerTime + windowSize / 2;

				filterCallback({
					mode: 'range',
					startTime: windowStart,
					endTime: windowEnd
				});
			}
		}
	};

	/**
	 * 重置时间过滤器
	 */
	/**
	 * 重置时间过滤器
	 */
	const resetTimeFilter = () => {
		if (originalData.value && originalData.value.timeRange) {
			const start = new Date(originalData.value.timeRange.start).getTime();
			const end = new Date(originalData.value.timeRange.end).getTime();

			// 重置为中间时间点
			currentTimePoint.value = start + (end - start) / 2;
			startTime.value = start;
			endTime.value = end;

			// 重置窗口百分比
			timeWindowPercentage.value = 5;

			// 应用过滤
			applyTimeFilter();
		}
	};

	/**
	 * 格式化时间显示
	 * @param {string|number} timestamp - 时间戳或ISO字符串
	 * @returns {string} 格式化后的时间字符串
	 */
	const formatDateTime = (timestamp) => {
		if (!timestamp) return "";

		try {
			const date = new Date(timestamp);
			return date.toLocaleString("zh-CN", {
				year: "numeric",
				month: "2-digit",
				day: "2-digit",
				hour: "2-digit",
				minute: "2-digit",
				second: "2-digit",
			});
		} catch (error) {
			console.error("格式化时间出错:", error);
			return String(timestamp);
		}
	};

	return {
		timeRange,
		currentTimePoint,
		timeWindowPercentage,
		startTime,
		endTime,
		applyTimeFilter,
		resetTimeFilter,
		formatDateTime,
	};
}

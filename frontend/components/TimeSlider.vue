<template>
  <div class="time-range-container">
    <div class="mode-header">
      <h6 class="time-heading">{{ modeTitle }}</h6>
    </div>

    <!-- 时间轴滑块 -->
    <div ref="timeSliderRef" class="time-slider"></div>

    <!-- 当前时间点/区间显示 -->
    <div class="current-time-display">
      <span v-if="mode === 'percentage'" class="time-badge">{{ formattedCurrentTime }}</span>
      <div v-else-if="mode === 'range'" class="range-time-display">
        <span class="time-badge">{{ formattedStartTime }}</span>
        <span class="time-separator">至</span>
        <span class="time-badge">{{ formattedEndTime }}</span>
      </div>
    </div>

    <!-- 仅在百分比模式下显示窗口调整 -->
    <div v-if="mode === 'percentage'" class="time-window-container">
      <div class="window-controls">
        <label for="timeWindowRange" class="form-label">时间窗口范围:</label>
        <input type="range" class="window-range" id="timeWindowRange" min="1" max="20" step="1"
          v-model="timeWindowPercentage" @input="handleWindowChange">
        <span class="window-badge">±{{ timeWindowPercentage }}%</span>
      </div>
    </div>

    <!-- 时间范围信息显示 -->
    <div class="time-range-labels">
      <span class="time-label">{{ formattedMinTime }}</span>
      <span class="time-label">{{ formattedMaxTime }}</span>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, watch, nextTick } from 'vue';
// 直接导入 noUiSlider 及其样式
import noUiSlider from 'nouislider';
import 'nouislider/distribute/nouislider.css';

export default {
  name: 'TimeSlider',
  props: {
    timeRange: {
      type: Object,
      required: true
    },
    mode: {
      type: String,
      default: 'percentage', // 'percentage' 或 'range'
      validator: value => ['percentage', 'range'].includes(value)
    }
  },
  emits: ['updateTimeFilter'],
  setup(props, { emit }) {
    // 引用DOM元素
    const timeSliderRef = ref(null);

    // 时间滑块实例
    let sliderInstance = null;

    // 状态 - 用于百分比模式
    const currentTime = ref(null);
    const timeWindowPercentage = ref(5);

    // 状态 - 用于区间模式
    const startTime = ref(null);
    const endTime = ref(null);

    // 根据模式显示不同的标题
    const modeTitle = computed(() => {
      return props.mode === 'percentage' ? '时间点选择' : '时间区间选择';
    });

    // 格式化时间显示
    const formattedCurrentTime = computed(() => {
      if (!currentTime.value) return '请选择时间点';
      return new Date(currentTime.value).toLocaleString('zh-CN');
    });

    const formattedStartTime = computed(() => {
      if (!startTime.value) return '起始时间';
      return new Date(startTime.value).toLocaleString('zh-CN');
    });

    const formattedEndTime = computed(() => {
      if (!endTime.value) return '结束时间';
      return new Date(endTime.value).toLocaleString('zh-CN');
    });

    const formattedMinTime = computed(() => {
      if (!props.timeRange.start) return '';
      return new Date(props.timeRange.start).toLocaleString('zh-CN');
    });

    const formattedMaxTime = computed(() => {
      if (!props.timeRange.end) return '';
      return new Date(props.timeRange.end).toLocaleString('zh-CN');
    });

    // 初始化时间滑块
    const initTimeSlider = () => {
      if (!props.timeRange.start || !props.timeRange.end) return;

      // 确保DOM元素已挂载
      if (!timeSliderRef.value) return;

      // 转换为时间戳
      const minTime = new Date(props.timeRange.start).getTime();
      const maxTime = new Date(props.timeRange.end).getTime();

      // 清除旧实例
      if (sliderInstance) {
        sliderInstance.destroy();
      }

      // 根据模式创建不同的滑块
      if (props.mode === 'percentage') {
        // 计算中间值作为初始值
        const centerTime = minTime + (maxTime - minTime) / 2;
        currentTime.value = centerTime;

        // 创建单点滑块
        sliderInstance = noUiSlider.create(timeSliderRef.value, {
          start: [centerTime],
          connect: false,
          range: {
            'min': minTime,
            'max': maxTime
          },
          step: 1000 * 60, // 1分钟步长
          tooltips: [
            {
              to: function (value) {
                return new Date(parseInt(value)).toLocaleString('zh-CN');
              }
            }
          ]
        });

        // 添加事件监听
        sliderInstance.on('update', function (values) {
          currentTime.value = parseInt(values[0]);
        });

      } else if (props.mode === 'range') {
        // 设置初始区间为整个时间范围的中间30%
        const timeSpan = maxTime - minTime;
        const initialStart = minTime + timeSpan * 0.35;
        const initialEnd = minTime + timeSpan * 0.65;

        startTime.value = initialStart;
        endTime.value = initialEnd;

        // 创建区间滑块
        sliderInstance = noUiSlider.create(timeSliderRef.value, {
          start: [initialStart, initialEnd],
          connect: true, // 连接两个手柄之间
          range: {
            'min': minTime,
            'max': maxTime
          },
          step: 1000 * 60, // 1分钟步长
          tooltips: [
            {
              to: function (value) {
                return new Date(parseInt(value)).toLocaleString('zh-CN');
              }
            },
            {
              to: function (value) {
                return new Date(parseInt(value)).toLocaleString('zh-CN');
              }
            }
          ]
        });

        // 添加事件监听
        sliderInstance.on('update', function (values, handle) {
          if (handle === 0) {
            startTime.value = parseInt(values[0]);
          } else {
            endTime.value = parseInt(values[1]);
          }
        });
      }

      // 添加滑块变化结束事件，自动应用筛选
      sliderInstance.on('change', function () {
        emitTimeFilterUpdate();
      });
    };

    // 处理时间窗口百分比变化
    const handleWindowChange = () => {
      emitTimeFilterUpdate();
    };

    // 计算最小和最大时间
    const minTime = computed(() => {
      return props.timeRange?.start ? new Date(props.timeRange.start).getTime() : null;
    });

    const maxTime = computed(() => {
      return props.timeRange?.end ? new Date(props.timeRange.end).getTime() : null;
    });

    // 发出更新事件
    const emitTimeFilterUpdate = () => {
      if (props.mode === 'percentage' && currentTime.value) {
        emit('updateTimeFilter', {
          mode: 'percentage',
          centerTime: currentTime.value,
          windowPercentage: parseInt(timeWindowPercentage.value)
        });
      } else if (props.mode === 'range' && startTime.value && endTime.value) {
        emit('updateTimeFilter', {
          mode: 'range',
          startTime: startTime.value,
          endTime: endTime.value
        });
      }
    };

    // 监听时间范围变化
    watch(() => props.timeRange, () => {
      nextTick(() => {
        initTimeSlider();
      });
    }, { deep: true });

    // 监听模式变化
    watch(() => props.mode, () => {
      nextTick(() => {
        initTimeSlider();
      });
    });

    // 组件挂载后初始化
    onMounted(() => {
      initTimeSlider();
    });

    return {
      timeSliderRef,
      currentTime,
      startTime,
      endTime,
      timeWindowPercentage,
      modeTitle,
      formattedCurrentTime,
      formattedStartTime,
      formattedEndTime,
      formattedMinTime,
      formattedMaxTime,
      handleWindowChange
    };
  }
}
</script>

<style scoped>
.time-range-container {
  padding: 20px 15px;
  margin-bottom: 15px;
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

.mode-header {
  display: flex;
  justify-content: flex-start;
  align-items: center;
  margin-bottom: 15px;
}

.time-heading {
  font-weight: 500;
  margin: 0;
}

.time-slider {
  margin: 25px 0;
  height: 6px;
}

.current-time-display {
  display: flex;
  justify-content: center;
  margin-top: 10px;
}

.range-time-display {
  display: flex;
  align-items: center;
  gap: 10px;
}

.time-separator {
  color: #6c757d;
}

.time-badge {
  background-color: #0d6efd;
  color: white;
  padding: 5px 10px;
  border-radius: 4px;
  font-size: 14px;
}

.time-window-container {
  margin-top: 20px;
}

.window-controls {
  display: flex;
  align-items: center;
  gap: 10px;
}

.window-range {
  flex: 1;
}

.window-badge {
  background-color: #6c757d;
  color: white;
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 13px;
}

.time-range-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 10px;
  color: #6c757d;
  font-size: 13px;
}

/* 自定义nouislider样式通过全局CSS设置 */
</style>
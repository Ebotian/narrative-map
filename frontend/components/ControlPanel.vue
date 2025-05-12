<template>
  <div class="controls-container">
    <h2 class="title">叙事网络可视化</h2>

    <!-- 文件选择区域 -->
    <div class="file-controls">
      <div class="form-row">
        <label for="jsonFile" class="form-label">选择JSON文件：</label>
        <input class="form-control" type="file" id="jsonFile" accept=".json" @change="handleFileChange">
        <button class="btn btn-primary" @click="$emit('loadDefaultData')">
          加载默认数据
        </button>
      </div>
    </div>

    <!-- 时间范围信息 -->
    <div class="time-range-info" :class="{ 'has-data': timeRange.start && timeRange.end }">
      {{ timeRangeInfoText }}
    </div>

    <!-- 时间滑块组件 -->
    <time-slider v-if="timeRange.start && timeRange.end" :timeRange="timeRange" @updateTimeFilter="updateTimeFilter" />

    <!-- 图例 -->
    <div class="legend">
      <div class="legend-item" v-for="(color, type) in nodeColorMap" :key="type">
        <div class="color-box" :style="{ backgroundColor: color.color }"></div>
        <span>{{ color.label }}</span>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import TimeSlider from './TimeSlider.vue';

export default {
  name: 'ControlPanel',
  components: {
    TimeSlider
  },
  props: {
    timeRange: {
      type: Object,
      default: () => ({ start: null, end: null })
    }
  },
  emits: ['loadData', 'loadDefaultData', 'updateTimeFilter'],
  setup(props, { emit }) {
    // 节点颜色映射
    const nodeColorMap = {
      'PERSON': { color: '#FF9999', label: '人物 (PERSON)' },
      'LOCATION': { color: '#99CC99', label: '地点 (LOCATION)' },
      'ORGANIZATION': { color: '#6A9DFF', label: '组织 (ORGANIZATION)' },
      'non-NE': { color: '#DDDDDD', label: '非实体 (non-NE)' }
    };

    // 计算时间范围显示文本
    const timeRangeInfoText = computed(() => {
      if (props.timeRange.start && props.timeRange.end) {
        return `可用时间范围: ${props.timeRange.start} 至 ${props.timeRange.end}`;
      }
      return '请加载数据以查看可用的时间范围';
    });

    // 处理文件上传
    const handleFileChange = (event) => {
      const file = event.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const jsonData = JSON.parse(e.target.result);
            emit('loadData', jsonData);
          } catch (error) {
            console.error('解析JSON文件时出错:', error);
            alert('解析JSON文件时出错: ' + error.message);
          }
        };
        reader.readAsText(file);
      }
    };

    // 更新时间过滤器
    const updateTimeFilter = (filterParams) => {
      emit('updateTimeFilter', filterParams);
    };

    return {
      nodeColorMap,
      timeRangeInfoText,
      handleFileChange,
      updateTimeFilter
    };
  }
}
</script>

<style scoped>
.controls-container {
  background-color: #f8f9fa;
  padding: 15px;
  border-bottom: 1px solid #dee2e6;
}

.title {
  margin-bottom: 15px;
  font-size: 1.8rem;
}

.file-controls {
  margin-bottom: 15px;
}

.form-row {
  display: flex;
  gap: 15px;
  align-items: center;
}

.time-range-info {
  padding: 10px;
  background-color: #f1f8ff;
  border-left: 4px solid #0d6efd;
  margin-bottom: 15px;
}

.time-range-info.has-data {
  background-color: #d1ecf1;
  border-left-color: #17a2b8;
}

.legend {
  margin-top: 15px;
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 5px;
}

.color-box {
  width: 15px;
  height: 15px;
  display: inline-block;
  border: 1px solid #333;
}
</style>
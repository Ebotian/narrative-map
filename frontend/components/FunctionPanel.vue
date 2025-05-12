<template>
  <div class="function-panel" :class="{ 'collapsed': !activePanel }">
    <div v-if="activePanel === 'file'" class="panel-content">
      <h3>文件操作</h3>
      <div class="form-row">
        <label for="jsonFile" class="form-label">选择JSON文件：</label>
        <input class="form-control" type="file" id="jsonFile" accept=".json" @change="handleFileChange">
      </div>
      <div class="form-row">
        <button class="btn btn-primary" @click="$emit('loadDefaultData')">
          加载默认数据
        </button>
      </div>
    </div>

    <div v-else-if="activePanel === 'percentTime'" class="panel-content">
      <h3>百分比时间</h3>
      <div v-if="timeRange.start && timeRange.end" class="time-controls">
        <!-- 时间范围信息 -->
        <div class="time-range-info">
          可用时间范围: {{ timeRange.start }} 至 {{ timeRange.end }}
        </div>

        <!-- 百分比时间滑块 -->
        <time-slider :timeRange="timeRange" mode="percentage" @updateTimeFilter="updateTimeFilter" />
      </div>
      <div v-else class="no-data-message">
        请先加载数据以使用时间过滤功能
      </div>
    </div>

    <div v-else-if="activePanel === 'rangeTime'" class="panel-content">
      <h3>区间时间</h3>
      <div v-if="timeRange.start && timeRange.end" class="time-controls">
        <!-- 时间范围信息 -->
        <div class="time-range-info">
          可用时间范围: {{ timeRange.start }} 至 {{ timeRange.end }}
        </div>

        <!-- 区间时间选择器 -->
        <time-slider :timeRange="timeRange" mode="range" @updateTimeFilter="updateTimeFilter" />
      </div>
      <div v-else class="no-data-message">
        请先加载数据以使用时间过滤功能
      </div>
    </div>

    <button v-if="activePanel" class="collapse-btn" @click="$emit('closePanel')">
      折叠 &lt;
    </button>
  </div>
</template>

<script>
import { computed } from 'vue';
import TimeSlider from './TimeSlider.vue';

export default {
  name: 'FunctionPanel',
  components: {
    TimeSlider
  },
  props: {
    activePanel: {
      type: String,
      default: null
    },
    timeRange: {
      type: Object,
      default: () => ({ start: null, end: null })
    }
  },
  emits: ['loadData', 'loadDefaultData', 'updateTimeFilter', 'closePanel'],
  setup(props, { emit }) {
    // 节点颜色映射
    const nodeColorMap = {
      'PERSON': { color: '#FF9999', label: '人物 (PERSON)' },
      'LOCATION': { color: '#99CC99', label: '地点 (LOCATION)' },
      'ORGANIZATION': { color: '#6A9DFF', label: '组织 (ORGANIZATION)' },
      'non-NE': { color: '#DDDDDD', label: '非实体 (non-NE)' }
    };

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
      handleFileChange,
      updateTimeFilter
    };
  }
}
</script>

<style scoped>
.function-panel {
  height: 100%;
  background-color: #f8f9fa;
  border-right: 1px solid #dee2e6;
  transition: width 0.3s ease;
  position: relative;
  overflow-y: auto;
}

.function-panel.collapsed {
  display: none;
}

.panel-content {
  padding: 20px;
}

h3 {
  margin-bottom: 15px;
  color: #333;
  border-bottom: 1px solid #dee2e6;
  padding-bottom: 10px;
}

.form-row {
  margin-bottom: 15px;
}

.time-range-info {
  padding: 10px;
  background-color: #d1ecf1;
  border-left: 4px solid #17a2b8;
  margin-bottom: 15px;
}

.time-controls {
  margin-top: 15px;
}

.no-data-message {
  padding: 15px;
  background-color: #f8d7da;
  color: #721c24;
  border-radius: 4px;
  margin-top: 15px;
}

.collapse-btn {
  position: absolute;
  top: 50%;
  right: -15px;
  transform: translateY(-50%);
  background-color: #e8f4f8;
  border: 1px solid #b0d8e8;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  z-index: 10;
  font-size: 10px;
}

.collapse-btn:hover {
  background-color: #d1e7f0;
}
</style>
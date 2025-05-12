<template>
  <div id="app-container">
    <div class="layout-container">
      <!-- 左侧岛状图标栏，比例1 -->
      <div class="icon-column">
        <icon-bar @iconSelected="handleIconSelect" />
      </div>

      <!-- 中间可折叠功能区，展开时比例3，折叠时不显示 -->
      <div v-if="activeFunctionPanel" class="function-column">
        <function-panel :activePanel="activeFunctionPanel" :timeRange="timeRange" @loadData="loadNetworkData"
          @loadDefaultData="loadDefaultData" @updateTimeFilter="handleTimeFilterUpdate"
          @closePanel="activeFunctionPanel = null" />
      </div>

      <!-- 右侧网络图可视化区域，比例默认9，功能区展开时为6 -->
      <div class="visualization-column">
        <network-visualization :networkData="processedData" :loading="loading" />

        <!-- 图例说明 -->
        <div class="legend">
          <div class="legend-item" v-for="(color, type) in nodeColorMap" :key="type">
            <div class="color-box" :style="{ backgroundColor: color.color }"></div>
            <span>{{ color.label }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue';
import IconBar from './components/IconBar.vue';
import FunctionPanel from './components/FunctionPanel.vue';
import NetworkVisualization from './components/NetworkVisualization.vue';
import { useNetworkData } from './composables/useNetworkData';
import { useTimeFiltering } from './composables/useTimeFiltering';

export default {
  name: 'App',
  components: {
    IconBar,
    FunctionPanel,
    NetworkVisualization
  },
  setup() {
    // 网络数据相关状态和方法
    const {
      originalData,
      processedData,
      loading,
      loadNetworkData,
      loadDefaultData,
      filterByTimeRange
    } = useNetworkData();

    // 时间过滤相关逻辑
    const {
      timeRange,
      currentTimePoint,
      timeWindowPercentage,
      applyTimeFilter
    } = useTimeFiltering(originalData, filterParams => {
      filterByTimeRange(filterParams);
    });

    // 节点颜色映射
    const nodeColorMap = {
      'PERSON': { color: '#FF9999', label: '人物 (PERSON)' },
      'LOCATION': { color: '#99CC99', label: '地点 (LOCATION)' },
      'ORGANIZATION': { color: '#6A9DFF', label: '组织 (ORGANIZATION)' },
      'non-NE': { color: '#DDDDDD', label: '非实体 (non-NE)' }
    };

    // 当前激活的功能面板
    const activeFunctionPanel = ref(null);

    // 处理图标选择
    const handleIconSelect = (iconId) => {
      activeFunctionPanel.value = iconId;
    };

    // 处理时间过滤器更新事件
    const handleTimeFilterUpdate = (filterParams) => {
      applyTimeFilter(filterParams);
    };

    // 生命周期钩子
    onMounted(() => {
      // 页面加载后自动加载默认数据
      loadDefaultData();
    });

    return {
      // 数据相关
      originalData,
      processedData,
      loading,
      loadNetworkData,
      loadDefaultData,

      // 时间相关
      timeRange,
      currentTimePoint,
      timeWindowPercentage,
      handleTimeFilterUpdate,

      // UI相关
      activeFunctionPanel,
      handleIconSelect,
      nodeColorMap
    };
  }
}
</script>

<style>
#app-container {
  height: 100vh;
  width: 100vw;
  margin: 0;
  padding: 0;
  overflow: hidden;
}

.layout-container {
  display: flex;
  height: 100%;
  width: 100%;
}

.icon-column {
  flex: 1;
  max-width: 70px;
  border-right: 1px solid #e1e1e1;
  background-color: #f0f0f0;
}

.function-column {
  flex: 0 0 30%;
  border-right: 1px solid #e1e1e1;
  position: relative;
  overflow-y: auto;
}

.visualization-column {
  flex: 6;
  /* 当功能区展开时为6 */
  flex-grow: 1;
  /* 当功能区折叠时占据剩余空间 */
  position: relative;
  height: 100%;
  overflow: hidden;
}

.legend {
  position: absolute;
  bottom: 120px;
  right: 20px;
  background-color: rgba(255, 255, 255, 0.85);
  border-radius: 8px;
  padding: 10px 15px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 100;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.color-box {
  width: 15px;
  height: 15px;
  border: 1px solid #333;
  border-radius: 3px;
}
</style>
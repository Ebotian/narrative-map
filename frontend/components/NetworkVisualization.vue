<template>
  <div class="network-container">
    <div v-if="loading" class="loading-overlay">
      <div class="spinner"></div>
      <div class="loading-text">加载数据中...</div>
    </div>
    <div v-else-if="!networkData || !networkData.nodes || networkData.nodes.length === 0" class="empty-state">
      <div class="empty-message">
        <i class="network-icon">📊</i>
        <p>暂无网络数据，请加载数据后查看</p>
      </div>
    </div>
    <div ref="networkContainer" class="vis-network"></div>
  </div>
</template>

<script>
import { ref, onMounted, watch, nextTick } from 'vue';

export default {
  name: 'NetworkVisualization',
  props: {
    networkData: {
      type: Object,
      default: () => ({})
    },
    loading: {
      type: Boolean,
      default: false
    }
  },
  setup(props) {
    // 网络容器引用
    const networkContainer = ref(null);

    // 网络实例
    let network = null;

    // 格式化节点数据
    const formatNodes = (nodesData) => {
      if (!nodesData) return [];
      return nodesData.map(node => {
        // 生成时间戳信息
        let timestampsInfo = '';
        if (node.timestamps && node.timestamps.length > 0) {
          timestampsInfo = '<br><br><strong>时间点:</strong><br>' +
            node.timestamps.map(ts => new Date(ts).toLocaleString('zh-CN')).join('<br>');
        }

        return {
          id: node.id,
          label: node.label,
          value: node.value,
          shape: "dot",
          color: getNodeColor(node.type),
          borderWidth: 2,
          font: { color: "black" },
          title: `${node.id}\n${node.value}\n${node.type}`
        };
      });
    };

    // 格式化边数据
    const formatEdges = (edgesData) => {
      if (!edgesData) return [];
      return edgesData.map((edge, index) => {
        const label = edge.label;
        let count = 1;
        const match = label.match(/\((\d+)\)/);
        if (match) {
          count = parseInt(match[1]);
        }

        // 生成时间戳信息
        let timestampsInfo = '';
        if (edge.timestamps && edge.timestamps.length > 0) {
          timestampsInfo = '<br><br><strong>发生时间:</strong><br>' +
            edge.timestamps.map(ts => new Date(ts).toLocaleString('zh-CN')).join('<br>');
        }

        return {
          id: `${edge.from}_${edge.to}_${index}`,
          from: edge.from,
          to: edge.to,
          label: label,
          title: `${edge.from}->-${edge.to}(${index})\n${label}`,
          color: "#666666",
          width: 0.15 + count * 0.05,
          arrows: { to: { enabled: true, scaleFactor: 0.5 } },
          physics: false,
          smooth: edge.from === edge.to ? false : {
            enabled: Math.random() > 0.5,
            type: Math.random() > 0.5 ? "curvedCW" : "curvedCCW",
            roundness: 0.3 + Math.random() * 0.6
          }
        };
      });
    };

    // 根据节点类型设置颜色
    const getNodeColor = (type) => {
      const colorMap = {
        'PERSON': '#FF9999',
        'LOCATION': '#99CC99',
        'ORGANIZATION': '#6A9DFF',
        'non-NE': '#DDDDDD'
      };
      return colorMap[type] || '#DDDDDD';
    };

    // 初始化并绘制网络图
    const initNetwork = async () => {
      if (!networkContainer.value) return;

      try {
        // 动态导入vis-network
        const { Network, DataSet } = await import('vis-network/standalone');

        // 使用数据创建节点和边
        const nodes = new DataSet(formatNodes(props.networkData.nodes || []));
        const edges = new DataSet(formatEdges(props.networkData.edges || []));

        // 创建数据对象
        const data = { nodes, edges };

        // 配置选项
        const options = {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "multiselect": true,
            "tooltipDelay": 100
          },
          "nodes": {
            "scaling": {
              "min": 10,
              "max": 30
            },
            "margin": 10 // 节点边距
          },
          "edges": {
            "hoverWidth": 2,
            "selectionWidth": 3,
            "smooth": {
              "type": "continuous"
            }
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "iterations": 150,
              "updateInterval": 25
            },
            "barnesHut": {
              "gravitationalConstant": -5000, // 增加引力使节点更紧凑
              "springLength": 150, // 减少弹簧长度
              "springConstant": 0.08, // 增加弹簧常数
              "damping": 0.3
            }
          },
          "layout": {
            "improvedLayout": true
          }
        };

        // 清理旧的网络实例
        if (network) {
          network.destroy();
          network = null;
        }

        // 创建新的网络实例
        network = new Network(networkContainer.value, data, options);

        // 事件处理
        network.on('selectNode', (params) => {
          console.log('选中节点:', params);
        });
      } catch (error) {
        console.error('初始化网络图失败:', error);
      }
    };

    // 监听数据变化重新绘制
    watch(() => props.networkData, () => {
      if (props.networkData && !props.loading) {
        nextTick(() => {
          initNetwork();
        });
      }
    }, { deep: true });

    // 监听loading状态
    watch(() => props.loading, () => {
      if (!props.loading && props.networkData) {
        nextTick(() => {
          initNetwork();
        });
      }
    });

    // 组件挂载后初始化
    onMounted(() => {
      if (props.networkData && !props.loading) {
        initNetwork();
      }
    });

    // 添加窗口resize监听
    const handleResize = () => {
      if (network) {
        network.redraw();
        network.fit({
          animation: {
            duration: 300,
            easingFunction: 'easeInOutQuad'
          }
        });
      }
    };

    onMounted(() => {
      window.addEventListener('resize', handleResize);
      return () => {
        window.removeEventListener('resize', handleResize);
      };
    });

    return {
      networkContainer,
      handleResize
    };
  }
}
</script>

<style scoped>
.network-container {
  position: relative;
  flex: 1;
  height: 100%;
  min-height: 0; /* 允许flex容器收缩 */
}

.vis-network {
  width: 100%;
  height: 100%;
  min-height: 300px; /* 保留最小高度但更小 */
  background-color: #ffffff;
  border: 1px solid #dee2e6;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(255, 255, 255, 0.7);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid rgba(0, 0, 0, 0.1);
  border-radius: 50%;
  border-top-color: #0d6efd;
  animation: spin 1s linear infinite;
}

.loading-text {
  margin-top: 10px;
  font-size: 16px;
  color: #666;
}

.empty-state {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  min-height: 400px;
  background-color: #f8f9fa;
}

.empty-message {
  text-align: center;
  color: #6c757d;
}

.network-icon {
  font-size: 48px;
  display: block;
  margin-bottom: 10px;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}
</style>
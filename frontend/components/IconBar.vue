<template>
  <div class="icon-bar">
    <div v-for="(icon, index) in icons" :key="index" class="icon-item" :class="{ 'active': activeIcon === icon.id }"
      @click="selectIcon(icon.id)" :title="icon.title">
      <span class="icon">{{ icon.symbol }}</span>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue';

export default {
  name: 'IconBar',
  emits: ['iconSelected'],
  setup(props, { emit }) {
    const icons = [
      { id: 'file', symbol: '📂', title: '文件' },
      { id: 'percentTime', symbol: '⏱️', title: '百分比时间' },
      { id: 'rangeTime', symbol: '⌛', title: '区间时间' },
    ];

    const activeIcon = ref(null);

    const selectIcon = (iconId) => {
      // 如果点击当前激活的图标，则切换状态（关闭面板）
      if (activeIcon.value === iconId) {
        activeIcon.value = null;
      } else {
        activeIcon.value = iconId;
      }

      emit('iconSelected', activeIcon.value);
    };

    return {
      icons,
      activeIcon,
      selectIcon
    };
  }
}
</script>

<style scoped>
.icon-bar {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: #f0f0f0;
  padding: 10px 0;
}

.icon-item {
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 10px 0;
  cursor: pointer;
  border-radius: 50%;
  background-color: #e8f4f8;
  transition: all 0.2s ease;
}

.icon-item:hover {
  background-color: #d1e7f0;
  transform: scale(1.05);
}

.icon-item.active {
  background-color: #b0d8e8;
  box-shadow: 0 0 0 2px #6a9dff;
}

.icon {
  font-size: 1.8rem;
}
</style>
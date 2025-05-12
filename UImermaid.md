graph TD
    A[TimeSlider.vue] -->|emitTimeFilterUpdate| B(FunctionPanel.vue)
    B -->|updateTimeFilter| C(App.vue)
    C -->|handleTimeFilterUpdate| D[useTimeFiltering.js]
    D -->|applyTimeFilter| E[filterCallback]
    E --> F[NetworkVisualization.vue]

    subgraph useTimeFiltering.js
        D1[timeRange computed]
        D2[currentTimePoint]
        D3[timeWindowPercentage]
        D4[applyTimeFilter]
    end

    style D fill:#f9f,stroke:#333
    style D1 fill:#bbf,stroke:#333
    style D4 fill:#f96,stroke:#333
```

时间过滤流程说明：
1. TimeSlider组件触发时间变化事件
2. FunctionPanel接收并转发事件到App组件
3. App组件调用useTimeFiltering的applyTimeFilter方法
4. applyTimeFilter根据当前模式(百分比/区间)计算时间窗口
5. 调用filterCallback应用过滤到网络数据
6. NetworkVisualization组件接收过滤后的数据并更新视图

关键点：
- timeRange依赖于originalData的初始化
- 需要确保在访问timeRange.value时originalData已准备好
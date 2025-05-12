from pyvis.network import Network
from collections import defaultdict
import random


class SemanticNetworkVisualizer:
    def __init__(self):
        self.input_data = {}
        self.node_freq = defaultdict(int)
        self.edge_data = []
        self.net = Network(
            height="100vh",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=True)
        # 关闭pyvis的隐式边合并功能
        self.net.barnes_hut(
            gravity=-3000,
            central_gravity=0.3,
            spring_length=150)

    def process_data(self, input_data):
        self.input_data = input_data
        # 处理数据
        for _, triple in input_data.items():
            arg0 = triple["ARG0"]
            pred = triple["PRED"]
            arg1 = triple["ARG1"]

            self.node_freq[arg0] += 1
            self.node_freq[arg1] += 1

            # 检查是否已存在相同的关系
            found = False
            for i, edge in enumerate(self.edge_data):
                if edge["from"] == arg0 and edge["to"] == arg1 and edge["pred"] == pred:
                    self.edge_data[i]["freq"] += 1
                    found = True
                    break

            # 如果不存在，添加新的关系
            if not found:
                self.edge_data.append({
                    "from": arg0,
                    "to": arg1,
                    "pred": pred,
                    "freq": 1
                })

    def add_nodes(self):
        # 定义不同实体类型的颜色
        type_colors = {
            "PERSON": "#FF9999",
            "ORGANIZATION": "#6A9DFF",
            "LOCATION": "#99CC99",
            "DATE": "#FFCC99",
            "TIME": "#CC99FF",
        }

        # 默认颜色
        default_color = "#DDDDDD"

        # 存储节点的类型
        node_types = {}

        # 首先从边数据中收集节点类型信息
        for edge in self.edge_data:
            from_node = edge["from"]
            to_node = edge["to"]

            for i, triple in enumerate(self.input_data.values()):
                if triple["ARG0"] == from_node:
                    node_types[from_node] = triple["ARG0_type"]
                if triple["ARG1"] == to_node:
                    node_types[to_node] = triple["ARG1_type"]

        # 添加节点
        for node, freq in self.node_freq.items():
            # 获取节点类型，如果没有则使用默认颜色
            node_type = node_types.get(node, "")
            color = type_colors.get(node_type, default_color)

            self.net.add_node(
                node,
                label=f"{node}({freq})",
                title=f"类型: {node_type if node_type else '未知'}, 频次: {freq}",
                value=freq,  # 节点大小与频次相关
                color=color,  # 根据实体类型设置颜色
                borderWidth=2,
                font={"size": freq * 3 + 12}  # 字体大小与频次相关
            )

    def add_edges(self):
        # 添加边
        # 记录节点对之间有多少条边，用于计算不同的曲度
        node_pairs_count = defaultdict(int)

        # 首先计算每对节点之间有多少条不同的边
        for edge in self.edge_data:
            pair = (edge["from"], edge["to"])
            node_pairs_count[pair] += 1

        # 为每对节点的边分配不同曲度
        node_pairs_current = defaultdict(int)
        # 统一边的颜色为深灰色
        edge_color = "#666666"

        for edge in self.edge_data:
            pair = (edge["from"], edge["to"])
            pair_count = node_pairs_count[pair]
            current_index = node_pairs_current[pair]
            node_pairs_current[pair] += 1

            # 为每条边添加唯一ID，防止边被合并
            # 添加随机性确保ID真正唯一
            rand_suffix = random.randint(1000, 9999)
            edge_id = f"{edge['from']}_{edge['to']}_{edge['pred']}_{rand_suffix}"

            # 对于多条边，使用显著不同的曲度值
            if pair_count == 1:
                smooth = False
            else:
                # 设置更大的曲率区分
                smoothness = 0.3 + (current_index * 0.3)
                # 交替使用顺时针和逆时针
                curve_type = "curvedCW" if current_index % 2 == 0 else "curvedCCW"

                # 直接使用字典而不是字符串
                smooth = {
                    "enabled": True,
                    "type": curve_type,
                    "roundness": smoothness
                }

            self.net.add_edge(
                edge["from"],
                edge["to"],
                id=edge_id,
                title=f"{edge['pred']}({edge['freq']})",
                label=f"{edge['pred']}({edge['freq']})",  # 在标签中显示频数
                color=edge_color,  # 统一边的颜色
                width=min((edge["freq"] + 1) / 20, 2),  # 边宽与频次相关，加1确保最小宽度
                arrows={
                    "to": {
                      "enabled": True,
                        "scaleFactor": 0.5  # 设置箭头大小为默认的一半
                    }
                },
                smooth=smooth,
                physics=False  # 禁用物理引擎对此边的影响
            )

    def configure_layout(self):
        # 设置选项，避免覆盖单个边的设置
        self.net.set_options("""
        {
          "interaction": {
            "hover": true,
            "navigationButtons": true
          },
          "nodes": {
            "scaling": {
              "min": 10,
              "max": 30
            }
          },
          "edges": {
            "hoverWidth": 2,
            "selectionWidth": 3
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "iterations": 100
            },
            "barnesHut": {
              "gravitationalConstant": -3000,
              "springLength": 200,
              "springConstant": 0.04
            }
          }
        }
        """)

    def generate_network(
            self,
            input_data,
            output_file="semantic_network.html"):
        self.process_data(input_data)
        self.add_nodes()
        self.add_edges()
        self.configure_layout()
        self.net.save_graph(output_file)


# 示例输入数据
sample_input = {
    0: {"ARG0": "政府", "PRED": "提高", "ARG1": "税收"},
    1: {"ARG0": "企业", "PRED": "抗议", "ARG1": "政策"},
    2: {"ARG0": "政府", "PRED": "调整", "ARG1": "政策"},
    3: {"ARG0": "消费者", "PRED": "抵制", "ARG1": "企业"},
    4: {"ARG0": "政府", "PRED": "出台", "ARG1": "政策"},
    5: {"ARG0": "企业", "PRED": "响应", "ARG1": "政策"},
    6: {"ARG0": "政府", "PRED": "提高", "ARG1": "税收"},
    7: {"ARG0": "消费者", "PRED": "抗议", "ARG1": "税收"},
    8: {"ARG0": "政府", "PRED": "提高", "ARG1": "税收"},
    9: {"ARG0": "企业", "PRED": "抗议", "ARG1": "政策"},
    10: {"ARG0": "政府", "PRED": "调整", "ARG1": "政策"},
    11: {"ARG0": "消费者", "PRED": "抵制", "ARG1": "企业"},
    12: {"ARG0": "政府", "PRED": "出台", "ARG1": "政策"},
    13: {"ARG0": "企业", "PRED": "响应", "ARG1": "政策"},
    14: {"ARG0": "政府", "PRED": "提高", "ARG1": "税收"},
    15: {"ARG0": "消费者", "PRED": "抗议", "ARG1": "税收"},
    16: {"ARG0": "消费者", "PRED": "反对", "ARG1": "税收"},
    17: {"ARG0": "消费者", "PRED": "抵抗", "ARG1": "税收"},
}

# 使用示例
if __name__ == "__main__":
    visualizer = SemanticNetworkVisualizer()
    visualizer.generate_network(
        input_data=sample_input,
        output_file="test.html")

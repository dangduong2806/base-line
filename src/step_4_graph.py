import networkx as nx
from .step_3_graph_builder import IsomorphismEngine

class ReasoningGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.iso_engine = IsomorphismEngine()
        self.node_counter = 0  # Đếm số node để gán ID duy nhất

    def get_equivalent_node(self, step_content, potential_nodes):
        """
        Tìm trong các node ứng viên xem có node nào tương đương (Isomorphic)
        với step_content hiện tại không.
        """
        for node_id in potential_nodes:
            existing_content = self.graph.nodes[node_id]['content']
            # Sử dụng IsomorphismEngine để kiểm tra
            if self.iso_engine.are_equivalent(existing_content, step_content):
                return node_id
        return None
    
    def build_graph(self, all_paths):
        """
        all_paths: List các list verified steps.
        Mỗi step có: {'content': '...', 'confidence': 0.9}
        """
        # Reset graph mỗi lần build mới để tránh rác từ lần chạy trước
        self.graph.clear()
        self.node_counter = 0
        # node gốc
        root_id = "ROOT"
        self.graph.add_node(root_id, content="Start", count=1)

        for path in all_paths:
            current_parent_id = root_id

            for step in path:
                step_content = step['content']
                step_conf = step['confidence'] # xác suất

                # 1 lấy danh sách các con của node cha hiện tại
                children_ids = list(self.graph.successors(current_parent_id))

                # 2. Kiểm tra xem bước này đã tồn tại trong đám con chưa (Isomorphism Check)
                matched_node_id = self.get_equivalent_node(step_content, children_ids)

                if matched_node_id:
                    # --- NODE MERGING (Gộp node) ---
                    # Nếu đã có, tăng trọng số (Count) và cập nhật confidence
                    self.graph.nodes[matched_node_id]['count'] += 1
                    # Cộng dồn confidence để tính trung bình sau này
                    self.graph.nodes[matched_node_id]['total_confidence'] += step_conf # xác suất

                    # Cập nhật trọng số cạnh
                    if self.graph.has_edge(current_parent_id, matched_node_id):
                        self.graph[current_parent_id][matched_node_id]['weight'] += 1
                    current_parent_id = matched_node_id # đi tiếp xuống dưới
                else:
                    # Thêm node mới
                    # nếu chưa có tạo node mới
                    new_node_id = f"node_{self.node_counter}"
                    self.node_counter += 1

                    self.graph.add_node(
                        new_node_id,
                        content=step_content,
                        count=1,
                        total_confidence=step_conf
                    )
                    # Thêm cạnh từ cha xuống con mới
                    self.graph.add_edge(current_parent_id, new_node_id, weight=1)
                    current_parent_id = new_node_id  # đi tiếp xuống dưới
        return self.graph


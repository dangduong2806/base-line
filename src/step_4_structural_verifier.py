import networkx as nx
import numpy as np

class StructuralVerifier:
    def __init__(self, config):
        """
        Khởi tạo với Config để lấy trọng số (weight).
        Không truyền graph vào đây vì class này dùng để xử lý nhiều graph khác nhau.
        """
        self.config = config
        # Lấy trọng số pha trộn giữa điểm gốc (Local) và điểm cấu trúc (Global)
        # Mặc định là 0.5 (50-50)
        self.centrality_weight = config['structural'].get('centrality_weight', 0.5)

    # def verify_structure(self, graph):
    #     """
    #     Input: networkx.DiGraph (Raw Graph từ bước 3)
    #     Output: networkx.DiGraph (Graph đã được update điểm số final_score)
    #     """
    #     if graph.number_of_nodes() == 0:
    #         return graph  # Trả về nguyên bản nếu graph rỗng
        
    #     # 1. Tính toán PageRank hoặc Centrality
    #     # Trong paper họ dùng Laplacian relaxation, nhưng PageRank là một approximation tốt
    #     # để tìm ra các "trụ cột" trong luồng suy luận.
    #     centrality_scores = nx.pagerank(self.graph, weight='weight')

    #     for node in self.graph.nodes():
    #         # Lấy điểm cục bộ (từ Atomic/Logical step)
    #         local_conf = self.graph.nodes[node].get('total_confidence', 0)
    #         count = self.graph.nodes[node].get('count', 1)

    #         # 2. Kết hợp với Structural Score (Centrality)
    #         # Một node đúng thường nằm trên các luồng chính (centrality cao)
    #         struct_score = centrality_scores.get(node, 0)

    #         # Cập nhật lại trọng số node
    #         # Công thức lai ghép: Local * Structural
    #         final_node_weight = (local_conf /count) * (1 + struct_score)

    #         self.graph.nodes[node]['final_score'] = final_node_weight

    #     return self.graph

    def verify_structure(self, graph):
        """
        Input: networkx.DiGraph (Raw Graph từ bước 3)
        Output: networkx.DiGraph (Graph đã được update điểm số final_score)
        """
        if graph.number_of_nodes() == 0:
            return graph

        # 1. Tính toán Centrality (Độ quan trọng toàn cục)
        # PageRank là thuật toán tốt nhất để tìm "Node đồng thuận" trong đồ thị suy luận
        # Các node được nhiều luồng suy luận đi qua sẽ có PageRank cao.
        try:
            # weight='weight' nghĩa là cạnh nào lặp lại nhiều lần (được vote nhiều) sẽ quan trọng hơn
            centrality_scores = nx.pagerank(graph, weight='weight', alpha=0.85)
        except Exception as e:
            # Fallback nếu graph quá nhỏ hoặc lỗi hội tụ -> Dùng Degree Centrality đơn giản
            centrality_scores = nx.degree_centrality(graph)

        # Tìm max centrality để chuẩn hóa về đoạn [0, 1]
        max_cent = max(centrality_scores.values()) if centrality_scores else 1.0

        # 2. Cập nhật điểm số cho từng Node
        for node in graph.nodes():
            node_data = graph.nodes[node]
            
            # a. Lấy điểm Local (từ Model Confidence)
            # node_data['total_confidence'] là tổng score, chia cho count để lấy trung bình
            count = node_data.get('count', 1)
            total_conf = node_data.get('total_confidence', 0.0)
            local_score = total_conf / count if count > 0 else 0.0
            
            # b. Lấy điểm Global (Structure)
            raw_cent = centrality_scores.get(node, 0)
            # Chuẩn hóa
            global_score = raw_cent / max_cent if max_cent > 0 else 0
            
            # c. Tính Final Score (Pha trộn)
            # Công thức: Final = (1 - alpha) * Local + alpha * Global
            alpha = self.centrality_weight
            final_score = (1 - alpha) * local_score + alpha * global_score # local_score là trung bình xác suất
            
            # Lưu lại vào graph để dùng cho bước Selection
            graph.nodes[node]['final_score'] = final_score
            graph.nodes[node]['local_score'] = local_score
            graph.nodes[node]['global_score'] = global_score # Lưu để debug

        return graph
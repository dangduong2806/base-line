import numpy as np
import networkx as nx
from .step_3_graph_builder import IsomorphismEngine # Tái sử dụng engine so sánh

class EntropySelector:
    def __init__(self):
        self.iso_engine = IsomorphismEngine()
    
    def select_answer(self, graph):
        """
        Input: Reasoning Graph đã được verify và re-weight (có 'final_score').
        Output: (best_answer_content, confidence_score, entropy)
        """
        # 1. Tìm tất cả các node lá (Leaf nodes - Out degree = 0)
        leaf_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        if not leaf_nodes:
            return {
                'final_answer': None,
                'final_path_content': "",
                'confidence': 0.0,
                'entropy': float('inf')
            }
        
        # 2. Gom nhóm các đáp án tương đương (Grouping Answer Clusters)
        # Structure: { "canonical_answer_text": { "score": float, "original_texts": [] } }
        # Value = {'score': sum_score, 'leaf_nodes': [list_of_node_ids]}

        clusters = {}
        for node in leaf_nodes:
            content = graph.nodes[node].get('content', '')
            # Lấy điểm số đã được tính từ bước Structural Verification
            # Đây chính là thành phần cốt lõi của công thức SC(global)
            score = graph.nodes[node].get('final_score', 0.0) # score là trung bình xác suất

            # Kiểm tra xem đáp án này đã có trong cluster nào chưa
            found_cluster = False
            for key in clusters.keys():
                if self.iso_engine.are_equivalent(content, key):
                    # Nếu tương đương, gộp vào cluster đó
                    clusters[key]['score'] += score
                    clusters[key]['original_texts'].append(content)
                    clusters[key]['leaf_nodes'].append(node)
                    found_cluster = True
                    break
            if not found_cluster:
                # Tạo cluster mới
                clusters[content] = {
                    'score': score,
                    'original_texts': [content],
                    'leaf_nodes': [node]
                }
        # 3. Chuẩn hóa thành xác suất (Probability Distribution)
        # P(a) = SC(a) / Sum(SC(all))
        candidates = []
        total_sc_score = sum(c['score'] for c in clusters.values())
        if total_sc_score == 0:
            return {
                'final_answer': None,
                'final_path_content': "",
                'confidence': 0.0,
                'entropy': float('inf')
            }
        
        for answer, data in clusters.items():
            prob = data['score'] / total_sc_score
            candidates.append({
                'answer': answer,
                'prob': prob,
                'raw_score': data['score'],
                'leaf_nodes': data['leaf_nodes']
            })

        # 4. Tính entropy của phân phối này
        # H(X) = - sum(p * log(p))
        probs = np.array([c['prob'] for c in candidates])
        # Cộng thêm epsilon để tránh log(0)
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # 5. Chọn đáp án tốt nhất (Max Probability = Minimized Risk)
        best_candidate = max(candidates, key=lambda x: x['prob'])

        # [QUAN TRỌNG] Tái tạo lại đường dẫn (Path Reconstruction)
        # Trong cụm thắng cuộc, chọn ra node lá có điểm cao nhất để đại diện
        best_leaf_nodes = best_candidate['leaf_nodes']
        # Tìm node có final_score cao nhất trong nhóm này
        representative_leaf = max(best_leaf_nodes, key=lambda n: graph.nodes[n].get('final_score', 0))

        # Truy vết từ Root -> Leaf
        full_path_text = self._reconstruct_path(graph, representative_leaf)
        return {
            'final_answer': best_candidate['answer'],
            'final_path_content': full_path_text,
            'confidence': best_candidate['prob'],
            'entropy': entropy,
            'all_candidates': candidates
        }
    
    def _reconstruct_path(self, graph, leaf_node_id):
        """
        Helper: Tìm đường đi ngắn nhất từ Root đến Leaf để lấy nội dung.
        """
        try:
            # Tìm đường đi từ ROOT đến node lá này
            # Vì graph là cây/DAG, shortest_path sẽ trả về chuỗi các node cha -> con
            path_nodes = nx.shortest_path(graph, source="ROOT", target=leaf_node_id)

            # Ghép nội dung các bước lại
            path_contents = []
            for node in path_nodes:
                # Bỏ qua node ROOT (thường chỉ là placeholder "Start Problem")
                if node == "ROOT": continue

                content = graph.nodes[node].get('content', '')
                path_contents.append(content)
            
            # Nối lại thành chuỗi văn bản (mỗi bước 1 dòng)
            return "\n".join(path_contents)
        except Exception as e:
            print(f"Error reconstructing path: {e}")
            return ""
        
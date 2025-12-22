import sys
import yaml
import logging
import argparse
import numpy as np

# Import các module từ source code
# Giả sử bạn đã đặt các file class vào đúng thư mục src/ và models/
from models.llm_engine import LLMEngine
from src.step_1_sampler import AdaptiveSampler
from src.step_2_verifier import LocalVerifier
# from src.step_3_graph_builder import ReasoningGraphBuilder
from src.step_4_graph import ReasoningGraph
from src.step_4_structural_verifier import StructuralVerifier
from src.step_5_selector import EntropySelector

# Cấu hình Logging (MLOps standard)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiment.log")
    ]
)
logger = logging.getLogger(__name__)

class ResearchPipeline:
    def __init__(self, config_path):
        # 1. Load Config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {config_path}")
        
        # 2. Initialize Model (Heavy component)
        logger.info("Initializing Llama-3-8B (4-bit)...")
        self.llm = LLMEngine(self.config['model']['name'])
        
        # 3. Initialize Pipeline Components
        self.sampler = AdaptiveSampler(self.llm, self.config)
        self.local_verifier = LocalVerifier(self.config, self.llm)
        self.graph_builder = ReasoningGraph() # Có chứa IsomorphismEngine bên trong
        self.struct_verifier = StructuralVerifier(self.config)
        self.selector = EntropySelector()

    def _build_prompt(self, problem_text):
        """
        Tạo prompt theo chuẩn Llama-3 Instruct, ép buộc định dạng Step-by-Step.
        """
        # System prompt định nghĩa vai trò và định dạng output bắt buộc
        system_prompt = (
            "You are a mathematics expert. Solve the following problem step-by-step.\n"
            "IMPORTANT FORMATTING RULES:\n"
            "1. Each step must be on a new line.\n"
            "2. Start each step with 'Step k:' (e.g., 'Step 1:', 'Step 2:').\n"
            "3. State the mathematical expression clearly in each step.\n"
            "4. The final answer must be boxed using LaTeX format: \\boxed{answer}.\n"
            "5. Do not output anything else (like conversational filler)."
        )
        
        # User prompt chứa bài toán
        user_prompt = f"Problem: {problem_text}\n\nSolution:"
        
        # Format chuẩn của Llama-3
        full_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return full_prompt

    def run(self, problem_text):
        """
        Thực thi quy trình 5 bước theo paper.
        """
        logger.info(f"--- STARTING PIPELINE FOR: {problem_text[:50]}... ---")
        
        # ---------------------------------------------------------
        # BƯỚC 1: Adaptive Sampling (Sinh mẫu thích ứng)
        # ---------------------------------------------------------
        logger.info(">>> Step 1: Adaptive Sampling...")
        
        full_prompt = self._build_prompt(problem_text=problem_text)
        raw_paths = self.sampler.sample(full_prompt)
        sample_count = len(raw_paths)
        logger.info(f"    Generated {len(raw_paths)} raw reasoning paths.")
        logger.info(f"All paths to data: {raw_paths}")

        # ---------------------------------------------------------
        # BƯỚC 2: Local Verification (Atomic + Logical Check)
        # ---------------------------------------------------------
        logger.info(">>> Step 2: Atomic & Logical Verification (SymPy + Logprobs)...")
        verified_paths = []
        valid_path_count = 0
        
        for idx, path in enumerate(raw_paths):
            # path là list các object/dict chứa text và logprobs của từng bước
            verified_steps = self.local_verifier.verify_path(path, problem_text=problem_text)
            
            # Chỉ giữ lại các path có ít nhất 1 bước đúng
            if verified_steps:
                verified_paths.append(verified_steps)
                valid_path_count += 1
            else:
                logger.debug(f"    Path {idx} rejected completely.")
        
        logger.info(f"    Retained {valid_path_count} valid paths after local filtering.")

        if not verified_paths:
            logger.warning("!!! No valid paths found. Pipeline aborted.")
            return None
        else:
            logger.info(f"Verified steps: {verified_paths}")

        # ---------------------------------------------------------
        # BƯỚC 3: Graph Construction (Isomorphism Isomorphism)
        # ---------------------------------------------------------
        logger.info(">>> Step 3: Building Reasoning Graph (SymPy Isomorphism)...")
        raw_graph = self.graph_builder.build_graph(verified_paths)
        logger.info(f"    Graph built with {raw_graph.number_of_nodes()} nodes and {raw_graph.number_of_edges()} edges.")

        # Gọi hàm đệ quy để in cây
        print("Raw graph:")
        self._print_graph_tree(raw_graph, "ROOT", indent="", visited_path=set())
        print("\n")

        # ---------------------------------------------------------
        # BƯỚC 4: Structural Verification (Global Dependency)
        # ---------------------------------------------------------
        logger.info(">>> Step 4: Structural Verification (Centrality Reweighting)...")
        refined_graph = self.struct_verifier.verify_structure(raw_graph)
        
        # Debug: In ra một vài node quan trọng
        top_nodes = sorted(refined_graph.nodes(data=True), key=lambda x: x[1].get('final_score', 0), reverse=True)[:3]
        logger.debug(f"    Top robust nodes: {[n[1].get('content') for n in top_nodes]}")
        logger.info(f"Structural Verification Graph: {refined_graph}")

        print("FULL REASONING GRAPH VISUALIZATION (Tree View)")
        print("Format: [Score] Node_ID: Content")
        print("Score = Final weighted score (Local + Global)")
        # Gọi hàm đệ quy để in cây
        self._print_graph_tree(refined_graph, "ROOT", indent="", visited_path=set())
        print("\n")

        # ---------------------------------------------------------
        # BƯỚC 5: Global Selection (Entropy Minimization)
        # ---------------------------------------------------------
        logger.info(">>> Step 5: Final Selection (Entropy Minimization)...")
        result = self.selector.select_answer(refined_graph)
        
        return result, sample_count
    
    def _print_graph_tree(self, graph, current_node, indent="", visited_path=None):
            """
            Hàm đệ quy in đồ thị dạng cây thư mục.
            """
            if visited_path is None: visited_path = set()
            
            # Lấy thông tin node
            if current_node == "ROOT":
                content = "[START PROBLEM]"
                score = 1.0
            else:
                data = graph.nodes[current_node]
                # Cắt ngắn nội dung nếu quá dài để hiển thị đẹp
                full_content = data.get('content', '').replace('\n', ' ').strip()
                content = (full_content[:75] + '...') if len(full_content) > 75 else full_content
                score = data.get('final_score', 0.0)

            # In node hiện tại
            # Ký hiệu: └── cho nhánh cuối, ├── cho nhánh giữa
            print(f"{indent}O-- [{score:.4f}] {current_node}: {content}")

            # Lấy các node con
            children = list(graph.successors(current_node))
            
            if not children:
                return

            # Đệ quy in con
            # Tránh lặp vô hạn nếu đồ thị có chu trình (mặc dù graph này thường là DAG)
            if current_node in visited_path:
                print(f"{indent}    ( ... Merge/Loop back to existing path ... )")
                return
                
            visited_path.add(current_node)
            
            for i, child in enumerate(children):
                is_last = (i == len(children) - 1)
                # Tạo thụt đầu dòng cho cấp con
                next_indent = indent + ("    " if is_last else "|   ")
                
                # Gọi đệ quy
                self._print_graph_tree(graph, child, next_indent, visited_path.copy())

def main():
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Run Hallucination Detection Pipeline")
    parser.add_argument("--config", type=str, default="configs/main_config.yaml", help="Path to config file")
    parser.add_argument("--problem", type=str, default=None, help="Math problem to solve")
    args = parser.parse_args()

    # Khởi tạo Pipeline
    try:
        pipeline = ResearchPipeline(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return

    # Bài toán mẫu (Lấy từ tập PRM800K hoặc Competition Math)
    # Đây là bài toán đại số cần biến đổi biểu thức
    default_problem = "Find the coefficient of x^3 in the expansion of (2*x + 1)^5."
    
    problem_to_solve = args.problem if args.problem else default_problem
    
    # Chạy
    result, _ = pipeline.run(problem_to_solve)
    
    # Hiển thị kết quả cuối cùng
    if result:
        print("\n" + "="*40)
        print("RESEARCH EXPERIMENT RESULT")
        print("="*40)
        print(f"PROBLEM: {problem_to_solve}")
        print("-" * 40)
        print(f"FINAL PATH CONTENT: {result['final_path_content']}")
        print(f"FINAL ANSWER: {result['final_answer']}")
        print(f"CONFIDENCE (SC Score): {result['confidence']:.4f}")
        print(f"SYSTEM ENTROPY:        {result['entropy']:.4f}")
        print("-" * 40)
        
        # Logic cảnh báo Hallucination dựa trên Entropy
        threshold = pipeline.config['selection']['entropy_threshold']
        if result['entropy'] > threshold:
             print("⚠️  STATUS: HIGH UNCERTAINTY (Potential Hallucination)")
             print("   Recommendation: Human verification needed.")
        else:
             print("✅  STATUS: VERIFIED (High Consistency)")
             print("   Recommendation: Answer is likely correct.")
        
        print("="*40 + "\n")
    else:
        print("❌ Failed to generate a valid answer.")

if __name__ == "__main__":
    main()
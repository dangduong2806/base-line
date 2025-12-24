from src.metrics import DeepMathMetrics
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import yaml
import time
import torch

from models.llm_engine import LLMEngine
from src.step_1_sampler import AdaptiveSampler
from src.step_2_verifier import LocalVerifier
# from src.step_3_graph_builder import ReasoningGraphBuilder
from src.step_4_graph import ReasoningGraph
from src.step_4_structural_verifier import StructuralVerifier
from src.step_5_selector import EntropySelector
from main import ResearchPipeline

logger = logging.getLogger(__name__)

def load_prm800k_test_set(file_path, limit=50):
    """
    Load dataset PRM800K. 
    Giữ nguyên cấu trúc JSON object để lấy thông tin Golden Path (steps, completions).
    """

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit: break
            try:
                item = json.loads(line)
                # Đảm bảo dữ liệu hợp lệ (có question và label)
                if 'question' in item and 'label' in item:
                    data.append(item)
            except json.JSONDecodeError:
                continue
    return data

def run_benchmark():
    # 1. Setup
    config_path = "configs/main_config.yaml"
    # Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load pipeline
    pipeline = ResearchPipeline(config_path=config_path)

    math_eval = DeepMathMetrics()
    
    # Load Data (Thay đường dẫn file thật của bạn vào đây)
    data_path = "data/raw/test_subset.jsonl"
    test_data = load_prm800k_test_set(data_path) 
    
    results = []
    
    print(f"Starting benchmark on {len(test_data)} samples...")
    for item in tqdm(test_data):
        problem = item['question']['problem']
        gt = item['question']['ground_truth_answer']

        try:
            start_time = time.time()
            torch.cuda.reset_peak_memory_stats()

            # Gọi pipeline
            result, sample_count = pipeline.run(problem_text=problem)

            end_time = time.time()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB

            pred = result['final_answer']
            best_path_text = result['final_path_content']

            metrics = math_eval.compute_metrics(
                generated_path_text = best_path_text,
                label_data = item
            )

            # Dù kết quả thế nào thì vẫn cứ in ra output của model
            print(f"PROBLEM: {problem}")
            print(f"Model output steps: \n{best_path_text}")
            print(f"Model last answer: \n{pred}\n")

            print(f"EE: {metrics['EE']}, ASS: {metrics['ASS']}, TSA: {metrics['TSA']}")
            record = {
                "problem": problem[:30] + "...",
                "EE": metrics['EE'],
                "ASS": metrics['ASS'],
                "TSA": metrics['TSA'],
                "SE": sample_count,
                "CO": end_time - start_time,
                "MF": peak_memory
            }
            results.append(record)
        except Exception as e:
            print(f"❌ Failed to generate a valid answer for problem: {problem[:30]}...")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # 3. Aggregation (Tính trung bình)
    df = pd.DataFrame(results)
    print("FINAL BENCHMARK RESULTS (PRM800K)")
    print("="*50)
    print(f"1. Expression Equivalence (EE - Accuracy): {df['EE'].mean():.4f}")
    print(f"2. Algebraic Simplification (ASS):         {df['ASS'].mean():.4f}")
    print(f"3. Transformation Step Acc (TSA):          {df['TSA'].mean():.4f}")
    print("-" * 30)
    print(f"4. Sample Efficiency (SE - Avg Samples):   {df['SE'].mean():.2f}")
    print(f"5. Comp Overhead (CO - Avg Time s):        {df['CO'].mean():.4f}")
    print(f"6. Memory Footprint (MF - Peak MB):        {df['MF'].max():.2f}") # Lấy Max thay vì Mean cho RAM
    print("="*50)

    # Lưu kết quả chi tiết
    # df.to_csv("benchmark_results.csv", index=False)
    # print("Detailed results saved to benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
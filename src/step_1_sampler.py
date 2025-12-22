import re
import numpy as np
from collections import Counter

class AdaptiveSampler:
    def __init__(self, llm_engine, config):
        self.llm = llm_engine
        self.min_samples = config['adaptive_sampling'].get('min_samples',3)
        self.max_samples = config['adaptive_sampling'].get('max_samples', 10)
        self.threshold = config['adaptive_sampling'].get('consistency_threshold', 0.7)

    def sample(self, prompt):
        """
        Thực hiện Adaptive Sampling theo thuật toán:
        Input: Prompt bài toán
        Output: List[List[Dict]] (Danh sách các path, mỗi path gồm các step kèm logprob)
        """
        # Cấu trúc dữ liệu
        all_paths_data = [] # lưu full data để đẩy xuống pipeline
        extracted_answers = [] # lưu đáp án để kiểm tra consistency

        print(f"   [Sampler] Starting adaptive sampling (Min: {self.min_samples}, Max: {self.max_samples}, Thr: {self.threshold})")

        # --- BƯỚC 1: Initial Batching (Sinh tập khởi tạo) ---
        # Paper luôn bắt đầu với k0 mẫu để có cái nhìn thống kê ban đầu

        initial_batch = self.llm.generate(prompt, num_return_sequences=self.min_samples)

        for text, conf in initial_batch:
            # Parse text thành cấu trúc steps cho bước sau
            steps = self._parse_text_to_steps(text, conf) # conf là log-prob
            all_paths_data.append(steps)

            # Trích xuất đán áp để vote
            ans = self.extract_answer(text)
            extracted_answers.append(ans)

        # --- BƯỚC 2: Adaptive Sampling Loop ---
        current_k = self.min_samples

        while current_k < self.max_samples:
            # a. Tính độ đồng thuận hiện tại (Consistency Check)
            # Lọc bỏ các mẫu lỗi không tìm thấy đáp án (None)
            valid_answers = [ans for ans in extracted_answers if ans is not None]

            if not valid_answers:
                # Nếu chưa tìm thấy đáp án nào hợp lệ, buộc phải sinh tiếp
                consistency_score = 0.0
            else:
                # Tìm đáp án xuất hiện nhiều nhất (Majority Vote)
                most_common_ans, count = Counter(valid_answers).most_common(1)[0]
                consistency_score = count / len(valid_answers)
            
            # b. Kiểm tra điều kiện dừng (Stopping Criterion)
            if consistency_score > self.threshold:
                print(f"   [Sampler] Consistency {consistency_score:.2f} > {self.threshold}, stopping at k={current_k}.")
                break
            
            # c. Nếu chưa đạt, sinh thêm (Incremental Sampling)
            # Sinh thêm 1 mẫu (hoặc batch nhỏ)
            print(f"   [Sampler] Consistency {consistency_score:.2f} < {self.threshold}. Generating +1 path...")
            new_batch = self.llm.generate(prompt, num_return_sequences=1)

            for text, conf in new_batch:
                steps = self._parse_text_to_steps(text, conf)
                all_paths_data.append(steps)
                extracted_answers.append(self.extract_answer(text))
            
            current_k += 1

        if current_k == self.max_samples:
            print(f"   [Sampler] Reached max samples k={self.max_samples}.")
        
        return all_paths_data

    def _parse_text_to_steps(self, text, path_confidene): # path_confidence là log-prob
        """
        Helper: Chuyển văn bản thô thành danh sách các bước (Atomic Steps).
        Gán logprob giả định (hoặc thật nếu model hỗ trợ) cho từng bước.
        """
        steps = []
        # Tách dòng, bỏ dòng trống
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Tính logprob từ confidence (score 0-1)
        # log(prob). Nếu prob ~ 0 thì gán giá trị rất nhỏ.
        safe_conf = max(path_confidene, 1e-10)
        # path_logprob = np.log(safe_conf)
        path_logprob = safe_conf
        for line in lines:
            # Lọc các dòng rác (Header/Footer của model)
            if any(k in line.lower() for k in ["solution:", "answer:", "###", "step-by-step"]):
                continue

            steps.append({
                'text': line,
                'logprob': path_logprob, # Tạm dùng path score cho step (đơn giản hóa)
                'confidence': safe_conf # Giá trị log-prob
            })
        return steps

    def extract_answer(self, text):
        """
        Helper: Trích xuất đáp án cuối cùng.
        Ưu tiên định dạng \boxed{}, sau đó đến số cuối cùng.
        """
        if not text:
            return None
        
        # 1. Chuẩn nhất: Tìm \boxed{ans} (Dữ liệu toán học thường dùng cái này)
        match = re.search(r'\\boxed\{(.*?)\}', text)
        if match:
            return self._normalize_answer(match.group(1))
        
        # 2. Fallback: Tìm "The answer is..."
        match = re.search(r'(?:answer|result) is\s*([0-9a-zA-Z\+\-\*\^\.]+)', text, re.IGNORECASE)
        if match:
            return self._normalize_answer(match.group(1))
        
        # 3. Fallback cuối cùng: Lấy cụm số/biểu thức cuối cùng của chuỗi
        lines = text.strip().split('\n')
        if lines:
            last_line = lines[-1]
            # Tìm cụm ký tự toán học ở cuối dòng
            last_math = re.search(r'([0-9x\+\-\*\^]+)$', last_line)
            if last_math:
                return self._normalize_answer(last_math.group(1))
        
        return None

    def _normalize_answer(self, ans_text):
        """Làm sạch đáp án để so sánh chuỗi chính xác hơn"""
        return ans_text.strip().lower().replace(" ", "")

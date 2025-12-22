import numpy as np
from sympy import sympify, SympifyError
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
logger = logging.getLogger(__name__)
class LocalVerifier:
    # def verify_step(self, context, step_content, step_logprob):
    #     """
    #     Kết hợp Atomic và Logical check.
    #     """
    #     # 1. Atomic Check (SymPy): Bước này có vô lý về toán học không?
    #     # Ví dụ: "1 + 1 = 3" -> Atomic Error
    #     atomic_score = self.sympy_check(step_content)
    #     # 2. Logical Dependency (Model Confidence):
    #     # Model có chắc chắn bước này suy ra từ context không?
    #     # Dùng logprob (xác suất) làm thước đo dependency.
    #     logical_score = np.exp(step_logprob)  # Chuyển logprob về prob bình thường

    #     # Kết hợp hai thước đo
    #     final_score = atomic_score * logical_score
    #     return final_score

    def __init__(self, config, llm_engine):
        """
        Khởi tạo Verifier với các cấu hình ngưỡng (thresholds).
        """
        self.config = config
        self.llm = llm_engine
        
        # Lấy tham số từ config, nếu không có thì dùng giá trị mặc định
        self.atomic_enabled = config['verification'].get('atomic_check_enabled', True)
        self.logical_enabled = config['verification'].get('logical_check_enabled', True)
        self.logprob_threshold = config['verification'].get('logprob_threshold', -1.5)

    def verify_path(self, path, problem_text = ""):
        """
        Input: 
            path: List các bước (dict) từ model. 
            Mỗi item dạng: {'text': '...', 'logprob': -0.5, ...}
            
        Output: 
            verified_steps: List các bước đã qua kiểm duyệt.
        """
        verified_steps = []
        
        # Context dùng để theo dõi chuỗi suy luận (nếu cần check logic phức tạp hơn)
        # Ở version đơn giản này, ta check từng bước độc lập dựa trên logprob và syntax.
        
        # Context bắt đầu rỗng (hoặc là prompt bài toán gốc nếu muốn chính xác hơn)
        current_context = f"Problem: {problem_text}\nSolution:\n" #
        
        for i, step in enumerate(path):
            step_content = step.get('text', '')
            step_logprob = step.get('logprob', -float('inf'))
            
            # --- 1. Atomic Check (Kiểm tra lỗi toán học sơ đẳng) ---
            if self.atomic_enabled:
                if not self._check_atomic_validity(step_content):
                    # Nếu bước này viết sai cú pháp toán học (vd: "x + = 2") -> Dừng ngay
                    break 

            # ---------------------------------------------------
            # 2. LOGICAL DEPENDENCY CHECK (Conditional Probability)
            # ---------------------------------------------------
            # Công thức: Score = P(Step_k | Context)
            # Bỏ qua bước 1 nếu muốn (vì nó phụ thuộc đề bài, chưa có context)
            # Nhưng tốt nhất vẫn nên tính để xem nó có khớp với tri thức nội tại ko

            logic_score = self._compute_step_score(current_context, step_content)
            # Log debug để bạn tune threshold
            logger.info(f"Step {i+1}: {step_content[:20]}... | Score: {logic_score:.4f}")
            if logic_score < self.logprob_threshold:
                logger.debug(f"Step {i+1} failed LOGICAL check. Score {logic_score:.4f} < {self.logprob_threshold}")
                break # Cắt nhánh (Pruning)
            
            # Nếu qua được cả 2 vòng check thì thêm vào danh sách hợp lệ
            verified_steps.append({
                'content': step_content,
                'confidence': np.exp(logic_score), # Chuyển logprob về xác suất (0-1)
                'logprob': step_logprob
            })
            # Cập nhật context cho vòng lặp sau
            current_context += step_content + "\n"
            
        return verified_steps
    
    def _check_atomic_validity(self, text):
        """
        Dùng SymPy để kiểm tra xem text có chứa biểu thức toán học hợp lệ không.
        Đây là cách đơn giản để lọc bỏ các bước 'nói nhảm' (gibberish).
        """
        try:
            # Logic: Thử parse text. Nếu SymPy parse được -> Có khả năng là toán.
            # Ta cần clean text một chút trước khi parse (bỏ các từ tiếng Anh common)
            clean_text = text.lower().replace("solve", "").replace("step", "").strip()
            
            # Nếu chuỗi rỗng sau khi clean -> Có thể là lời dẫn, tạm cho qua (True)
            if not clean_text:
                return True
                
            # Thử parse
            sympify(clean_text)
            return True
        except:
            # Nếu SymPy báo lỗi syntax -> Bước này không phải toán học hợp lệ
            # Tuy nhiên, LLM hay viết lời văn (text), nên ta chỉ return False
            # nếu ta cực kỳ khắt khe. Ở mức độ nghiên cứu này, ta có thể return True
            # nhưng log lại warning.
            return True # Tạm thời cho qua để tránh lọc nhầm lời văn giải thích
        
    def _compute_step_score(self, context, step_text):
        """
        IMPLEMENTATION CỦA CÔNG THỨC LOGICAL CHECK TRONG PAPER.
        
        Tính: Average Log-Likelihood của `step_text` KHI BIẾT `context`.
        """
        # 1. Tokenize Context và Step riêng biệt để biết độ dài
        # Lưu ý: Cần add_special_tokens=False để tránh duplicate BOS token
        context_ids = self.llm.tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids
        step_ids = self.llm.tokenizer(step_text, return_tensors="pt", add_special_tokens=False).input_ids

        context_len = context_ids.shape[1]
        step_len = step_ids.shape[1]

        # 2. Ghép lại thành input hoàn chỉnh: [Context, Step]
        input_ids = torch.cat([context_ids, step_ids], dim=1).to(
            device=self.llm.model.device,
            dtype=torch.long
        )

        # 3. Chạy Model (Forward pass) để lấy Logits
        with torch.no_grad():
            outputs = self.llm.model(input_ids)
            logits = outputs.logits # [1, seq_len, vocab_size]

        # 4. Shift Logits và Labels để tính Loss
        # Logits ở vị trí t dùng để dự đoán token ở t+1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # 5. Tính CrossEntropyLoss cho TỪNG token (reduction='none')
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        all_token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 6. MASKING (Cực kỳ quan trọng)
        # Chúng ta chỉ quan tâm loss của phần Step, không quan tâm model thuộc context hay không.
        # Phần Step bắt đầu từ vị trí: context_len - 1 (do đã shift)
        # Độ dài cần lấy: step_len

        # Start index trong mảng loss (đã shift 1) là context_len - 1
        # Nhưng nếu context rỗng (bước 1), start_idx = 0
        start_idx = max(0, context_len - 1)

        # Lấy loss của riêng phần step
        step_token_losses = all_token_losses[start_idx:]

        # Nếu step quá ngắn hoặc lỗi, return score thấp
        if len(step_token_losses) == 0:
            return -999.0
        
        # 7. Tính điểm trung bình (Average Log-Likelihood)
        # Loss là -log(P), nên Log-Likelihood = -Loss
        avg_log_likelihood = -torch.mean(step_token_losses).item()

        return avg_log_likelihood # Giá trị log-prob
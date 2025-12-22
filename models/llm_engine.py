import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np

class LLMEngine:
    def __init__(self, model_name):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            device_map="auto"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, num_return_sequences=1, temperature=0.7):
        """
        Sinh văn bản và trả về kèm độ tự tin (Confidence Score).
        Input:
            prompt: str
            num_return_sequences: int (Số lượng mẫu muốn sinh cùng lúc)
        Output:
            List[Tuple(text, score)]: [(text_1, 0.85), (text_2, 0.92), ...]
        """
        # 1. Chuẩn bị Input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        # 2. Gọi Model Generate (Batch Processing)
        # Quan trọng: Bật return_dict_in_generate=True và output_scores=True để lấy logprobs
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,         # Độ dài tối đa của câu trả lời
                do_sample=True,             # Bắt buộc True để sinh đa dạng (cho Adaptive Sampling)
                temperature=temperature,
                top_k=40,                   # Lấy top 40 token tốt nhất
                top_p=0.95,                 # Nucleus sampling
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True          # YÊU CẦU ĐỂ TÍNH ĐIỂM
            )

        # 3. Tính toán Confidence Score & Decode Text
        results = []
        
        # Hàm compute_transition_scores giúp tính log_prob của các token ĐƯỢC SINH RA
        # (Nó tự động căn chỉnh scores với sequences)
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        
        # Duyệt qua từng chuỗi trong batch
        for i in range(num_return_sequences):
            # a. Lấy generated tokens (bỏ phần prompt input đi)
            generated_tokens = outputs.sequences[i][input_length:]
            
            # b. Decode ra text
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # c. Tính điểm (Score)
            # Lấy log_probs của chuỗi thứ i
            # transition_scores[i] chứa log_prob của từng token
            # Lọc bỏ các token padding (nếu có, logprob sẽ là -inf)
            log_probs = transition_scores[i]
            valid_log_probs = log_probs[~torch.isinf(log_probs)]
            
            if len(valid_log_probs) > 0:
                # Cách tính 1: Trung bình cộng Log-prob (Mean Log Likelihood)
                # mean_log_prob = valid_log_probs.mean().item()
                # score = np.exp(mean_log_prob) # Chuyển về xác suất (0-1)
                
                # Cách tính 2 (Khuyên dùng): Tổng Log-prob (Sequence Probability)
                # Tuy nhiên, câu dài sẽ có điểm thấp hơn câu ngắn. 
                # Để ổn định, ta dùng Mean exp (Geometric Mean)
                score = np.exp(valid_log_probs.mean().item())
            else:
                score = 0.0

            results.append((text, score)) # Score ở đây là mang giá trị log-prob

        return results
    
    def generate_with_confidence(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Bật return_dict_in_generate để lấy logits và output_scores để lấy xs
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True
        )
        # Decode text
        generated_text = self.tokenizer.decode(outputs.seuqences[0], skip_special_tokens=True)
        # 2. Tính confidence score từ logits
        # transition_scores trả về log_softmax của các token được chọn
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scors, normalize_logits=True
        )
        # Tính trung bình log-prob (Geometric mean của probability)
        # Đây là metric đo độ tự tin nội tại của model
        confidence_score = torch.mean(transition_scores).item()
        return generated_text, np.exp(confidence_score)  # Chuyển log-prob về prob bình thường

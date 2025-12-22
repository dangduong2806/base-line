import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy import Eq, simplify, count_ops, Symbol, Number, Float, Integer
import re

class DeepMathMetrics:
    def __init__(self):
        # Cấu hình parse mạnh mẽ (hiểu 2x là 2*x)
        self.transformations = (standard_transformations + (implicit_multiplication_application,))

    def compute_all_metrics(self, generated_path_text, ground_truth_value_str):
        """
        Hàm chính để tính toán cả 3 chỉ số cho 1 lời giải (path).
        
        Input:
            generated_path_text: Toàn bộ chuỗi suy luận của model.
            ground_truth_value_str: Đáp án đúng (VD: "5", "x=5").
            
        Output: dict {EE, ASS, TSA}
        """
        # 1. Tiền xử lý: Tách các bước và trích xuất biểu thức toán học
        steps = self._extract_steps_with_math(generated_path_text)
        # 2. Parsing Ground Truth (QUAN TRỌNG: Cần làm sạch trước)
        # Ép kiểu string để tránh lỗi nếu JSON trả về số int/float
        clean_gt_str = self._clean_latex(str(ground_truth_value_str))
        clean_gt_str = re.sub(r'_\{?\d+\}?', '', clean_gt_str) # Bỏ _9, _{10}
        gt_expr = self._safe_parse(clean_gt_str)

        if not steps:
            print(f"No found steps")
            return {"EE": 0.0, "ASS": 0.0, "TSA": 0.0}
        
        # Parse Ground Truth thành giá trị số/biểu thức (VD: x=5 -> {x: 5})
        # gt_solution = self._parse_ground_truth(ground_truth_value_str)

        # --- Metric 1: Expression Equivalence (EE) ---
        # Kiểm tra tính liên kết logic giữa các bước liền kề
        valid_transitions = 0
        total_transitions = 0
        for i in range(len(steps) - 1):
            expr_curr = steps[i]['expr']
            expr_next = steps[i+1]['expr']

            if expr_curr is None or expr_next is None:
                continue
            # Nếu 2 bước giống hết nhau, skip để tránh bias điểm cao
            if steps[i]['text'] == steps[i+1]['text']:
                continue

            total_transitions += 1
            # Kiểm tra: Liệu expr_curr có tương đương expr_next?
            # Lưu ý: Model thường biến đổi phương trình. A=B => C=D.
            # Ta check simplify(curr - next) == 0 (nếu là biểu thức) 
            # hoặc check tập nghiệm (nếu là phương trình)
            if self._check_equivalence(expr_curr, expr_next):
                valid_transitions += 1
        ee_score = (valid_transitions / total_transitions) if total_transitions > 0 else 0.0

        # --- Metric 2: Transformation Step Accuracy (TSA) ---
        # Kiểm tra từng bước có đúng với đáp án thực tế không
        correct_steps = 0
        checkable_steps = 0 # Chỉ đếm những bước có thể check được
        uncheckable_steps = 0
        for step in steps:
            expr = step['expr']
            if expr is None: continue

            is_correct, is_checkable = self._check_consistency_robust(expr, gt_expr)
            if is_checkable:
                checkable_steps += 1
                if is_correct:
                    correct_steps += 1
                else:
                    # Debug nhẹ: In ra bước sai để kiểm tra
                    print(f"TSA Fail: {step['text']} vs GT: {ground_truth_value_str}")
                    pass
            else:
                uncheckable_steps += 1
        print(f"Tổng số steps: {len(steps)}")
        print(f"Số steps không check được: {uncheckable_steps}")
        tsa_score = (correct_steps / checkable_steps) if checkable_steps > 0 else 0.0

        # --- Metric 3: Algebraic Simplification Score (ASS) ---
        # Chỉ tính trên bước cuối cùng (Final Answer)
        last_expr = steps[-1]['expr']
        ass_score = 0.0
        if last_expr is not None:
            ass_score = self._calculate_ass(last_expr)
        
        return {
            "EE": round(ee_score, 4),
            "ASS": round(ass_score, 4),
            "TSA": round(tsa_score, 4)
        }
    

    def _extract_steps_with_math(self, text):
        """
        Phiên bản Robust hơn: Chấp nhận cả những dòng không có 'Step X:'
        miễn là nó chứa biểu thức toán học (dấu =, >, <, +, -...).
        """
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        parsed_steps = []
        
        for line in lines:
            # 1. Bỏ qua các dòng rác (lời dẫn không có số/toán)
            # Kiểm tra xem dòng này có chứa ít nhất 1 toán tử hoặc số không
            if not any(char in line for char in "=+-*/^<>") and not any(char.isdigit() for char in line):
                continue

            # 2. Trích xuất phần toán học
            # Nếu có dấu ":", khả năng cao là "Step 1: x=5" -> lấy phần sau dấu :
            if ":" in line:
                potential_math = line.split(":", 1)[1].strip()
            else:
                potential_math = line # Lấy cả dòng nếu không có label

            # Clean các ký tự latex thì mới check được
            cleaned_math = self._clean_latex(potential_math)
            # 3. Thử parse
            expr = self._safe_parse(cleaned_math)
            
            # Nếu parse được (không phải None), tính là 1 bước hợp lệ
            if expr is not None:
                parsed_steps.append({'text': line, 'expr': expr})
            
        return parsed_steps
    
    def _clean_latex(self, text):
        """
        Làm sạch các ký tự LaTeX để SymPy có thể hiểu được.
        """
        if not text: return ""
        text = str(text).strip()
        
        # 1. Xử lý các command bao đóng thường gặp
        text = text.replace(r"\boxed", "").replace(r"\overline", "")
        text = text.replace(r"\$", "") # Bỏ dấu $
        text = text.replace("\\", "")  # Bỏ backslash còn lại
        
        # 2. Xử lý phân số \frac{a}{b} -> (a)/(b)
        # Regex này tìm \frac{...}{...} và thay thế bằng phép chia
        # Lưu ý: Regex đơn giản, không đệ quy, nhưng đủ cho PRM800K cơ bản
        text = re.sub(r'frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', text)
        
        # 3. Thay ngoặc nhọn còn sót lại bằng ngoặc đơn (cho an toàn)
        text = text.replace("{", "(").replace("}", ")")
        
        return text
    
    def _safe_parse(self, text):
        try:
            # Xử lý trường hợp "x = 5" -> Eq(x, 5)
            if "=" in text:
                parts = text.split("=")
                if len(parts) == 2:
                    lhs = parse_expr(parts[0], transformations=self.transformations)
                    rhs = parse_expr(parts[1], transformations=self.transformations)
                    return Eq(lhs, rhs)
            
            # Xử lý biểu thức thường
            return parse_expr(text, transformations=self.transformations)
        except:
            return None
        
    def _check_equivalence(self, expr1, expr2):
        """Kiểm tra 2 biểu thức/phương trình có tương đương ngữ nghĩa không"""
        try:
            # Case 1: Cả 2 là phương trình (Eq)
            if isinstance(expr1, Eq) and isinstance(expr2, Eq):
                # Cách đơn giản: Chuyển về dạng A - B = 0 và so sánh
                return simplify((expr1.lhs - expr1.rhs) - (expr2.lhs - expr2.rhs)) == 0
            
            # Case 2: Cả 2 là biểu thức
            return simplify(expr1 - expr2) == 0
        except:
            return False
    
    # def _check_consistency_with_ground_truth(self, step_expr, gt_value):
    #     """
    #     Thay Ground Truth vào phương trình bước hiện tại xem có đúng không.
    #     VD: Step: '2x = 10', GT: 5. Thay x=5 -> 10=10 (True).
    #     """
    #     try:
    #         if gt_value is None: return False

    #         # Nếu step là phương trình Eq(lhs, rhs)
    #         if isinstance(step_expr, Eq):
    #             # Tìm biến trong phương trình
    #             symbols = step_expr.free_symbols
    #             if not symbols: return False # Phương trình hằng số 1=2
    #             # Thay thế tất cả biến bằng gt_value (Giả sử bài toán 1 biến)
    #             # Lưu ý: Logic này đúng cho bài toán tìm x.
    #             # Với bài toán rút gọn, gt_value là kết quả cuối.

    #             # Thử check: lhs - rhs = 0?
    #             # Cần subs biến. VD: step: 2*x - 10 = 0. GT: x=5.
    #             check = step_expr.subs({list(symbols)[0]: gt_value})
    #             return simplify(check.lhs - check.rhs) == 0
            
    #         # Nếu step là biểu thức (VD đang rút gọn): 'x + x' -> '2x'
    #         # Check xem biểu thức này có bằng GT không? (Không khả thi cho bài rút gọn từng bước)
    #         # Với bài rút gọn: GT là kết quả cuối.
    #         # Nếu bước hiện tại biến đổi đúng, giá trị của nó với x bất kỳ phải bằng GT? 
    #         # Không, bài rút gọn thì biểu thức thay đổi hình dạng nhưng giá trị giữ nguyên.
    #         # -> Check: simplify(step_expr - gt_value) == 0 ?? 
    #         # (Chỉ đúng nếu GT là biểu thức gốc chưa rút gọn hoặc đã rút gọn).
            
    #         # Tạm thời implement cho bài toán tìm nghiệm (Solving):
    #         symbols = step_expr.free_symbols
    #         if symbols:
    #             val = step_expr.subs({list(symbols)[0]: gt_value})
    #             if isinstance(val, Eq): return simplify(val.lhs - val.rhs) == 0

    #         return False
    #     except:
    #         return False

    def _check_consistency_robust(self, step_expr, gt_expr):
        """
        Kiểm tra tính đúng đắn (Correctness) của bước trung gian so với GT.
        Cover đầy đủ: Số học, Thay thế nghiệm, và Biến đổi đại số.
        
        Returns: (is_correct, is_checkable)
        """
        try:
            if gt_expr is None: return False, False

            # Lấy tập hợp biến
            step_vars = step_expr.free_symbols
            gt_vars = gt_expr.free_symbols if hasattr(gt_expr, 'free_symbols') else set()

            # --- GROUP 1: ARITHMETIC (Không có biến) ---
            # VD: "29/100 = 0.29" hoặc "1 + 1 = 2"
            if not step_vars:
                if isinstance(step_expr, Eq):
                    # Check 2 vế bằng nhau (sai số nhỏ cho float)
                    is_eq = abs(float(step_expr.lhs) - float(step_expr.rhs)) < 1e-6
                    return is_eq, True
                # [FIX]: Nếu bước là 1 con số (VD: 29) và GT là số (29) -> Check luôn!
                if isinstance(step_expr, (Number, Float, Integer)) and not gt_vars:
                    # GT có thể là Eq(x, 29) hoặc số 29
                    gt_val = gt_expr.rhs if isinstance(gt_expr, Eq) else gt_expr
                    is_eq = abs(float(step_expr) - float(gt_val)) < 1e-6
                    return is_eq, True
                
                return False, False

            # --- GROUP 2: ALGEBRAIC EQUIVALENCE (Cả 2 đều có biến) ---
            # VD: GT="2x", Step="x+x". Dùng cho bài toán rút gọn.
            if step_vars and gt_vars:
                # Nếu tập biến khớp nhau (cùng là x, hoặc cùng là a,b)
                # Hoặc step là phương trình, gt là biểu thức...
                
                # Logic: Hiệu số giữa Step và GT phải bằng 0 (hoặc tương đương)
                # Lưu ý: Nếu Step là Eq(A, B) và GT là C. Thì A-B phải tương đương C? Không hẳn.
                # Thường bài rút gọn: Step là Eq(P, simplified_P) hoặc chỉ là simplified_P.
                
                # Chuyển đổi Step về dạng biểu thức (LHS - RHS)
                val_step = (step_expr.lhs - step_expr.rhs) if isinstance(step_expr, Eq) else step_expr
                val_gt = (gt_expr.lhs - gt_expr.rhs) if isinstance(gt_expr, Eq) else gt_expr
                
                # Check: simplify(val_step - val_gt) == 0
                # VD: Step: x+x, GT: 2x. -> (2x) - (2x) = 0 -> True.
                if simplify(val_step - val_gt) == 0:
                    return True, True
                
                # Nếu không bằng 0, có thể do Step đang biến đổi trung gian chưa về đích.
                # Rất khó check intermediate của bài rút gọn nếu không có full context.
                # Tạm thời return False, True (Checkable nhưng sai so với GT cuối)
                return False, True

            # --- GROUP 3: SUBSTITUTION (Step có biến, GT là hằng số) ---
            # VD: Step: 2x = 58, GT: 29
            if step_vars and not gt_vars:
                # Xử lý GT nếu nó là Eq (VD: x=29 -> lấy 29)
                target_val = gt_expr.rhs if isinstance(gt_expr, Eq) else gt_expr
                
                # CASE 3a: Step chỉ có 1 biến (x) -> Thay GT vào x
                if len(step_vars) == 1:
                    var = list(step_vars)[0] # Lấy biến duy nhất (dù là x hay y)
                    
                    # Thay thế
                    substituted = step_expr.subs(var, target_val)
                    
                    # Kiểm tra kết quả sau thay thế
                    if isinstance(substituted, Eq):
                        # VD: 2(29) = 58 -> 58=58 -> True
                        is_correct = simplify(substituted.lhs - substituted.rhs) == 0
                        return is_correct, True
                    else:
                        # VD: Step là "x + 1". GT là 5. Thay vào ra 6. 
                        # Không phải mệnh đề đúng/sai -> Không check được
                        return False, False
                
                # CASE 3b: Step có nhiều biến (3x + y = 124) -> BÓ TAY
                # Vì ta chỉ có GT cho đáp án cuối (ví dụ x), không biết y.
                return False, False 

            return False, False
            
        except Exception as e:
            # print(f"TSA Check Error: {e}") 
            return False, False
    
    def _calculate_ass(self, expr):
        """
        ASS = 1 - (khoảng cách đến dạng canonical)
        Dùng count_ops để đo độ phức tạp.
        """
        try:
            # Dạng tối giản lý tưởng
            canonical = simplify(expr)
            ops_generated = count_ops(expr)
            ops_canonical = count_ops(canonical)

            if ops_generated <= ops_canonical:
                return 1.0 # Đã tối giản tốt
            
            # Phạt dựa trên độ phức tạp thừa
            # VD: Gen: 2+2 (ops=1). Canon: 4 (ops=0). Diff=1.
            # Score = 1 / (1 + diff)
            return 1.0 / (1.0 + (ops_generated - ops_canonical))
        
        except:
            return 0.0
        
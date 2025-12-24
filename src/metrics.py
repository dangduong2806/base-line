import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy import Eq, simplify, count_ops, Symbol, Number, Float, Integer, solve
import re

class DeepMathMetrics:
    def __init__(self):
        # Cấu hình parse mạnh mẽ (hiểu 2x là 2*x)
        self.transformations = (standard_transformations + (implicit_multiplication_application,))

    # --------------------------------------------------------------------------
    # 1. HELPER FUNCTIONS (Xử lý chuỗi & Toán học)
    # --------------------------------------------------------------------------

    def _clean_latex(self, text):
        if not text: return ""
        text = str(text).strip()
        # Xử lý các format latex thường gặp
        text = text.replace(r"\boxed", "").replace(r"\overline", "")
        text = text.replace(r"\$", "").replace("\\", "")
        text = text.replace("{", "(").replace("}", ")")
        
        # Xử lý phân số lồng nhau bằng vòng lặp
        while "frac(" in text:
            new_text = re.sub(r'frac\(([^()]+)\)\(([^()]+)\)', r'(\1)/(\2)', text)
            if new_text == text: break 
            text = new_text
        return text
    
    def _extract_math_from_text(self, text):
        """
        Hàm phụ trợ: Dùng Regex để tìm toán trong câu văn tiếng Anh/Việt.
        Ưu tiên:
        1. Dạng phương trình: x = 29, 2x = 58
        2. Dạng số cuối cùng trong câu: ... is 29.
        """
        text = str(text).strip()
        
        # Pattern 1: Tìm phương trình (VD: x = 29, val = 1.5)
        # Regex này tìm: [Biến] [dấu =] [Số]
        eq_pattern = r'([a-zA-Z_]+[a-zA-Z0-9_]*)\s*=\s*([-+]?\d*\.?\d+)'
        matches = re.findall(eq_pattern, text)
        if matches:
            # Lấy phương trình cuối cùng tìm được
            var, val = matches[-1]
            return f"{var} = {val}"
            
        # Pattern 2: Tìm số đứng một mình (thường là đáp án cuối)
        # Tránh lấy số của Step (VD: "Step 7" -> đừng lấy số 7)
        # Logic: Lấy số nằm ở cuối câu hoặc sau các từ khóa "is", "are", "result", "answer"
        
        # Tìm tất cả các số thực
        num_pattern = r'[-+]?\d*\.?\d+'
        nums = re.findall(num_pattern, text)
        
        if nums:
            # Heuristic: Nếu câu bắt đầu bằng "Step N", bỏ số đầu tiên đi nếu nó là số nguyên nhỏ
            if text.lower().startswith("step") and len(nums) > 1:
                return nums[-1] # Lấy số cuối cùng (thường là đáp án 29)
            return nums[-1]
            
        return text # Trả về nguyên gốc để Sympy tự xử lý (hy vọng)

    def _safe_parse(self, text):
        """Parse text thành biểu thức Sympy một cách an toàn và mạnh mẽ hơn"""
        if not text: return None
        
        # BƯỚC 1: Thử parse trực tiếp (cho trường hợp text là toán thuần: "x=29")
        clean_text = self._clean_latex(text)
        try:
            if "=" in clean_text:
                parts = clean_text.split("=")
                if len(parts) == 2:
                    lhs = parse_expr(parts[0], transformations=self.transformations)
                    rhs = parse_expr(parts[1], transformations=self.transformations)
                    return Eq(lhs, rhs)
            return parse_expr(clean_text, transformations=self.transformations)
        except:
            pass # Thất bại thì sang bước 2

        # BƯỚC 2: Fallback - Trích xuất toán từ văn bản (Cho trường hợp "The answer is 29")
        extracted_math = self._extract_math_from_text(clean_text)
        
        # Nếu trích xuất ra vẫn y nguyên text cũ (không lọc được gì) mà bước 1 đã fail -> Fail
        if extracted_math == clean_text: 
            return None

        try:
            # Parse lại phần đã trích xuất (VD: "29" hoặc "x=29")
            if "=" in extracted_math:
                parts = extracted_math.split("=")
                if len(parts) == 2:
                    lhs = parse_expr(parts[0], transformations=self.transformations)
                    rhs = parse_expr(parts[1], transformations=self.transformations)
                    return Eq(lhs, rhs)
            return parse_expr(extracted_math, transformations=self.transformations)
        except:
            return None
    
    def _get_solution_set(self, expr):
        """
        Trả về tập nghiệm (các giá trị số) của biểu thức/phương trình.
        Bỏ qua tên biến (x, y, z...).
        """
        try:
            if expr is None: return set()
            if isinstance(expr, (Number, float, int)):
                return {float(expr)}
            
            if isinstance(expr, Eq):
                free_syms = list(expr.free_symbols)
                if not free_syms: # Dạng 6=6
                    return {float(expr.lhs)} if abs(float(expr.lhs) - float(expr.rhs)) < 1e-6 else set()
                
                # Giải phương trình để lấy giá trị
                solutions = solve(expr)
                values = set()
                for sol in solutions:
                    if isinstance(sol, dict):
                        for v in sol.values(): values.add(v)
                    else:
                        values.add(sol)
                return values
            return set()
        except:
            return set()
        
    # --------------------------------------------------------------------------
    # 2. CORE METRIC CALCULATIONS
    # --------------------------------------------------------------------------
    def _calculate_ee(self, model_final_expr, golden_final_expr):
        """
        Expression Equivalence (EE):
        So sánh đáp án cuối cùng của Model với đáp án chuẩn (Golden).
        Dùng tập nghiệm để bỏ qua sự khác biệt về tên biến.
        """
        if model_final_expr is None or golden_final_expr is None:
            return 0.0
        
        try:
            # Cách 1: So sánh tập nghiệm (Mạnh nhất)
            # VD: Model "y=6", Golden "x=6" -> Đều là {6} -> Match
            model_sols = self._get_solution_set(model_final_expr)
            golden_sols = self._get_solution_set(golden_final_expr)
            
            if model_sols and golden_sols:
                # Kiểm tra xem tập nghiệm có khớp nhau không
                # Dùng giao thoa với sai số nhỏ cho float
                matched = False
                for m in model_sols:
                    for g in golden_sols:
                        try:
                            if abs(float(m) - float(g)) < 1e-7:
                                matched = True; break
                        except: pass
                    if matched: break
                return 1.0 if matched else 0.0

            # Cách 2: Symbolic Equivalence (Fallback cho bài toán rút gọn)
            # VD: Model "x+x", Golden "2x"
            # Helper: Chuyển Eq thành biểu thức
            def _to_expr(e):
                return (e.lhs - e.rhs) if isinstance(e, Eq) else e
            
            model_expr = _to_expr(model_final_expr)
            golden_expr = _to_expr(golden_final_expr)
            
            diff = simplify(model_expr - golden_expr)
            if diff == 0:
                return 1.0
            
        except Exception as e:
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            pass
            
        return 0.0
    
    def _calculate_ass(self, expr):
        """
        Algebraic Simplification Score (ASS):
        Đo lường mức độ tối giản. 
        Nếu biểu thức của model (generated) phức tạp hơn dạng canonical -> điểm thấp.
        Phạt các bước "verbose" hoặc không cần thiết.
        """
        try:
            if expr is None: return 0.0
            
            # Nếu là phương trình Eq(L, R), chuyển thành biểu thức L-R để đo độ phức tạp
            target_expr = (expr.lhs - expr.rhs) if isinstance(expr, Eq) else expr
            
            # Tính độ phức tạp hiện tại
            ops_generated = count_ops(target_expr)
            
            # Tính độ phức tạp lý tưởng (canonical)
            canonical_expr = simplify(target_expr)
            ops_canonical = count_ops(canonical_expr)

            # Nếu model viết còn gọn hơn hoặc bằng canonical -> 1.0 tuyệt đối
            if ops_generated <= ops_canonical:
                return 1.0
            
            # Nếu dài dòng hơn, phạt điểm
            # Nếu dài dòng, phạt theo mức độ:
            # dài dòng 20% → 0.8
            # dài dòng 50% → 0.5
            # dài dòng 100% → 0.0
            complexity_ratio = (ops_generated - ops_canonical) / (ops_canonical + 1e-6)
            # ✅ Sử dụng exponential penalty thay vì linear
            ass_score = 1.0 / (1.0 + complexity_ratio)  # Sigmoid-like
            
            return max(0.0, ass_score)

        except Exception as e:
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0
    

    def _verify_transition(self, expr_prev, expr_curr):
        """
        Kiểm tra logic giữa 2 biểu thức SymPy liền kề (A -> B).
        Logic 3 lớp: Equivalence -> Implication -> Subset.
        """
        try:
            # --- CHECK 1: EQUIVALENCE (Tương đương đại số) ---
            # A tương đương B nếu simplify(A - B) == 0
            diff = simplify(expr_prev - expr_curr)
            if diff == 0:
                return 1.0
            
            # --- CHECK 2: IMPLICATION (Quan hệ kéo theo A => B) ---
            # Giải A tìm nghiệm, thay vào B.
            free_symbols = list(expr_prev.free_symbols)
            
            # Chỉ áp dụng nếu biểu thức đơn giản (1 biến) để tránh treo máy
            if len(free_symbols) == 1:
                variable = free_symbols[0]
                
                # Giải phương trình bước trước
                try:
                    solutions_prev = solve(expr_prev, variable)
                except:
                    solutions_prev = []

                if solutions_prev:
                    all_implied = True
                    for sol in solutions_prev:
                        # Thay nghiệm A vào biểu thức B
                        try:
                            check_val = expr_curr.subs(variable, sol)
                            # Kiểm tra xem thay vào có ra 0 không (hoặc xấp xỉ 0)
                            if simplify(check_val) != 0:
                                # Check sai số float (nếu có)
                                if abs(float(check_val)) > 1e-9:
                                    all_implied = False
                                    break
                        except:
                            all_implied = False
                            break
                    
                    if all_implied:
                        return 1.0 # Logic đúng chiều (A => B)

                # --- CHECK 3: SUBSET (Tập con nghiệm) ---
                # Logic đúng nếu: Tập nghiệm A là TẬP CON của tập nghiệm B
                try:
                    sols_curr = solve(expr_curr, variable)
                    # Chuyển về set số phức để so sánh (tránh lỗi định dạng)
                    set_prev = set([complex(s) for s in solutions_prev]) 
                    set_curr = set([complex(s) for s in sols_curr])
                    
                    if set_prev.issubset(set_curr) and len(set_prev) > 0:
                        return 1.0
                except:
                    pass

        except Exception:
            pass # Nếu lỗi tính toán quá phức tạp, bỏ qua

        return 0.1 # Nếu trượt hết các check -> Logic sai
    
    def _check_step_logic(self, gen_exprs, golden_exprs):
        """
        Tính điểm logic dựa trên chuỗi các biểu thức đã parse.
        Input:
            gen_exprs: List[SymPy Object] (Model generated steps)
            golden_exprs: List[SymPy Object] (Ground Truth steps - dùng để tham chiếu nếu cần)
        Output:
            Float (0.0 -> 1.0)
        """
        # Nếu không có biểu thức nào hoặc chỉ có 1 bước -> Coi như logic OK (hoặc 0 tùy policy)
        if not gen_exprs or len(gen_exprs) < 2:
            return 1.0 

        total_score = 0.0
        transitions = 0

        # Duyệt qua từng cặp bước liền kề trong gen_exprs
        for i in range(len(gen_exprs) - 1):
            prev = gen_exprs[i]
            curr = gen_exprs[i+1]
            
            # Kiểm tra logic chuyển từ prev -> curr
            score = self._verify_transition(prev, curr)
            
            total_score += score
            transitions += 1
        
        # Trả về điểm trung bình cộng logic của cả chuỗi
        return total_score / transitions if transitions > 0 else 0.1
    
    # def _check_step_logic(self, gen_exprs, golden_exprs):
    #     """
    #     Kiểm tra logic của chuỗi bước.
        
    #     Ví dụ bad logic:
    #     - "x = 5" rồi "2x = 10" (đảo ngược)
    #     - "x = 5" rồi "x = 3" (mâu thuẫn)
    #     """
        
        # if len(gen_exprs) < 2:
        #     return 1.0  # Chỉ 1 bước thì ok
        
        # # Lấy solution của mỗi bước
        # step_solutions = []
        # for step in gen_exprs:
        #     sols = self._get_solution_set(step)
        #     step_solutions.append(sols)
        
        # # Kiểm tra không có mâu thuẫn
        # for i in range(len(step_solutions) - 1):
        #     curr_sols = step_solutions[i]
        #     next_sols = step_solutions[i + 1]
            
        #     if curr_sols and next_sols:
        #         # Kiểm tra: Các solutions có tương thích không?
        #         compatible = False
        #         for c in curr_sols:
        #             for n in next_sols:
        #                 try:
        #                     if abs(float(c) - float(n)) < 1e-6:
        #                         compatible = True
        #                         break
        #                 except: pass
        #             if compatible: break
                
        #         if not compatible:
        #             return 0.5  # ← Phạt nếu có mâu thuẫn logic
        
        # return 1.0
    
    def _check_tsa_step(self, step_expr, gt_final_expr, golden_exprs):
        """
        Transformation Step Accuracy (TSA) cho 1 bước.
        Check xem bước này có khớp với bất kỳ bước nào trong Golden Path hoặc GT cuối không.
        """
        if step_expr is None: return False # [Safe check]
        # (Logic TSA đã tối ưu ở câu trả lời trước, dùng lại _get_solution_set)
        step_sols = self._get_solution_set(step_expr)
        
        # Check với toàn bộ Golden Path
        all_targets = golden_exprs + [gt_final_expr]
        
        for target in all_targets:
            target_sols = self._get_solution_set(target)
            if step_sols and target_sols:
                 for s in step_sols:
                    for t in target_sols:
                        try:
                            if abs(float(s) - float(t)) < 1e-8:
                                return True # Có khớp
                        except: pass
        return False
    
    # --------------------------------------------------------------------------
    # 3. MAIN COMPUTE FUNCTION
    # --------------------------------------------------------------------------
    def compute_metrics(self, generated_path_text, label_data):
        """
        Hàm chính gọi từ bên ngoài.
        Input:
            generated_path_text: String chứa toàn bộ lời giải model.
            label_data: Dict JSON chứa thông tin ground truth & golden steps.
        """
        # --- A. PREPARE DATA ---
        # 1. Parse các bước giải của Model (Giả định phân tách bằng dòng mới)
        # Bạn có thể thay bằng regex logic riêng của bạn để tách bước
        gen_lines = [line for line in generated_path_text.split('\n') if line.strip()]
        gen_exprs = [self._safe_parse(line) for line in gen_lines]
        gen_exprs = [e for e in gen_exprs if e is not None] # Lọc cái nào parse được
        
        if not gen_exprs:
            print(f"Không parse được bước nào từ lời giải của mô hình")
            return {"EE": 0.0, "TSA": 0.0, "ASS": 0.0}

        # 2. Extract Golden Path (Biểu thức chuẩn từ dữ liệu)
        golden_exprs = []
        if 'label' in label_data and 'steps' in label_data['label']:
            for step in label_data['label']['steps']:
                txt = None
                if step.get('human_completion'): txt = step['human_completion']['text']
                elif step.get('chosen_completion') is not None:
                    idx = step['chosen_completion']
                    if step.get('completions'): txt = step['completions'][idx]['text']
                if txt:
                    parsed = self._safe_parse(txt)
                    if parsed is not None: golden_exprs.append(parsed)

        # 3. Ground Truth Final (Đáp án cuối cùng của đề)
        gt_str = label_data['question']['ground_truth_answer']
        gt_final_expr = self._safe_parse(gt_str)

        # --- B. COMPUTE METRICS ---

        # 1. EE (Expression Equivalence) - Chỉ tính trên bước cuối cùng của Model
        # So sánh bước cuối model vs (Golden Final Step HOẶC GT Final)
        model_final = gen_exprs[-1]
        golden_final = golden_exprs[-1] if golden_exprs else gt_final_expr
        
        ee_score = self._calculate_ee(model_final, golden_final)

        # 2. TSA (Transformation Step Accuracy) - Tính trên từng bước
        tsa_hits = 0
        for step in gen_exprs:
            if self._check_tsa_step(step, gt_final_expr, golden_exprs):
                tsa_hits += 1
        tsa_score = tsa_hits / len(gen_exprs) if gen_exprs else 0.0

        # 3. ASS (Algebraic Simplification Score) - Trung bình cộng các bước
        ass_scores = [self._calculate_ass(step) for step in gen_exprs]
        ass_score = sum(ass_scores) / len(ass_scores) if ass_scores else 0.0

        # Thêm logic check
        logic_score = self._check_step_logic(gen_exprs, golden_exprs)

        return {
            "EE": ee_score * logic_score,
            "TSA": tsa_score * logic_score,
            "ASS": ass_score * logic_score,
            "num_steps": len(gen_exprs)
        }
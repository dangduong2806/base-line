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
    
    def _safe_parse(self, text):
        """Parse text thành biểu thức Sympy một cách an toàn"""
        try:
            text = self._clean_latex(text)
            # Xử lý dấu bằng để tạo phương trình
            if "=" in text:
                parts = text.split("=")
                if len(parts) == 2:
                    lhs = parse_expr(parts[0], transformations=self.transformations)
                    rhs = parse_expr(parts[1], transformations=self.transformations)
                    return Eq(lhs, rhs)
            return parse_expr(text, transformations=self.transformations)
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
        if not model_final_expr or not golden_final_expr:
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
                            if abs(float(m) - float(g)) < 1e-6:
                                matched = True; break
                        except: pass
                    if matched: break
                return 1.0 if matched else 0.0

            # Cách 2: Symbolic Equivalence (Fallback cho bài toán rút gọn)
            # VD: Model "x+x", Golden "2x"
            diff = simplify(model_final_expr - golden_final_expr) # Lưu ý: nếu là Eq thì cần xử lý lhs-rhs
            if diff == 0: return 1.0
            
        except:
            pass
            
        return 0.0
    
    def _calculate_ass(self, expr):
        """
        Algebraic Simplification Score (ASS):
        Đo lường mức độ tối giản. 
        Nếu biểu thức của model (generated) phức tạp hơn dạng canonical -> điểm thấp.
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
            # Công thức: 1 - (phần thừa / phần gốc)
            return max(0.0, 1.0 - (ops_generated - ops_canonical) / (ops_generated + 1e-6))
        except:
            return 0.0
    
    def _check_tsa_step(self, step_expr, gt_final_expr, golden_exprs):
        """
        Transformation Step Accuracy (TSA) cho 1 bước.
        Check xem bước này có khớp với bất kỳ bước nào trong Golden Path hoặc GT cuối không.
        """
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
                            if abs(float(s) - float(t)) < 1e-6:
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
                    if parsed: golden_exprs.append(parsed)

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

        return {
            "EE": ee_score,
            "TSA": tsa_score,
            "ASS": ass_score,
            "num_steps": len(gen_exprs)
        }
import re
from sympy import sympify, simplify, parse_expr, SympifyError
from sympy.parsing.latex import parse_latex # Cần cài antlr4-python3-runtime nếu dùng cái này
# Trên Kaggle nếu cài parse_latex khó, ta có thể dùng mẹo regex để clean latex
import difflib
class IsomorphismEngine:
    def __init__(self):
        # Các từ khóa thường gặp trong lời giải để loại bỏ
        self.stop_words = ["solve", "step", "calculate", "therefore", "hence", "we get", "implies", "so", "first", "then"]

    def are_equivalent(self, text1, text2):
        """
        Hàm chính: Kiểm tra 2 bước có tương đương không.
        Kết hợp: Symbolic Check (Mạnh nhất) + String Cleaning (Phụ trợ)
        Quy trình: 
        1. Clean Text -> So sánh chuỗi (Exact Match)
        2. Thử Parse Toán -> So sánh hiệu số (Math Match)
        3. Nếu Toán lỗi -> So sánh độ tương đồng văn bản (Fuzzy Text Match)
        """

        # --- BƯỚC 1: TIỀN XỬ LÝ & SO SÁNH CHUỖI CƠ BẢN ---
        # Xóa số thứ tự đầu dòng (VD: "1: First..." -> "First...")
        clean_t1 = self._clean_structure(text1)
        clean_t2 = self._clean_structure(text2)
        # 1. Nếu chuỗi y hệt nhau -> True luôn (nhanh)
        if clean_t1.strip().lower() == clean_t2.strip().lower():
            return True
        
        # --- BƯỚC 2: SO SÁNH TOÁN HỌC (SYMPY) ---
        # Cố gắng trích xuất và tính toán nếu là công thức
        # 2. Cố gắng trích xuất biểu thức toán học từ text
        expr1 = self._extract_and_parse(clean_t1)
        expr2 = self._extract_and_parse(clean_t2)

        # 3. Dùng SymPy để so sánh hiệu
        if expr1 is not None and expr2 is not None:
            try:
                # Kiểm tra hiệu số: expr1 - expr2 == 0 ?
                # simplify() là hàm mạnh nhất của sympy để rút gọn
                diff = simplify(expr1 - expr2)
                # ✅ FIX: Kiểm tra diff có phải số 0 hay không
                # Có 3 trường hợp:
                # 1. diff == 0 (số 0)
                # 2. diff == S(0) (SymPy zero)
                # 3. diff là biểu thức số nhỏ < 1e-10
                if diff == 0:
                    return True
                
                # Nếu diff là số thực, kiểm tra giá trị tuyệt đối
                try:
                    diff_value = float(diff)
                    if abs(diff_value) < 1e-9:
                        return True
                except (TypeError, ValueError):
                    # diff là biểu thức SymPy chứ không phải số
                    # Ví dụ: diff = x - 1 (không thể so sánh với số)
                    pass
                
            except Exception:
                pass # Nếu lỗi tính toán, fallback xuống so sánh text
        
        # --- BƯỚC 3: SO SÁNH VĂN XUÔI THÔNG MINH (FUZZY MATCH) ---
        # Nếu toán học thất bại (do là văn xuôi hoặc parse lỗi), ta dùng Fuzzy Match
        # Logic: Nếu 2 câu văn giống nhau > 85% -> Coi là tương đương
        matcher = difflib.SequenceMatcher(None, clean_t1.lower(), clean_t2.lower())
        similarity = matcher.ratio() # Trả về từ 0.0 đến 1.0
        if similarity > 0.85: # Ngưỡng 85% giống nhau
            return True
        
        return False
    
    def _clean_structure(self, text):
        """Xóa số thứ tự, dấu hai chấm đầu dòng."""
        # Regex xóa: "Step 1:", "1.", "1: ", "###"
        text = re.sub(r'^\s*(?:Step\s+)?\d+[:.]\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[#*]', '', text) # Xóa ký tự markdown
        return text.strip()

    def _extract_and_parse(self, text):
        """
        Trích xuất phần toán học từ câu văn và chuyển thành SymPy Object.
        Ví dụ: "So, x + 1 = 2" -> Sympy(x + 1 - 2) (Chuyển vế đổi dấu)
        """
        try:
            # 1. Làm sạch text (Lower case, bỏ các từ nối)
            clean_text = text.lower()
            for word in self.stop_words:
                clean_text = clean_text.replace(word, "")
            
            # 2. Xử lý dấu "=" (Phương trình)
            # SymPy hiểu biểu thức, không hiểu phương trình theo kiểu "A=B" trực tiếp trong tính toán
            # Mẹo: Chuyển "A = B" thành "A - B"
            if "=" in clean_text:
                parts = clean_text.split("=")
                if len(parts) == 2:
                    lhs = parts[0].strip()
                    rhs = parts[1].strip()
                    # Parse riêng lẻ và trừ nhau
                    expr_lhs = self._safe_parse(lhs)
                    expr_rhs = self._safe_parse(rhs)
                    if expr_lhs and expr_rhs:
                        return expr_lhs - expr_rhs
            # 3. Nếu không có dấu "=", parse bình thường
            return self._safe_parse(clean_text)
        except Exception:
            return None
        
    def _safe_parse(self, expr_text):
        """
        Parse text thành SymPy object. 
        Nếu lỗi (do là văn xuôi) -> Return None im lặng.
        """
        if not expr_text or len(expr_text.strip()) < 2:
            return None
        
        # Nếu chứa quá nhiều từ tiếng Anh thông thường -> Khả năng cao là văn xuôi -> Bỏ qua Parse Toán
        # Đếm số lượng từ dài (trên 3 ký tự)
        words = re.findall(r'[a-z]{3,}', expr_text.lower())
        common_words = set(["the", "and", "for", "with", "that", "this", "have", "from", "boys", "girls", "ways", "seats"])
        # Nếu câu chứa từ ngữ văn xuôi thông thường -> Return None ngay để đỡ tốn thời gian parse
        if any(w in common_words for w in words):
             return None
        
        try:
            # Xử lý các ký hiệu LaTeX cơ bản thường gặp
            str_expr = expr_text.replace("^", "**") # Python dùng ** cho lũy thừa
            str_expr = str_expr.replace("\\frac", "frac") # Bỏ backslash
            str_expr = re.sub(r'\\boxed\{(.*?)\}', r'\1', str_expr) # Lấy nội dung trong boxed
            str_expr = str_expr.rstrip(".") # Bỏ dấu chấm câu

            # Lọc bỏ các ký tự lạ không phải toán học (chữ cái text)
            # Bước này tricky, ở mức đơn giản ta cứ thử parse
            return sympify(str_expr)
        except (SympifyError, SyntaxError, TypeError, AttributeError):
            return None
    
if __name__ == "__main__":
    iso = IsomorphismEngine()
    
    # Test case 1: Đổi vế
    t1 = "x + 5 = 10"
    t2 = "x = 5" 
    print(f"Test 1 (Equation): {iso.are_equivalent(t1, t2)}") # Kỳ vọng: True (vì x+5-10 = x-5)
    
    # Test case 2: Biến đổi đại số
    t3 = "(x + 1)^2"
    t4 = "x^2 + 2*x + 1"
    print(f"Test 2 (Expansion): {iso.are_equivalent(t3, t4)}") # Kỳ vọng: True
    
    # Test case 3: Lời văn khác nhau
    t5 = "Therefore, the value is 2*x"
    t6 = "2*x"
    print(f"Test 3 (Text noise): {iso.are_equivalent(t5, t6)}") # Kỳ vọng: True
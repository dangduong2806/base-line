import re
from sympy import sympify, simplify, parse_expr
from sympy.parsing.latex import parse_latex # Cần cài antlr4-python3-runtime nếu dùng cái này
# Trên Kaggle nếu cài parse_latex khó, ta có thể dùng mẹo regex để clean latex
class IsomorphismEngine:
    def __init__(self):
        # Các từ khóa thường gặp trong lời giải để loại bỏ
        self.stop_words = ["solve", "step", "calculate", "therefore", "hence", "we get", "implies"]

    def are_equivalent(self, text1, text2):
        """
        Hàm chính: Kiểm tra 2 bước có tương đương không.
        Kết hợp: Symbolic Check (Mạnh nhất) + String Cleaning (Phụ trợ)
        """
        # 1. Nếu chuỗi y hệt nhau -> True luôn (nhanh)
        if text1.strip().lower() == text2.strip().lower():
            return True
        
        # 2. Cố gắng trích xuất biểu thức toán học từ text
        expr1 = self._extract_and_parse(text1)
        expr2 = self._extract_and_parse(text2)

        # 3. Dùng SymPy để so sánh hiệu
        if expr1 is not None and expr2 is not None:
            try:
                # Kiểm tra hiệu số: expr1 - expr2 == 0 ?
                # simplify() là hàm mạnh nhất của sympy để rút gọn
                diff = simplify(expr1 - expr2)
                if diff == 0:
                    return True
            except Exception:
                pass # Nếu lỗi tính toán, fallback xuống so sánh text
        return False

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
        except Exception as e:
            return None
    def _safe_parse(self, expr_text):
        """
        Wrapper để parse chuỗi thành SymPy object, xử lý lỗi cú pháp
        """
        try:
            # Xử lý các ký hiệu LaTeX cơ bản thường gặp
            str_expr = str_expr.replace("^", "**") # Python dùng ** cho lũy thừa
            str_expr = str_expr.replace("\\frac", "frac") # Bỏ backslash
            str_expr = re.sub(r'\\boxed\{(.*?)\}', r'\1', str_expr) # Lấy nội dung trong boxed

            # Lọc bỏ các ký tự lạ không phải toán học (chữ cái text)
            # Bước này tricky, ở mức đơn giản ta cứ thử parse
            return sympify(str_expr)
        except (SyntaxError):
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
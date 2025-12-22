from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Trỏ đến thư mục chứa các file đã lưu ở trên
model_path = "tools/prm_specialist_deberta-v3-base" 

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Test thử một bước giải
text = "1+1=? [SEP] Step 1: 1+1=2"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
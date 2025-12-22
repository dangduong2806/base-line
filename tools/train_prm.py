import json
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import yaml

# Cấu hình
config_path = "configs/main_config.yaml"

class PRMDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=512):    
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load data PRM800K (Phase 2 - label từng bước)
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                
                # Kiểm tra dữ liệu đầu vào cơ bản
                if 'question' not in item or 'problem' not in item['question']:
                    continue
                if 'label' not in item or 'steps' not in item['label'] or item['label']['steps'] is None:
                    continue

                problem = item['question']['problem']

                for step in item['label']['steps']:
                    # Kiểm tra nếu completions bị None thì bỏ qua
                    if step.get('completions') is None:
                        continue
                    for comp in step['completions']:
                        if comp['rating'] is None:
                            continue

                        text = problem + " [SEP] " + comp["text"]
                        label = 1 if comp["rating"] > 0 else 0

                        self.data.append({
                            'text': text,
                            'label': label
                        })
        print("Loaded PRM samples:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(item['label'], dtype=torch.long)
        }

def train():
    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    MODEL_NAME = config['tool']['name1']
    OUTPUT_DIR = config['tool']['output_dir1']
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Load dataset (Bạn cần trỏ đúng file phase2_train.jsonl)
    full_dataset = PRMDataset("data/raw/phase1_train.jsonl", tokenizer)
    
    # Chia 90% train, 10% validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=2e-5,
        fp16=True, # Tiết kiệm VRAM
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none", # Tránh lỗi nếu bạn chưa cài wandb
        greater_is_better=True, # accuracy càng cao càng tốt
    )

    # Định nghĩa hàm tính toán độ chính xác (Accuracy) cho tập Eval
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        return {"accuracy": (predictions == torch.tensor(labels)).float().mean().item()}
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Bắt đầu huán luyện")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("PRM Specialist Model trained and saved!")

if __name__ == "__main__":
    train() # Uncomment để chạy
    pass
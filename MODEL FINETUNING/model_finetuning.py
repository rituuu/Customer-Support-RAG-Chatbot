#THIS CODE SHOULD BE EXECUTED IN GOOGLE COLAB WITH RUNTIME OF EITHER A100 GPU OR T4 GPU

#!pip install -q transformers accelerate peft trl bitsandbytes datasets wandb

import os
from google.colab import userdata

import os
from google.colab import userdata

# Load HuggingFace token securely
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN") or "your_hf_token_here"

# Step 2: Load base model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "google/gemma-1.1-2b-it"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16")

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=os.environ["HF_TOKEN"])
tokenizer.pad_token = tokenizer.eos_token  # Set <eos> token as padding

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    use_auth_token=os.environ["HF_TOKEN"]
)

# Step 3: Apply QLoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Step 4: Load and preprocess dataset
from datasets import load_dataset

def format_chat(example):
    messages = example["messages"]
    formatted = ""
    for msg in messages:
        if msg["role"] == "user":
            formatted += f"<|user|>\n{msg['content'].strip()}\n"
        elif msg["role"] == "assistant":
            formatted += f"<|assistant|>\n{msg['content'].strip()}\n"
    return {"text": formatted.strip()}

data_path = "/content/Processed_FAQ_data.jsonl"
dataset = load_dataset("json", data_files={"train": data_path})
dataset["train"] = dataset["train"].map(format_chat)

# Step 5: Tokenization with labels = input_ids
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    tokens["labels"] = tokens["input_ids"].clone()
    return {k: v[0] for k, v in tokens.items()}  # Remove batch dim

dataset["train"] = dataset["train"].map(tokenize, remove_columns=["messages", "text"])
dataset["train"].set_format(type="torch")

# Step 6: Fine-tuning
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="gemma-qlora-customer-support2.0",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    report_to="wandb",  # optional
    gradient_checkpointing=True,
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    args=training_args
)

trainer.train()

#THIS CODE SHOULD BE EXECUTED IN GOOGLE COLAB WITH RUNTIME OF EITHER A100 GPU OR T4 GPU
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

#  Step 1: Define model paths
base_model_id = "google/gemma-1.1-2b-it"
adapter_dir = "/content/gemma-qlora-customer-support2.0/checkpoint-69"  # fine-tuned adapter

#  Step 2: Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

#  Step 3: Load adapter correctly (local dir)
model = PeftModel.from_pretrained(
    model,
    adapter_dir,
    from_transformers=True  # <--- THIS FIXES THE ERROR
)

#  Step 4: Merge adapter weights with base model
model = model.merge_and_unload()

#  Step 5: Save merged model
save_path = "/content/gemma-qlora-merged"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

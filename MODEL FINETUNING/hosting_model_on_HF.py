#THIS CODE SHOULD BE EXECUTED IN GOOGLE COLAB WITH RUNTIME OF EITHER A100 GPU OR T4 GPU
from huggingface_hub import notebook_login

notebook_login()

# Step 1: Install required library
#!pip install huggingface_hub --quiet

# Step 2: Import necessary modules
from huggingface_hub import login, upload_folder
from google.colab import userdata

# Step 3: Securely retrieve Hugging Face token from Colab secret manager
hf_token = userdata.get("model_upload")

# Step 4: Login to Hugging Face Hub
login(token=hf_token)

# Step 5: Define path to the merged model folder
model_path = "/content/gemma-qlora-merged"  # Update this path if needed

# Step 6: Define your Hugging Face Hub repo ID (namespace/repo_name)
repo_id = "RituGujela100/gemma-qlora-customer-support-v2.0"  # Your repo ID created from HF Hub

# Step 7: Upload the entire folder to the Hugging Face Hub
upload_folder(
    repo_id=repo_id,
    folder_path=model_path,
    token=hf_token,          # Ensures it uses your write token
    repo_type="model"        # Default is "model", change to "dataset" if you're uploading datasets
)

print(f" Model successfully uploaded to: https://huggingface.co/{repo_id}")

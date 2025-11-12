from langchain.chains import RetrievalQA 
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import boto3

# Function to load the model
def get_gemma_llm():
    model_id = "RituGujela100/gemma-qlora-customer-support-v2.0"  # Your fine-tuned Gemma model
    device = 0 if torch.cuda.is_available() else -1

    try:
        # Try to load using the standard pipeline approach
        pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=model_id,
            device=device,
            max_new_tokens=300,
            return_full_text=False,
            do_sample=False,
            temperature=0.3,
            repetition_penalty=1.15,
        )
        print("Loaded Gemma model successfully using Transformers pipeline.")
    except KeyError as e:
        if "gemma" in str(e).lower():
            print("⚠️ 'gemma' model type not recognized. Falling back to AutoModelForCausalLM...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=300,
                return_full_text=False,
                do_sample=False,
                temperature=0.3,
                repetition_penalty=1.15,
            )
            print("Loaded Gemma model successfully using AutoModelForCausalLM fallback.")
        else:
            raise e

    # Cleaning function for model responses
    def clean_response(text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = [
            line for line in lines
            if not line.strip().lower().startswith(('assistant:', 'customer:', 'ps:', 'p.s.'))
        ]
        cleaned_text = '\n'.join(cleaned_lines)
        for noise in [
            "Assistant:", "Customer:", "Helpful Assistant", "<br />", "[Insert Reply]",
            "[Company Name]", "[insert email address]", "[customer name]", "[Your Name]"
        ]:
            cleaned_text = cleaned_text.replace(noise, "")
        return cleaned_text.strip()

    # Custom wrapper class for HuggingFacePipeline
    class CustomHuggingFacePipeline(HuggingFacePipeline):
        def __call__(self, prompt: str, **kwargs):
            raw = self.pipeline(prompt, **kwargs)[0]['generated_text']
            return clean_response(raw)

    return CustomHuggingFacePipeline(pipeline=pipe)


# Updated prompt for concise, relevant answers
prompt_template = """
You are a helpful, polite, and professional Amazon customer support assistant. Always respond in natural human language, using complete sentences and up to 250 words if needed. Be empathetic if the customer is upset, and never include assistant labels in your reply. You have to provide a logical solution that makes sense to the customer's problem. Provide solution in minimum 250 words and maximum 1000 words. But make sure you don't make up anything on your own and always comply with Amazon's policies.
Reference:
{context}
Customer query:
{question}
Response:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]  
)


# Retrieval function
def get_response_llm(llm, vectorstore_faiss, query):
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=False,
            chain_type_kwargs={"prompt": PROMPT}
        )
        result = qa.invoke({"query": query})
        return result["result"]
    except Exception as e:
        return f"Error during retrieval: {str(e)}"


from langchain.chains import RetrievalQA  
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import gradio as gr
import os

def get_gemma_llm():
    model_id = "RituGujela100/gemma-qlora-customer-support-v2.0"
    device = -1
    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=model_id,
        device=device,
        max_new_tokens=300,
        return_full_text=False,
        do_sample=False,
        temperature=0.4,
        repetition_penalty=1.35,  
    )
    print("Loaded Gemma model successfully using Transformers pipeline")

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

    class CustomHuggingFacePipeline(HuggingFacePipeline):
        def __call__(self, prompt: str, **kwargs):
            raw = self.pipeline(prompt, **kwargs)[0]['generated_text']
            return clean_response(raw)

    return CustomHuggingFacePipeline(pipeline=pipe)


prompt_template = """
You are a helpful, polite, and professional Amazon customer support assistant. Always respond in natural human language, using complete sentences. Give a complete, concise and logical solution. You have to provide a logical solution that makes sense to the customer's problem. Be empathetic if the customer is upset, and never include assistant labels in your reply. Make sure you don't made up anything on your own and always comply with Amazon's policies. Don't extend your answer unnecessarily, Keep it to the point. Always end your answer with complete sentences. If you don't know anything, politely suggest the customer to contact Amazon Customer Support directly.
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

def get_response_llm(llm, vectorstore_faiss, query, threshold=0.35, top_k=6):
    try:
        # Get top-k documents + scores from FAISS
        docs_and_scores = vectorstore_faiss.similarity_search_with_score(query, k=top_k)

        # Convert distance → pseudo-similarity and filter by threshold
        filtered_docs = []
        for doc, score in docs_and_scores:
            similarity = 1 - score  # FAISS distance → approximate similarity
            if similarity > threshold:
                doc.metadata["score"] = round(similarity, 3)
                filtered_docs.append(doc)

        if not filtered_docs:
            return "Sorry, I couldn’t assist in this case. I can't find enough relevant information. Please contact Amazon Customer Support. If you need any more help, feel free to ask!"

        # Concatenate only relevant chunks
        context = "\n\n".join([doc.page_content for doc in filtered_docs])

        # Build the final prompt
        final_prompt = PROMPT.format(context=context, question=query)

        # Generate response
        response = llm(final_prompt)
        return response.strip()

    except Exception as e:
        return f"Error during retrieval: {str(e)}"



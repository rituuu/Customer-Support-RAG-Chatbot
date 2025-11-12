import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from QASystem.retrievalll import get_gemma_llm, get_response_llm
from QASystem.ingestionnn import data_ingestion, get_vector_store

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

faiss_index = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Load your fine-tuned Gemma model (hosted on Hugging Face)
llm = get_gemma_llm()
# Main prediction function (with error handling)
def answer_question_from_pdf(question):
    """Answer question based on PDF content using RAG."""
    if not question.strip():
        return "Please enter a valid question."
    try:
        answer = get_response_llm(llm, faiss_index, question)
        return answer.strip()  # Ensures only the answer is returned, cleanly
    except Exception as e:
        return f"Error: {str(e)}"  # Errors are captured and shown in the answer box

# Gradio Interface
gradio_app = gr.Interface(
    fn=answer_question_from_pdf,
    inputs=gr.Textbox(lines=2, label="Ask a question from the PDF"),
    outputs=gr.Textbox(label="Answer"),
    title="Welcome to QuickQuery AI, Your AI-Powered Support Assistant. Ask Anything, Get Instant Help",
    description="Twisting complexity into clarity – one question at a time. Illuminate your queries with next-gen support. Smarter answers. Faster resolutions. Just ask! How Can I Help You Today?",
)

# App launch
if __name__ == "__main__":
    gradio_app.launch(server_name="127.0.0.1", server_port=7860)

print("App successfully executed")

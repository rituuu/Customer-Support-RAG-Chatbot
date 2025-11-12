# DATA INGESTION AND PREPROCESSING
import os
import json
import fitz  # PyMuPDF

DATA_DIR = "./data"
OUTPUT_FILE = "Processed_FAQ_data22.jsonl"


def extract_qa_from_pdf_text(text):
    """
    Parses the extracted PDF text and segments it into clean Q/A pairs.
    Returns a list of dicts in {"messages": [...]} format.
    """
    lines = text.split('\n')
    qa_pairs = []
    current_question = ""
    current_answer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.lower().startswith("q") or line.endswith("?") or any(kw in line.lower() for kw in ["what should i do", "how can i", "how do i"]):
            if current_question and current_answer:
                qa_pairs.append({
                    "messages": [
                        {"role": "user", "content": current_question},
                        {"role": "assistant", "content": " ".join(current_answer)}
                    ]
                })
                current_answer = []
            current_question = line
        else:
            current_answer.append(line)

    if current_question and current_answer:
        qa_pairs.append({
            "messages": [
                {"role": "user", "content": current_question},
                {"role": "assistant", "content": " ".join(current_answer)}
            ]
        })

    return qa_pairs


def convert_pdf_to_jsonl(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    qa_examples = extract_qa_from_pdf_text(full_text)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in qa_examples:
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved {len(qa_examples)} Q/A pairs to {output_path}")


if __name__ == "__main__":
    print("Reading PDF and converting to JSONL...")
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_DIR, file)
            convert_pdf_to_jsonl(pdf_path, OUTPUT_FILE)
    print(" PDF to JSONL conversion complete. File ready for fine-tuning!")

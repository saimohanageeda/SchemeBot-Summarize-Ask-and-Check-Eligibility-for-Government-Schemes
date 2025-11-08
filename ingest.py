import os
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR = "data/pdfs"
INDEX_DIR = "data/faiss_index"

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text")
    return text

def build_index():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    all_docs = []
    all_metadata = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            scheme_name = os.path.splitext(filename)[0].lower()
            print(f"📄 Processing {filename}")
            pdf_path = os.path.join(DATA_DIR, filename)
            text = extract_text_from_pdf(pdf_path)

            if text.strip():
                all_docs.append(text)
                all_metadata.append({"scheme_name": scheme_name})

    if not all_docs:
        print("❌ No PDF text found.")
        return

    print("✅ Using HuggingFace embeddings (offline)...")
    vs = FAISS.from_texts(all_docs, embeddings, metadatas=all_metadata)
    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"✅ Index built and saved in {INDEX_DIR}")

if __name__ == "__main__":
    build_index()
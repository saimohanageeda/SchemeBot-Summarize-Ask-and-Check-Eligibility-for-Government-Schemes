import os
import re
import json
import torch
from difflib import SequenceMatcher
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------
# Device setup
# ----------------------------
device = 0 if torch.cuda.is_available() else -1
print(f"✅ Using {'GPU' if device == 0 else 'CPU'}")

# ----------------------------
# Paths
# ----------------------------
ELIGIBILITY_PATH = os.path.join("rules", "eligibility.json")
INDEX_PATH = os.path.join("data", "faiss_index")

# ----------------------------
# Load eligibility JSON
# ----------------------------
if not os.path.exists(ELIGIBILITY_PATH):
    raise FileNotFoundError(f"❌ Eligibility file not found at {ELIGIBILITY_PATH}")

with open(ELIGIBILITY_PATH, "r", encoding="utf-8") as f:
    eligibility_data = json.load(f)

# ----------------------------
# Load FAISS index
# ----------------------------
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"❌ FAISS index not found at {INDEX_PATH}. Run ingestion first.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# ----------------------------
# Load QA model
# ----------------------------
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2",
    device=device
)

# ----------------------------
# Helper functions
# ----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_summary_sections(scheme_name: str) -> dict:
    path = os.path.join("summaries", f"{scheme_name.lower().replace(' ', '_')}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sections", {})

def get_hint_category(query: str) -> str:
    query_lower = query.lower()
    if any(k in query_lower for k in ["income", "eligibility", "criteria", "limit"]):
        return "income"
    if any(k in query_lower for k in ["subsidy", "financial assistance", "benefit"]):
        return "subsidy"
    return None

def match_section_heading(sections: dict, query: str) -> str:
    hint = get_hint_category(query)
    best_match, best_ratio = None, 0.0
    for title, text in sections.items():
        ratio = SequenceMatcher(None, query.lower(), title.lower()).ratio()
        title_lower = title.lower()
        if hint and hint in title_lower:
            ratio += 0.25
        if ratio > best_ratio:
            best_match, best_ratio = text, ratio
    return best_match if best_ratio > 0.4 else None

def filter_section_by_keywords(section_text: str, query: str) -> str:
    query_keywords = re.findall(r'\w+', query.lower())
    sentences = re.split(r'(?<=[.!?]) +', section_text)
    relevant = [s for s in sentences if any(k in s.lower() for k in query_keywords)]
    if relevant:
        return " ".join(relevant)
    words = section_text.split()
    return " ".join(words[:120])

def keyword_search_in_summary(sections: dict, query: str) -> str:
    query_keywords = re.findall(r'\w+', query.lower())
    all_sentences = []
    for text in sections.values():
        sentences = re.split(r'(?<=[.!?]) +', text)
        all_sentences.extend(sentences)
    relevant = [s for s in all_sentences if any(k in s.lower() for k in query_keywords)]
    if relevant:
        return " ".join(relevant[:3])
    return None

# ----------------------------
# Main QA function
# ----------------------------
def ask_question(scheme_name: str, query: str) -> str:

    query_lower = query.lower().strip()

    # 🔹 Case-insensitive scheme lookup
    scheme_info = None
    for key in eligibility_data:
        if key.lower() == scheme_name.lower():
            scheme_info = eligibility_data[key]
            break

    # =====================================================
    # 1️⃣ Check eligibility.json first (Structured answers)
    # =====================================================
    if scheme_info:

        # -------- INCOME LIMIT --------
        if "income" in query_lower:

            income_info = scheme_info.get("income_limit")

            if income_info is None:
                return "💡 There is no income limit. It is not income based."

            if isinstance(income_info, dict) and income_info:
                return "💡 Income Limits:\n" + "\n".join(
                    [f"{k}: ₹{v:,}" for k, v in income_info.items()]
                )

        # -------- AGE LIMIT --------
        if "age" in query_lower:

            age_limit = scheme_info.get("age_limit")

            if age_limit is None:
                return "💡 There is no age limit. It is not age based."

            return f"💡 Minimum Age Required: {age_limit} years"

        # -------- BENEFITS --------
        if any(k in query_lower for k in ["subsidy", "financial assistance", "benefit"]):

            subsidy_info = scheme_info.get("subsidy_amount")

            if isinstance(subsidy_info, dict) and subsidy_info:
                return "💡 Benefits:\n" + "\n".join(
                    [f"{k}: ₹{v:,}" for k, v in subsidy_info.items()]
                )

    # =====================================================
    # 2️⃣ Try cached summary
    # =====================================================
    sections = load_summary_sections(scheme_name)
    context = match_section_heading(sections, query)

    if context:
        context = filter_section_by_keywords(context, query)
        try:
            result = qa_pipeline(question=query, context=context)
            answer = clean_text(result["answer"])
            if answer:
                return f"💡 Answer (from summary): {answer}"
        except Exception as e:
            print(f"⚠️ QA failed on summary: {e}")

    # =====================================================
    # 3️⃣ Keyword search in summary
    # =====================================================
    context = keyword_search_in_summary(sections, query)

    if context:
        try:
            result = qa_pipeline(question=query, context=context)
            answer = clean_text(result["answer"])
            if answer:
                return f"💡 Answer (from summary): {answer}"
        except Exception as e:
            print(f"⚠️ QA failed on summary sentences: {e}")

    # =====================================================
    # 4️⃣ FAISS fallback
    # =====================================================
    docs = retriever.invoke(query)
    scheme_key = scheme_name.lower().replace(" ", "_")

    scheme_docs = [
        d for d in docs
        if scheme_key in d.metadata.get("source", "").lower()
        or scheme_name.lower() in d.page_content.lower()
    ]

    if not scheme_docs:
        return f"❌ No relevant information found for {scheme_name}."

    context = " ".join([d.page_content for d in scheme_docs])
    context = clean_text(context)

    try:
        result = qa_pipeline(question=query, context=context)
        answer = clean_text(result["answer"])
    except Exception as e:
        return f"⚠️ QA failed on FAISS: {e}"

    if not answer or answer.lower() in ["", "none", "no answer"]:
        snippet = context[:300] + "..." if len(context) > 300 else context
        return f"ℹ️ Couldn’t find a precise answer. Here's some relevant info:\n{snippet}"

    return f"💡 Answer (from FAISS): {answer}"


# ----------------------------
# CLI Entry
# ----------------------------
if __name__ == "__main__":
    scheme_name = input("Ask a Question about which Scheme? ").strip()
    query = input("Enter your question: ").strip()
    print(ask_question(scheme_name, query))

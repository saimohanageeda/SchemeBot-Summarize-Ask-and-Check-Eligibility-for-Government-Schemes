import os
import re
import json
import torch
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------
# Device setup
# ----------------------------
device = 0 if torch.cuda.is_available() else -1
print(f"✅ Using {'GPU' if device == 0 else 'CPU'} for summarization")

# ----------------------------
# Load FAISS index
# ----------------------------
INDEX_PATH = "data/faiss_index"
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError("❌ FAISS index not found. Run ingestion first.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# ----------------------------
# Summarization model
# ----------------------------
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device
)

# ----------------------------
# Helper functions
# ----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").strip()
    return text

def chunk_text(text: str, max_words: int = 400) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def deduplicate_sentences(sentences: list[str]) -> list[str]:
    seen = set()
    unique = []
    for s in sentences:
        s_clean = s.strip().lower()
        if s_clean not in seen and len(s_clean) > 10:
            seen.add(s_clean)
            unique.append(s.strip())
    return unique

def extract_numeric_facts(text: str) -> list[str]:
    """Extract numeric facts like ₹ amounts, percentages, years, limits."""
    pattern = r"(₹\s?\d[\d,.]*\s?(lakh|crore)?)|(\d+%?)|(\b\d{4}\b)"
    return list(set(re.findall(pattern, text)))

def summarize_chunks(chunks: list[str]) -> str:
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
            summaries.append(summary)
        except Exception:
            continue
    full_summary = " ".join(summaries)
    # Split into sentences, deduplicate
    sentences = re.split(r'(?<=[.!?]) +', full_summary)
    sentences = deduplicate_sentences(sentences)
    # Keep numeric facts
    numeric_facts = ["• " + "".join(fact).strip() for fact in extract_numeric_facts(full_summary)]
    bullets = ["• " + s.strip() for s in sentences if s.strip()]
    # Numeric facts first
    return "\n".join(numeric_facts + bullets)

# ----------------------------
# Sections & Keywords
# ----------------------------
SECTIONS = [
    "Overview",
    "Objectives",
    "Eligibility",
    "Benefits",
    "Documents Required",
    "Application Process",
    "Conditions / Monitoring",
    "Innovation & Sustainability"
]

SECTION_KEYWORDS = {
    "Overview": ["overview", "introduction", "aim", "purpose", "mission", "launch"],
    "Objectives": ["objective", "goal", "aim", "purpose", "vision"],
    "Eligibility": ["eligibility", "criteria", "who can apply", "category", "income", "beneficiary"],
    "Benefits": ["benefit", "subsidy", "grant", "loan", "financial assistance"],
    "Documents Required": ["document", "proof", "certificate", "Aadhaar", "photo", "income proof"],
    "Application Process": ["apply", "procedure", "portal", "registration", "process"],
    "Conditions / Monitoring": ["condition", "monitoring", "responsibility", "ownership", "transfer"],
    "Innovation & Sustainability": ["innovation", "green", "sustainable", "eco-friendly", "technology"]
}

def extract_relevant_text(text: str, keywords: list[str]) -> str:
    if not keywords:
        return text
    pattern = "|".join(re.escape(k.lower()) for k in keywords)
    sentences = re.split(r'(?<=[.!?]) +', text)
    relevant = [s for s in sentences if re.search(pattern, s.lower())]
    return " ".join(relevant) if relevant else text

# ----------------------------
# Main summarizer
# ----------------------------
SUMMARY_CACHE = "summaries"
os.makedirs(SUMMARY_CACHE, exist_ok=True)

def get_cache_path(scheme_name: str) -> str:
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', scheme_name.lower())
    return os.path.join(SUMMARY_CACHE, f"{safe_name}.json")

def summarize_scheme(scheme_name: str, db=db) -> str:
    cache_path = get_cache_path(scheme_name)

    # Check cache
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"📂 Retrieved cached summary for '{scheme_name}' ✅\n")
        return cached["final_summary"]

    # Retrieve relevant docs from FAISS
    scheme_key = scheme_name.lower().replace(" ", "_")
    all_docs = db.similarity_search(scheme_name, k=8)
    scheme_docs = [
        d for d in all_docs
        if scheme_key in d.metadata.get("source", "").lower()
        or scheme_name.lower() in d.page_content.lower()
    ]
    if not scheme_docs:
        return f"❌ No information found for {scheme_name}"

    text = clean_text(" ".join([d.page_content for d in scheme_docs]))
    words = text.split()
    if len(words) > 4000:
        text = " ".join(words[:4000])

    final_summary = f"📘 **Summary for {scheme_name}:**\n\n"
    structured_summary = {}

    for section in SECTIONS:
        keywords = SECTION_KEYWORDS.get(section, [])
        section_text = extract_relevant_text(text, keywords)
        chunks = chunk_text(section_text)
        section_summary = summarize_chunks(chunks)
        structured_summary[section] = section_summary
        final_summary += f"### 🟢 {section}\n{section_summary}\n\n"

    # Cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"scheme_name": scheme_name, "final_summary": final_summary, "sections": structured_summary}, f, indent=2)

    print(f"💾 Saved summary for '{scheme_name}' ✅")
    return final_summary.strip()

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    scheme_name = input("Enter scheme name to summarize: ").strip()
    print(summarize_scheme(scheme_name))
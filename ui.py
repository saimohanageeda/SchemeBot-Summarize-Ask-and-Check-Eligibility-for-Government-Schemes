import sys
import os
import base64

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app.qa import ask_question
from app.summarize import summarize_scheme
from app.eligibility import check_eligibility

# -------------------------------------------------
# 🔹 CONFIG
# -------------------------------------------------

st.set_page_config(page_title="🏛️ SchemeBot", layout="centered")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

st.title("🏛️ SchemeBot: Summarize, Ask and Check Eligibility for Government Schemes")

# -------------------------------------------------
# 🔹 BACKGROUND FUNCTION
# -------------------------------------------------

def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background:
                    linear-gradient(
                        rgba(0,0,0,0.75),
                        rgba(0,0,0,0.75)
                    ),
                    url("data:image/png;base64,{encoded}");
                background-size: contain;
                background-position: center;
                background-repeat: no-repeat;
                background-color: #000;
                min-height: 100vh;
            }}

            .main .block-container {{
                max-width: 950px;
                margin: auto;
                padding: 2rem;
                background: rgba(15,15,15,0.90);
                border-radius: 18px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.6);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------------------
# 🔹 IMAGE DETECTION
# -------------------------------------------------

def get_scheme_image(scheme_name):
    base_name = scheme_name.lower().replace(" ", "_")
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(ASSETS_DIR, base_name + ext)
        if os.path.exists(path):
            return path
    return None

def get_home_image():
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(ASSETS_DIR, "home_banner" + ext)
        if os.path.exists(path):
            return path
    return None

# -------------------------------------------------
# 🔹 LOAD SCHEMES
# -------------------------------------------------

PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")

AVAILABLE_SCHEMES = []

if os.path.exists(PDF_DIR):
    AVAILABLE_SCHEMES = [
        os.path.splitext(f)[0]
        for f in os.listdir(PDF_DIR)
        if f.endswith(".pdf")
    ]

if not AVAILABLE_SCHEMES:
    st.warning("⚠️ No schemes found in data/pdfs folder.")

# -------------------------------------------------
# 🔹 SESSION STATE
# -------------------------------------------------

if "stage" not in st.session_state:
    st.session_state["stage"] = "choose_scheme"

if "scheme" not in st.session_state:
    st.session_state["scheme"] = None

# -------------------------------------------------
# 🔹 STAGE 1: CHOOSE SCHEME
# -------------------------------------------------

if st.session_state["stage"] == "choose_scheme":

    home_image = get_home_image()
    if home_image:
        set_background(home_image)

    st.subheader("📌 Select a Government Scheme")

    if AVAILABLE_SCHEMES:
        selected_scheme = st.selectbox(
            "👉 Choose a scheme:",
            AVAILABLE_SCHEMES
        )

        if st.button("Submit Scheme"):
            st.session_state["scheme"] = selected_scheme
            st.session_state["stage"] = "show_summary"
            st.rerun()

# -------------------------------------------------
# 🔹 STAGE 2: SHOW SUMMARY
# -------------------------------------------------

elif st.session_state["stage"] == "show_summary":

    scheme = st.session_state["scheme"]

    scheme_image = get_scheme_image(scheme)
    if scheme_image:
        set_background(scheme_image)

    st.subheader(f"📄 Summary for {scheme}")

    with st.spinner("Generating summary..."):
        summary = summarize_scheme(scheme)
        st.write(summary)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💬 Ask Questions"):
            st.session_state["stage"] = "ask_questions"
            st.rerun()

    with col2:
        if st.button("🔄 Start Over"):
            st.session_state.clear()
            st.rerun()

# -------------------------------------------------
# 🔹 STAGE 3: ASK QUESTIONS (UPDATED WITH FAQ)
# -------------------------------------------------

elif st.session_state["stage"] == "ask_questions":

    scheme = st.session_state["scheme"]

    scheme_image = get_scheme_image(scheme)
    if scheme_image:
        set_background(scheme_image)

    st.subheader(f"💬 Ask a Question about {scheme}")

    # Frequently Asked Questions
    COMMON_QUESTIONS = [
        "Who is eligible?",
        "What is the income limit?",
        "What is the age limit?",
        "What are the benefits?",
        "Type my own question"
    ]

    selected_question = st.selectbox(
        "Choose a common question:",
        COMMON_QUESTIONS
    )

    if selected_question == "Type my own question":
        query = st.text_input("Enter your custom question:")
    else:
        query = selected_question

    if st.button("Get Answer"):
        if query:
            with st.spinner("Searching answer..."):
                answer = ask_question(scheme, query)
                st.success(answer)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Check Eligibility"):
            st.session_state["stage"] = "check_eligibility"
            st.rerun()

    with col2:
        if st.button("🔄 Start Over"):
            st.session_state.clear()
            st.rerun()

# -------------------------------------------------
# 🔹 STAGE 4: CHECK ELIGIBILITY
# -------------------------------------------------

elif st.session_state["stage"] == "check_eligibility":

    scheme = st.session_state["scheme"]

    scheme_image = get_scheme_image(scheme)
    if scheme_image:
        set_background(scheme_image)

    st.subheader(f"✅ Check Eligibility for {scheme}")

    age = st.number_input("Enter your age:", min_value=0, max_value=120)
    income = st.number_input("Enter your annual income (₹):", min_value=0)
    state = st.text_input("Enter your state:")

    if st.button("Check Eligibility"):
        user_data = {
            "age": age,
            "income": income,
            "state": state
        }

        result = check_eligibility(scheme, user_data)
        st.success(result)

    if st.button("🔄 Start Over"):
        st.session_state.clear()
        st.rerun()

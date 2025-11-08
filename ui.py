import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from app.qa import ask_question
from app.summarize import summarize_scheme
from app.eligibility import check_eligibility

st.set_page_config(page_title="🏛️ SchemeBot", layout="centered")
st.title("🏛️ SchemeBot (Offline HuggingFace Version)")

# Available schemes (replace with dynamic list if needed)
AVAILABLE_SCHEMES = ["Housing Scheme","Health Scheme","Scholarship Scheme","Farming Scheme","Women Scheme"]

# Initialize session state
if "stage" not in st.session_state:
    st.session_state["stage"] = "choose_scheme"
if "scheme" not in st.session_state:
    st.session_state["scheme"] = None

# Stage 1: Choose a scheme
if st.session_state["stage"] == "choose_scheme":
    st.subheader("Available Schemes:")
    for s in AVAILABLE_SCHEMES:
        st.markdown(f"- {s}")

    scheme = st.text_input("👉 Enter a scheme name:")
    if st.button("Submit Scheme"):
        if scheme:
            st.session_state["scheme"] = scheme.strip()
            st.session_state["stage"] = "show_summary"
            st.rerun()

# Stage 2: Show summary
elif st.session_state["stage"] == "show_summary":
    scheme = st.session_state["scheme"]
    st.subheader(f"📄 Summary for {scheme}")
    st.write(summarize_scheme(scheme))

    if st.button("Next ➡️ Ask Questions"):
        st.session_state["stage"] = "ask_questions"
        st.rerun()

# Stage 3: QA
elif st.session_state["stage"] == "ask_questions":
    scheme = st.session_state["scheme"]
    st.subheader(f"💬 Ask a Question about {scheme}")
    query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if query:
            st.write(ask_question(scheme, query))

    if st.button("Next ➡️ Check Eligibility"):
        st.session_state["stage"] = "check_eligibility"
        st.rerun()

# Stage 4: Eligibility
elif st.session_state["stage"] == "check_eligibility":
    scheme = st.session_state["scheme"]
    st.subheader(f"✅ Check Eligibility for {scheme}")

    age = st.number_input("Enter your age:", 0, 100)
    income = st.number_input("Enter your annual income:")
    occupation = st.text_input("Enter your occupation:")
    state = st.text_input("Enter your state:")

    if st.button("Check Eligibility"):
        user_data = {"age": age, "income": income, "occupation": occupation, "state": state}
        st.write(check_eligibility(scheme, user_data))

    if st.button("🔄 Start Over"):
        st.session_state.clear()
        st.rerun()
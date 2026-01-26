import os
import streamlit as st

from backend import fetch_company_data, read_pdf_text, run_full_analysis_with_eval


st.set_page_config(page_title="Omaha Forensic Engine", layout="wide")

st.title("The Omaha Forensic Engine")
st.caption("Forensic, Buffett & Munger-style memo generation.")

st.sidebar.header("Input")
ticker = (st.sidebar.text_input("Enter Ticker Symbol (e.g., NVDA)") or "").strip()
gold_path = os.getenv(
    "GOLD_STANDARD_PATH",
    "/Users/manmohansharma/Downloads/Goog Stock Report (1).pdf",
)

run_button = st.button("Run Omaha Chain", type="primary", disabled=not ticker)

if run_button:
    with st.status("Fetching EDGAR filings...", expanded=True) as status:
        try:
            filings_text = fetch_company_data(ticker)
        except Exception as exc:
            status.update(label="EDGAR fetch failed.", state="error")
            message = str(exc)
            if "rate limit" in message.lower() or "429" in message:
                st.error(
                    "SEC rate limit exceeded. Wait at least 10 minutes before retrying, "
                    "and consider lowering EDGAR_RATE_LIMIT_PER_SEC in secrets."
                )
            else:
                st.error(f"Error: {exc}")
        else:
            status.update(label="Running sequential chain...", state="running")
            output_placeholder = st.empty()
            streamed_text = ""

            try:
                gold_text = read_pdf_text(gold_path)
                for chunk in run_full_analysis_with_eval(filings_text, gold_text):
                    streamed_text += chunk
                    output_placeholder.markdown(streamed_text)
            except Exception as exc:
                status.update(label="Chain failed.", state="error")
                st.error(f"Error: {exc}")
            else:
                status.update(label="Complete.", state="complete")

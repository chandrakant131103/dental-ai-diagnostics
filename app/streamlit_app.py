"""
Streamlit UI for the Dental Diagnostic AI platform.

Run locally:
    streamlit run app/streamlit_app.py

In Colab, tunnel with pyngrok:
    from pyngrok import ngrok
    !streamlit run app/streamlit_app.py &>/content/logs.txt &
    print(ngrok.connect(8501))

Set API_URL below to wherever api/main.py is running (localhost if
co-located, or your deployed FastAPI URL e.g. HF Spaces / Render).
"""
import io
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("DENTAL_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Dental Diagnostic AI", layout="wide")
st.title("🦷 Dental Diagnostic AI")
st.caption("Upload a panoramic X-ray to get automated tooth-by-tooth findings, severity grading, and a downloadable report.")

with st.sidebar:
    st.header("Settings")
    patient_name = st.text_input("Patient name (optional)", value="")
    st.markdown("---")
    st.markdown(
        "**Pipeline**\n"
        "1. YOLO tooth/finding localization\n"
        "2. U-Net pathology segmentation\n"
        "3. Feature-based severity grading\n"
    )
    st.markdown(f"API: `{API_URL}`")

uploaded = st.file_uploader("Upload panoramic X-ray", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    col1, col2 = st.columns([1, 1])
    image = Image.open(uploaded)
    with col1:
        st.subheader("Uploaded X-ray")
        st.image(image, use_column_width=True)

    if st.button("Run diagnosis", type="primary"):
        with st.spinner("Running detection → segmentation → severity grading..."):
            uploaded.seek(0)
            files = {"file": (uploaded.name, uploaded.read(), uploaded.type)}
            try:
                resp = requests.post(f"{API_URL}/predict", files=files, timeout=120)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        findings = result["findings"]
        with col2:
            st.subheader(f"Findings ({result['num_teeth_detected']} detected)")
            if findings:
                df = pd.DataFrame(findings)
                df = df[["finding_class", "detector_confidence", "lesion_area_ratio", "severity"]]
                df.columns = ["Finding", "Confidence", "Lesion Area Ratio", "Severity"]

                def highlight_severity(row):
                    color = {"mild": "#c8e6c9", "moderate": "#ffe0b2", "severe": "#ffcdd2"}.get(
                        row["Severity"], "white"
                    )
                    return [f"background-color: {color}"] * len(row)

                st.dataframe(df.style.apply(highlight_severity, axis=1), use_container_width=True)

                severe_count = sum(1 for f in findings if f["severity"] == "severe")
                if severe_count:
                    st.warning(f"⚠️ {severe_count} finding(s) graded SEVERE - recommend clinician review.")
            else:
                st.success("No findings detected above the confidence threshold.")

        st.markdown("---")
        st.subheader("Download full PDF report")
        uploaded.seek(0)
        files = {"file": (uploaded.name, uploaded.read(), uploaded.type)}
        pdf_resp = requests.post(
            f"{API_URL}/report", files=files, params={"patient_name": patient_name or "N/A"}, timeout=120
        )
        if pdf_resp.status_code == 200:
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_resp.content,
                file_name="dental_diagnostic_report.pdf",
                mime="application/pdf",
            )
        else:
            st.error("Report generation failed.")
else:
    st.info("Upload a panoramic X-ray image to begin.")

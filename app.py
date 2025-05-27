import streamlit as st
import requests
import base64

# Streamlit app
st.title("Weekly Ad Page Comparison Tool")

# File upload
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Upload First Ad Page (PDF/Image)", type=["pdf", "png", "jpg"])
with col2:
    file2 = st.file_uploader("Upload Second Ad Page (PDF/Image)", type=["pdf", "png", "jpg"])

# Process files
if file1 and file2:
    with st.spinner("Processing files..."):
        # Prepare files for backend
        files = {
            "file1": (file1.name, file1, file1.type),
            "file2": (file2.name, file2, file2.type)
        }
        # Send to backend (update URL with your Heroku app URL)
        response = requests.post("http://127.0.0.1:8000/upload", files=files)
        
        if response.status_code == 200:
            results = response.json()
            st.success("Comparison complete!")
            
            # Display results
            st.subheader("Comparison Results")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**File 1 Text**")
                st.write(results["text1"])
            with col2:
                st.write("**File 2 Text**")
                st.write(results["text2"])
            
            st.subheader("Differences")
            st.write(results["differences"])
            
            # Download report
            report_data = results["report"]
            st.download_button(
                label="Download Report (CSV)",
                data=report_data,
                file_name="comparison_report.csv",
                mime="text/csv"
            )
        else:
            st.error("Error processing files. Please try again.")
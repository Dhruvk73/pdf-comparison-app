import streamlit as st
import pandas as pd
import io
import json
import logging
import re
from dotenv import load_dotenv
import os
import time


# --- Logging Setup ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(levelname)s - PID:%(process)d - [%(name)s - %(funcName)s:%(lineno)d] - %(message)s'
)
# ---------------------

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logging.info(f"SUCCESS: frontend.py loaded .env file from: {dotenv_path}")
else:
    logging.warning(f".env file not found at {dotenv_path}. Relying on system environment variables.")

# Import backend processing function
try:
    from backend_processor import process_files_for_comparison, POPPLER_BIN_PATH
    BACKEND_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import backend_processor: {e}", exc_info=True)
    BACKEND_AVAILABLE = False
    POPPLER_BIN_PATH = os.getenv('POPPLER_PATH_OVERRIDE', None) 
    def process_files_for_comparison(file1_bytes, file1_name, file2_bytes, file2_name):
        st.error("Backend processor module is not available. Please check the import.")
        logging.error("process_files_for_comparison called but backend is not available.")
        return {
            "error": "Backend processing module not found.", "message": "Error: Backend not available.",
            "product_items_file1_count": 0, "product_items_file2_count": 0,
            "product_comparison_details": [], "report_csv_data": "Error,Backend not available\n",
            "all_product_details_file1": [], "all_product_details_file2": []
        }

st.set_page_config(layout="wide", page_title="PDF Comparison App")

# --- Styling (Removed image-related CSS) ---
st.markdown("""
    <style>
        body, .stApp {
            background-color: #ffffff; font-family: Inter, "Noto Sans", sans-serif; color: #141414;
        }
        .header-logo h2, .header-nav-links a, .main-title, .form-label { color: #141414; }
        .subtitle { color: #6b7280; }
        .header-button, .stButton button[kind="primary"] { background-color: #141414; color: #f9fafb; }
        button[data-testid="stDownloadButton-button"][key="export_button"] {
            background-color: #f0f2f6 !important; color: #141414 !important;
            border: 1px solid #d0d0d0 !important;
        }
        button[data-testid="stDownloadButton-button"][key="export_button"]:hover {
            background-color: #e0e2e6 !important;
        }
        .stDataFrame table { width: 100%; border-collapse: collapse; }
        .stDataFrame th, .stDataFrame td { border: 1px solid #ededed; padding: 8px; text-align: left; color: #141414; }
        .stDataFrame th { background-color: #f5f5f5; }
        .header { display: flex; align-items: center; justify-content: space-between; white-space: nowrap; border-bottom: 1px solid #ededed; padding: 0.75rem 2.5rem; background-color: #ffffff;}
        .header-logo { display: flex; align-items: center; gap: 1rem;}
        .header-logo svg { width: 1rem; height: 1rem; fill: currentColor;}
        .header-nav { display: flex; flex: 1; justify-content: flex-end; gap: 2rem;}
        .header-nav-links { display: flex; align-items: center; gap: 2.25rem;}
        .main-title { letter-spacing: -0.01em; font-size: 2rem; font-weight: bold; line-height: 1.25;}
        .form-label { font-size: 1rem; font-weight: 500; line-height: normal; padding-bottom: 0.5rem;}
        .centered-button-container { display: flex; justify-content: center; margin-top: 1rem; margin-bottom: 1rem;}
        .stButton>button { min-width: 200px; border-radius: 0.5rem; height: 2.5rem; font-weight: bold; font-size: 0.875rem;}
    </style>
    <div class="header">
        <div class="header-logo">
            <svg viewBox="0 0 48 48"><path d="M24 45.8096C19.6865 45.8096 15.4698 44.5305 11.8832 42.134C8.29667 39.7376 5.50128 36.3314 3.85056 32.3462C2.19985 28.361 1.76794 23.9758 2.60947 19.7452C3.451 15.5145 5.52816 11.6284 8.57829 8.5783C11.6284 5.52817 15.5145 3.45101 19.7452 2.60948C23.9758 1.76795 28.361 2.19986 32.3462 3.85057C36.3314 5.50129 39.7376 8.29668 42.134 11.8833C44.5305 15.4698 45.8096 19.6865 45.8096 24L24 24L24 45.8096Z"></path></svg>
            <h2>PDF Comparison App</h2>
        </div>
        <div class="header-nav">
            <div class="header-nav-links"><a href="#"></a> <a href="#"></a> <a href="#"></a> <a href="#"></a></div>
            <button class="header-button">Get Started</button>
        </div>
    </div>
""", unsafe_allow_html=True)

_, main_content_column, _ = st.columns([1, 2, 1]) 

with main_content_column:
    st.markdown("<div style='text-align: left; margin-top: 1.25rem; margin-bottom: 1.25rem;'>", unsafe_allow_html=True)
    st.markdown("<p class='main-title'>Compare PDFs</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload two PDF files to compare and identify differences.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='max-width: 480px;'>", unsafe_allow_html=True)
    st.markdown("<p class='form-label'>PDF 1</p>", unsafe_allow_html=True)
    uploaded_file1 = st.file_uploader("Drag and drop PDF 1 or click to browse", type="pdf", key="pdf1_uploader", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='max-width: 480px; margin-top: 0.75rem; margin-bottom: 0.75rem;'>", unsafe_allow_html=True)
    st.markdown("<p class='form-label'>PDF 2</p>", unsafe_allow_html=True)
    uploaded_file2 = st.file_uploader("Drag and drop PDF 2 or click to browse", type="pdf", key="pdf2_uploader", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='centered-button-container'>", unsafe_allow_html=True)
    if st.button("Compare", key="compare_button_main", type="primary"):
        if not BACKEND_AVAILABLE:
            st.error("Backend processing module is not available. Please check the application setup.")
        elif uploaded_file1 is not None and uploaded_file2 is not None:
            progress_bar = st.progress(0) 
            status_text = st.empty()
            
            # Clear previous results from session state before starting a new comparison
            if 'comparison_results' in st.session_state: del st.session_state.comparison_results

                      
            if 'table_data_for_display' in st.session_state: del st.session_state.table_data_for_display
            if 'csv_export_content' in st.session_state: del st.session_state.csv_export_content

            try:
                st.session_state.file1_bytes = uploaded_file1.getvalue()
                st.session_state.file2_bytes = uploaded_file2.getvalue()
                st.session_state.file1_name = uploaded_file1.name
                st.session_state.file2_name = uploaded_file2.name

                progress_bar.progress(5)
                status_text.text("✅ Files loaded.")
                time.sleep(0.2)

                with st.spinner("⏳ Processing files and generating comparison... This may take several minutes."):
                    progress_bar.progress(15)
                    status_text.text("Analyzing PDFs with backend...")
                    results = process_files_for_comparison(
                        st.session_state.file1_bytes, st.session_state.file1_name,
                        st.session_state.file2_bytes, st.session_state.file2_name
                    )
                    st.session_state.comparison_results = results
                    
                    # Initialize lists for table display and CSV export
                    table_data_for_display = []
                    csv_export_data_list = []

                    if results and not results.get("error"):
                        progress_bar.progress(65)
                        status_text.text("⚙️ Backend analysis complete. Preparing display...")
                        time.sleep(0.5)

                        product_comparison_details = results.get("product_comparison_details", [])
                        issues_list_for_table = [
                            item for item in product_comparison_details 
                            if item.get("Comparison_Type") != "Product Match - Attributes OK"
                        ]
                        
                        if issues_list_for_table:
                            total_issues = len(issues_list_for_table)
                            for i, issue in enumerate(issues_list_for_table):
                                current_progress_issues = 65 + int((i / total_issues) * 30)
                                progress_bar.progress(current_progress_issues)
                                status_text.text(f"📝 Preparing issue {i+1}/{total_issues}...")

                                product_name = "Unknown Product"  # Default value
                                p1_name_from_issue = issue.get('P1_Name') # Make sure this line exists and is correct
                                p2_name_from_issue = issue.get('P2_Name') # Make sure this line exists and is correct

                                if p1_name_from_issue and isinstance(p1_name_from_issue, str) and p1_name_from_issue.strip():
                                        product_name = p1_name_from_issue.strip()
                                elif p2_name_from_issue and isinstance(p2_name_from_issue, str) and p2_name_from_issue.strip():
                                        product_name = p2_name_from_issue.strip()

                                file1_issue_val, file2_issue_val = "N/A", "N/A" # Default values
                                diff_text = issue.get('Differences', '')

                                    # ... (inside the loop after product_name is determined) ...

                                file1_issue_val, file2_issue_val = "N/A", "N/A" # Default values
                                diff_text = issue.get('Differences', '')

                                if diff_text:
                                        # Dynamically get escaped filenames from session state for regex
                                        f1_name_escaped = re.escape(st.session_state.file1_name)
                                        f2_name_escaped = re.escape(st.session_state.file2_name)

                                        # Iterate over each part of the difference string (e.g., "Price diff; Metadata diff")
                                        for d_part in diff_text.split('; '):
                                            d_part = d_part.strip() # Remove leading/trailing whitespace

                                            # Try to match Sale Price differences
                                            # Pattern: Sale Price: FILENAME=$VALUE vs FILENAME=$VALUE
                                            # Captures only the value part after $
                                            sp_match = re.search(rf"Sale Price: {f1_name_escaped}=\$(.*?) vs {f2_name_escaped}=\$(.*)", d_part)
                                            if sp_match:
                                                val1, val2 = sp_match.groups()
                                                logging.debug(f"SP Match for {product_name} ({d_part}): val1='{val1}', val2='{val2}'") # ADD THIS
                                                file1_issue_val = f"${val1.strip()}" # Add $ back, strip potential spaces
                                                file2_issue_val = f"${val2.strip()}"
                                                break # Found the first difference type, display it

                                            # Try to match Regular Price differences
                                            # Pattern: Regular Price: FILENAME=$VALUE vs FILENAME=$VALUE
                                            rp_match = re.search(rf"Regular Price: {f1_name_escaped}=\$(.*?) vs {f2_name_escaped}=\$(.*)", d_part)
                                            if rp_match:
                                                val1, val2 = rp_match.groups()
                                                file1_issue_val = f"${val1.strip()}"
                                                file2_issue_val = f"${val2.strip()}"
                                                break

                                            # Try to match Metadata differences
                                            # Pattern: Metadata: FILENAME='VALUE' vs FILENAME='VALUE'
                                            md_match = re.search(rf"Metadata: {f1_name_escaped}='(.*?)' vs {f2_name_escaped}='(.*?)'", d_part)
                                            if md_match:
                                                val1, val2 = md_match.groups()
                                                # Metadata can be long, decide how to display.
                                                # Option 1: Show full values (might clutter table)
                                                file1_issue_val = f"'{val1}'"
                                                file2_issue_val = f"'{val2}'"
                                                # Option 2: Indicate a difference (more concise for table)
                                                # file1_issue_val = "Differs"
                                                # file2_issue_val = "(see CSV/details)"
                                                break
                                        
                                        # If diff_text was present but no regex matched (e.g., new difference type not handled by regex)
                                        # file1_issue_val and file2_issue_val will remain "N/A" unless a generic fallback is added here.
                                        # For example, if no specific regex above matched:
                                        if file1_issue_val == "N/A" and file2_issue_val == "N/A":
                                            file1_issue_val = "Mismatch detected"
                                            file2_issue_val = "(check details)"


                                elif issue.get('Comparison_Type') == "Unmatched Product in File 1":
                                        file1_issue_val, file2_issue_val = "Product Present", "Product Missing"
                                elif issue.get('Comparison_Type') == "Unmatched Product in File 2 (Extra)":
                                        file1_issue_val, file2_issue_val = "Product Missing", "Product Present"

                                table_data_for_display.append({
                                        "Product Name": product_name,
                                        f"{st.session_state.file1_name} Issue": file1_issue_val,
                                        f"{st.session_state.file2_name} Issue": file2_issue_val
                                })
# ...
                        else:
                            status_text.text("✅ All products matched perfectly or no discrepancies found.")
                            time.sleep(1)

                    st.session_state.table_data_for_display = table_data_for_display
                    
                    # Prepare CSV content from the simplified data
                    if csv_export_data_list:
                        csv_df = pd.DataFrame(csv_export_data_list)
                        st.session_state.csv_export_content = csv_df.to_csv(index=False).encode('utf-8')
                    else:
                        st.session_state.csv_export_content = "No issues to export.".encode('utf-8')
                
                progress_bar.progress(100)
                if results.get("error"):
                    status_text.error(f"Backend Error: {results.get('error')}")
                else:
                    status_text.success("🎉 Comparison Complete! Results are ready.")
                time.sleep(1)
                status_text.empty()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.error("Critical error in Streamlit frontend during comparison:", exc_info=True)
                if 'progress_bar' in locals(): progress_bar.progress(0)
                if 'status_text' in locals(): status_text.error("Comparison failed due to an application error.")
        else:
            st.warning("Please upload both PDF files to compare.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Display Results Section
    # In your frontend.py

# ... (code before the Display Results Section) ...

# Display Results Section
if 'comparison_results' in st.session_state:
    results = st.session_state.comparison_results # 'results' is defined here for this entire block

    if results.get("error"):
        # This 'pass' was in your original code. If there's a global error from the backend,
        # it's usually displayed when the 'Compare' button is pressed via status_text.error().
        # You might choose to explicitly display results.get("error") here too if needed.
        pass
    else:
        # No global error from the comparison process, proceed to display all results.

        # --- Existing Comparison Issues Display (as in your original code) ---
        st.markdown("<h2 class='main-title' style='font-size: 1.375rem; margin-top:1.25rem; margin-bottom:0.75rem;'>Comparison Issues</h2>", unsafe_allow_html=True)
        st.markdown(f"<p class='subtitle' style='margin-top:0.25rem; margin-bottom:0.75rem;'>Review differences found between '{st.session_state.get('file1_name', 'File 1')}' and '{st.session_state.get('file2_name', 'File 2')}'.</p>", unsafe_allow_html=True)

        # Download button for the CSV content
        if 'csv_export_content' in st.session_state and st.session_state.csv_export_content:
            st.markdown("<div class='centered-button-container'>", unsafe_allow_html=True)
            try:
                st.download_button(
                    label="Export Issues to CSV",
                    data=st.session_state.csv_export_content,
                    file_name=f"comparison_summary_{st.session_state.get('file1_name', 'f1').replace('.pdf','').replace(' ','_')}_vs_{st.session_state.get('file2_name', 'f2').replace('.pdf','').replace(' ','_')}.csv",
                    mime="text/csv", key="export_button"
                )
            except Exception as e_dl:
                st.error(f"Could not prepare download: {e_dl}")
            st.markdown("</div><br>", unsafe_allow_html=True)

        # Display table from pre-computed data
        table_data_to_show = st.session_state.get('table_data_for_display', [])
        if table_data_to_show:
            df_display = pd.DataFrame(table_data_to_show)
            column_order_final = [
                "Product Name",
                f"{st.session_state.file1_name} Issue",
                f"{st.session_state.file2_name} Issue"
            ]
            df_display_columns = [col for col in column_order_final if col in df_display.columns]

            if df_display_columns:
                st.dataframe(df_display[df_display_columns], use_container_width=True, hide_index=True)
            else:
                st.info("No data columns to display for issues.")
        elif results.get("product_comparison_details"): # This means comparison ran, found details, but 'table_data_for_display' was empty
            st.success("🎉 No significant issues found requiring table display, or all products matched perfectly!")
        else:
            st.info("No comparison details were processed or available for the issues table.")
        # --- End of Existing Comparison Issues Display ---

        # --- NEW: Raw Vision Model Output Display ---
        # This new section is correctly placed inside the 'else' block,
        # so 'results' is already defined and checked for global errors.
        st.markdown("---") # Add a visual separator
        st.markdown("<h2 class='main-title' style='font-size: 1.375rem; margin-top:1.25rem; margin-bottom:0.75rem;'>Raw Vision Model Output Details</h2>", unsafe_allow_html=True)

        # Check for and display raw data for file 1
        # The 'results' variable is accessible here.
        if 'all_product_details_file1' in results and results['all_product_details_file1']:
            with st.expander(f"Extracted Product Data from: {st.session_state.get('file1_name', 'File 1')}"):
                st.json(results['all_product_details_file1'])
        else:
            st.info(f"No raw product data was extracted or available for {st.session_state.get('file1_name', 'File 1')}.")

        # Check for and display raw data for file 2
        if 'all_product_details_file2' in results and results['all_product_details_file2']:
            with st.expander(f"Extracted Product Data from: {st.session_state.get('file2_name', 'File 2')}"):
                st.json(results['all_product_details_file2'])
        else:
            st.info(f"No raw product data was extracted or available for {st.session_state.get('file2_name', 'File 2')}.")
        # --- End of NEW Raw Vision Model Output Display ---

else: # This 'else' corresponds to "if 'comparison_results' NOT in st.session_state"
    st.markdown("<div style='text-align:center; margin-top:1rem; margin-bottom:1rem;'>", unsafe_allow_html=True)
    st.caption("Upload two PDF files and click 'Compare' to see the results.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6b7280; font-size: 0.875rem;'>© Your Company Name.</p>", unsafe_allow_html=True)
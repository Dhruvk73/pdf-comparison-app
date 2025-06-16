import streamlit as st
import pandas as pd
import io
import json
import logging
import re
import base64
from dotenv import load_dotenv
import os
import time
import tempfile
from pathlib import Path

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

# --- MODIFIED SECTION: Backend Import and Main Processing Function ---
# This flag will be set within the process_files_for_comparison function
BACKEND_AVAILABLE = False

# This function is now defined in the frontend to bridge the gap with the backend.
def process_files_for_comparison(file1_bytes, file1_name, file2_bytes, file2_name):
    global BACKEND_AVAILABLE

    try:
        # Attempt to import the specific backend function needed
        from visual_layout_backend import simple_catalog_comparison
        BACKEND_AVAILABLE = True
        logging.info("Successfully imported 'simple_catalog_comparison' from backend.")
    except ImportError as e:
        logging.error(f"Failed to import 'simple_catalog_comparison' from visual_layout_backend: {e}", exc_info=True)
        st.error("Critical backend function 'simple_catalog_comparison' is not available. Please check the backend script ('visual_layout_backend.py') and dependencies.")
        BACKEND_AVAILABLE = False
        # Return the error structure frontend expects
        return {
            "error": "Backend processing function 'simple_catalog_comparison' not found in 'visual_layout_backend.py'.",
            "message": "Error: Backend not available or function missing.",
            "product_items_file1_count": 0, "product_items_file2_count": 0,
            "product_comparison_details": [], "report_csv_data": "Error,Backend not available\n",
            "all_product_details_file1": [], "all_product_details_file2": [],
            "highlighted_pages_file1": [], "highlighted_pages_file2": []
        }

    # Create a temporary directory for processing files
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf1_path = os.path.join(tmpdir, file1_name)
        pdf2_path = os.path.join(tmpdir, file2_name)

        # Save uploaded PDF bytes to temporary files
        with open(pdf1_path, "wb") as f:
            f.write(file1_bytes)
        with open(pdf2_path, "wb") as f:
            f.write(file2_bytes)
        logging.info(f"Temporary PDF files created: {pdf1_path}, {pdf2_path}")

        # Your actual base path for templates
        app_root_dir = Path(__file__).resolve().parent
        template_base_dir = app_root_dir / "Templates" # Relative path

        # Replace these with the ACTUAL NAMES of your template files
        actual_template1_filename = "template1.jpg"
        actual_template2_filename = "template2.jpg"
        actual_template3_filename = "template3.jpg"

        template1_path = template_base_dir / actual_template1_filename
        template2_path = template_base_dir / actual_template2_filename
        template3_path = template_base_dir / actual_template3_filename

        # Check if the template files exist at the specified paths
        missing_templates = []
        if not os.path.exists(template1_path):
            missing_templates.append(template1_path)
        if not os.path.exists(template2_path):
            missing_templates.append(template2_path)
        if not os.path.exists(template3_path):
            missing_templates.append(template3_path)

        if missing_templates:
            error_message = "**Error: The following template files were not found:**\n"
            for mt in missing_templates:
                error_message += f"- `{mt}`\n"
            error_message += "\nPlease ensure the template files exist at the specified paths and filenames are correct in `new_frontend.py`."
            st.error(error_message)
            logging.error(f"Missing template files: {', '.join([str(p) for p in missing_templates])}")

        logging.info(f"Using template files: {template1_path}, {template2_path}, {template3_path}")

        try:
            logging.info("Calling backend 'simple_catalog_comparison' function...")
            pipeline_output_dir = os.path.join(tmpdir, "backend_pipeline_output")
            os.makedirs(pipeline_output_dir, exist_ok=True)

            pipeline_results = simple_catalog_comparison(
                pdf1_path=pdf1_path,
                pdf2_path=pdf2_path,
                template1_path=template1_path,
                template2_path=template2_path,
                template3_path=template3_path
            )
            logging.info("Backend 'simple_catalog_comparison' call finished.")

            frontend_results_to_display = {
                "error": None, "message": "Comparison complete.",
                "product_items_file1_count": 0, "product_items_file2_count": 0,
                "product_comparison_details": [], "report_csv_data": "No issues to export.\n",
                "all_product_details_file1": [], "all_product_details_file2": [],
                "highlighted_pages_file1": [], "highlighted_pages_file2": []
            }

            # --- ADD THIS NEW BLOCK HERE ---
            # This block populates the details for the frontend summary
            if "step3_vlm_comparison" in pipeline_results and pipeline_results["step3_vlm_comparison"].get("page_1"):
                # Assuming single-page comparison for now.
                vlm_page_results = pipeline_results["step3_vlm_comparison"]["page_1"].get("results", {})
                frontend_results_to_display["product_comparison_details"] = vlm_page_results.get("comparison_rows", [])
            # --- END OF NEW BLOCK ---

            if pipeline_results.get("errors"):
                frontend_results_to_display["error"] = "; ".join(map(str,pipeline_results["errors"]))
                frontend_results_to_display["message"] = "Comparison completed with errors."
                logging.error(f"Backend pipeline reported errors: {frontend_results_to_display['error']}")

            if "step1_pdf_processing" in pipeline_results and isinstance(pipeline_results["step1_pdf_processing"], dict):
                pdf_proc_res = pipeline_results["step1_pdf_processing"]
                frontend_results_to_display["product_items_file1_count"] = pdf_proc_res.get("total_products_catalog1", 0)
                frontend_results_to_display["product_items_file2_count"] = pdf_proc_res.get("total_products_catalog2", 0)

                def get_visualization_images_as_base64(catalog_main_path_str, num_pages, catalog_id_prefix):
                    images_b64 = []
                    logging.info(f"[{catalog_id_prefix}] Attempting to load visualizations. Path: '{catalog_main_path_str}', Expected Pages: {num_pages}")
                    if not catalog_main_path_str or not os.path.exists(catalog_main_path_str):
                        logging.warning(f"[{catalog_id_prefix}] Catalog base path not found or invalid: {catalog_main_path_str}")
                        return [None] * num_pages

                    catalog_main_path = Path(catalog_main_path_str)
                    for page_num in range(1, num_pages + 1):
                        viz_filename = f"{catalog_id_prefix}_p{page_num}_ranking_visualization.jpg"
                        viz_filepath = catalog_main_path / f"page_{page_num}" / viz_filename
                        logging.info(f"[{catalog_id_prefix}] Checking for file for Page {page_num}: {viz_filepath}")
                        if viz_filepath.exists():
                            try:
                                with open(viz_filepath, "rb") as img_f:
                                    encoded_img = base64.b64encode(img_f.read()).decode('utf-8')
                                    images_b64.append(encoded_img)
                                    logging.info(f"[{catalog_id_prefix}] Successfully loaded and encoded Page {page_num} from: {viz_filepath}")
                            except Exception as img_err:
                                logging.error(f"[{catalog_id_prefix}] Error reading or encoding visualization {viz_filepath} for Page {page_num}: {img_err}")
                                images_b64.append(None)
                        else:
                            logging.warning(f"[{catalog_id_prefix}] Visualization file NOT FOUND for Page {page_num}: {viz_filepath}")
                            images_b64.append(None)
                    logging.info(f"[{catalog_id_prefix}] Finished loading. Total images/placeholders returned: {len(images_b64)}")
                    return images_b64

                frontend_results_to_display["highlighted_pages_file1"] = get_visualization_images_as_base64(
                    pdf_proc_res.get("catalog1_path"), pdf_proc_res.get("catalog1_pages", 0), "c1"
                )
                frontend_results_to_display["highlighted_pages_file2"] = get_visualization_images_as_base64(
                    pdf_proc_res.get("catalog2_path"), pdf_proc_res.get("catalog2_pages", 0), "c2"
                )

            logging.info("Frontend results processing complete.")
            return frontend_results_to_display

        except Exception as pipeline_err:
            logging.error(f"Error during backend pipeline execution or result processing: {pipeline_err}", exc_info=True)
            st.error(f"An error occurred while processing with the backend: {pipeline_err}")
            return {
                "error": f"Pipeline execution error: {str(pipeline_err)}",
                "message": "Error during backend processing.",
                "product_items_file1_count": 0, "product_items_file2_count": 0,
                "product_comparison_details": [], "report_csv_data": f"Error,{str(pipeline_err)}\n",
                "all_product_details_file1": [], "all_product_details_file2": [],
                "highlighted_pages_file1": [], "highlighted_pages_file2": []
            }

st.set_page_config(layout="wide", page_title="PDF Comparison App")

# --- CORRECTED CSS SECTION ---
st.markdown("""
<style>
    /* General body and app styling */
    body, .stApp {
        background-color: #ffffff;
        font-family: Inter, "Noto Sans", sans-serif;
        color: #141414;
    }
    /* Main titles and form labels */
    .main-title { letter-spacing: -0.01em; font-size: 2rem; font-weight: bold; line-height: 1.25; color: #141414; }
    .subtitle { color: #6b7280; }
    .form-label { font-size: 1rem; font-weight: 500; line-height: normal; padding-bottom: 0.5rem; color: #141414; }

    /* Button styling */
    .stButton>button { min-width: 200px; border-radius: 0.5rem; height: 2.5rem; font-weight: bold; font-size: 0.875rem;}
    .centered-button-container { display: flex; justify-content: center; margin-top: 1rem; margin-bottom: 1rem;}
    .stButton button[kind="primary"] { background-color: #141414; color: #f9fafb; }
    
    /* Styles for the visual comparison display */
    .image-display-frame {
        width: 100%;
        box-sizing: border-box;
        border: 1px solid #e2e8f0; /* Softer border color */
        padding: 1rem;
        border-radius: 0.5rem; /* Consistent radius */
        background-color: #f8fafc; /* Lighter background */
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 1rem; /* Space below each frame */
    }
    
    .image-wrapper {
        width: 100%;
        height: auto; /* Allow height to be determined by content */
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .image-display-frame img {
        display: block;
        max-width: 100%; /* Image will scale with the container width */
        height: auto;    /* Height adjusts to maintain aspect ratio */
        object-fit: contain;
        border-radius: 4px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
    }
    
    .image-caption {
        font-size: 0.9em;
        color: #4a5568; /* Darker grey for better readability */
        margin-top: 1rem;
        text-align: center;
        font-weight: 500;
    }

    /* Page analysis header */
    .page-analysis-header {
        font-size: 1.25rem;
        font-weight: 600;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
            
    /* Override Streamlit success message text color */
    div[data-testid="stAlert"][kind="success"] p {
        color: #000000 !important; /* Black text color */
    }
    
            /* Target Streamlit's metric components for better visibility */
    div[data-testid="stMetric"] label {
        color: #6b7280 !important; /* Sets the label (e.g., "Price Issues") to gray */
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #141414 !important; /* Sets the value (e.g., "12") to black */
    }
</style>
""", unsafe_allow_html=True)


# --- Main content layout & File Uploaders ---
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

# --- Compare Button Logic ---
st.markdown("<div class='centered-button-container'>", unsafe_allow_html=True)
if st.button("Compare", key="compare_button_main", type="primary"):
    if uploaded_file1 is not None and uploaded_file2 is not None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Clear previous results
        for key in st.session_state.keys():
            if key.startswith('comparison_') or key.startswith('highlighted_') or key.startswith('file_'):
                del st.session_state[key]
        
        try:
            st.session_state.file1_bytes = uploaded_file1.getvalue()
            st.session_state.file2_bytes = uploaded_file2.getvalue()
            st.session_state.file1_name = uploaded_file1.name
            st.session_state.file2_name = uploaded_file2.name

            with st.spinner("⏳ Processing files and generating comparison... This may take several minutes."):
                status_text.text("Calling backend for analysis...")
                progress_bar.progress(15)
                
                results = process_files_for_comparison(
                    st.session_state.file1_bytes, st.session_state.file1_name,
                    st.session_state.file2_bytes, st.session_state.file2_name
                )
                st.session_state.comparison_results = results
                st.session_state.highlighted_pages_file1 = results.get("highlighted_pages_file1", [])
                st.session_state.highlighted_pages_file2 = results.get("highlighted_pages_file2", [])
                logging.info("Backend call finished. Results stored in session state.")

            progress_bar.progress(100)
            if results and results.get("error"):
                status_text.error(f"Comparison Error: {results.get('error')}")
            else:
                status_text.success("🎉 Comparison Complete! Results are ready below.")

        except Exception as e:
            st.error(f"An unexpected error occurred in the frontend: {str(e)}")
            logging.error("Critical error in Streamlit frontend during comparison initiation:", exc_info=True)
            if 'progress_bar' in locals(): progress_bar.progress(0)
            if 'status_text' in locals(): status_text.error("Comparison failed due to an application error.")
    else:
        st.warning("Please upload both PDF files to compare.")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# --- Display Results Section ---
# --- Display Results Section ---
if 'comparison_results' in st.session_state:
    results = st.session_state.comparison_results

    if results.get("error"):
        st.error(f"An error occurred during the comparison process: {results.get('error')}")
    else:
        # --- Summary Metrics ---
        st.markdown("<h2 class='main-title' style='font-size: 1.375rem; margin-top:1.25rem; margin-bottom:0.75rem;'>Error Summary</h2>", unsafe_allow_html=True)

        # ### THIS IS THE KEY FIX ###
        # It now reads the summary directly from the backend results instead of recalculating.
        summary = results.get("summary_metrics", {})
        total_mistakes = summary.get('incorrect_matches', 0)
        price_mistakes = summary.get('price_issues', 0)
        text_mistakes = summary.get('text_issues', 0)
        image_mistakes = summary.get('image_issues', 0)

        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        sum_col1.metric("Total Discrepancies", total_mistakes)
        sum_col2.metric("Price Issues", price_mistakes)
        sum_col3.metric("Text/Brand Issues", text_mistakes)
        sum_col4.metric("Image Mismatches", image_mistakes)

        # --- Visual Comparison Section (Full Pages) ---
        highlighted_pages_file1 = st.session_state.get('highlighted_pages_file1', [])
        highlighted_pages_file2 = st.session_state.get('highlighted_pages_file2', [])

        if (highlighted_pages_file1 and any(img for img in highlighted_pages_file1)) or \
           (highlighted_pages_file2 and any(img for img in highlighted_pages_file2)):

            st.markdown("<h2 class='main-title' style='font-size: 1.375rem; margin-top:2.5rem; margin-bottom:0.5rem;'>Visual Page Analysis</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='subtitle' style='margin-bottom:1.5rem;'>Products with detected errors are outlined in yellow with an indicator at the top-left.</p>", unsafe_allow_html=True)

            num_pages_to_display = max(len(highlighted_pages_file1), len(highlighted_pages_file2))

            for page_idx in range(num_pages_to_display):
                st.markdown("---")
                st.markdown(f"<p class='page-analysis-header'>Page {page_idx + 1} Analysis</p>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                # Display for PDF 1
                with col1:
                    if page_idx < len(highlighted_pages_file1) and highlighted_pages_file1[page_idx]:
                        st.markdown(f"""
                            <div class="image-display-frame">
                                <div class="image-wrapper">
                                    <img src="data:image/jpeg;base64,{highlighted_pages_file1[page_idx]}"
                                         alt="{st.session_state.get('file1_name', 'File 1')} Page {page_idx + 1} Visualization">
                                </div>
                                <div class="image-caption">{st.session_state.get('file1_name', 'File 1')}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(f"No visualization available for {st.session_state.get('file1_name', 'File 1')} - Page {page_idx + 1}.")

                # Display for PDF 2
                with col2:
                    if page_idx < len(highlighted_pages_file2) and highlighted_pages_file2[page_idx]:
                        st.markdown(f"""
                            <div class="image-display-frame">
                                <div class="image-wrapper">
                                    <img src="data:image/jpeg;base64,{highlighted_pages_file2[page_idx]}"
                                         alt="{st.session_state.get('file2_name', 'File 2')} Page {page_idx + 1} Visualization">
                                </div>
                                <div class="image-caption">{st.session_state.get('file2_name', 'File 2')}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(f"No visualization available for {st.session_state.get('file2_name', 'File 2')} - Page {page_idx + 1}.")
        else:
            st.info("No full-page visualizations were generated by the backend.")
            logging.info("No highlighted pages data to display.")
else:
    st.markdown("<div style='text-align:center; margin-top:1rem; margin-bottom:1rem;'>", unsafe_allow_html=True)
    st.caption("Upload two PDF files and click 'Compare' to see the results.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6b7280; font-size: 0.875rem;'>© PDF Comparison Tool.</p>", unsafe_allow_html=True)
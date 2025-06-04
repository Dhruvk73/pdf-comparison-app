# new_frontend.py

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
import tempfile # Added for temporary files
from pathlib import Path # Added for path manipulation

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

        # !!! IMPORTANT: Template file paths are now set to your specified directory.
        # Ensure the template filenames below match your actual files in that directory.
        
        # Your actual base path for templates
        script_dir = Path(__file__).resolve().parent 
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
            # Optionally, you might want to return an error structure here if templates are critical
            # For now, it will proceed, and the backend will likely handle the missing files (or error out).
            # However, the backend `catalog_comparison_pipeline` expects these paths to be valid.
            # To prevent an immediate crash if the backend doesn't handle it gracefully, you might stop here:
            # return {
            #     "error": f"Missing template files: {', '.join(missing_templates)}",
            #     "message": "Template files not found. Please check paths.",
            #     # ... (rest of the error structure) ...
            # }
        
        logging.info(f"Using template files: {template1_path}, {template2_path}, {template3_path}")
        st.success(f"""
        **Using actual template files for backend processing:**
        - `{template1_path}`
        - `{template2_path}`
        - `{template3_path}`
        Ensure these files exist and are the correct templates for your comparison.
        """)

        # API keys (ROBOFLOW_API_KEY, OPENAI_API_KEY) are expected to be in the environment (e.g., from .env)
        # as 'simple_catalog_comparison' will attempt to load them if not passed.

        try:
            logging.info("Calling backend 'simple_catalog_comparison' function...")
            # The 'simple_catalog_comparison' function from backend calls 'catalog_comparison_pipeline'.
            # 'catalog_comparison_pipeline' saves results in a subdirectory "catalog_comparison_results" by default.
            # We need to ensure output_directory in the backend is either configurable or we know where to find outputs.
            # For now, we assume the pipeline output structure as defined in visual_layout_backend.py
            
            # The `catalog_comparison_pipeline` (called by `simple_catalog_comparison`)
            # uses `output_directory` parameter with default "catalog_comparison_results".
            # Let's make the output go into our temp dir so it's cleaned up.
            pipeline_output_dir = os.path.join(tmpdir, "backend_pipeline_output")
            os.makedirs(pipeline_output_dir, exist_ok=True)

            # Modifying simple_catalog_comparison to accept output_directory might be needed in backend,
            # or adapting catalog_comparison_pipeline call within simple_catalog_comparison.
            # For now, assuming simple_catalog_comparison itself does not take output_directory.
            # The results dictionary it returns should contain paths to where data was stored.
            
            # Re-checking `simple_catalog_comparison`: it directly calls `catalog_comparison_pipeline`
            # but doesn't pass `output_directory` to it. So `catalog_comparison_pipeline` will use its
            # default "catalog_comparison_results" relative to where the script is run, or needs modification.
            # For a robust solution, `simple_catalog_comparison` should accept `output_directory`.
            # Given the constraint of changing frontend only, this is a limitation.
            # We will have to parse paths from the default output or what's returned.

            pipeline_results = simple_catalog_comparison(
                pdf1_path=pdf1_path,
                pdf2_path=pdf2_path,
                template1_path=template1_path,
                template2_path=template2_path,
                template3_path=template3_path
                # roboflow_project and roboflow_version have defaults in simple_catalog_comparison
            )
            logging.info("Backend 'simple_catalog_comparison' call finished.")
            # logging.debug(f"Raw pipeline results from backend: {json.dumps(pipeline_results, indent=2, default=str)}")

            # --- Attempt to map pipeline_results to the structure frontend expects ---
            frontend_results_to_display = {
                "error": None, "message": "Comparison complete.",
                "product_items_file1_count": 0, "product_items_file2_count": 0,
                "product_comparison_details": [], "report_csv_data": "No issues to export.\n",
                "all_product_details_file1": [], "all_product_details_file2": [],
                "highlighted_pages_file1": [], "highlighted_pages_file2": []
            }

            if pipeline_results.get("errors"):
                frontend_results_to_display["error"] = "; ".join(map(str,pipeline_results["errors"]))
                frontend_results_to_display["message"] = "Comparison completed with errors."
                logging.error(f"Backend pipeline reported errors: {frontend_results_to_display['error']}")


            # Extract product counts from step1_pdf_processing
            if "step1_pdf_processing" in pipeline_results and isinstance(pipeline_results["step1_pdf_processing"], dict):
                pdf_proc_res = pipeline_results["step1_pdf_processing"]
                frontend_results_to_display["product_items_file1_count"] = pdf_proc_res.get("total_products_catalog1", 0)
                frontend_results_to_display["product_items_file2_count"] = pdf_proc_res.get("total_products_catalog2", 0)

                # Extract highlighted pages (ranking visualizations)
                def get_visualization_images_as_base64(catalog_main_path_str, num_pages, catalog_id_prefix):
                    images_b64 = []
                    if not catalog_main_path_str or not os.path.exists(catalog_main_path_str):
                        logging.warning(f"Catalog base path not found or invalid: {catalog_main_path_str}")
                        return [None] * num_pages # Return list of Nones

                    catalog_main_path = Path(catalog_main_path_str)
                    for page_num in range(1, num_pages + 1):
                        # Backend saves visualization as: {page_prefix}_ranking_visualization.jpg
                        # page_prefix is like "c1_p{page_num}"
                        viz_filename = f"{catalog_id_prefix}_p{page_num}_ranking_visualization.jpg"
                        viz_filepath = catalog_main_path / f"page_{page_num}" / viz_filename
                        if viz_filepath.exists():
                            try:
                                with open(viz_filepath, "rb") as img_f:
                                    images_b64.append(base64.b64encode(img_f.read()).decode('utf-8'))
                            except Exception as img_err:
                                logging.error(f"Error reading visualization {viz_filepath}: {img_err}")
                                images_b64.append(None)
                        else:
                            logging.warning(f"Visualization file not found: {viz_filepath}")
                            images_b64.append(None)
                    return images_b64

                frontend_results_to_display["highlighted_pages_file1"] = get_visualization_images_as_base64(
                    pdf_proc_res.get("catalog1_path"), pdf_proc_res.get("catalog1_pages", 0), "c1"
                )
                frontend_results_to_display["highlighted_pages_file2"] = get_visualization_images_as_base64(
                    pdf_proc_res.get("catalog2_path"), pdf_proc_res.get("catalog2_pages", 0), "c2"
                )

            # Map VLM comparison results (product_comparison_details and report_csv_data)
            # This is complex as backend returns paths to Excel files per page.
            # For frontend's current expectation of a list of dicts & a single CSV string:
            all_comparison_issue_rows_for_csv = []
            if "step3_vlm_comparison" in pipeline_results and isinstance(pipeline_results["step3_vlm_comparison"], dict):
                vlm_results_by_page = pipeline_results["step3_vlm_comparison"]
                for page_id, page_data in vlm_results_by_page.items():
                    if isinstance(page_data, dict) and "results" in page_data and isinstance(page_data["results"], dict):
                        page_comparison_rows = page_data["results"].get("comparison_rows", [])
                        cat1_name_on_page = page_data["results"].get("catalog1_name", "File1") # e.g. Catalog1_Page1
                        cat2_name_on_page = page_data["results"].get("catalog2_name", "File2") # e.g. Catalog2_Page1

                        for row in page_comparison_rows:
                            # The backend's `create_practical_comparison_row` structures this.
                            # We need to map it to the simpler structure expected by the frontend's CSV generation.
                            issue_type = row.get("issue_type", "N/A")
                            if issue_type != "Match Confirmed": # Only process actual differences/issues for CSV
                                p1_details_from_row = row.get(f"{cat1_name_on_page}_details", "N/A")
                                p2_details_from_row = row.get(f"{cat2_name_on_page}_details", "N/A")
                                
                                # Attempt to get a "Product Name" - can be from p1 or p2 details string
                                product_name_guess = p1_details_from_row.split(" - ")[0] if p1_details_from_row != "Product Missing" else \
                                                     (p2_details_from_row.split(" - ")[0] if p2_details_from_row != "Product Missing" else "Unknown Product")

                                all_comparison_issue_rows_for_csv.append({
                                    "Product Name": product_name_guess,
                                    "Issue Type": issue_type,
                                    f"{file1_name} Details": p1_details_from_row, # Original PDF name for clarity in CSV
                                    f"{file2_name} Details": p2_details_from_row, # Original PDF name
                                    "Raw Differences": row.get("details", "N/A"),
                                    # Add more specific P1/P2 attributes if needed and if they can be parsed from row[*_details]
                                })
                
                # Also populate the product_comparison_details (which was used by commented-out table in frontend)
                # For now, let's use the same transformed data.
                frontend_results_to_display["product_comparison_details"] = all_comparison_issue_rows_for_csv


            if all_comparison_issue_rows_for_csv:
                try:
                    csv_df = pd.DataFrame(all_comparison_issue_rows_for_csv)
                    frontend_results_to_display["report_csv_data"] = csv_df.to_csv(index=False).encode('utf-8')
                except Exception as df_err:
                    logging.error(f"Error creating CSV from comparison rows: {df_err}")
                    frontend_results_to_display["report_csv_data"] = "Error generating CSV report.\n".encode('utf-8')
            else:
                 frontend_results_to_display["report_csv_data"] = "No significant discrepancies found to export.".encode('utf-8')


            logging.info("Frontend results processing complete.")
            return frontend_results_to_display

        except Exception as pipeline_err:
            logging.error(f"Error during backend pipeline execution or result processing: {pipeline_err}", exc_info=True)
            st.error(f"An error occurred while processing with the backend: {pipeline_err}")
            return { # Return error structure
                "error": f"Pipeline execution error: {str(pipeline_err)}",
                "message": "Error during backend processing.",
                "product_items_file1_count": 0, "product_items_file2_count": 0,
                "product_comparison_details": [], "report_csv_data": f"Error,{str(pipeline_err)}\n",
                "all_product_details_file1": [], "all_product_details_file2": [],
                "highlighted_pages_file1": [], "highlighted_pages_file2": []
            }
# --- END OF MODIFIED SECTION ---

st.set_page_config(layout="wide", page_title="PDF Comparison App")

# JavaScript and CSS for interactive zoom functionality
st.markdown("""
<script>
window.zoomImage = function(imageId, event) {
    event.preventDefault();
    event.stopPropagation();
    
    const container = document.querySelector('[data-image-id="' + imageId + '"]');
    const img = container.querySelector('img');
    const icon = container.querySelector('.zoom-icon');
    
    if (!container.classList.contains('zoomed')) {
        // Zoom in
        const rect = img.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        
        // Calculate click position relative to the image
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Calculate the transform origin as percentages
        const originX = (x / rect.width) * 100;
        const originY = (y / rect.height) * 100;
        
        img.style.transformOrigin = originX + '% ' + originY + '%';
        img.style.transform = 'scale(2)';
        container.classList.add('zoomed');
        
        // Change icon to zoom out
        icon.innerHTML = '🔍−'; // Using Unicode minus
        icon.title = 'Click to zoom out';
        
        // Store zoom state in Streamlit (optional, if you need to persist server-side)
        // window.parent.postMessage({
        //     type: 'streamlit:setComponentValue',
        //     value: {action: 'zoom_in', imageId: imageId, x: originX, y: originY}
        // }, '*');
        
    } else {
        // Zoom out
        img.style.transform = 'scale(1)';
        img.style.transformOrigin = 'center center';
        container.classList.remove('zoomed');
        
        // Change icon back to zoom in
        icon.innerHTML = '🔍+';
        icon.title = 'Click to zoom in';
        
        // Store zoom state in Streamlit (optional)
        // window.parent.postMessage({
        //     type: 'streamlit:setComponentValue',
        //     value: {action: 'zoom_out', imageId: imageId}
        // }, '*');
    }
};

// Add event listeners when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    function setupZoomListeners() {
        const containers = document.querySelectorAll('.image-display-frame[data-image-id]');
        containers.forEach(container => {
            const imageId = container.getAttribute('data-image-id');
            const icon = container.querySelector('.zoom-icon');
            
            if (icon && !icon.hasAttribute('data-listener-added')) {
                icon.addEventListener('click', function(e) {
                    window.zoomImage(imageId, e);
                });
                icon.setAttribute('data-listener-added', 'true');
            }
        });
    }
    setupZoomListeners(); // Initial setup

    // Re-run for Streamlit updates (e.g., after button click and content reload)
    // Using MutationObserver for better reliability with Streamlit's dynamic content
    const streamlitAppRoot = document.getElementById('root'); // Streamlit typically renders into a div with id 'root'
    if (streamlitAppRoot) {
        const observer = new MutationObserver(function(mutationsList, observer) {
            for(let mutation of mutationsList) {
                if (mutation.type === 'childList' || mutation.type === 'subtree') {
                    setupZoomListeners(); // Re-apply listeners if DOM changes
                    break; 
                }
            }
        });
        observer.observe(streamlitAppRoot, { childList: true, subtree: true });
    } else { // Fallback if root element ID changes or is not standard
         new MutationObserver(() => {
            constรหัสโปรแกรม = setTimeout(setupZoomListeners, 50); // Debounce
        }).observe(document.body, {childList: true, subtree: true});
    }
});
</script>

<style>
    /* General body and app styling */
    body, .stApp {
        background-color: #ffffff; 
        font-family: Inter, "Noto Sans", sans-serif; 
        color: #141414;
    }
    /* Header and titles */
    .header-logo h2, .header-nav-links a, .main-title, .form-label { color: #141414; }
    .subtitle { color: #6b7280; }
    .header-button, .stButton button[kind="primary"] { background-color: #141414; color: #f9fafb; }
    /* Download button specific styling */
    button[data-testid="stDownloadButton-button"][key="export_button"] {
        background-color: #f0f2f6 !important; 
        color: #141414 !important;
        border: 1px solid #d0d0d0 !important;
    }
    button[data-testid="stDownloadButton-button"][key="export_button"]:hover {
        background-color: #e0e2e6 !important;
    }
    /* DataFrame styling */
    .stDataFrame table { width: 100%; border-collapse: collapse; }
    .stDataFrame th, .stDataFrame td { border: 1px solid #ededed; padding: 8px; text-align: left; color: #141414; }
    .stDataFrame th { background-color: #f5f5f5; }
    /* Header layout */
    .header { display: flex; align-items: center; justify-content: space-between; white-space: nowrap; border-bottom: 1px solid #ededed; padding: 0.75rem 2.5rem; background-color: #ffffff;}
    .header-logo { display: flex; align-items: center; gap: 1rem;}
    .header-logo svg { width: 1rem; height: 1rem; fill: currentColor;}
    .header-nav { display: flex; flex: 1; justify-content: flex-end; gap: 2rem;}
    .header-nav-links { display: flex; align-items: center; gap: 2.25rem;}
    .main-title { letter-spacing: -0.01em; font-size: 2rem; font-weight: bold; line-height: 1.25;}
    .form-label { font-size: 1rem; font-weight: 500; line-height: normal; padding-bottom: 0.5rem;}
    .centered-button-container { display: flex; justify-content: center; margin-top: 1rem; margin-bottom: 1rem;}
    .stButton>button { min-width: 200px; border-radius: 0.5rem; height: 2.5rem; font-weight: bold; font-size: 0.875rem;}

    /* Custom styles for image comparison */
    .image-comparison-page-container {
        display: flex;
        justify-content: space-around;
        align-items: flex-start;
        gap: 20px;
        margin-top: 20px;
        flex-wrap: wrap;
    }
    .image-display-frame {
        flex: 1; 
        min-width: 45%; /* Ensure responsiveness */
        max-width: 48%; /* Prevent taking full width on very large screens if only one column */
        border: 1px solid #eee;
        padding: 10px;
        border-radius: 8px;
        background-color: #f9f9f9;
        display: flex;
        flex-direction: column;
        align-items: center;
        position: relative; /* For zoom icon positioning */
        overflow: hidden; /* Important for zoom effect */
    }
    .image-wrapper {
        width: 100%;
        height: auto; /* Adjust height dynamically */
        max-height: 600px; /* Optional: constrain max height */
        display: flex; 
        justify-content: center;
        align-items: center;
        position: relative;
        overflow: hidden; /* Crucial for transform-origin to work as expected */
        border-radius: 4px;
    }
    .image-display-frame img {
        display: block; /* Remove extra space below img */
        max-width: 100%; 
        max-height: 100%; /* Fill wrapper height */
        height: auto;    /* Maintain aspect ratio */
        object-fit: contain; /* Ensure entire image is visible */
        border-radius: 4px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        cursor: crosshair; /* Indicates clickability for zoom */
    }
    .image-display-frame.zoomed img {
        cursor: zoom-out;
    }
    .image-caption {
        font-size: 0.9em;
        color: #666;
        margin-top: 10px;
        text-align: center;
        font-weight: 500;
    }
    
    /* Zoom icon styling */
    .zoom-icon {
        position: absolute;
        top: 15px;
        right: 15px;
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px solid #007bff;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        color: #007bff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        z-index: 10;
        transition: all 0.2s ease;
        opacity: 0; /* Hidden by default */
        transform: scale(0.8);
        user-select: none; /* Prevent text selection */
    }
    
    .image-display-frame:hover .zoom-icon { /* Show on hover over frame */
        opacity: 1;
        transform: scale(1);
    }
    
    .zoom-icon:hover {
        background-color: #007bff;
        color: white;
        transform: scale(1.1);
    }
    
    .zoom-icon:active {
        transform: scale(0.95);
    }
    
    /* Show zoom icon when zoomed and different style */
    .image-display-frame.zoomed .zoom-icon {
        opacity: 1; /* Ensure it's visible when zoomed */
        background-color: rgba(220, 53, 69, 0.9); /* Reddish for zoom out */
        border-color: #dc3545;
        color: white; /* White icon on red background */
    }
    
    .image-display-frame.zoomed .zoom-icon:hover {
        background-color: #c82333; /* Darker red on hover */
        color: white;
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
        
        # Clear previous results from session state
        for key in ['comparison_results', 'table_data_for_display', 'csv_export_content', 
                    'highlighted_pages_file1', 'highlighted_pages_file2', 
                    'file1_bytes', 'file1_name', 'file2_bytes', 'file2_name']:
            if key in st.session_state:
                del st.session_state[key]
        # Reset zoom states for images might be needed if they are persisted in session_state elsewhere
        # st.session_state.zoom_states = {} 
        # st.session_state.zoom_positions = {}

        try:
            st.session_state.file1_bytes = uploaded_file1.getvalue()
            st.session_state.file2_bytes = uploaded_file2.getvalue()
            st.session_state.file1_name = uploaded_file1.name
            st.session_state.file2_name = uploaded_file2.name

            progress_bar.progress(5)
            status_text.text("✅ Files loaded. Preparing for backend processing...")
            time.sleep(0.2)

            with st.spinner("⏳ Processing files with backend and generating comparison... This may take several minutes."):
                progress_bar.progress(15)
                status_text.text("Calling backend for analysis...")
                
                # This now calls the newly defined wrapper function in this frontend script
                results = process_files_for_comparison(
                    st.session_state.file1_bytes, st.session_state.file1_name,
                    st.session_state.file2_bytes, st.session_state.file2_name
                )
                st.session_state.comparison_results = results
                
                st.session_state.highlighted_pages_file1 = results.get("highlighted_pages_file1", [])
                st.session_state.highlighted_pages_file2 = results.get("highlighted_pages_file2", [])
                
                # The 'product_comparison_details' from results will be used for CSV generation.
                # The old 'table_data_for_display' logic can be removed if not used for a direct table.
                # The CSV content is now directly in results.get("report_csv_data")
                st.session_state.csv_export_content = results.get("report_csv_data", "No data to export.\n".encode('utf-8'))

            progress_bar.progress(100)
            if results and results.get("error"):
                status_text.error(f"Comparison Error: {results.get('error')}")
                st.error(f"Details: {results.get('message', 'No additional details.')}")
            elif not BACKEND_AVAILABLE: # Check the global flag updated by process_files_for_comparison
                 status_text.error("Backend processor module is not available. Please check the application setup.")
            else:
                status_text.success("🎉 Comparison Complete! Results are ready below.")
            time.sleep(1)
            # status_text.empty() # Keep status message

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
if 'comparison_results' in st.session_state:
    results = st.session_state.comparison_results

    if results.get("error"):
        # Error message already shown by the button logic
        pass
    elif not BACKEND_AVAILABLE and not results.get("error"): # Handle case where BACKEND_AVAILABLE was false from start
        st.error("Backend is not available. Cannot display results.")
    else:
        # --- Summary Metrics ---
        st.markdown("<h2 class='main-title' style='font-size: 1.375rem; margin-top:1.25rem; margin-bottom:0.75rem;'>Comparison Summary</h2>", unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric(label=f"Products in {st.session_state.get('file1_name', 'File 1')}", value=results.get("product_items_file1_count", "N/A"))
        col_m2.metric(label=f"Products in {st.session_state.get('file2_name', 'File 2')}", value=results.get("product_items_file2_count", "N/A"))
        
        # Count issues for the third metric
        num_issues = 0
        if "product_comparison_details" in results and isinstance(results["product_comparison_details"], list):
            num_issues = len([
                item for item in results["product_comparison_details"]
                # Assuming 'Match Confirmed' or similar indicates no issue for that specific comparison item.
                # The exact condition depends on how 'Comparison_Type' is populated by your backend/mapping.
                # if item.get("Comparison_Type") != "Match Confirmed" 
            ]) # For now, let's count all rows returned in product_comparison_details as potential items of interest.
            num_issues = len(results["product_comparison_details"])


        col_m3.metric(label="Discrepancies / Items of Interest", value=num_issues)

        # --- Download Report Button ---
        if 'csv_export_content' in st.session_state and st.session_state.csv_export_content:
            try:
                # Attempt to decode if it's bytes, or use as is if it's already a string (though it should be bytes)
                csv_string_check = st.session_state.csv_export_content
                if isinstance(csv_string_check, bytes):
                    csv_string_check = csv_string_check.decode('utf-8')
                
                if "No issues to export." not in csv_string_check and "No significant discrepancies found to export." not in csv_string_check and "No comparison details to export." not in csv_string_check and len(csv_string_check.splitlines()) > 1 : # Check if there's more than just header or empty message
                    st.download_button(
                        label="📥 Export Discrepancies Report (CSV)",
                        data=st.session_state.csv_export_content,
                        file_name=f"comparison_report_{st.session_state.get('file1_name', 'f1')}_vs_{st.session_state.get('file2_name', 'f2')}.csv",
                        mime="text/csv",
                        key="export_button"
                    )
                else:
                    st.info("No significant discrepancies were found to include in the CSV export.")
            except Exception as e_csv_btn:
                st.error(f"Could not prepare download button: {e_csv_btn}")
                logging.error(f"Error with CSV download button content: {e_csv_btn}")


        # --- Visual Comparison Section (Full Pages) ---
        highlighted_pages_file1 = st.session_state.get('highlighted_pages_file1', [])
        highlighted_pages_file2 = st.session_state.get('highlighted_pages_file2', [])

        if (highlighted_pages_file1 and any(highlighted_pages_file1)) or \
           (highlighted_pages_file2 and any(highlighted_pages_file2)): # Check if there are actual images
            st.markdown("<h2 class='main-title' style='font-size: 1.375rem; margin-top:1.25rem; margin-bottom:0.75rem;'>Visual Page Analysis (Ranked Boxes)</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='subtitle' style='margin-top:0.25rem; margin-bottom:0.75rem;'>These images show detected product boxes and their ranking order on each page. They are not direct pixel-by-pixel difference highlights.</p>", unsafe_allow_html=True)
            st.markdown("<p style='color: #666; font-size: 0.9em; margin-bottom: 1rem;'>💡 Hover over images and click the '🔍+' icon to zoom into specific areas. Click '🔍−' to zoom out.</p>", unsafe_allow_html=True)

            num_pages_to_display = max(len(highlighted_pages_file1), len(highlighted_pages_file2))

            for page_idx in range(num_pages_to_display):
                st.markdown(f"--- \n**Page {page_idx + 1} Analysis**")
                col1, col2 = st.columns(2)

                image_id_f1 = f"img_f1_p{page_idx}"
                image_id_f2 = f"img_f2_p{page_idx}"

                with col1:
                    if page_idx < len(highlighted_pages_file1) and highlighted_pages_file1[page_idx]:
                        st.markdown(f"""
                            <div class="image-display-frame" data-image-id="{image_id_f1}">
                                <div class="zoom-icon" title="Click to zoom in">🔍+</div>
                                <div class="image-wrapper">
                                    <img src="data:image/jpeg;base64,{highlighted_pages_file1[page_idx]}" 
                                         alt="{st.session_state.get('file1_name', 'File 1')} Page {page_idx + 1} Visualization">
                                </div>
                                <div class="image-caption">{st.session_state.get('file1_name', 'File 1')} - Page {page_idx + 1}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(f"No visualization available for {st.session_state.get('file1_name', 'File 1')} - Page {page_idx + 1}.")
                
                with col2:
                    if page_idx < len(highlighted_pages_file2) and highlighted_pages_file2[page_idx]:
                        st.markdown(f"""
                            <div class="image-display-frame" data-image-id="{image_id_f2}">
                                <div class="zoom-icon" title="Click to zoom in">🔍+</div>
                                <div class="image-wrapper">
                                    <img src="data:image/jpeg;base64,{highlighted_pages_file2[page_idx]}" 
                                         alt="{st.session_state.get('file2_name', 'File 2')} Page {page_idx + 1} Visualization">
                                </div>
                                <div class="image-caption">{st.session_state.get('file2_name', 'File 2')} - Page {page_idx + 1}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(f"No visualization available for {st.session_state.get('file2_name', 'File 2')} - Page {page_idx + 1}.")
                st.markdown("<br>", unsafe_allow_html=True) # Adds a bit of space between page pairs

        else:
            st.info("No full-page visualizations (ranked box highlights) available from the backend or images could not be generated/found.")
            logging.info("No highlighted pages data to display.")
else:
    st.markdown("<div style='text-align:center; margin-top:1rem; margin-bottom:1rem;'>", unsafe_allow_html=True)
    st.caption("Upload two PDF files and click 'Compare' to see the results.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6b7280; font-size: 0.875rem;'>© PDF Comparison Tool.</p>", unsafe_allow_html=True) # Placeholder company name
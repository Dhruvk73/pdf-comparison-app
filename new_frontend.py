# frontend.py

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
    # MODIFIED IMPORT: Changed 'visual_layout_backend' to 'backend_processor'
    from backend_processor import process_files_for_comparison 
    BACKEND_AVAILABLE = True
    logging.info("SUCCESS: backend_processor.process_files_for_comparison imported.")
except ImportError as e:
    logging.error(f"Failed to import backend_processor: {e}", exc_info=True)
    BACKEND_AVAILABLE = False
    def process_files_for_comparison(file1_bytes, file1_name, file2_bytes, file2_name):
        st.error("Backend processor module is not available. Please check the import.")
        logging.error("process_files_for_comparison called but backend is not available.")
        return {
            "error": "Backend processing module not found.", "message": "Error: Backend not available.",
            "product_items_file1_count": 0, "product_items_file2_count": 0,
            "product_comparison_details": [], "report_csv_data": "Error,Backend not available\n",
            "all_product_details_file1": [], "all_product_details_file2": [],
            "highlighted_pages_file1": [], "highlighted_pages_file2": []
        }

st.set_page_config(layout="wide", page_title="PDF Comparison App")

# Initialize zoom state for each page using session state
if 'zoom_states' not in st.session_state:
    st.session_state.zoom_states = {}
if 'zoom_positions' not in st.session_state:
    st.session_state.zoom_positions = {}

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
        icon.innerHTML = '🔍−';
        icon.title = 'Click to zoom out';
        
        // Store zoom state in Streamlit
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: {action: 'zoom_in', imageId: imageId, x: originX, y: originY}
        }, '*');
        
    } else {
        // Zoom out
        img.style.transform = 'scale(1)';
        img.style.transformOrigin = 'center center';
        container.classList.remove('zoomed');
        
        // Change icon back to zoom in
        icon.innerHTML = '🔍+';
        icon.title = 'Click to zoom in';
        
        // Store zoom state in Streamlit
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: {action: 'zoom_out', imageId: imageId}
        }, '*');
    }
};

// Add event listeners when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Re-run this function whenever Streamlit updates the page
    setTimeout(function() {
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
    }, 100);
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
        min-width: 45%; 
        border: 1px solid #eee;
        padding: 10px;
        border-radius: 8px;
        background-color: #f9f9f9;
        display: flex;
        flex-direction: column;
        align-items: center;
        position: relative;
        overflow: hidden;
    }
    .image-wrapper {
        width: 100%;
        height: auto;
        display: flex; 
        justify-content: center;
        align-items: center;
        position: relative;
        overflow: hidden;
        border-radius: 4px;
    }
    .image-display-frame img {
        max-width: 100%; 
        height: auto;
        border-radius: 4px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        cursor: crosshair;
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
        opacity: 0;
        transform: scale(0.8);
        user-select: none;
    }
    
    .image-display-frame:hover .zoom-icon {
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
    
    /* Show zoom icon when zoomed */
    .image-display-frame.zoomed .zoom-icon {
        opacity: 1;
        background-color: rgba(220, 53, 69, 0.9);
        border-color: #dc3545;
        color: #dc3545;
    }
    
    .image-display-frame.zoomed .zoom-icon:hover {
        background-color: #dc3545;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Main content layout
st.markdown("<div style='text-align: left; margin-top: 1.25rem; margin-bottom: 1.25rem;'>", unsafe_allow_html=True)
st.markdown("<p class='main-title'>Compare PDFs</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload two PDF files to compare and identify differences.</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# File uploaders
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
        
        if 'comparison_results' in st.session_state: del st.session_state.comparison_results
        if 'table_data_for_display' in st.session_state: del st.session_state.table_data_for_display
        if 'csv_export_content' in st.session_state: del st.session_state.csv_export_content
        if 'highlighted_pages_file1' in st.session_state: del st.session_state.highlighted_pages_file1
        if 'highlighted_pages_file2' in st.session_state: del st.session_state.highlighted_pages_file2
        st.session_state.zoom_states = {} # Reset zoom states
        st.session_state.zoom_positions = {} # Reset zoom positions

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
                
                st.session_state.highlighted_pages_file1 = results.get("highlighted_pages_file1", [])
                st.session_state.highlighted_pages_file2 = results.get("highlighted_pages_file2", [])

                table_data_for_display = [] # Kept for CSV generation
                csv_export_data_list = [] # Kept for CSV generation

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
                        # Prepare CSV export data (even if table display is commented)
                        for i, issue in enumerate(issues_list_for_table):
                            product_name = "Unknown Product"
                            p1_name_from_issue = issue.get('P1_Name_Core')
                            p2_name_from_issue = issue.get('P2_Name_Core')

                            if p1_name_from_issue and isinstance(p1_name_from_issue, str) and p1_name_from_issue.strip():
                                product_name = p1_name_from_issue.strip()
                            elif p2_name_from_issue and isinstance(p2_name_from_issue, str) and p2_name_from_issue.strip():
                                product_name = p2_name_from_issue.strip()
                            
                            if not product_name or product_name == "Unknown Product":
                                p1_variant_from_issue = issue.get('P1_Variant')
                                p2_variant_from_issue = issue.get('P2_Variant')
                                if p1_variant_from_issue and isinstance(p1_variant_from_issue, str) and p1_variant_from_issue.strip():
                                    product_name = p1_variant_from_issue.strip()
                                elif p2_variant_from_issue and isinstance(p2_variant_from_issue, str) and p2_variant_from_issue.strip():
                                    product_name = p2_variant_from_issue.strip()

                            file1_issue_val, file2_issue_val = "N/A", "N/A"
                            diff_text = issue.get('Differences', '')

                            if diff_text:
                                match_op = re.search(r"Offer Price: F1=\$(.*?) vs F2=\$(.*)", diff_text)
                                match_rp = re.search(r"Regular Price: F1=\$(.*?) vs F2=\$(.*)", diff_text)
                                match_size = re.search(r"Size: F1='(.*?)' vs F2='(.*?)'", diff_text)

                                if match_op:
                                    file1_issue_val = f"Offer Price: ${match_op.group(1)}"
                                    file2_issue_val = f"Offer Price: ${match_op.group(2)}"
                                elif match_rp:
                                    file1_issue_val = f"Regular Price: ${match_rp.group(1)}"
                                    file2_issue_val = f"Regular Price: ${match_rp.group(2)}"
                                elif match_size:
                                    file1_issue_val = f"Size: '{match_size.group(1)}'"
                                    file2_issue_val = f"Size: '{match_size.group(2)}'"
                                else: 
                                    file1_issue_val = "Attribute Mismatch"
                                    file2_issue_val = "(check details)"

                            elif issue.get('Comparison_Type') == "Unmatched Product in File 1":
                                file1_issue_val, file2_issue_val = "Product Present", "Product Missing"
                            elif issue.get('Comparison_Type') == "Unmatched Product in File 2 (Extra)":
                                file1_issue_val, file2_issue_val = "Product Missing", "Product Present"
                            
                            csv_export_data_list.append({
                                "Product Name": product_name,
                                "Issue Type": issue.get('Comparison_Type'),
                                f"{st.session_state.file1_name} Details": file1_issue_val,
                                f"{st.session_state.file2_name} Details": file2_issue_val,
                                "Raw Differences": diff_text,
                                "P1_Brand": issue.get("P1_Brand"),
                                "P1_Name_Core": issue.get("P1_Name_Core"),
                                "P1_Variant": issue.get("P1_Variant"),
                                "P1_Size_Orig": issue.get("P1_Size_Orig"),
                                "P1_Size_Norm": issue.get("P1_Size_Norm"),
                                "P1_Offer_Price": issue.get("P1_Offer_Price"),
                                "P1_Regular_Price": issue.get("P1_Regular_Price"),
                                "P2_Brand": issue.get("P2_Brand"),
                                "P2_Name_Core": issue.get("P2_Name_Core"),
                                "P2_Variant": issue.get("P2_Variant"),
                                "P2_Size_Orig": issue.get("P2_Size_Orig"),
                                "P2_Size_Norm": issue.get("P2_Size_Norm"),
                                "P2_Offer_Price": issue.get("P2_Offer_Price"),
                                "P2_Regular_Price": issue.get("P2_Regular_Price"),
                            })

                    else:
                        status_text.text("✅ All products matched perfectly or no discrepancies found.")
                        time.sleep(1)

                    st.session_state.table_data_for_display = table_data_for_display # Keep for CSV
                    
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
if 'comparison_results' in st.session_state:
    results = st.session_state.comparison_results

    if results.get("error"):
        pass
    else:
        # --- Visual Comparison Section ---
        highlighted_pages_file1 = st.session_state.get('highlighted_pages_file1', [])
        highlighted_pages_file2 = st.session_state.get('highlighted_pages_file2', [])

        if highlighted_pages_file1 or highlighted_pages_file2:
            st.markdown("<h2 class='main-title' style='font-size: 1.375rem; margin-top:1.25rem; margin-bottom:0.75rem;'>Visual Differences (Full Pages)</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='subtitle' style='margin-top:0.25rem; margin-bottom:0.75rem;'>Red/Orange indicates differences in '{st.session_state.get('file1_name', 'File 1')}', Green/Blue indicates differences in '{st.session_state.get('file2_name', 'File 2')}'.</p>", unsafe_allow_html=True)
            st.markdown("<p style='color: #666; font-size: 0.9em; margin-bottom: 1rem;'>💡 Hover over images to see zoom controls, click to zoom into specific areas</p>", unsafe_allow_html=True)

            num_pages_to_display = max(len(highlighted_pages_file1), len(highlighted_pages_file2))

            for page_idx in range(num_pages_to_display):
                st.markdown(f"**Page {page_idx + 1} Comparison**")
                col1, col2 = st.columns(2)

                # Unique IDs for each image
                image_id_f1 = f"img_f1_p{page_idx}"
                image_id_f2 = f"img_f2_p{page_idx}"

                with col1:
                    if page_idx < len(highlighted_pages_file1) and highlighted_pages_file1[page_idx]:
                        st.markdown(f"""
                            <div class="image-display-frame" data-image-id="{image_id_f1}">
                                <div class="zoom-icon" title="Click to zoom in">🔍+</div>
                                <div class="image-wrapper">
                                    <img src="data:image/jpeg;base64,{highlighted_pages_file1[page_idx]}" 
                                         alt="{st.session_state.get('file1_name', 'File 1')} Page {page_idx + 1}">
                                </div>
                                <div class="image-caption">{st.session_state.get('file1_name', 'File 1')} - Page {page_idx + 1}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(f"No image available for {st.session_state.get('file1_name', 'File 1')} - Page {page_idx + 1}.")
                
                with col2:
                    if page_idx < len(highlighted_pages_file2) and highlighted_pages_file2[page_idx]:
                        st.markdown(f"""
                            <div class="image-display-frame" data-image-id="{image_id_f2}">
                                <div class="zoom-icon" title="Click to zoom in">🔍+</div>
                                <div class="image-wrapper">
                                    <img src="data:image/jpeg;base64,{highlighted_pages_file2[page_idx]}" 
                                         alt="{st.session_state.get('file2_name', 'File 2')} Page {page_idx + 1}">
                                </div>
                                <div class="image-caption">{st.session_state.get('file2_name', 'File 2')} - Page {page_idx + 1}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(f"No image available for {st.session_state.get('file2_name', 'File 2')} - Page {page_idx + 1}.")
                
                st.markdown("<br><br>", unsafe_allow_html=True)

        else:
            st.info("No full-page visual differences available or images could not be generated.")

else:
    st.markdown("<div style='text-align:center; margin-top:1rem; margin-bottom:1rem;'>", unsafe_allow_html=True)
    st.caption("Upload two PDF files and click 'Compare' to see the results.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6b7280; font-size: 0.875rem;'>© Your Company Name.</p>", unsafe_allow_html=True)
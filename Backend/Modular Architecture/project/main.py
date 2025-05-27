# project/main.py
import os
import openai
import time
import logging
import tempfile
import json
import re
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import pandas as pd
import io
from io import BytesIO
from pdf2image import convert_from_path # pdfinfo_from_path removed as not used
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError, PDFPopplerTimeoutError # Added more specific exceptions
from PIL import Image

# Import configurations
import config

# Import services and utils
from services.s3_service import S3Service, S3ServiceInterface
from services.ocr_service import OCRService, OCRServiceInterface
from services.detection_service import DetectionService, DetectionServiceInterface
from services.extraction_service import ExtractionService, ExtractionServiceInterface
from services.comparison_service import ComparisonService, ComparisonServiceInterface
# Utils are used within services, direct import here might not be needed unless for main orchestration logic.

# --- Roboflow SDK Conditional Import ---
Roboflow = None
try:
    from roboflow import Roboflow as RoboflowSDK
    Roboflow = RoboflowSDK # Assign to the global-like variable if import succeeds
    config.ROBOFLOW_SDK_AVAILABLE = True
    logging.info("Roboflow SDK imported successfully.")
except ImportError:
    config.ROBOFLOW_SDK_AVAILABLE = False
    logging.warning("Could not import 'Roboflow'. Ensure 'roboflow' package is installed if detection features are needed.")
# --- End Roboflow SDK ---

# --- Flask App and Logging Setup ---
app = Flask(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__) # Main application logger
# --- End Flask App and Logging Setup ---

# --- Initialize Clients ---
s3_client = None
textract_client = None
roboflow_model_object = None
openai_client_instance = None # Renamed to avoid conflict with openai module

try:
    logger.info(f"Initializing Boto3 clients for region: {config.AWS_DEFAULT_REGION}...")
    s3_client = boto3.client(
        's3',
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        region_name=config.AWS_DEFAULT_REGION
    )
    textract_client = boto3.client(
        'textract',
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        region_name=config.AWS_DEFAULT_REGION
    )
    # Test S3 client (optional but good for startup check)
    s3_client.list_buckets() # This will raise an error if creds are bad
    logger.info("Boto3 S3 and Textract clients initialized successfully.")
except (NoCredentialsError, PartialCredentialsError) as e:
    logger.error(f"AWS credentials not found or incomplete: {e}")
except ClientError as e: # Catch other Boto3 client errors (e.g., invalid region, network issues)
    logger.error(f"Error initializing Boto3 clients: {e}", exc_info=True)
except Exception as e:
    logger.error(f"Unexpected error initializing Boto3 clients: {e}", exc_info=True)


if config.ROBOFLOW_SDK_AVAILABLE and Roboflow and \
   config.ROBOFLOW_API_KEY and config.ROBOFLOW_PROJECT_ID and config.ROBOFLOW_VERSION_NUMBER:
    try:
        logger.info("Initializing Roboflow model object...")
        rf = Roboflow(api_key=config.ROBOFLOW_API_KEY)
        project = rf.project(config.ROBOFLOW_PROJECT_ID)
        roboflow_model_object = project.version(int(config.ROBOFLOW_VERSION_NUMBER)).model
        logger.info("Roboflow model object initialized successfully.")
    except ValueError as e_val:
        logger.error(f"Error with Roboflow configuration (e.g., version number not an int): {e_val}", exc_info=True)
    except Exception as e_rf:
        logger.error(f"Error initializing Roboflow model object: {e_rf}", exc_info=True)
else:
    if config.ROBOFLOW_SDK_AVAILABLE: # SDK is there, but config is missing
        logger.warning("Roboflow SDK is available, but API key, project ID, or version number is missing. Roboflow features will be disabled.")
    # If SDK not available, warning already logged.

if config.OPENAI_API_KEY:
    try:
        logger.info("Initializing OpenAI client...")
        openai_client_instance = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        # Test call (optional)
        # openai_client_instance.models.list()
        logger.info("OpenAI client initialized successfully.")
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI API Key is invalid or authentication failed: {e}")
        openai_client_instance = None
    except Exception as e_openai:
        logger.error(f"Error initializing OpenAI client: {e_openai}", exc_info=True)
        openai_client_instance = None
else:
    logger.warning("OPENAI_API_KEY not found. LLM-based extraction features will be disabled.")
# --- End Initialize Clients ---


# --- Initialize Services ---
# Ensure clients are passed only if they were successfully initialized
s3_service: S3ServiceInterface = S3Service(s3_client, config.S3_BUCKET_NAME) if s3_client and config.S3_BUCKET_NAME else None
ocr_service: OCRServiceInterface = OCRService(textract_client, config.S3_BUCKET_NAME) if textract_client and config.S3_BUCKET_NAME else None
detection_service: DetectionServiceInterface = DetectionService(roboflow_model_object, config.ROBOFLOW_SDK_AVAILABLE) if config.ROBOFLOW_SDK_AVAILABLE else None
extraction_service: ExtractionServiceInterface = ExtractionService(openai_client_instance) if openai_client_instance else None
comparison_service: ComparisonServiceInterface = ComparisonService() # No external clients needed for this simple version

# Service availability checks (optional, but good for early warning)
if not s3_service: logger.warning("S3Service could not be initialized. File uploads to S3 will fail.")
if not ocr_service: logger.warning("OCRService could not be initialized. Textract processing will fail.")
if not detection_service : logger.warning("DetectionService could not be initialized. Roboflow predictions will be unavailable.")
if not extraction_service : logger.warning("ExtractionService could not be initialized. LLM data extraction will be unavailable.")
# --- End Initialize Services ---


def process_single_file_for_items(
    file_storage,
    original_filename: str,
    file_bytes: bytes
) -> list[dict[str, any]]:
    """
    Processes a single uploaded file (PDF or image) and returns a list of extracted product items.
    """
    current_file_product_items = []
    pil_images: List[Image.Image] = []
    temp_pdf_path_for_conversion = None # For PDF conversion only

    # Ensure temp_uploads directory exists for PDF conversion and Roboflow temp images
    os.makedirs(config.TEMP_UPLOADS_DIR, exist_ok=True)

    try:
        if original_filename.lower().endswith(".pdf"):
            logger.info(f"Processing PDF: {original_filename}. Converting to images...")
            # Save PDF to a temporary file for poppler to read
            # Use a unique name to avoid collisions if multiple requests happen
            ts = int(time.time() * 1000)
            temp_pdf_filename = f"{ts}_{secure_filename(original_filename)}"
            temp_pdf_path_for_conversion = os.path.join(config.TEMP_UPLOADS_DIR, temp_pdf_filename)

            with open(temp_pdf_path_for_conversion, "wb") as f_pdf:
                f_pdf.write(file_bytes)
            
            try:
                pil_images = convert_from_path(
                    temp_pdf_path_for_conversion,
                    dpi=200, # Configurable?
                    poppler_path=config.POPPLER_PATH if config.POPPLER_PATH else None,
                    timeout=300 # 5 minutes timeout for conversion
                )
                logger.info(f"Converted PDF {original_filename} to {len(pil_images)} image(s).")
            except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError, PDFPopplerTimeoutError) as e_pdf:
                logger.error(f"PDF conversion error for {original_filename}: {e_pdf}", exc_info=True)
                # Consider raising a custom exception or returning an error indicator
                raise ValueError(f"PDF processing error for {original_filename}: {str(e_pdf)}") from e_pdf
            except Exception as e_conv: # Catch any other conversion errors
                logger.error(f"Unexpected PDF conversion error for {original_filename}: {e_conv}", exc_info=True)
                raise ValueError(f"Unexpected PDF conversion error for {original_filename}: {str(e_conv)}") from e_conv
            finally:
                if temp_pdf_path_for_conversion and os.path.exists(temp_pdf_path_for_conversion):
                    try:
                        os.remove(temp_pdf_path_for_conversion)
                        logger.debug(f"Removed temporary PDF: {temp_pdf_path_for_conversion}")
                    except OSError as e_del_pdf:
                        logger.warning(f"Could not delete temporary PDF {temp_pdf_path_for_conversion}: {e_del_pdf}")
        
        elif original_filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")): # Added webp
            logger.info(f"Processing image file: {original_filename}")
            try:
                pil_images = [Image.open(BytesIO(file_bytes))]
            except IOError as e_img:
                logger.error(f"Could not open image {original_filename}: {e_img}")
                raise ValueError(f"Invalid or corrupted image file: {original_filename}") from e_img
        else:
            logger.error(f"Unsupported file type for {original_filename}")
            raise ValueError(f"Unsupported file type: {original_filename}. Please upload PDF, PNG, JPG, or JPEG.")

        # --- Process each page/image ---
        s3_page_keys_for_cleanup_on_error = []
        for page_idx, page_image_pil in enumerate(pil_images):
            page_desc = f"page {page_idx + 1} of {original_filename}"
            logger.info(f"Processing {page_desc}...")
            image_width_px, image_height_px = page_image_pil.size
            
            # 1. Detection (Roboflow)
            roboflow_preds = None
            if detection_service:
                # Pass original_filename and page_idx for unique temp image naming in service
                temp_robo_filename = f"{secure_filename(os.path.splitext(original_filename)[0])}_p{page_idx}.jpg"
                roboflow_preds = detection_service.get_roboflow_predictions(
                    page_image_pil,
                    config.TEMP_UPLOADS_DIR,
                    temp_robo_filename
                )
            if not roboflow_preds: # Handles None or empty list
                logger.warning(f"No Roboflow detections (or service unavailable) for {page_desc}. Skipping Textract and LLM for this page.")
                continue # Move to next page if no boxes detected

            # 2. OCR (Textract via S3)
            textract_all_blocks_page = None
            blocks_map_page = None
            s3_page_object_name_for_textract = None

            if ocr_service and s3_service: # Both must be available
                img_byte_arr_for_s3 = BytesIO()
                page_image_pil.save(img_byte_arr_for_s3, format='JPEG') # Textract prefers JPEG/PNG
                img_byte_arr_for_s3.seek(0) # Reset stream position

                timestamp = int(time.time() * 1000)
                s3_page_object_name_for_textract = f"ocr_inputs/{timestamp}_{secure_filename(os.path.splitext(original_filename)[0])}_p{page_idx}.jpg"
                
                uploaded_s3_key = s3_service.upload_fileobj(img_byte_arr_for_s3, s3_page_object_name_for_textract)
                
                if uploaded_s3_key:
                    s3_page_keys_for_cleanup_on_error.append(uploaded_s3_key) # Add for potential cleanup
                    textract_all_blocks_page = ocr_service.analyze_document_from_s3(uploaded_s3_key)
                    # S3 cleanup is now handled after this block, or in a broader try/finally for the endpoint
                else:
                    logger.warning(f"S3 upload failed for {page_desc}. Cannot perform OCR.")
                    # No S3 key to clean up here if upload failed
            
            if not textract_all_blocks_page: # Handles None or empty list
                logger.warning(f"Textract analysis failed or returned no blocks (or service unavailable) for {page_desc}. Skipping LLM for this page.")
                if s3_page_object_name_for_textract and s3_service: # Attempt cleanup if uploaded
                    s3_service.delete_object(s3_page_object_name_for_textract)
                    if s3_page_object_name_for_textract in s3_page_keys_for_cleanup_on_error:
                        s3_page_keys_for_cleanup_on_error.remove(s3_page_object_name_for_textract)
                continue

            blocks_map_page = {block['Id']: block for block in textract_all_blocks_page}

            # 3. Collate text within detected boxes (using OCR service method)
            collated_snippets = ocr_service.collate_text_for_product_boxes(
                roboflow_preds,
                textract_all_blocks_page, # Pass the blocks directly
                blocks_map_page,          # Pass the map directly
                image_width_px,
                image_height_px
            )
            
            # Cleanup S3 object for Textract *after* collation (if successful or not)
            if s3_page_object_name_for_textract and s3_service:
                s3_service.delete_object(s3_page_object_name_for_textract)
                if s3_page_object_name_for_textract in s3_page_keys_for_cleanup_on_error:
                    s3_page_keys_for_cleanup_on_error.remove(s3_page_object_name_for_textract)


            # 4. Extraction (LLM) and Normalization for each snippet
            if not extraction_service:
                logger.warning("ExtractionService is not available. Cannot extract structured data from snippets.")
            else:
                for snippet_info in collated_snippets:
                    if not snippet_info.get("collated_text", "").strip():
                        logger.debug(f"Skipping empty collated text for box {snippet_info.get('product_box_id', 'Unknown')}")
                        continue

                    llm_extracted_data = extraction_service.extract_product_data_with_llm(snippet_info["collated_text"])
                    
                    if llm_extracted_data.get("llm_call_failed"):
                        logger.warning(f"LLM extraction failed for snippet from box {snippet_info.get('product_box_id')}. Snippet:\n{snippet_info['collated_text']}\nError: {llm_extracted_data.get('error_message')}")
                        # Optionally, store raw snippet if LLM fails
                        # current_file_product_items.append({ ... "error_llm_extraction": True, ...})
                        continue # Skip to next snippet

                    normalized_product_data = extraction_service.normalize_product_data(llm_extracted_data)

                    # Basic validation of extracted data (e.g., has a name or brand)
                    if normalized_product_data and \
                       (normalized_product_data.get("product_name_core") or normalized_product_data.get("product_brand")):
                        
                        product_data_for_list = {
                            "source_file": original_filename,
                            "page_number": page_idx + 1,
                            "roboflow_box_id": snippet_info.get("product_box_id"),
                            "roboflow_confidence": snippet_info.get("roboflow_confidence"),
                            "roboflow_class_name": snippet_info.get("class_name"),
                            "roboflow_box_coords_pixels_center_wh": snippet_info.get("roboflow_box_coords_pixels_center_wh"),
                            "original_collated_text": snippet_info["collated_text"],
                            **normalized_product_data # Add all fields from normalized_product_data
                        }
                        current_file_product_items.append(product_data_for_list)
                    else:
                        logger.warning(f"LLM extraction deemed incomplete or invalid for snippet from box {snippet_info.get('product_box_id')}. Missing core name/brand.")
                        logger.debug(f"Problematic Snippet Text:\n{snippet_info['collated_text']}")
                        logger.debug(f"Problematic LLM Output (normalized):\n{normalized_product_data}")

        return current_file_product_items

    except ValueError as ve: # Catch custom ValueErrors raised for file processing issues
        logger.error(f"File processing error for {original_filename}: {ve}")
        # Re-raise to be caught by the main endpoint handler if needed, or return error indicator
        raise
    except Exception as e_main_proc:
        logger.error(f"Major unexpected error during processing of {original_filename}: {e_main_proc}", exc_info=True)
        raise # Re-raise to be caught by the main endpoint handler
    finally:
        # Ensure any S3 objects uploaded for OCR are cleaned up if an error occurred mid-processing
        if s3_service and s3_page_keys_for_cleanup_on_error:
            logger.info(f"Performing final S3 cleanup for {len(s3_page_keys_for_cleanup_on_error)} objects due to processing stage completion or error.")
            for key_to_clean in s3_page_keys_for_cleanup_on_error:
                s3_service.delete_object(key_to_clean)
        # Temporary PDF for conversion is cleaned up within its own block


@app.route("/upload", methods=["POST"])
def process_uploaded_files_route():
    logger.info(f"Received request at /upload endpoint. Request Files: {list(request.files.keys())}")
    if 'file1' not in request.files or 'file2' not in request.files:
        logger.warning("Missing 'file1' or 'file2' in request.")
        return jsonify({"error": "Two files ('file1' and 'file2') are required."}), 400

    file1_storage = request.files['file1']
    file2_storage = request.files['file2']

    if not file1_storage.filename or not file2_storage.filename:
        logger.warning("One or both files are missing filenames.")
        return jsonify({"error": "Files must have names."}), 400
        
    # Secure filenames early
    filename1 = secure_filename(file1_storage.filename)
    filename2 = secure_filename(file2_storage.filename)

    # Read file bytes early to avoid issues with stream being consumed
    try:
        file1_bytes = file1_storage.read()
        file2_bytes = file2_storage.read()
        file1_storage.seek(0) # Reset stream pointer (though bytes are now primary)
        file2_storage.seek(0)
    except Exception as e_read:
        logger.error(f"Error reading file bytes: {e_read}", exc_info=True)
        return jsonify({"error": "Could not read uploaded files."}), 400


    processed_outputs_data = {"file1_items": [], "file2_items": []}
    processing_errors = []

    # Process file 1
    try:
        logger.info(f"--- Starting processing for File 1: {filename1} ---")
        processed_outputs_data["file1_items"] = process_single_file_for_items(
            file1_storage, filename1, file1_bytes
        )
        logger.info(f"--- Finished processing for File 1: {filename1}. Found {len(processed_outputs_data['file1_items'])} items. ---")
    except ValueError as ve: # Specific errors from process_single_file_for_items
        logger.error(f"Error processing File 1 ({filename1}): {ve}", exc_info=True)
        processing_errors.append(f"File 1 ({filename1}): {str(ve)}")
    except Exception as e:
        logger.error(f"General error processing File 1 ({filename1}): {e}", exc_info=True)
        processing_errors.append(f"File 1 ({filename1}): An unexpected error occurred during processing.")

    # Process file 2
    try:
        logger.info(f"--- Starting processing for File 2: {filename2} ---")
        processed_outputs_data["file2_items"] = process_single_file_for_items(
            file2_storage, filename2, file2_bytes
        )
        logger.info(f"--- Finished processing for File 2: {filename2}. Found {len(processed_outputs_data['file2_items'])} items. ---")
    except ValueError as ve:
        logger.error(f"Error processing File 2 ({filename2}): {ve}", exc_info=True)
        processing_errors.append(f"File 2 ({filename2}): {str(ve)}")
    except Exception as e:
        logger.error(f"General error processing File 2 ({filename2}): {e}", exc_info=True)
        processing_errors.append(f"File 2 ({filename2}): An unexpected error occurred during processing.")

    if processing_errors:
        # If critical errors occurred (e.g., PDF conversion), might be best to return error now
        # For now, we'll proceed to comparison if at least one file yielded items
        if not processed_outputs_data["file1_items"] and not processed_outputs_data["file2_items"]:
             return jsonify({
                "error": "Processing failed for both files.",
                "details": processing_errors
            }), 500


    # --- Compare products ---
    product_items1 = processed_outputs_data["file1_items"]
    product_items2 = processed_outputs_data["file2_items"]
    product_centric_comparison_report = []

    if comparison_service:
        if product_items1 or product_items2: # Only compare if there's something to compare
            product_centric_comparison_report = comparison_service.compare_product_items(
                product_items1, product_items2, similarity_threshold=70 # Configurable?
            )
            logger.info(f"Comparison complete. Report items: {len(product_centric_comparison_report)}")
        else:
            logger.info("No items extracted from either file to compare.")
    else:
        logger.warning("ComparisonService not available. Skipping product comparison.")
    
    # --- Prepare final response ---
    # Grammar check placeholders (not implemented in this refactor)
    grammar_issues1 = []
    grammar_issues2 = []

    comparison_results = {
        "message": "File processing and comparison complete.",
        "processing_status": "Partial success" if processing_errors else "Success",
        "processing_errors": processing_errors if processing_errors else None,
        "product_items_file1_count": len(product_items1),
        "extracted_items_file1": product_items1, # Include extracted items for review
        "product_items_file2_count": len(product_items2),
        "extracted_items_file2": product_items2, # Include extracted items for review
        "product_comparison_details": product_centric_comparison_report,
        "grammar_issues_file1_placeholder": grammar_issues1, # Placeholder
        "grammar_issues_file2_placeholder": grammar_issues2, # Placeholder
    }

    # --- Generate CSV report ---
    report_lines_for_csv = []
    if product_centric_comparison_report:
        for diff_item in product_centric_comparison_report:
            report_lines_for_csv.append({
                "Comparison_Type": diff_item.get("type", "N/A"),
                "P1_Brand": diff_item.get("product1_brand", diff_item.get("product_brand", "")),
                "P1_Name_Core": diff_item.get("product1_name_core", diff_item.get("product_name_core", "")),
                "P1_Variant": diff_item.get("product1_variant", diff_item.get("product_variant", "")),
                "P1_Size": diff_item.get("product1_size", diff_item.get("product_size", "")),
                "P1_Offer_Price": str(diff_item.get("offer_price1", diff_item.get("offer_price", ""))),
                "P1_Regular_Price": str(diff_item.get("regular_price1", diff_item.get("regular_price", ""))),
                "P1_Unit_Indicator": diff_item.get("unit_indicator1", diff_item.get("unit_indicator", "")),
                "P1_Store_Terms": diff_item.get("store_terms1", diff_item.get("store_terms", "")),
                "P2_Brand": diff_item.get("product2_brand", ""),
                "P2_Name_Core": diff_item.get("product2_name_core", ""),
                "P2_Variant": diff_item.get("product2_variant", ""),
                "P2_Size": diff_item.get("product2_size", ""),
                "P2_Offer_Price": str(diff_item.get("offer_price2", "")),
                "P2_Regular_Price": str(diff_item.get("regular_price2", "")),
                "P2_Unit_Indicator": diff_item.get("unit_indicator2", ""),
                "P2_Store_Terms": diff_item.get("store_terms2", ""),
                "Similarity_Percent": diff_item.get("text_similarity_percent", ""),
                "Differences": diff_item.get("differences", "")
            })
    else: # No comparison happened or no matches
        report_lines_for_csv.append({
            "Comparison_Type": "Summary",
            "P1_Name_Core": f"{len(product_items1)} items extracted from File 1",
            "P2_Name_Core": f"{len(product_items2)} items extracted from File 2",
            "Differences": "No comparison performed or no matches found." if not (product_items1 and product_items2) else "No items to compare or comparison service unavailable."
        })
        if processing_errors: # Add errors to CSV if any
            for i, err_msg in enumerate(processing_errors):
                 report_lines_for_csv.append({"Comparison_Type": f"Processing Error {i+1}", "Differences": err_msg})


    report_df = pd.DataFrame(report_lines_for_csv)
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)
    comparison_results["report_csv_content"] = csv_buffer.getvalue() # Changed key for clarity

    # Determine overall status code
    status_code = 200
    if processing_errors and not (product_items1 or product_items2): # Both failed completely
        status_code = 500 # Internal Server Error
    elif processing_errors: # Some partial success
        status_code = 207 # Multi-Status, if you want to be very specific, or stick to 200 with error details

    logger.info(f"Request to /upload completed. Status Code: {status_code}. File1 items: {len(product_items1)}, File2 items: {len(product_items2)}.")
    return jsonify(comparison_results), status_code


if __name__ == '__main__':
    # Create temp_uploads directory if it doesn't exist at startup
    os.makedirs(config.TEMP_UPLOADS_DIR, exist_ok=True)
    logger.info(f"Flask app starting on port {config.PORT} with debug_mode={config.DEBUG_MODE}")
    app.run(debug=config.DEBUG_MODE, host='0.0.0.0', port=config.PORT)
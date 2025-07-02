import os
import pickle
import json
import logging
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv


# Import the specific functions you want to test from your backend
from visual_layout_backend import (
    process_files_for_comparison, # We'll run this once to get the data
    compare_product_items,
    run_vision_review_for_pair,
    check_for_photo_mistake,
    get_segment_image_bytes,
    post_process_and_validate_item_data,
    draw_detailed_highlights,
    parse_price_string
)

load_dotenv() # <-- ADD THIS to load the .env file
POPPLER_BIN_PATH = os.getenv('POPPLER_PATH_OVERRIDE', None) # <-- ADD THIS
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Set the file paths for the two PDFs you want to test with
FILE1_PATH = r"C:\Users\khura\Desktop\Comparison Files\Compare 1\Shopper-21-al-27-mayo-2025 3.pdf"
FILE2_PATH = r"C:\Users\khura\Desktop\Comparison Files\Compare 2\21 al 27 mayo 2025_03.pdf"
CACHE_FILE = "cached_data/full_extraction_cache.pkl"

# Create necessary directories
os.makedirs("cached_data", exist_ok=True)
os.makedirs("output", exist_ok=True)


def capture_and_cache_data():
    """
    Runs the full, slow pipeline ONCE and saves the result to a cache file.
    """
    logging.info("--- CAPTURE MODE: Running full pipeline to generate cache ---")
    if not os.path.exists(FILE1_PATH) or not os.path.exists(FILE2_PATH):
        logging.error(f"Test files not found. Please update FILE1_PATH and FILE2_PATH in this script.")
        return

    with open(FILE1_PATH, "rb") as f1, open(FILE2_PATH, "rb") as f2:
        file1_bytes = f1.read()
        file2_bytes = f2.read()
        file1_name = Path(FILE1_PATH).name
        file2_name = Path(FILE2_PATH).name
        
        # This is the only time we run the full, slow process
        results = process_files_for_comparison(file1_bytes, file1_name, file2_bytes, file2_name)

        if "error" in results:
            logging.error(f"An error occurred during data capture: {results['error']}")
        else:
            # Save the essential results needed for fast re-runs
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(results, f)
            logging.info(f"Successfully captured and saved data to {CACHE_FILE}")


def run_fast_debug_from_cache():
    """
    Loads data from the cache and re-runs only the fast comparison and drawing logic.
    """
    logging.info(f"--- DEBUG MODE: Loading data from {CACHE_FILE} ---")
    if not os.path.exists(CACHE_FILE):
        logging.error("Cache file not found. Run this script in CAPTURE_MODE first.")
        return

    with open(CACHE_FILE, "rb") as f:
        cached_results = pickle.load(f)

    # Load all the pre-processed data from the cache
    final_product_items_file1 = cached_results["all_product_details_file1"]
    final_product_items_file2 = cached_results["all_product_details_file2"]
    
    # We need the original images for drawing highlights
    # You might need to adjust how PIL images are loaded if they don't pickle well
    # For this example, let's assume they were part of the results or re-load them.
    # NOTE: The original `process_files_for_comparison` does not return the page pils.
    # You would need to modify it or re-convert here for a complete test.
    # For simplicity, let's assume the highlight drawing function can be tested with a blank image.
    
    from pdf2image import convert_from_path
    page_pils_file1 = convert_from_path(FILE1_PATH, dpi=300, fmt='jpeg', poppler_path=POPPLER_BIN_PATH)
    page_pils_file2 = convert_from_path(FILE2_PATH, dpi=300, fmt='jpeg', poppler_path=POPPLER_BIN_PATH)


    logging.info("Running fast comparison and categorization logic...")
    # Re-run the comparison logic on the cached data
    # (The logic from your main loop)
    preliminary_comparison_report = compare_product_items(final_product_items_file1, final_product_items_file2)
    final_comparison_report = []

    for report_item in preliminary_comparison_report:
            
            if "Mismatch" in report_item.get("Comparison_Type", ""):
                p1_box_id = report_item.get("P1_Box_ID")
                p2_box_id = report_item.get("P2_Box_ID")
                item1 = next((item for item in final_product_items_file1 if item.get("product_box_id") == p1_box_id), None)
                item2 = next((item for item in final_product_items_file2 if item.get("product_box_id") == p2_box_id), None)

                # This block now runs for ANY matched pair, not just mismatches.
                item_data = report_item 
                if item1 and item2:
                    
                    page_idx1 = item1["page_idx_for_reprocessing"]
                    page_idx2 = item2["page_idx_for_reprocessing"]

                    segment_bytes1 = get_segment_image_bytes(page_pils_file1[page_idx1], item1["roboflow_box_coords_pixels_center_wh"], item1["product_box_id"])
                    segment_bytes2 = get_segment_image_bytes(page_pils_file2[page_idx2], item2["roboflow_box_coords_pixels_center_wh"], item2["product_box_id"])
                    
                    is_photo_mistake = check_for_photo_mistake(segment_bytes1, segment_bytes2, item1["product_box_id"])
                    if is_photo_mistake:
                        current_diffs = item_data.get("Differences", "")
                        photo_diff_str = "Photo Mistake: Core products appear different."
                        if photo_diff_str not in current_diffs:
                            item_data["Differences"] = f"{current_diffs}; {photo_diff_str}".strip('; ')
                        if "Mismatch" not in item_data.get("Comparison_Type", ""):
                            item_data["Comparison_Type"] = "Product Match - Attribute Mismatch"

                    if "Mismatch" in item_data.get("Comparison_Type", "") and not is_photo_mistake:
                        # The text/price re-check with GPT-4V still only runs for initial mismatches
                        vision_data = run_vision_review_for_pair(item1, item2, page_pils_file1[page_idx1], page_pils_file2[page_idx2])
                        
                        if vision_data:
                            # (vision re-check logic remains the same)
                            v1_details = vision_data.get("item1_details", {})
                            v2_details = vision_data.get("item2_details", {})
                            v1_price = parse_price_string(v1_details.get("offer_price"))
                            v2_price = parse_price_string(v2_details.get("offer_price"))
                            prices_match_vision = (v1_price is not None and v2_price is not None and abs(v1_price - v2_price) < 0.02)
                            
                            if v1_details:
                                if v1_price is not None: item1['offer_price'] = v1_price
                                if v1_details.get('size_and_quantity'): item1['size_quantity_info'] = v1_details.get('size_and_quantity')
                            
                            if v2_details:
                                if v2_price is not None: item2['offer_price'] = v2_price
                                if v2_details.get('size_and_quantity'): item2['size_quantity_info'] = v2_details.get('size_and_quantity')

                            if prices_match_vision:
                                logger.warning(f"VISION CORRECTION: Mismatch for {p1_box_id} resolved by vision.")
                                item_data["Comparison_Type"] = "Product Match - Attributes OK (Corrected by Vision)"
                                item_data["Differences"] = ""
                            else:
                                logger.warning(f"VISION CONFIRMATION: Mismatch for {p1_box_id} confirmed by vision.")
                            
                            post_process_and_validate_item_data(item1, item1.get('price_candidates', []), item1.get('collated_text', ''), p1_box_id)
                            post_process_and_validate_item_data(item2, item2.get('price_candidates', []), item2.get('collated_text', ''), p2_box_id)

                # Categorization logic now correctly modifies item_data
                categories = []
                differences_str = item_data.get("Differences", "")
                
                if "Price" in differences_str: categories.append("Price Issue")
                if any(term in differences_str for term in ["Size", "Variant", "Name", "Brand"]): categories.append("Text Issue")
                if "Photo:" in differences_str: categories.append("Photo") # <-- CHANGE THIS LINE
                
                item_data['issue_categories'] = list(set(categories))
                
                final_comparison_report.append(item_data)



    logging.info("Generating new highlighted images...")
    # Re-run the drawing logic and save the output locally
    max_pages = max(len(page_pils_file1), len(page_pils_file2))
    for page_idx in range(max_pages):
        if page_idx < len(page_pils_file1):
            page_pil_copy1 = page_pils_file1[page_idx].copy()
            items_on_page1 = [item for item in final_product_items_file1 if item.get("page_idx_for_reprocessing") == page_idx]
            report_items_on_page1 = [r for r in final_comparison_report if r.get("P1_Box_ID") and f"-Page{page_idx}-" in r["P1_Box_ID"]]
            
            img_bytes1 = draw_detailed_highlights(page_pil_copy1, items_on_page1, report_items_on_page1, "file1")
            with open(f"output/file1_page_{page_idx + 1}_highlighted.jpg", "wb") as f_out:
                f_out.write(img_bytes1.getbuffer())
            logging.info(f"Saved test output to output/file1_page_{page_idx + 1}_highlighted.jpg")

        if page_idx < len(page_pils_file2):
            page_pil_copy2 = page_pils_file2[page_idx].copy()
            items_on_page2 = [item for item in final_product_items_file2 if item.get("page_idx_for_reprocessing") == page_idx]
            report_items_on_page2 = [r for r in final_comparison_report if r.get("P2_Box_ID") and f"-Page{page_idx}-" in r["P2_Box_ID"]]

            img_bytes2 = draw_detailed_highlights(page_pil_copy2, items_on_page2, report_items_on_page2, "file2")
            with open(f"output/file2_page_{page_idx + 1}_highlighted.jpg", "wb") as f_out:
                f_out.write(img_bytes2.getbuffer())
            logging.info(f"Saved test output to output/file2_page_{page_idx + 1}_highlighted.jpg")


if __name__ == '__main__':
    # --- CHOOSE YOUR MODE ---
    # Set to True to run the slow data capture once.
    # Set to False to run fast debugging using the cached data.
    CAPTURE_MODE = False

    if CAPTURE_MODE:
        capture_and_cache_data()
    else:
        run_fast_debug_from_cache()
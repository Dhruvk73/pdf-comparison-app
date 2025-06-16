import os
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import tempfile
import numpy as np
from sklearn.cluster import KMeans
import math
from pathlib import Path
import cv2
from typing import List, Optional, Tuple
import re
import json
import pandas as pd
import io
import base64
import logging
import time
from typing import Dict, List, Optional, Tuple
import openai
from fuzzywuzzy import fuzz

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# PDF PROCESSING CONFIGURATION
# ========================================

# Configure poppler path for pdf2image
POPPLER_PATH = r'C:\Program Files\poppler-0.68.0\bin'  # Default Windows path
# For other systems, you can modify this path or set it as None to use system PATH

def get_poppler_path():
    """
    Get the poppler path for pdf2image conversion.
    Returns None if poppler is in system PATH or on non-Windows systems.
    """
    # Check if running on Windows and if the default path exists
    if os.name == 'nt':  # Windows
        if os.path.exists(POPPLER_PATH):
            return POPPLER_PATH
        else:
            # Try some common alternative paths
            alternative_paths = [
                r'C:\Program Files\poppler\bin',
                r'C:\poppler\bin',
                r'C:\poppler-0.68.0\bin',
                r'C:\tools\poppler\bin'
            ]
            for path in alternative_paths:
                if os.path.exists(path):
                    return path
            
            # If no path found, print warning but continue (might work if in PATH)
            print(f"Warning: Poppler not found at {POPPLER_PATH} or alternative paths.")
            print("Make sure poppler is installed and either:")
            print("1. Update POPPLER_PATH variable in the script")
            print("2. Add poppler to your system PATH")
            return None
    else:
        # Non-Windows systems typically have poppler in PATH
        return None

# ========================================
# SCRIPT 1: PDF Processing and Ranking
# ========================================

def is_valid_product_box(width, height, image_width, image_height,
                         min_width_ratio=0.08, min_height_ratio=0.08,
                         min_area_ratio=0.01, max_aspect_ratio=5.0):
    """
    Determine if a bounding box represents a valid product (not a small banner/header)
    """
    # Calculate relative dimensions
    width_ratio = width / image_width
    height_ratio = height / image_height
    area_ratio = (width * height) / (image_width * image_height)

    # Calculate aspect ratio (prevent division by zero)
    aspect_ratio = max(width, height) / max(min(width, height), 1)

    # Apply filters
    if width_ratio < min_width_ratio:
        return False  # Too narrow

    if height_ratio < min_height_ratio:
        return False  # Too short

    if area_ratio < min_area_ratio:
        return False  # Too small overall

    if aspect_ratio > max_aspect_ratio:
        return False  # Too elongated (likely a banner)

    # Additional absolute minimum sizes
    if width < 80 or height < 80:
        return False  # Too small in absolute terms

    return True

def improved_grid_ranking(boxes, tolerance_factor=0.3):
    """
    Improved grid-based ranking that follows strict top-to-bottom, left-to-right pattern

    Args:
        boxes: List of box dictionaries with center_x, center_y
        tolerance_factor: How much vertical tolerance to allow for "same row" (0.1-0.5)
    """
    if not boxes:
        return boxes

    print(f"Improved grid ranking for {len(boxes)} boxes...")

    # Sort all boxes by Y coordinate first to identify rows
    boxes_by_y = sorted(boxes, key=lambda b: b["center_y"])

    # Group boxes into rows based on Y-coordinate clustering
    rows = []
    current_row = [boxes_by_y[0]]

    for i in range(1, len(boxes_by_y)):
        current_box = boxes_by_y[i]
        last_box_in_row = current_row[-1]

        # Calculate dynamic tolerance based on box heights
        avg_height = np.mean([b.get("height", 100) for b in current_row + [current_box]])
        y_tolerance = avg_height * tolerance_factor

        # If Y difference is small enough, add to current row
        if abs(current_box["center_y"] - last_box_in_row["center_y"]) <= y_tolerance:
            current_row.append(current_box)
        else:
            # Start new row
            rows.append(current_row)
            current_row = [current_box]

    # Don't forget the last row
    if current_row:
        rows.append(current_row)

    print(f"Identified {len(rows)} rows:")
    for i, row in enumerate(rows):
        avg_y = np.mean([b["center_y"] for b in row])
        print(f"  Row {i+1}: {len(row)} boxes at Y≈{avg_y:.1f}")

    # Sort each row by X coordinate (left to right)
    ranked_boxes = []
    for row_idx, row in enumerate(rows):
        sorted_row = sorted(row, key=lambda b: b["center_x"])
        ranked_boxes.extend(sorted_row)

        # Debug: print row details
        x_positions = [b["center_x"] for b in sorted_row]
        print(f"  Row {row_idx+1} X positions: {[f'{x:.1f}' for x in x_positions]}")

    return ranked_boxes

def adaptive_row_detection(boxes, min_boxes_per_row=2):
    """
    Advanced row detection using density-based clustering
    """
    if len(boxes) < min_boxes_per_row:
        return boxes

    # Extract Y coordinates
    y_coords = np.array([b["center_y"] for b in boxes]).reshape(-1, 1)

    # Use DBSCAN-like approach or simple gap detection
    y_sorted = sorted([b["center_y"] for b in boxes])

    # Find gaps in Y coordinates to determine row breaks
    gaps = []
    for i in range(1, len(y_sorted)):
        gap = y_sorted[i] - y_sorted[i-1]
        gaps.append(gap)

    # Determine threshold for row separation (larger gaps = new rows)
    if gaps:
        gap_threshold = np.percentile(gaps, 60)  # Use 70th percentile as threshold
        print(f"Row separation threshold: {gap_threshold:.1f} pixels")
    else:
        gap_threshold = 50  # Default fallback

    # Group into rows based on gaps
    rows = []
    current_row = []

    boxes_by_y = sorted(boxes, key=lambda b: b["center_y"])

    for i, box in enumerate(boxes_by_y):
        if i == 0:
            current_row = [box]
        else:
            prev_box = boxes_by_y[i-1]
            y_gap = box["center_y"] - prev_box["center_y"]

            if y_gap <= gap_threshold:
                current_row.append(box)
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [box]

    # Add final row
    if current_row:
        rows.append(current_row)

    # Sort each row by X coordinate and combine
    ranked_boxes = []
    for row in rows:
        sorted_row = sorted(row, key=lambda b: b["center_x"])
        ranked_boxes.extend(sorted_row)

    return ranked_boxes

# In SCRIPT 1

def extract_ranked_boxes_from_image(pil_img, roboflow_model, output_folder, page_prefix="page1",
                                    filter_small_boxes=True, ranking_method="improved_grid",
                                    confidence_threshold=25):
    """
    Extract product boxes, rank them, save cropped images.
    Returns the list of ranked box dictionaries and paths to saved cropped files.
    Visualization is NOT created here.
    """
    # Save PIL image as a temp file for Roboflow
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_img_path = tmp.name
        pil_img.save(temp_img_path, "JPEG")

    # print(f"Running Roboflow detection on {page_prefix} with confidence threshold {confidence_threshold}%...")
    rf_result = roboflow_model.predict(temp_img_path, confidence=confidence_threshold, overlap=30)
    boxes_from_rf = rf_result.json().get("predictions", [])
    # print(f"Roboflow detected {len(boxes_from_rf)} boxes for {page_prefix}")

    sortable_boxes = []
    filtered_count = 0
    for pred in boxes_from_rf:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        cls = pred.get("class", "unknown")
        if is_valid_product_box(w, h, pil_img.width, pil_img.height):
            sortable_boxes.append({
                "pred": pred, "center_x": x, "center_y": y,
                "left": x - w / 2, "top": y - h / 2,
                "right": x + w / 2, "bottom": y + h / 2,
                "width": w, "height": h, "class": cls
            })
        else:
            filtered_count += 1
    # print(f"Kept {len(sortable_boxes)} valid product boxes for {page_prefix}, filtered out {filtered_count}")

    if not sortable_boxes:
        os.remove(temp_img_path)
        return [], [] # Return empty lists if no valid boxes

    # Apply selected ranking method
    if ranking_method == "improved_grid":
        sortable_boxes = improved_grid_ranking(sortable_boxes)
    elif ranking_method == "adaptive":
        sortable_boxes = adaptive_row_detection(sortable_boxes)
    elif ranking_method == "kmeans":
        sortable_boxes = rank_by_kmeans_rows(sortable_boxes)
    else:  # reading_order
        sortable_boxes = rank_by_reading_order(sortable_boxes)

    os.makedirs(output_folder, exist_ok=True) # For cropped images

    saved_cropped_files = []
    for idx, b in enumerate(sortable_boxes):
        pred_box = b["pred"]
        x, y, w, h = pred_box["x"], pred_box["y"], pred_box["width"], pred_box["height"]
        padding = 10
        left = max(0, int(x - w / 2) - padding)
        upper = max(0, int(y - h / 2) - padding)
        right = min(pil_img.width, int(x + w / 2) + padding)
        lower = min(pil_img.height, int(y + h / 2) + padding)
        
        if right <= left or lower <= upper: # Check for invalid crop dimensions
            # print(f"Warning: Invalid crop dimensions for box {idx+1} in {page_prefix}. Skipping crop.")
            continue

        cropped = pil_img.crop((left, upper, right, lower))
        save_path = os.path.join(output_folder, f"{page_prefix}_rank_{idx+1}.jpg")
        cropped.save(save_path, "JPEG", quality=95) # Explicit high quality for VLM
        saved_cropped_files.append(save_path)

    os.remove(temp_img_path)
    return sortable_boxes, saved_cropped_files

# In SCRIPT 1

# Replace the ENTIRE create_ranking_visualization function with this one:

def create_ranking_visualization(pil_img: Image.Image, boxes: List[Dict], output_path: str, issue_details_per_rank: Optional[Dict] = None):
    """
    Creates a visualization with PROMINENT red boxes for specific error areas and yellow outlines for products with issues.
    FIXED VERSION with proper coordinate handling and larger text.
    """
    img_copy = pil_img.copy()
    draw = ImageDraw.Draw(img_copy, "RGBA")

    # FIXED: Try to load larger fonts with fallbacks
    font = None
    label_font = None
    
    try:
        # Try different font sizes - much larger than before
        font = ImageFont.truetype("arial.ttf", 60)  # Increased from 40
        label_font = ImageFont.truetype("arialbd.ttf", 50)  # Increased from 36
        logger.info("Successfully loaded Arial fonts with large sizes")
    except IOError:
        try:
            # Try alternative font names
            font = ImageFont.truetype("Arial.ttf", 60)
            label_font = ImageFont.truetype("Arial Bold.ttf", 50)
            logger.info("Successfully loaded Arial fonts (alternative names)")
        except IOError:
            try:
                # Try Windows system fonts
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 60)
                label_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 50)
                logger.info("Successfully loaded Windows system fonts")
            except IOError:
                try:
                    # Try macOS system fonts
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 60)
                    label_font = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", 50)
                    logger.info("Successfully loaded macOS system fonts")
                except IOError:
                    # Use default fonts as final fallback
                    font = ImageFont.load_default()
                    label_font = ImageFont.load_default()
                    logger.warning("Using default fonts - text may be small")
    
    # Ensure fonts are set (should never happen, but safety check)
    if font is None:
        font = ImageFont.load_default()
    if label_font is None:
        label_font = ImageFont.load_default()

    if issue_details_per_rank is None:
        issue_details_per_rank = {}

    # Enhanced colors for better visibility
    error_box_color = (255, 0, 0, 160)  # Bright red with transparency
    error_outline_color = (255, 255, 255)  # White outline for contrast
    product_outline_color = (255, 255, 0)  # Yellow for product outline
    text_color = (255, 255, 255)  # White text
    text_stroke_color = (0, 0, 0)  # Black stroke for text
    text_bg_color = (0, 0, 0, 220)  # Semi-transparent black background

    logger.info(f"Processing {len(issue_details_per_rank)} products with issues")

    # Process only the products that have reported issues
    for rank, differences in issue_details_per_rank.items():
        if not differences:  # Skip if no issues
            continue

        logger.info(f"Processing rank {rank} with {len(differences)} differences")

        # Find the main bounding box for this rank
        if (rank - 1) < len(boxes):
            box_data = boxes[rank - 1]
            product_box_coords = [box_data["left"], box_data["top"], box_data["right"], box_data["bottom"]]
            
            # Draw THICK yellow outline around entire product with issues
            draw.rectangle(product_box_coords, outline=product_outline_color, width=8)  # Increased width
            logger.info(f"Drew yellow outline for rank {rank} at {product_box_coords}")

            # Process each specific difference within this product
            for diff_idx, diff in enumerate(differences):
                diff_type = diff.get("type", "Unknown")
                diff_box_coords = diff.get("box1")  # Use box1 for first catalog
                
                logger.info(f"Processing difference {diff_idx} for rank {rank}: type={diff_type}, box={diff_box_coords}")
                
                if not diff_box_coords:
                    logger.warning(f"No bounding box coordinates for rank {rank}, difference {diff_idx}")
                    continue

                # Validate bounding box format
                if not isinstance(diff_box_coords, list) or len(diff_box_coords) != 4:
                    logger.warning(f"Invalid bounding box format for rank {rank}: {diff_box_coords}")
                    continue

                try:
                    # FIXED: Better coordinate handling
                    x1, y1, x2, y2 = [float(coord) for coord in diff_box_coords]
                    
                    # FIXED: Check if coordinates are relative (0-1) or absolute
                    if all(0 <= coord <= 1 for coord in [x1, y1, x2, y2]):
                        # Relative coordinates - convert to absolute within the product box
                        product_width = product_box_coords[2] - product_box_coords[0]
                        product_height = product_box_coords[3] - product_box_coords[1]
                        
                        abs_x1 = product_box_coords[0] + (x1 * product_width)
                        abs_y1 = product_box_coords[1] + (y1 * product_height)
                        abs_x2 = product_box_coords[0] + (x2 * product_width)
                        abs_y2 = product_box_coords[1] + (y2 * product_height)
                        
                        abs_box = [abs_x1, abs_y1, abs_x2, abs_y2]
                        logger.info(f"Converted relative coords {diff_box_coords} to absolute {abs_box}")
                    else:
                        # FIXED: For absolute coordinates, handle them more carefully
                        product_width = product_box_coords[2] - product_box_coords[0]
                        product_height = product_box_coords[3] - product_box_coords[1]
                        
                        # If coordinates seem to be within product bounds, treat as relative to product
                        if x2 <= product_width and y2 <= product_height:
                            abs_x1 = product_box_coords[0] + x1
                            abs_y1 = product_box_coords[1] + y1
                            abs_x2 = product_box_coords[0] + x2
                            abs_y2 = product_box_coords[1] + y2
                            abs_box = [abs_x1, abs_y1, abs_x2, abs_y2]
                            logger.info(f"Treated as product-relative coords: {diff_box_coords} -> {abs_box}")
                        else:
                            # Use as absolute page coordinates
                            abs_box = [x1, y1, x2, y2]
                            logger.info(f"Using as absolute page coords: {abs_box}")
                    
                    # EXPANDED error box for better visibility
                    expansion = 25  # Increased expansion
                    abs_box[0] = max(0, abs_box[0] - expansion)  # left
                    abs_box[1] = max(0, abs_box[1] - expansion)  # top
                    abs_box[2] = min(img_copy.width, abs_box[2] + expansion)  # right
                    abs_box[3] = min(img_copy.height, abs_box[3] + expansion)  # bottom
                    
                    # FIXED: Ensure minimum box size for visibility
                    min_width, min_height = 150, 80  # Increased minimum size
                    current_width = abs_box[2] - abs_box[0]
                    current_height = abs_box[3] - abs_box[1]
                    
                    if current_width < min_width:
                        center_x = (abs_box[0] + abs_box[2]) / 2
                        abs_box[0] = max(0, center_x - min_width/2)
                        abs_box[2] = min(img_copy.width, abs_box[0] + min_width)
                    
                    if current_height < min_height:
                        center_y = (abs_box[1] + abs_box[3]) / 2
                        abs_box[1] = max(0, center_y - min_height/2)
                        abs_box[3] = min(img_copy.height, abs_box[1] + min_height)
                    
                    # Ensure coordinates are within image bounds
                    abs_box = [
                        max(0, min(abs_box[0], img_copy.width)),
                        max(0, min(abs_box[1], img_copy.height)), 
                        max(abs_box[0], min(abs_box[2], img_copy.width)),
                        max(abs_box[1], min(abs_box[3], img_copy.height))
                    ]
                    
                    # Only draw if the box has valid dimensions
                    if abs_box[2] > abs_box[0] and abs_box[3] > abs_box[1]:
                        # Draw PROMINENT red error box with thick outline
                        draw.rectangle(abs_box, fill=error_box_color, outline=error_outline_color, width=6)
                        
                        # Add a second inner outline for extra visibility
                        inner_box = [abs_box[0]+4, abs_box[1]+4, abs_box[2]-4, abs_box[3]-4]
                        if inner_box[2] > inner_box[0] and inner_box[3] > inner_box[1]:
                            draw.rectangle(inner_box, outline=(255, 0, 0), width=4)
                        
                        # FIXED: Draw error type label with much better visibility
                        label_text = diff_type
                        
                        # Position label ABOVE the error box for better visibility
                        label_x = abs_box[0]
                        label_y = max(10, abs_box[1] - 80)  # Position above the box
                        
                        # FIXED: Calculate text size properly
                        try:
                            # Try to get text bounding box
                            text_bbox = draw.textbbox((label_x, label_y), label_text, font=label_font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]
                        except:
                            # Fallback text size estimation
                            text_width = len(label_text) * 30  # Rough estimation
                            text_height = 50
                        
                        # Ensure label fits within image bounds
                        if label_x + text_width > img_copy.width:
                            label_x = max(0, img_copy.width - text_width - 10)
                        
                        # Draw larger text background for better readability
                        bg_box = [
                            label_x - 10, 
                            label_y - 10, 
                            label_x + text_width + 20, 
                            label_y + text_height + 10
                        ]
                        draw.rectangle(bg_box, fill=text_bg_color)
                        
                        # Draw the text with thick stroke and larger size
                        draw.text((label_x, label_y), label_text, fill=text_color, font=label_font, 
                                 stroke_width=4, stroke_fill=text_stroke_color)
                        
                        final_width = abs_box[2] - abs_box[0]
                        final_height = abs_box[3] - abs_box[1]
                        logger.info(f"Drew ENHANCED red error box for rank {rank}, type {diff_type} at {abs_box} (size: {final_width}x{final_height})")
                    else:
                        logger.warning(f"Invalid box dimensions after processing: {abs_box}")
                        
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing coordinates for rank {rank}: {e}")
                    continue
        else:
            logger.warning(f"No box data found for rank {rank}")

    img_copy.save(output_path, "JPEG", quality=95)
    logger.info(f"Enhanced red error box visualization saved to: {output_path}")
    logger.info(f"Total products with issues highlighted: {len(issue_details_per_rank)}")

    # ADDED: Debug information about image and text
    logger.info(f"Image dimensions: {img_copy.width}x{img_copy.height}")
    logger.info(f"Font used: {type(font)} at size 60")
    logger.info(f"Label font used: {type(label_font)} at size 50")

    

def create_ranking_visualization_fallback(pil_img: Image.Image, boxes: List[Dict], output_path: str, issue_details_per_rank: Optional[Dict] = None):
    """
    Fallback visualization when no specific bounding boxes are available - highlights entire products
    """
    img_copy = pil_img.copy()
    draw = ImageDraw.Draw(img_copy, "RGBA")

    try:
        font = ImageFont.truetype("arial.ttf", 32)
        label_font = ImageFont.truetype("arialbd.ttf", 28)
    except IOError:
        font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    if issue_details_per_rank is None:
        issue_details_per_rank = {}

    # Yellow highlighting for products with issues
    product_highlight_color = (255, 255, 0, 100)  # Yellow with transparency
    text_color = (255, 255, 255)

    for rank, differences in issue_details_per_rank.items():
        if not differences:
            continue

        if (rank - 1) < len(boxes):
            box_data = boxes[rank - 1]
            product_box_coords = [box_data["left"], box_data["top"], box_data["right"], box_data["bottom"]]
            
            # Highlight entire product in yellow
            draw.rectangle(product_box_coords, fill=product_highlight_color, outline="yellow", width=3)
            
            # Add issue count label
            issue_types = [d.get("type", "Unknown") for d in differences]
            label_text = f"Issues: {len(differences)}"
            
            label_y = product_box_coords[1] - 25
            if label_y > 0:
                draw.text((product_box_coords[0], label_y), label_text, fill=text_color, font=label_font, 
                         stroke_width=2, stroke_fill="black")

    img_copy.save(output_path, "JPEG", quality=90)
    logger.info(f"Fallback visualization saved to: {output_path}")

def validate_same_product(self, product1_path: str, product2_path: str) -> bool:
    """Optional: Quick brand validation to ensure we're comparing same products"""
    
    # Extract just brand names quickly
    brand1 = self.extract_brand_only(product1_path)
    brand2 = self.extract_brand_only(product2_path)
    
    if brand1 and brand2:
        similarity = fuzz.ratio(brand1.lower(), brand2.lower())
        return similarity > 70  # 70% similarity threshold
    
    return True  # Assume same product if can't extract brands

def process_dual_pdfs_for_comparison(pdf_path1, pdf_path2, output_root="catalog_comparison",
                                     ranking_method="improved_grid", filter_small_boxes=True,
                                     confidence_threshold=25): # Removed generate_initial_visualizations
    print("="*60)
    print("DUAL PDF PROCESSING - STEP 1 (Box Detection & Cropping)")
    print(f"Detection Confidence Threshold: {confidence_threshold}%")
    print("="*60)

    poppler_path_to_use = get_poppler_path()
    # ... (logging for poppler path) ...

    output_path = Path(output_root)
    catalog1_base_path = output_path / "catalog1"
    catalog2_base_path = output_path / "catalog2"
    catalog1_base_path.mkdir(parents=True, exist_ok=True)
    catalog2_base_path.mkdir(parents=True, exist_ok=True)

    results = {
        "catalog1_path": str(catalog1_base_path), # Base path for catalog 1 outputs (page folders)
        "catalog2_path": str(catalog2_base_path), # Base path for catalog 2 outputs
        "catalog1_files": [], # List of paths to cropped product images from catalog 1
        "catalog2_files": [], # List of paths to cropped product images from catalog 2
        "catalog1_pages": 0,
        "catalog2_pages": 0,
        "total_products_catalog1": 0,
        "total_products_catalog2": 0,
        "confidence_threshold": confidence_threshold,
        "page_level_data_catalog1": {}, # Stores {page_num: {'image_pil': PILObject, 'ranked_boxes': list, 'page_folder_path': str}}
        "page_level_data_catalog2": {},
    }

    # Process PDF 1
    print(f"\nPROCESSING PDF 1 (Box Detection): {Path(pdf_path1).name}")
    print("-" * 50)
    try:
        # For PDF 1
        logger.info(f"Converting PDF 1 ({Path(pdf_path1).name}) to images with DPI 300...")
        pages1_pil_list = convert_from_path(pdf_path1, dpi=300, poppler_path=poppler_path_to_use) # Changed DPI
        results["catalog1_pages"] = len(pages1_pil_list)
        logger.info(f"Converted PDF 1 to {len(pages1_pil_list)} pages")

        for page_idx, pil_page_image in enumerate(pages1_pil_list):
            page_num = page_idx + 1
            page_folder_for_crops = catalog1_base_path / f"page_{page_num}"
            page_folder_for_crops.mkdir(exist_ok=True)

            # print(f"\nCatalog 1 - Page {page_num} (Box Detection & Cropping)...")
            # Note: Roboflow model is passed via GLOBAL_ROBOFLOW_MODEL through main_dual_pdf_processing
            ranked_boxes_on_page, saved_cropped_on_page = extract_ranked_boxes_from_image(
                pil_page_image,
                roboflow_model=None, # Will be picked up by patched function
                output_folder=str(page_folder_for_crops), # For saving cropped images
                page_prefix=f"c1_p{page_num}",
                filter_small_boxes=filter_small_boxes,
                ranking_method=ranking_method,
                confidence_threshold=confidence_threshold
            )
            results["catalog1_files"].extend(saved_cropped_on_page)
            results["total_products_catalog1"] += len(saved_cropped_on_page)
            results["page_level_data_catalog1"][page_num] = {
                'image_pil': pil_page_image, # Store the original PIL image of the page
                'ranked_boxes': ranked_boxes_on_page, # Store the detected & ranked boxes
                'page_folder_path': str(page_folder_for_crops) # Path where cropped images for this page are
            }
            # print(f"Page {page_num}: {len(saved_cropped_on_page)} product images cropped and data stored.")

    except Exception as e:
        logger.error(f"Error processing PDF 1: {e}", exc_info=True)
        results["catalog1_error"] = str(e)

    # Process PDF 2 (similar logic)
    print(f"\nPROCESSING PDF 2 (Box Detection): {Path(pdf_path2).name}")
    print("-" * 50)
    try:
    # For PDF 2
        logger.info(f"Converting PDF 2 ({Path(pdf_path2).name}) to images with DPI 300...")
        pages2_pil_list = convert_from_path(pdf_path2, dpi=300, poppler_path=poppler_path_to_use) # Changed DPI
        results["catalog2_pages"] = len(pages2_pil_list)
        logger.info(f"Converted PDF 2 to {len(pages2_pil_list)} pages")

        for page_idx, pil_page_image in enumerate(pages2_pil_list):
            page_num = page_idx + 1
            page_folder_for_crops = catalog2_base_path / f"page_{page_num}"
            page_folder_for_crops.mkdir(exist_ok=True)

            # print(f"\nCatalog 2 - Page {page_num} (Box Detection & Cropping)...")
            ranked_boxes_on_page, saved_cropped_on_page = extract_ranked_boxes_from_image(
                pil_page_image,
                roboflow_model=None, # Will be picked by patched function
                output_folder=str(page_folder_for_crops),
                page_prefix=f"c2_p{page_num}",
                filter_small_boxes=filter_small_boxes,
                ranking_method=ranking_method,
                confidence_threshold=confidence_threshold
            )
            results["catalog2_files"].extend(saved_cropped_on_page)
            results["total_products_catalog2"] += len(saved_cropped_on_page)
            results["page_level_data_catalog2"][page_num] = {
                'image_pil': pil_page_image,
                'ranked_boxes': ranked_boxes_on_page,
                'page_folder_path': str(page_folder_for_crops)
            }
            # print(f"Page {page_num}: {len(saved_cropped_on_page)} product images cropped and data stored.")
    except Exception as e:
        logger.error(f"Error processing PDF 2: {e}", exc_info=True)
        results["catalog2_error"] = str(e)

    

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Catalog 1 ({Path(pdf_path1).name}):")
    print(f"Pages: {results['catalog1_pages']}")
    print(f"Total Products: {results['total_products_catalog1']}")
    print(f"Output: {results['catalog1_path']}")

    print(f"\nCatalog 2 ({Path(pdf_path2).name}):")
    print(f"Pages: {results['catalog2_pages']}")
    print(f"Total Products: {results['total_products_catalog2']}")
    print(f"Output: {results['catalog2_path']}")

    print(f"\nRanking Method Used: {ranking_method}")
    print(f"Small Box Filtering: {'Enabled' if filter_small_boxes else 'Disabled'}")
    print(f"Detection Confidence: {confidence_threshold}%")  # Show threshold
    print(f"Poppler Path Used: {poppler_path_to_use if poppler_path_to_use else 'System PATH'}")

    return results

# Legacy functions for backward compatibility
def rank_by_kmeans_rows(boxes):
    """Use K-means clustering to identify rows, then sort within rows"""
    if len(boxes) < 2:
        return boxes

    y_coords = np.array([b["center_y"] for b in boxes]).reshape(-1, 1)
    n_boxes = len(boxes)
    estimated_cols = int(math.sqrt(n_boxes))
    estimated_rows = math.ceil(n_boxes / estimated_cols)
    n_clusters = min(estimated_rows, n_boxes)

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        row_labels = kmeans.fit_predict(y_coords)

        rows = {}
        for i, box in enumerate(boxes):
            row_id = row_labels[i]
            if row_id not in rows:
                rows[row_id] = []
            rows[row_id].append(box)

        sorted_row_ids = sorted(rows.keys(),
                               key=lambda rid: np.mean([b["center_y"] for b in rows[rid]]))

        ranked_boxes = []
        for row_id in sorted_row_ids:
            row_boxes = sorted(rows[row_id], key=lambda b: b["center_x"])
            ranked_boxes.extend(row_boxes)

        return ranked_boxes

    except Exception as e:
        print(f"K-means clustering failed: {e}")
        return rank_by_reading_order(boxes)

def rank_by_reading_order(boxes):
    """Rank boxes in reading order using distance-based approach"""
    if not boxes:
        return boxes

    remaining_boxes = boxes.copy()
    ranked_boxes = []

    start_box = min(remaining_boxes, key=lambda b: b["center_y"] + b["center_x"] * 0.1)
    ranked_boxes.append(start_box)
    remaining_boxes.remove(start_box)

    while remaining_boxes:
        current_box = ranked_boxes[-1]
        best_box = None
        best_score = float('inf')

        for box in remaining_boxes:
            dx = box["center_x"] - current_box["center_x"]
            dy = box["center_y"] - current_box["center_y"]

            if abs(dy) < 50:  # Same row threshold
                score = abs(dy) + max(0, -dx) * 2
            else:
                score = dy + abs(box["center_x"] - min(b["center_x"] for b in boxes)) * 0.1

            if score < best_score:
                best_score = score
                best_box = box

        if best_box:
            ranked_boxes.append(best_box)
            remaining_boxes.remove(best_box)

    return ranked_boxes

# Main function for easy usage
# In SCRIPT 1
def main_dual_pdf_processing(pdf_path1, pdf_path2, roboflow_model,
                             output_root="catalog_comparison", ranking_method="improved_grid",
                             confidence_threshold=25): # Removed generate_initial_visualizations
    global GLOBAL_ROBOFLOW_MODEL
    GLOBAL_ROBOFLOW_MODEL = roboflow_model

    original_extract = globals().get('extract_ranked_boxes_from_image')
    # It's good practice to ensure original_extract is not None before proceeding
    if original_extract is None:
        # This should not happen if the script is intact
        raise RuntimeError("extract_ranked_boxes_from_image function not found in globals.")


    def patched_extract(*args, **kwargs):
        if kwargs.get('roboflow_model') is None:
            kwargs['roboflow_model'] = GLOBAL_ROBOFLOW_MODEL
        # Ensure original_extract is callable
        if callable(original_extract):
            return original_extract(*args, **kwargs)
        else:
            # Handle error: original_extract is not what we expect
            raise TypeError("The original extract_ranked_boxes_from_image is not callable.")


    globals()['extract_ranked_boxes_from_image'] = patched_extract

    try:
        results = process_dual_pdfs_for_comparison(
            pdf_path1, pdf_path2, output_root, ranking_method,
            confidence_threshold=confidence_threshold
            # generate_initial_visualizations is removed
        )
        return results
    finally:
        globals()['extract_ranked_boxes_from_image'] = original_extract
        if 'GLOBAL_ROBOFLOW_MODEL' in globals():
            del globals()['GLOBAL_ROBOFLOW_MODEL']

# ========================================
# SCRIPT 2: Image Template Matching
# ========================================

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image using OpenCV.

    Args:
        image_path: Path to the image file

    Returns:
        Image as numpy array or None if failed to load
    """
    try:
        image = cv2.imread(image_path)
        if image is not None:
            # Convert BGR to RGB for consistency
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def resize_image_for_comparison(image: np.ndarray, max_size: int = 500) -> np.ndarray:
    """
    Resize image for faster comparison while maintaining aspect ratio.

    Args:
        image: Input image as numpy array
        max_size: Maximum dimension size

    Returns:
        Resized image
    """
    height, width = image.shape[:2]

    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image

def calculate_image_similarity(img1: np.ndarray, img2: np.ndarray, method: str = "structural") -> float:
    """
    Calculate similarity between two images using different methods.

    Args:
        img1: First image
        img2: Second image
        method: Comparison method ("structural", "histogram", "template")

    Returns:
        Similarity score (0.0 to 1.0, where 1.0 is identical)
    """
    # Resize images to same size for comparison
    img1_resized = resize_image_for_comparison(img1)
    img2_resized = resize_image_for_comparison(img2)

    # Resize both to the same dimensions
    target_size = (300, 300)
    img1_resized = cv2.resize(img1_resized, target_size)
    img2_resized = cv2.resize(img2_resized, target_size)

    if method == "structural":
        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)

        # Calculate Structural Similarity Index
        from skimage.metrics import structural_similarity
        similarity = structural_similarity(gray1, gray2)
        return max(0.0, similarity)  # Ensure non-negative

    elif method == "histogram":
        # Calculate histogram similarity
        hist1 = cv2.calcHist([img1_resized], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2_resized], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])

        # Normalize histograms
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        # Compare histograms using correlation
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0.0, similarity)

    elif method == "template":
        # Template matching
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)

        # Use the smaller image as template
        if gray1.size < gray2.size:
            template, image = gray1, gray2
        else:
            template, image = gray2, gray1

        # Perform template matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        return max(0.0, max_val)

    # Fallback: Mean Squared Error based similarity
    mse = np.mean((img1_resized.astype(float) - img2_resized.astype(float)) ** 2)
    similarity = 1.0 / (1.0 + mse / 1000.0)  # Normalize MSE
    return similarity

def find_matching_images_by_template(template_paths: List[str], folder_path: str,
                                   similarity_threshold: float = 0.8,
                                   comparison_method: str = "structural") -> List[str]:
    """
    Find images in folder that match the template images using image comparison.

    Args:
        template_paths: List of paths to template images
        folder_path: Path to folder containing images to search
        similarity_threshold: Minimum similarity score to consider a match (0.0 to 1.0)
        comparison_method: Method to use for comparison ("structural", "histogram", "template")

    Returns:
        List of matching image filenames
    """
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found!")
        return []

    # Load template images
    template_images = []
    for template_path in template_paths:
        if os.path.exists(template_path):
            template_img = load_image(template_path)
            if template_img is not None:
                template_images.append((template_path, template_img))
                print(f"Loaded template: {os.path.basename(template_path)}")
            else:
                print(f"Failed to load template: {template_path}")
        else:
            print(f"Template not found: {template_path}")

    if not template_images:
        print("No valid template images loaded!")
        return []

    # Get all image files in folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    folder_images = []

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            folder_images.append(file)

    print(f"Found {len(folder_images)} images in folder to compare")

    # Find matching images
    matching_images = []

    for folder_image in folder_images:
        folder_image_path = os.path.join(folder_path, folder_image)
        folder_img = load_image(folder_image_path)

        if folder_img is None:
            continue

        # Compare with each template
        for template_path, template_img in template_images:
            try:
                similarity = calculate_image_similarity(template_img, folder_img, comparison_method)

                print(f"Comparing {os.path.basename(template_path)} vs {folder_image}: {similarity:.3f}")

                if similarity >= similarity_threshold:
                    matching_images.append(folder_image)
                    print(f"  ✓ MATCH FOUND! Similarity: {similarity:.3f}")
                    break  # Don't check other templates for this image

            except Exception as e:
                print(f"Error comparing {template_path} with {folder_image}: {e}")

    return matching_images

def extract_number_from_filename(filename: str) -> Optional[int]:
    """Extract number from filename for sorting purposes."""
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else None

def get_base_name_pattern(filename: str) -> str:
    """Extract base name pattern without the number."""
    name_without_ext = os.path.splitext(filename)[0]
    numbers = re.findall(r'\d+', name_without_ext)
    if numbers:
        last_number = numbers[-1]
        last_num_pos = name_without_ext.rfind(last_number)
        return name_without_ext[:last_num_pos]
    return name_without_ext

def rename_images_sequentially(folder_path: str, remaining_images: List[str]):
    """
    Rename all remaining images sequentially starting from 1.
    Uses temporary files to avoid conflicts and prevent creating duplicates.

    Args:
        folder_path: Path to the folder containing images
        remaining_images: List of remaining image filenames after deletion
    """
    if not remaining_images:
        print(f"No remaining images to rename in {os.path.basename(folder_path)}")
        return

    # Sort remaining images by their current numbers
    images_with_numbers = []
    for img in remaining_images:
        num = extract_number_from_filename(img)
        if num is not None:
            images_with_numbers.append((img, num))

    # Sort by current number
    images_with_numbers.sort(key=lambda x: x[1])

    print(f"\nRenaming {len(images_with_numbers)} images sequentially...")

    # Create rename operations for sequential numbering (1, 2, 3, ...)
    rename_operations = []

    for new_index, (img, old_number) in enumerate(images_with_numbers, 1):
        if old_number != new_index:  # Only rename if number needs to change
            base_pattern = get_base_name_pattern(img)
            extension = os.path.splitext(img)[1]

            # Determine number format (preserve padding style)
            # Check if original files used zero-padding
            has_zero_padding = any(
                re.search(r'0\d+', os.path.splitext(original_img)[0])
                for original_img, _ in images_with_numbers
            )

            if has_zero_padding:
                # Use zero-padding to match original style
                max_digits = len(str(len(images_with_numbers)))
                new_num_str = f"{new_index:0{max_digits}d}"
            else:
                new_num_str = str(new_index)

            new_filename = f"{base_pattern}{new_num_str}{extension}"

            old_path = os.path.join(folder_path, img)
            new_path = os.path.join(folder_path, new_filename)

            rename_operations.append((old_path, new_path, img, new_filename))

    # Execute rename operations using temporary files to prevent conflicts
    successful_renames = 0
    temp_files = []  # Track temporary files for cleanup

    try:
        # Step 1: Move files to temporary names first to avoid conflicts
        for old_path, new_path, old_name, new_name in rename_operations:
            try:
                # Create a unique temporary filename
                temp_dir = os.path.dirname(old_path)
                temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir, suffix=os.path.splitext(old_path)[1])
                os.close(temp_fd)  # Close the file descriptor
                os.unlink(temp_path)  # Remove the empty temp file

                # Move original file to temp location
                os.rename(old_path, temp_path)
                temp_files.append((temp_path, new_path, old_name, new_name))
                print(f"Staged for rename: {old_name} -> {new_name}")

            except Exception as e:
                print(f"Error staging {old_name}: {e}")

        # Step 2: Move from temporary names to final names
        for temp_path, new_path, old_name, new_name in temp_files:
            try:
                # Double-check the target doesn't exist
                if os.path.exists(new_path):
                    print(f"Warning: Target file {new_name} already exists, skipping rename of {old_name}")
                    # Move back to original name to avoid losing the file
                    original_path = os.path.join(folder_path, old_name)
                    if not os.path.exists(original_path):
                        os.rename(temp_path, original_path)
                    continue

                os.rename(temp_path, new_path)
                print(f"Renamed: {old_name} -> {new_name}")
                successful_renames += 1

            except Exception as e:
                print(f"Error finalizing rename of {old_name}: {e}")
                # Try to restore original file
                try:
                    original_path = os.path.join(folder_path, old_name)
                    if not os.path.exists(original_path):
                        os.rename(temp_path, original_path)
                        print(f"Restored original file: {old_name}")
                except Exception as restore_error:
                    print(f"Error restoring {old_name}: {restore_error}")

    except Exception as e:
        print(f"Critical error during renaming process: {e}")

    finally:
        # Cleanup any remaining temporary files
        for temp_path, new_path, old_name, new_name in temp_files:
            if os.path.exists(temp_path):
                try:
                    # Try to restore to original name if something went wrong
                    original_path = os.path.join(folder_path, old_name)
                    if not os.path.exists(original_path) and not os.path.exists(new_path):
                        os.rename(temp_path, original_path)
                        print(f"Cleanup: Restored {old_name}")
                    else:
                        os.unlink(temp_path)
                        print(f"Cleanup: Removed temp file for {old_name}")
                except Exception as cleanup_error:
                    print(f"Cleanup error for {old_name}: {cleanup_error}")

    print(f"Successfully renamed {successful_renames} images sequentially")

def process_folders_with_image_templates(template1_path: str, template2_path: str, template3_path: str,
                                       folder1_path: str, folder2_path: str,
                                       similarity_threshold: float = 0.55,
                                       comparison_method: str = "structural",
                                       dry_run: bool = True):
    """
    Process two folders using image template matching to find and delete similar images,
    then rename remaining images sequentially.

    Args:
        template1_path: Path to first template image
        template2_path: Path to second template image
        template3_path: Path to third template image
        folder1_path: Path to first folder to process
        folder2_path: Path to second folder to process
        similarity_threshold: Minimum similarity score for match (0.0 to 1.0)
        comparison_method: Method for comparison ("structural", "histogram", "template")
        dry_run: If True, only show what would be done
    """
    template_paths = [template1_path, template2_path, template3_path]
    folder_paths = [folder1_path, folder2_path]

    print(f"Image Template Matching Process with Sequential Renaming")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Comparison method: {comparison_method}")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print("-" * 60)

    print("Template images:")
    for i, template_path in enumerate(template_paths, 1):
        print(f"  {i}. {template_path}")
    print()

    # Process each folder
    for folder_idx, folder_path in enumerate(folder_paths, 1):
        print(f"Processing Folder {folder_idx}: {folder_path}")
        print("-" * 40)

        # Get all images in folder before processing
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        all_images = [f for f in os.listdir(folder_path)
                     if any(f.lower().endswith(ext) for ext in image_extensions)]

        print(f"Found {len(all_images)} images in folder")

        # Find matching images using template matching
        matching_images = find_matching_images_by_template(
            template_paths, folder_path, similarity_threshold, comparison_method
        )

        if not matching_images:
            print(f"No matching images found in folder {folder_idx}")
            # Still rename sequentially even if no deletions
            if not dry_run:
                print("Renaming all images sequentially...")
                rename_images_sequentially(folder_path, all_images)
            else:
                print("DRY RUN - Would rename all images sequentially")
            print()
            continue

        print(f"\nFound {len(matching_images)} matching images to delete:")
        for img in matching_images:
            print(f"  - {img}")

        # Calculate remaining images after deletion
        remaining_images = [img for img in all_images if img not in matching_images]

        if dry_run:
            print(f"\nDRY RUN - Would perform the following:")
            print(f"1. Delete {len(matching_images)} matching images")
            print(f"2. Rename {len(remaining_images)} remaining images sequentially (1, 2, 3, ...)")
            print(f"\nRemaining images after deletion would be:")
            for i, img in enumerate(remaining_images, 1):
                print(f"  {i}. {img}")
        else:
            # Delete matching images
            deleted_count = 0
            actually_deleted = []

            for img in matching_images:
                img_path = os.path.join(folder_path, img)
                try:
                    os.remove(img_path)
                    deleted_count += 1
                    actually_deleted.append(img)
                    print(f"Deleted: {img}")
                except Exception as e:
                    print(f"Error deleting {img}: {e}")

            # Get fresh list of current images from folder after deletions
            current_images = [f for f in os.listdir(folder_path)
                            if any(f.lower().endswith(ext) for ext in image_extensions)]

            # Rename remaining images sequentially
            if current_images:
                rename_images_sequentially(folder_path, current_images)
            else:
                print("No images remaining after deletion")

            print(f"Completed folder {folder_idx}: Deleted {deleted_count} images, renamed remaining sequentially")

        print()

    print("Processing complete!")

# ========================================
# SCRIPT 3: VLM Catalog Comparison
# ========================================

class PracticalCatalogComparator:
    """
    Practical VLM-based catalog comparison focused on what matters:
    - Price differences
    - Different products/brands
    - Ignores minor text variations
    """

    def __init__(self, openai_api_key: str, vlm_model: str = "gpt-4o", price_tolerance: float = 0.01):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.vlm_model = vlm_model
        self.price_tolerance = price_tolerance  # Allow small price differences

        # Enhanced brand normalization for fuzzy matching
        self.brand_corrections = {
            'downy': 'Downy', 'gain': 'Gain', 'tide': 'Tide', 'glad': 'Glad',
            'scott': 'Scott', 'raid': 'Raid', 'ace': 'Ace', 'purex': 'Purex',
            'clorox': 'Clorox', 'bounty': 'Bounty', 'lysol': 'Lysol',
            'palmolive': 'Palmolive', 'ajax': 'Ajax', 'dawn': 'Dawn',
            'charmin': 'Charmin', 'kleenex': 'Kleenex', 'febreze': 'Febreze'
        }

        # Enhanced unit normalizations
        self.unit_normalizations = {
            'onzas': 'oz', 'onza': 'oz', 'ozs': 'oz', 'ounces': 'oz',
            'libras': 'lb', 'libra': 'lb', 'lbs': 'lb', 'pounds': 'lb',
            'galones': 'gal', 'galon': 'gal', 'gallons': 'gal',
            'litro': 'L', 'litros': 'L', 'lt': 'L', 'liter': 'L',
            'mililitros': 'mL', 'ml': 'mL', 'milliliters': 'mL',
            'gramos': 'g', 'gramo': 'g', 'grams': 'g',
            'rollos': 'rolls', 'rollo': 'roll',
            'ct': 'ct', 'count': 'ct', 'unidades': 'ct', 'piezas': 'ct'
        }

    def load_ranked_images_from_folder(self, folder_path: str) -> List[Dict]:
        """Load ranked images from folder"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image_files = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        for file_path in folder_path.iterdir():
            if file_path.suffix.lower() in supported_extensions:
                # Enhanced rank extraction patterns
                rank_patterns = [
                    r'rank[_\s]*(\d+)',
                    r'(\d+)[_\s]*rank',
                    r'position[_\s]*(\d+)',
                    r'(?:^|[_\s])(\d+)(?:[_\s]|$)'
                ]

                rank = None
                for pattern in rank_patterns:
                    rank_match = re.search(pattern, file_path.stem, re.IGNORECASE)
                    if rank_match:
                        rank = int(rank_match.group(1))
                        break

                if rank is None:
                    rank = len(image_files) + 1

                image_files.append({
                    'file_path': str(file_path),
                    'filename': file_path.name,
                    'rank': rank,
                    'folder_name': folder_path.name
                })

        image_files.sort(key=lambda x: x['rank'])
        logger.info(f"Loaded {len(image_files)} ranked images from {folder_path}")
        return image_files

    def extract_product_data_with_vlm(self, image_path: str, image_rank: int,
                                     catalog_name: str) -> Dict:
        """Extract focused product data - only what matters for comparison"""
        item_id_for_log = f"{catalog_name}-Rank{image_rank}"

        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Focused VLM prompt - only extract what we need for comparison
            # In find_differences_with_vlm, replace the entire system_prompt variable with this:

            system_prompt = """
            You are a precise visual comparison inspector. Your task is to compare two product images and identify key differences in three specific categories.

            Your rules are:
            1.  **Price Difference**: If the main offer prices are numerically different, report this.
            2.  **Text Difference**: Focus ONLY on the primary Brand Name and the product's listed Size/Count (e.g., "50 oz", "12 rolls", "8 pack"). Report a difference if these do not match. You MUST ignore minor changes in descriptive words, slogans, or word order.
            3.  **Image Difference**: Report this ONLY if the product shown is fundamentally different (e.g., a bottle vs. a box, a different product line). You MUST ignore small changes in angle, lighting, or position.

            For EACH valid difference you find based on these rules, return a JSON object with:
            - "type": "Price", "Text", or "Image".
            - "box1": The bounding box [x1, y1, x2, y2] of the specific difference in the FIRST image.
            - "box2": The bounding box [x1, y1, x2, y2] of the specific difference in the SECOND image.
            - "description": A brief explanation (e.g., "$12.97 vs $14.97", "Brand: Tide vs Gain", "Size: 50 oz vs 75 oz").

            If you find no differences according to these specific rules, return an empty list. Respond ONLY with a valid JSON object formatted as: {"differences": []}
            """
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]

            logger.info(f"ITEM_ID: {item_id_for_log} - Extracting focused product data")

            response = self.openai_client.chat.completions.create(
                model=self.vlm_model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0.1
            )

            response_content = response.choices[0].message.content

            if response_content is None:
                return {"error_message": "VLM returned no content", "item_id": item_id_for_log}

            extracted_data = json.loads(response_content)

            # Ensure expected fields exist
            default_fields = {
                'offer_price': None,
                'regular_price': None,
                'product_brand': None,
                'product_type': None,
                'size_quantity': None,
                'product_status': 'Product Present',
                'confidence_score': 5
            }

            for field, default_value in default_fields.items():
                if field not in extracted_data:
                    extracted_data[field] = default_value

            # Parse prices
            extracted_data['offer_price'] = self.parse_price_string(
                extracted_data.get('offer_price'), f"{item_id_for_log}-offer"
            )
            extracted_data['regular_price'] = self.parse_price_string(
                extracted_data.get('regular_price'), f"{item_id_for_log}-regular"
            )

            # Add metadata
            extracted_data.update({
                'item_id': item_id_for_log,
                'image_path': image_path,
                'rank': image_rank,
                'catalog_name': catalog_name,
                'filename': Path(image_path).name
            })

            # Normalize brand
            extracted_data = self.normalize_product_data(extracted_data)

            logger.info(f"ITEM_ID: {item_id_for_log} - Extraction complete: "
                       f"Brand: {extracted_data.get('product_brand', 'No Brand')}, "
                       f"Price: ${extracted_data.get('offer_price', 'No Price')}, "
                       f"Size: {extracted_data.get('size_quantity', 'No Size')}, "
                       f"Confidence: {extracted_data.get('confidence_score', 0)}/10")

            return extracted_data

        except Exception as e:
            logger.error(f"ITEM_ID: {item_id_for_log} - Extraction error: {e}")
            return {"error_message": str(e), "item_id": item_id_for_log}

    def parse_price_string(self, price_input, item_id_for_log="N/A"):
        """Enhanced price parsing for retail formats"""
        if price_input is None or price_input == "":
            return None

        if isinstance(price_input, (int, float)):
            if price_input < 0:
                return None
            if isinstance(price_input, int) and 100 <= price_input <= 99999:
                s_price = str(price_input)
                if len(s_price) == 3:
                    return float(f"{s_price[0]}.{s_price[1:]}")
                if len(s_price) == 4:
                    return float(f"{s_price[:2]}.{s_price[2:]}")
            return float(price_input)

        price_str = str(price_input).strip()

        # Remove currency symbols and clean
        price_str = re.sub(r'[$¢€£]', '', price_str)
        price_str = re.sub(r'[^\d\s\.]', '', price_str)
        price_str = price_str.strip()

        if not price_str:
            return None

        # Space separated (e.g., "8 87", "10 49")
        space_match = re.match(r'^(\d{1,2})\s+(\d{2})(?:\s*(?:c/u|each|ea)?.*)?$', price_str)
        if space_match:
            return float(f"{space_match.group(1)}.{space_match.group(2)}")

        # 3-digit (e.g., "887" -> 8.87)
        if re.fullmatch(r'[1-9]\d{2}', price_str):
            return float(f"{price_str[0]}.{price_str[1:]}")

        # 4-digit (e.g., "1097" -> 10.97)
        if re.fullmatch(r'[1-9]\d{3}', price_str):
            return float(f"{price_str[:2]}.{price_str[2:]}")

        # Standard decimal
        decimal_match = re.search(r'(\d+)\.(\d{1,2})', price_str)
        if decimal_match:
            return float(f"{decimal_match.group(1)}.{decimal_match.group(2)}")

        return None

    def normalize_product_data(self, product_data):
        """Normalize product data for better comparison"""
        if not isinstance(product_data, dict):
            return product_data

        # Safe brand normalization
        if product_data.get('product_brand'):
            brand_raw = product_data['product_brand']
            if isinstance(brand_raw, str):
                brand_lower = brand_raw.lower().strip()
                product_data['product_brand'] = self.brand_corrections.get(
                    brand_lower, product_data['product_brand']
                )

        return product_data

    def are_brands_same_product(self, brand1: str, brand2: str) -> Tuple[bool, float]:
        """Check if two brands represent the same product using fuzzy matching"""
        if not brand1 or not brand2:
            return False, 0

        # Normalize brands for comparison
        b1 = str(brand1).lower().strip()
        b2 = str(brand2).lower().strip()

        # Exact match
        if b1 == b2:
            return True, 100

        # Remove common words that don't affect product identity
        common_words = ['simply', 'ultra', 'advanced', 'new', 'improved', 'original', 'classic']

        def clean_brand(brand):
            words = brand.split()
            return ' '.join([w for w in words if w not in common_words])

        b1_clean = clean_brand(b1)
        b2_clean = clean_brand(b2)

        # Check if core brand names match
        scores = [
            fuzz.ratio(b1_clean, b2_clean),
            fuzz.partial_ratio(b1_clean, b2_clean),
            fuzz.token_sort_ratio(b1_clean, b2_clean)
        ]

        max_score = max(scores)

        # More lenient threshold - 80% similarity means same product
        is_same = max_score >= 80

        logger.debug(f"Brand comparison: '{brand1}' vs '{brand2}' -> {max_score}% (Same: {is_same})")

        return is_same, max_score

    def generate_practical_comparison(self, folder1_path: str, folder2_path: str,
                           catalog1_name: str = None, catalog2_name: str = None) -> Dict:
        """Generate practical comparison focused on what matters with enhanced debugging"""
        if not catalog1_name:
            catalog1_name = Path(folder1_path).name
        if not catalog2_name:
            catalog2_name = Path(folder2_path).name

        logger.info(f"Starting practical comparison: {catalog1_name} vs {catalog2_name}")
        logger.info(f"Focus: Position-based comparison for quality control")

        # Load and process images
        catalog1_images = self.load_ranked_images_from_folder(folder1_path)
        catalog2_images = self.load_ranked_images_from_folder(folder2_path)

        # Create lookup dictionaries by rank for easy access
        cat1_by_rank = {img['rank']: img for img in catalog1_images}
        cat2_by_rank = {img['rank']: img for img in catalog2_images}

        comparison_rows = []
        
        # Get maximum rank to compare
        max_rank = max(
            max(cat1_by_rank.keys()) if cat1_by_rank else 0,
            max(cat2_by_rank.keys()) if cat2_by_rank else 0
        )
        
        logger.info(f"Comparing {max_rank} positions between catalogs")

        # POSITION-BASED COMPARISON (no brand matching needed)
        for rank in range(1, max_rank + 1):
            product1_info = cat1_by_rank.get(rank)  # Product at this position in catalog 1
            product2_info = cat2_by_rank.get(rank)  # Product at this position in catalog 2
            
            # Create comparison row using existing function
            row = self.create_practical_comparison_row(
                product1_info, product2_info, rank, catalog1_name, catalog2_name
            )
            if row:
                comparison_rows.append(row)
                
                # Debug logging for each row
                result_status = row.get("comparison_result", "UNKNOWN")
                issue_type = row.get("issue_type", "N/A")
                differences_count = len(row.get("granular_differences", []))
                
                if result_status == "INCORRECT":
                    logger.info(f"Position {rank}: {result_status} - {issue_type} - {differences_count} differences")

        result = {
            "catalog1_name": catalog1_name,
            "catalog2_name": catalog2_name,
            "comparison_rows": comparison_rows,
            "catalog1_total_products": len(catalog1_images),
            "catalog2_total_products": len(catalog2_images),
            "total_positions_compared": max_rank,
            "comparison_criteria": {
                "comparison_type": "Position-based Quality Control",
                "focus": "Position-to-position comparison for production error detection"
            }
        }

        # ADD DEBUG CALL HERE
        debug_summary = self.debug_comparison_results(result)
        
        logger.info(f"Position-based comparison complete. Generated {len(comparison_rows)} comparison rows.")
        logger.info(f"Debug summary: {debug_summary['correct_count']} correct, {debug_summary['incorrect_count']} incorrect")
        
        return result

    def create_practical_comparison_row(self, product1: Dict, product2: Dict, rank: int,
                             catalog1_name: str, catalog2_name: str) -> Dict:
        """Create comparison row for position-based comparison with CORRECTED status logic"""

        # Handle missing products at this position
        if not product1 and not product2:
            return None

        if not product1:
            return {
                f"{catalog1_name}_details": "MISSING PRODUCT",
                f"{catalog2_name}_details": f"Position {rank} - Product Present",
                "comparison_result": "INCORRECT",  # FIXED: Was missing
                "issue_type": "Missing Product",
                "details": f"Product missing in {catalog1_name} at position {rank}",
                "granular_differences": []
            }

        if not product2:
            return {
                f"{catalog1_name}_details": f"Position {rank} - Product Present",
                f"{catalog2_name}_details": "MISSING PRODUCT", 
                "comparison_result": "INCORRECT",  # FIXED: Was missing
                "issue_type": "Missing Product",
                "details": f"Product missing in {catalog2_name} at position {rank}",
                "granular_differences": []
            }

        # Both products exist at this position - compare them directly
        item_id_for_log = f"Position{rank}"
        
        # Call VLM to find specific differences
        differences = self.find_differences_with_vlm(
            product1['file_path'],
            product2['file_path'],
            item_id_for_log
        )

        if not differences:
            # No differences found - products are identical
            return {
                f"{catalog1_name}_details": f"Position {rank} - Product Present",
                f"{catalog2_name}_details": f"Position {rank} - Product Present",
                "comparison_result": "CORRECT",
                "issue_type": "Match Confirmed",
                "details": "Products are identical at this position",
                "granular_differences": []
            }
        else:
            # FIXED: Differences found - categorize them properly
            issue_types = []
            detailed_descriptions = []
            
            for diff in differences:
                diff_type = diff.get('type', 'Unknown Error')
                diff_desc = diff.get('description', 'No description')
                
                issue_types.append(diff_type)
                detailed_descriptions.append(f"{diff_type}: {diff_desc}")
            
            # Create combined issue type and details
            combined_issue_type = ", ".join(sorted(list(set(issue_types))))
            combined_details = "; ".join(detailed_descriptions)
            
            return {
                f"{catalog1_name}_details": f"Position {rank} - Product Present",
                f"{catalog2_name}_details": f"Position {rank} - Product Present",
                "comparison_result": "INCORRECT",  # FIXED: Ensure this is set correctly
                "issue_type": combined_issue_type,
                "details": combined_details,
                "granular_differences": differences  # FIXED: Store the full differences
            }

    # Also add this debugging function to help track what's happening
    def debug_comparison_results(self, comparison_result: Dict):
        """Debug function to log what's in the comparison results"""
        comparison_rows = comparison_result.get("comparison_rows", [])
        
        logger.info(f"=== COMPARISON RESULTS DEBUG ===")
        logger.info(f"Total comparison rows: {len(comparison_rows)}")
        
        correct_count = 0
        incorrect_count = 0
        
        for i, row in enumerate(comparison_rows):
            result = row.get("comparison_result", "UNKNOWN")
            issue_type = row.get("issue_type", "N/A")
            differences = row.get("granular_differences", [])
            
            if result == "CORRECT":
                correct_count += 1
            elif result == "INCORRECT":
                incorrect_count += 1
                logger.info(f"  Row {i+1}: {result} - {issue_type} - {len(differences)} differences")
            else:
                logger.warning(f"  Row {i+1}: UNKNOWN STATUS '{result}' - {issue_type}")
        
        logger.info(f"Summary: {correct_count} CORRECT, {incorrect_count} INCORRECT")
        logger.info(f"=== END DEBUG ===")
        
        return {
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "total_count": len(comparison_rows)
        }

    def format_product_display(self, product: Dict) -> str:
        """Format product for display with safe None handling"""
        if not product:
            return "Product Missing"

        # Safe brand extraction
        brand_raw = product.get('product_brand')
        brand = brand_raw if brand_raw else 'Unknown Brand'

        offer_price = product.get('offer_price')
        size_raw = product.get('size_quantity')

        price_display = f"${offer_price}" if offer_price is not None else "No Price"
        size = size_raw if size_raw else 'Unknown Size'

        return f"{brand} - {price_display} - {size}"

    def export_practical_comparison(self, comparison_result: Dict, output_path: str):
        """Export practical comparison results with CORRECTED summary calculations"""

        # Create main comparison DataFrame
        df = pd.DataFrame(comparison_result["comparison_rows"])

        # Define columns
        catalog1_name = comparison_result["catalog1_name"]
        catalog2_name = comparison_result["catalog2_name"]

        column_order = [
            f"{catalog1_name}_details",
            f"{catalog2_name}_details",
            "comparison_result",
            "issue_type",
            "details"
        ]

        # Ensure all columns exist
        for col in column_order:
            if col not in df.columns:
                df[col] = "N/A"

        # Only include columns that actually exist
        existing_columns = [col for col in column_order if col in df.columns]
        df = df.reindex(columns=existing_columns)

        # Rename columns
        new_column_names = [
            f"{catalog1_name} Details",
            f"{catalog2_name} Details",
            "Comparison Result",
            "Issue Type",
            "Details"
        ]
        df.columns = new_column_names[:len(existing_columns)]

        # CORRECTED SUMMARY CALCULATIONS
        comparison_rows = comparison_result["comparison_rows"]
        total_rows = len(comparison_rows)
        
        # Initialize counters
        correct_matches = 0
        incorrect_matches = 0
        missing_products = 0
        
        # Count issue types
        price_issues = 0
        text_issues = 0
        image_issues = 0
        
        # Process each comparison row
        for row in comparison_rows:
            comparison_result_status = row.get("comparison_result", "")
            issue_type = row.get("issue_type", "")
            details = row.get("details", "")
            granular_differences = row.get("granular_differences", [])
            
            # Count by result status
            if comparison_result_status == "CORRECT":
                correct_matches += 1
            elif comparison_result_status == "INCORRECT":
                incorrect_matches += 1
                
                # FIXED: Check for missing products first
                if "Missing Product" in issue_type or "missing" in details.lower():
                    missing_products += 1
                    logger.debug(f"Found missing product: {issue_type} - {details}")
                
                # Count specific issue types from granular differences
                if granular_differences:
                    for diff in granular_differences:
                        diff_type = diff.get("type", "").lower()
                        if "price" in diff_type:
                            price_issues += 1
                            logger.debug(f"Found price issue: {diff_type}")
                        elif "text" in diff_type:
                            text_issues += 1
                            logger.debug(f"Found text issue: {diff_type}")
                        elif "image" in diff_type:
                            image_issues += 1
                            logger.debug(f"Found image issue: {diff_type}")
                else:
                    # FIXED: Better fallback logic for issue_type field
                    # Only count if it's not already counted as missing product
                    if "Missing Product" not in issue_type:
                        issue_type_lower = issue_type.lower()
                        
                        # Handle multiple issues in one field (e.g., "Price Error, Text Error")
                        if "price" in issue_type_lower:
                            price_issues += 1
                            logger.debug(f"Found price issue from issue_type: {issue_type}")
                        if "text" in issue_type_lower:
                            text_issues += 1
                            logger.debug(f"Found text issue from issue_type: {issue_type}")
                        if "image" in issue_type_lower:
                            image_issues += 1
                            logger.debug(f"Found image issue from issue_type: {issue_type}")

        # Calculate match rate
        match_rate = (correct_matches / max(total_rows, 1)) * 100
        error_rate = (incorrect_matches / max(total_rows, 1)) * 100

        # FIXED SUMMARY DATA with correct calculations
        summary_data = {
            "Metric": [
                "Total Comparisons",
                "Correct Matches", 
                "Incorrect Matches",
                "Products with Issues",
                "Price Issues Found",
                "Text Issues Found", 
                "Image Issues Found",
                "Missing Products",
                "Match Rate (%)",
                "Error Rate (%)",
                "Comparison Type",
                "Focus"
            ],
            "Value": [
                total_rows,
                correct_matches,
                incorrect_matches,
                incorrect_matches,  # Products with issues = incorrect matches
                price_issues,
                text_issues,
                image_issues,
                missing_products,
                f"{match_rate:.1f}%",
                f"{error_rate:.1f}%",
                comparison_result['comparison_criteria'].get('comparison_type', 'Position-based'),
                comparison_result['comparison_criteria'].get('focus', 'Quality Control')
            ]
        }
        summary_df = pd.DataFrame(summary_data)

        # Export to Excel
        output_path = Path(output_path)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Position_Comparison', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        logger.info(f"Position-based comparison exported to {output_path}")
        
        # ENHANCED CONSOLE OUTPUT WITH DEBUGGING
        print(f"\nPOSITION-BASED COMPARISON SUMMARY:")
        print(f"Total comparisons: {total_rows}")
        print(f"Correct matches: {correct_matches}")
        print(f"Incorrect matches: {incorrect_matches}")
        print(f"Products with issues: {incorrect_matches}")
        print(f"Price issues found: {price_issues}")
        print(f"Text issues found: {text_issues}")
        print(f"Image issues found: {image_issues}")
        print(f"Missing products: {missing_products}")
        print(f"Match rate: {match_rate:.1f}%")
        print(f"Error rate: {error_rate:.1f}%")
        
        # ENHANCED DEBUG LOGGING
        logger.info(f"DETAILED SUMMARY CALCULATION DEBUG:")
        logger.info(f"  Total rows processed: {total_rows}")
        logger.info(f"  Rows with CORRECT status: {correct_matches}")
        logger.info(f"  Rows with INCORRECT status: {incorrect_matches}")
        logger.info(f"  Individual issue counts - Price: {price_issues}, Text: {text_issues}, Image: {image_issues}")
        logger.info(f"  Missing products: {missing_products}")
        
        # Additional debugging: Show breakdown by row
        for i, row in enumerate(comparison_rows):
            result = row.get("comparison_result", "UNKNOWN")
            if result == "INCORRECT":
                issue_type = row.get("issue_type", "N/A")
                granular_diffs = row.get("granular_differences", [])
                logger.info(f"  Row {i+1}: {result} - {issue_type} - {len(granular_diffs)} granular differences")
                for j, diff in enumerate(granular_diffs):
                    diff_type = diff.get("type", "Unknown")
                    logger.info(f"    Diff {j+1}: {diff_type}")
            
        return {
            "total_comparisons": total_rows,
            "correct_matches": correct_matches,
            "incorrect_matches": incorrect_matches,
            "price_issues": price_issues,
            "text_issues": text_issues,
            "image_issues": image_issues,
            "missing_products": missing_products,
            "match_rate": match_rate
        }
    def find_differences_with_vlm(self, image1_path: str, image2_path: str, item_id_for_log: str) -> List[Dict]:
        """
        Enhanced VLM function with better bounding box detection guidance
        """
        
        system_prompt = """
        You are a quality control expert comparing two retail product images to find PRODUCTION ERRORS with precise locations.

        TASK: Find obvious quality control issues and provide EXACT bounding box coordinates for each error.

        ERROR TYPES TO DETECT:

        1. **Price Errors**: 
        - Significantly different prices (>$3 difference) for similar products
        - Obviously incorrect prices ($0.00, $999.99, etc.)
        - Missing price information
        - Malformed price text

        2. **Text Errors**:
        - Spelling mistakes in product names (e.g., "clorox" vs "Clorox")
        - Garbled or corrupted text (e.g., "Variedad 57 tandas 60 tandas")
        - Missing or incomplete product descriptions
        - Wrong product names for the shown image

        3. **Image Errors**:
        - Wrong product photos (completely different products)
        - Corrupted, pixelated, or missing images
        - Severely misaligned elements

        IGNORE these normal variations:
        - Different brands in same position (Tide vs Gain = normal)
        - Different product categories (detergent vs paper towels = normal)
        - Minor price differences (<$3)
        - Different promotional banners
        - Slight color/lighting variations

        CRITICAL BOUNDING BOX INSTRUCTIONS:
        
        1. **For Price Errors**: Draw box around the ENTIRE price area including dollar sign and cents
        - Example: For "$12.97" draw box covering the full price text
        - Make box generous enough to include any price formatting
        
        2. **For Text Errors**: Draw box around the ENTIRE problematic text area
        - Example: For misspelled "clorox" draw box covering the whole brand name area
        - Include surrounding context if text is part of larger description
        
        3. **Box Size Guidelines**:
        - Minimum box size: 60x30 pixels (width x height)
        - For prices: typically 80-150 pixels wide, 30-50 pixels tall
        - For product names: typically 100-250 pixels wide, 30-60 pixels tall
        - For descriptions: can be 200-400 pixels wide, 40-80 pixels tall
        
        4. **Coordinate System**: Use absolute pixel coordinates within each image
        - (0,0) is top-left corner of the image
        - X increases going right, Y increases going down
        - Typical product image size is 400-600 pixels wide, 400-800 pixels tall

        RESPONSE FORMAT - Return this EXACT JSON structure:
        {
            "differences": [
                {
                    "type": "Price Error",
                    "description": "Specific detailed description of the error",
                    "box1": [x1, y1, x2, y2],
                    "box2": [x1, y1, x2, y2]
                }
            ]
        }

        Where:
        - box1 = coordinates in FIRST image [left, top, right, bottom]
        - box2 = coordinates in SECOND image [left, top, right, bottom]
        - Coordinates must be integers
        - Ensure x2 > x1 and y2 > y1
        - Make boxes large enough to clearly see the error area

        EXAMPLES of good bounding boxes:
        - Price "$12.47": [50, 20, 130, 55]
        - Brand name "Clorox": [20, 100, 150, 135]  
        - Product description line: [15, 200, 350, 240]

        If no production errors found, return: {"differences": []}

        Focus on OBVIOUS errors that clearly need fixing, not minor variations.
        """
        
        try:
            with open(image1_path, "rb") as f1, open(image2_path, "rb") as f2:
                b64_img1 = base64.b64encode(f1.read()).decode('utf-8')
                b64_img2 = base64.b64encode(f2.read()).decode('utf-8')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {
                            "type": "text", 
                            "text": "FIRST IMAGE - Look for errors and provide box1 coordinates:"
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img1}"}},
                        {
                            "type": "text", 
                            "text": "SECOND IMAGE - Look for errors and provide box2 coordinates:"
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img2}"}}
                    ]
                }
            ]

            response = self.openai_client.chat.completions.create(
                model=self.vlm_model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=2000,  # Increased for detailed analysis
                temperature=0.1   # Low temperature for consistent results
            )

            response_content = response.choices[0].message.content
            logger.info(f"ITEM_ID: {item_id_for_log} - VLM Response: {response_content}")
            
            differences_data = json.loads(response_content)
            raw_differences = differences_data.get("differences", [])
            
            # Enhanced normalization with better bounding box validation
            normalized_differences = []
            
            for i, diff in enumerate(raw_differences):
                if not isinstance(diff, dict):
                    logger.warning(f"ITEM_ID: {item_id_for_log} - Skipping non-dict difference at index {i}: {diff}")
                    continue
                
                normalized_diff = {}
                
                # Handle type field
                if "type" in diff:
                    normalized_diff["type"] = diff["type"]
                elif "issue" in diff:
                    normalized_diff["type"] = diff["issue"]
                elif "category" in diff:
                    normalized_diff["type"] = diff["category"]
                else:
                    desc = str(diff.get("description", diff.get("details", "")))
                    if "price" in desc.lower():
                        normalized_diff["type"] = "Price Error"
                    elif "text" in desc.lower() or "name" in desc.lower():
                        normalized_diff["type"] = "Text Error"
                    else:
                        normalized_diff["type"] = "Unknown Error"
                    logger.warning(f"ITEM_ID: {item_id_for_log} - No 'type' field found, inferred: {normalized_diff['type']}")
                
                # Handle description field
                if "description" in diff:
                    normalized_diff["description"] = diff["description"]
                elif "details" in diff:
                    normalized_diff["description"] = diff["details"]
                elif "message" in diff:
                    normalized_diff["description"] = diff["message"]
                else:
                    product = diff.get("product", "")
                    issue = diff.get("issue", diff.get("type", ""))
                    normalized_diff["description"] = f"{product}: {issue}" if product else str(issue)
                
                # Enhanced bounding box validation and normalization
                box1 = diff.get("box1")
                box2 = diff.get("box2")
                
                # Validate and potentially fix bounding boxes
                if self._is_valid_bounding_box(box1):
                    # Ensure minimum box size
                    box1 = self._ensure_minimum_box_size(box1)
                    normalized_diff["box1"] = box1
                    logger.info(f"ITEM_ID: {item_id_for_log} - Valid box1 found: {box1}")
                else:
                    logger.warning(f"ITEM_ID: {item_id_for_log} - Invalid box1: {box1}")
                    # Try to create a reasonable default box
                    normalized_diff["box1"] = self._create_default_box(normalized_diff["type"])
                    logger.info(f"ITEM_ID: {item_id_for_log} - Using default box1: {normalized_diff['box1']}")
                
                if self._is_valid_bounding_box(box2):
                    box2 = self._ensure_minimum_box_size(box2)
                    normalized_diff["box2"] = box2
                    logger.info(f"ITEM_ID: {item_id_for_log} - Valid box2 found: {box2}")
                else:
                    logger.warning(f"ITEM_ID: {item_id_for_log} - Invalid box2: {box2}")
                    # Try to create a reasonable default box
                    normalized_diff["box2"] = self._create_default_box(normalized_diff["type"])
                    logger.info(f"ITEM_ID: {item_id_for_log} - Using default box2: {normalized_diff['box2']}")
                
                # Copy over any other fields
                for key, value in diff.items():
                    if key not in ["type", "description", "details", "issue", "message", "box1", "box2"]:
                        normalized_diff[key] = value
                
                normalized_differences.append(normalized_diff)
                
                # Enhanced logging
                has_boxes = normalized_diff.get("box1") and normalized_diff.get("box2")
                box1_size = self._get_box_size(normalized_diff.get("box1"))
                box2_size = self._get_box_size(normalized_diff.get("box2"))
                
                logger.info(f"ITEM_ID: {item_id_for_log} - Normalized Diff {i}: "
                        f"Type='{normalized_diff['type']}', "
                        f"Desc='{normalized_diff.get('description', 'N/A')[:50]}...', "
                        f"HasBoxes={has_boxes}, "
                        f"Box1Size={box1_size}, Box2Size={box2_size}")
            
            logger.info(f"ITEM_ID: {item_id_for_log} - Successfully processed {len(normalized_differences)} differences")
            return normalized_differences

        except json.JSONDecodeError as e:
            logger.error(f"ITEM_ID: {item_id_for_log} - JSON parsing error: {e}")
            logger.error(f"ITEM_ID: {item_id_for_log} - Raw response: {response_content}")
            return []
        except Exception as e:
            logger.error(f"ITEM_ID: {item_id_for_log} - VLM comparison error: {e}")
            return []

    def _is_valid_bounding_box(self, box) -> bool:
        """Validate bounding box format and dimensions"""
        if not isinstance(box, list) or len(box) != 4:
            return False
        
        try:
            x1, y1, x2, y2 = [float(coord) for coord in box]
            
            # Basic validation: x2 > x1 and y2 > y1
            if x2 <= x1 or y2 <= y1:
                return False
            
            # Reasonable coordinate ranges (assuming reasonable image sizes)
            if any(coord < 0 or coord > 2000 for coord in [x1, y1, x2, y2]):
                return False
            
            # Minimum size requirement
            width = x2 - x1
            height = y2 - y1
            if width < 20 or height < 15:  # Too small to be useful
                return False
            
            return True
        
        except (ValueError, TypeError):
            return False

    def _ensure_minimum_box_size(self, box, min_width=60, min_height=30):
        """Ensure bounding box meets minimum size requirements"""
        if not box:
            return box
        
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Expand box if too small
        if width < min_width:
            expand_x = (min_width - width) / 2
            x1 = max(0, x1 - expand_x)
            x2 = x1 + min_width
        
        if height < min_height:
            expand_y = (min_height - height) / 2
            y1 = max(0, y1 - expand_y)
            y2 = y1 + min_height
        
        return [int(x1), int(y1), int(x2), int(y2)]

    def _create_default_box(self, error_type):
        """Create a reasonable default bounding box based on error type"""
        if "price" in error_type.lower():
            # Price area typically in upper right
            return [300, 20, 450, 60]
        elif "text" in error_type.lower():
            # Text area typically in center/left
            return [50, 150, 300, 200]
        else:
            # Generic center box
            return [100, 100, 300, 200]

    def _get_box_size(self, box):
        """Get readable box size description"""
        if not box or len(box) != 4:
            return "Invalid"
        
        width = box[2] - box[0]
        height = box[3] - box[1]
        return f"{width}x{height}"
        
    def _is_valid_bounding_box(self, box) -> bool:
        """
        Validate that a bounding box has the correct format [x1, y1, x2, y2]
        """
        if not isinstance(box, list):
            return False
        
        if len(box) != 4:
            return False
        
        try:
            x1, y1, x2, y2 = [float(coord) for coord in box]
            
            # Basic validation: x2 > x1 and y2 > y1
            if x2 <= x1 or y2 <= y1:
                return False
            
            # Reasonable coordinate ranges (assuming images are reasonably sized)
            if any(coord < 0 or coord > 5000 for coord in [x1, y1, x2, y2]):
                return False
            
            return True
        
        except (ValueError, TypeError):
            return False    
        
def main_vlm_comparison(openai_api_key: str, folder1_path: str, folder2_path: str,
                       catalog1_name: str = None, catalog2_name: str = None,
                       output_path: str = None, price_tolerance: float = 0.01):
    """Main function for practical catalog comparison"""

    if not openai_api_key:
        raise ValueError("Please provide OPENAI_API_KEY")

    if not output_path:
        output_path = "practical_catalog_comparison.xlsx"

    # Initialize practical comparator
    comparator = PracticalCatalogComparator(
        openai_api_key,
        price_tolerance=price_tolerance
    )

    try:
        logger.info("Starting PRACTICAL catalog comparison - focusing on what matters...")
        results = comparator.generate_practical_comparison(
            folder1_path,
            folder2_path,
            catalog1_name,
            catalog2_name
        )

        # ADD THIS DEBUG SECTION BEFORE EXPORT:
        print("\n" + "="*60)
        print("PRE-EXPORT DEBUGGING")
        print("="*60)
        
        comparison_rows = results.get("comparison_rows", [])
        print(f"Total comparison rows: {len(comparison_rows)}")
        
        # Count manually to verify
        manual_correct = 0
        manual_incorrect = 0
        manual_price = 0
        manual_text = 0
        manual_missing = 0
        
        for i, row in enumerate(comparison_rows, 1):
            result = row.get("comparison_result", "UNKNOWN")
            issue_type = row.get("issue_type", "")
            details = row.get("details", "")
            granular_diffs = row.get("granular_differences", [])
            
            if result == "CORRECT":
                manual_correct += 1
            elif result == "INCORRECT":
                manual_incorrect += 1
                
                # Check for missing products
                if "Missing Product" in issue_type or "missing" in details.lower():
                    manual_missing += 1
                    print(f"Row {i}: MISSING PRODUCT - {issue_type}")
                else:
                    print(f"Row {i}: {result} - {issue_type} - {len(granular_diffs)} granular diffs")
                    
                    # Count issue types
                    if granular_diffs:
                        for diff in granular_diffs:
                            diff_type = diff.get("type", "").lower()
                            print(f"  -> Diff type: '{diff_type}'")
                            if "price" in diff_type:
                                manual_price += 1
                            elif "text" in diff_type:
                                manual_text += 1
                    else:
                        # Fallback to issue_type
                        if "price" in issue_type.lower():
                            manual_price += 1
                        if "text" in issue_type.lower():
                            manual_text += 1
        
        print(f"\nMANUAL COUNT VERIFICATION:")
        print(f"Correct: {manual_correct}")
        print(f"Incorrect: {manual_incorrect}")
        print(f"Price Issues: {manual_price}")
        print(f"Text Issues: {manual_text}")
        print(f"Missing Products: {manual_missing}")
        print("="*60)

        # Export results (this should now show the correct counts)
        export_result = comparator.export_practical_comparison(results, output_path)
        
        print(f"\nEXPORT RESULT VERIFICATION:")
        print(f"Export returned: {export_result}")

        print(f"PRACTICAL RESULTS SAVED TO: {output_path}")
        return results

    except Exception as e:
        logger.error(f"Error in practical comparison: {e}")
        raise

# ========================================
# STREAMLINED CATALOG COMPARISON PIPELINE
# ========================================

def catalog_comparison_pipeline(
    pdf_path1: str,
    pdf_path2: str,
    template1_path: str,
    template2_path: str,
    template3_path: str,
    roboflow_api_key: str,
    roboflow_project_name: str,
    roboflow_version: int,
    openai_api_key: str,
    output_directory: str = "catalog_comparison_results",
    confidence_threshold: float = 25,
    similarity_threshold: float = 0.55,
    price_tolerance: float = 0.01,
    ranking_method: str = "improved_grid",
    comparison_method: str = "structural",
    dry_run: bool = False
):
    """
    Complete catalog comparison pipeline that processes two PDFs and template images.

    Args:
        pdf_path1: Path to first PDF catalog
        pdf_path2: Path to second PDF catalog
        template1_path: Path to first template image for filtering
        template2_path: Path to second template image for filtering
        template3_path: Path to third template image for filtering
        roboflow_api_key: Roboflow API key for object detection
        roboflow_project_name: Roboflow project name
        roboflow_version: Roboflow model version
        openai_api_key: OpenAI API key for VLM analysis
        output_directory: Base output directory for all results
        confidence_threshold: Detection confidence threshold (default: 25)
        similarity_threshold: Image similarity threshold for template matching (default: 0.55)
        price_tolerance: Price difference tolerance for comparison (default: 0.01)
        ranking_method: Ranking algorithm to use (default: "improved_grid")
        comparison_method: Image comparison method (default: "structural")
        dry_run: If True, show what would be done without executing (default: False)

    Returns:
        Dictionary containing all pipeline results and file paths
    """

    print("=" * 80)
    print("CATALOG COMPARISON PIPELINE - COMPLETE AUTOMATION")
    print("=" * 80)
    print(f"PDF 1: {Path(pdf_path1).name}")
    print(f"PDF 2: {Path(pdf_path2).name}")
    print(f"Templates: {Path(template1_path).name}, {Path(template2_path).name}, {Path(template3_path).name}")
    print(f"Output Directory: {output_directory}")
    print(f"Detection Confidence: {confidence_threshold}%")
    print(f"Similarity Threshold: {similarity_threshold}")
    print(f"Price Tolerance: ${price_tolerance}")
    print(f"Poppler Path: {get_poppler_path() if get_poppler_path() else 'System PATH'}")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print("=" * 80)

    pipeline_results = {
        "step1_pdf_processing": None,
        "step2_template_filtering": None,
        "step3_vlm_comparison": None,
        "final_output_path": None,
        "pipeline_summary": {},
        "errors": []
    }

    try:
        # ==============================
        # STEP 1: PDF PROCESSING AND RANKING
        # ==============================
        print("\nSTEP 1: PDF PROCESSING AND PRODUCT EXTRACTION")
        print("-" * 50)

        # Initialize Roboflow model
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=roboflow_api_key)
            project = rf.project(roboflow_project_name)
            model = project.version(roboflow_version).model
            print(f"Roboflow model initialized: {roboflow_project_name} v{roboflow_version}")
        except Exception as e:
            error_msg = f"Failed to initialize Roboflow model: {e}"
            pipeline_results["errors"].append(error_msg)
            raise Exception(error_msg)

        # Process PDFs
        pdf_output_dir = os.path.join(output_directory, "01_pdf_processing")

        pdf_results = main_dual_pdf_processing(
            pdf_path1=pdf_path1,
            pdf_path2=pdf_path2,
            roboflow_model=model,
            output_root=pdf_output_dir,
            ranking_method=ranking_method,
            confidence_threshold=confidence_threshold
        )

        pipeline_results["step1_pdf_processing"] = pdf_results
        print(f"PDF processing complete:")
        print(f"Catalog 1: {pdf_results['total_products_catalog1']} products")
        print(f"Catalog 2: {pdf_results['total_products_catalog2']} products")

        # ==============================
        # STEP 2: TEMPLATE-BASED FILTERING
        # ==============================
        print("\n🧹 STEP 2: TEMPLATE-BASED IMAGE FILTERING")
        print("-" * 50)

        template_results = {}

        # Process each page in both catalogs
        catalog1_path = Path(pdf_results["catalog1_path"])
        catalog2_path = Path(pdf_results["catalog2_path"])

        # Find all page folders
        catalog1_pages = [d for d in catalog1_path.iterdir() if d.is_dir() and d.name.startswith("page_")]
        catalog2_pages = [d for d in catalog2_path.iterdir() if d.is_dir() and d.name.startswith("page_")]

        all_page_pairs = []
        max_pages = max(len(catalog1_pages), len(catalog2_pages))

        for i in range(max_pages):
            folder1 = str(catalog1_pages[i]) if i < len(catalog1_pages) else None
            folder2 = str(catalog2_pages[i]) if i < len(catalog2_pages) else None
            if folder1 or folder2:
                all_page_pairs.append((folder1, folder2))

        print(f"Found {len(all_page_pairs)} page pairs to process")

        for page_idx, (folder1, folder2) in enumerate(all_page_pairs, 1):
            print(f"\nProcessing Page {page_idx}...")

            if folder1 and folder2:
                try:
                    process_folders_with_image_templates(
                        template1_path=template1_path,
                        template2_path=template2_path,
                        template3_path=template3_path,
                        folder1_path=folder1,
                        folder2_path=folder2,
                        similarity_threshold=similarity_threshold,
                        comparison_method=comparison_method,
                        dry_run=dry_run
                    )
                    template_results[f"page_{page_idx}"] = "Processed successfully"
                    print(f"Page {page_idx} template filtering complete")

                except Exception as e:
                    error_msg = f"Template filtering failed for page {page_idx}: {e}"
                    template_results[f"page_{page_idx}"] = error_msg
                    pipeline_results["errors"].append(error_msg)
                    print(f"{error_msg}")
            else:
                print(f"Skipping page {page_idx} - missing folder(s)")

        pipeline_results["step2_template_filtering"] = template_results

        # ==============================
        # STEP 3: VLM-BASED COMPARISON
        # ==============================
        print("\nSTEP 3: VLM-BASED CATALOG COMPARISON")
        print("-" * 50)

        comparison_results = {}

        # Compare each page pair using VLM
        for page_idx, (folder1, folder2) in enumerate(all_page_pairs, 1):
            if folder1 and folder2 and os.path.exists(folder1) and os.path.exists(folder2):
                print(f"\n Analyzing Page {page_idx} with VLM...")

                try:
                    # Create output path for this page comparison
                    page_output_path = os.path.join(output_directory, "03_vlm_comparison", f"page_{page_idx}_comparison.xlsx")
                    os.makedirs(os.path.dirname(page_output_path), exist_ok=True)

                    vlm_results = main_vlm_comparison(
                        openai_api_key=openai_api_key,
                        folder1_path=folder1,
                        folder2_path=folder2,
                        catalog1_name=f"Catalog1_Page{page_idx}",
                        catalog2_name=f"Catalog2_Page{page_idx}",
                        output_path=page_output_path,
                        price_tolerance=price_tolerance
                    )

                    comparison_results[f"page_{page_idx}"] = {
                        "results": vlm_results,
                        "output_path": page_output_path
                    }
                    print(f"Page {page_idx} VLM comparison complete")

                except Exception as e:
                    error_msg = f"VLM comparison failed for page {page_idx}: {e}"
                    comparison_results[f"page_{page_idx}"] = {"error": error_msg}
                    pipeline_results["errors"].append(error_msg)
                    print(f"{error_msg}")
            else:
                print(f"Skipping VLM comparison for page {page_idx} - missing folders")

        pipeline_results["step3_vlm_comparison"] = comparison_results


        if "step3_vlm_comparison" in pipeline_results and pipeline_results["step3_vlm_comparison"]:
            vlm_all_pages_data = pipeline_results["step3_vlm_comparison"]
            logger.info("Updating visualizations with VLM comparison results...")

            # Iterate through each page's VLM results
            for page_id_key, page_vlm_data in vlm_all_pages_data.items(): # e.g., page_id_key = "page_1"
                if "error" in page_vlm_data or "results" not in page_vlm_data:
                    logger.warning(f"Skipping VLM viz update for {page_id_key} due to error or missing results.")
                    continue

                vlm_results_for_page = page_vlm_data["results"]
                page_num_match = re.search(r'(\d+)', page_id_key) # Get page number
                if not page_num_match:
                    logger.warning(f"Could not extract page number from VLM result key: {page_id_key}")
                    continue
                page_num = int(page_num_match.group(1))

                comparison_rows = vlm_results_for_page.get("comparison_rows", [])
                
                # Determine issue ranks for catalog 1 and catalog 2 for the current page
                issue_ranks_cat1 = set()
                issue_ranks_cat2 = set()

                # The comparison_rows are generated by iterating ranks from 1 to max_rank.
                # So, the index of the row + 1 gives the rank on the page.
                for rank_idx, row_data in enumerate(comparison_rows):
                    current_rank_on_page = rank_idx + 1 
                    
                    comparison_status = row_data.get("comparison_result", "")
                    issue_type = row_data.get("issue_type", "")
                    details = row_data.get("details", "")
                    
                    # Catalog names as used in VLM results (e.g., "Catalog1_Page1")
                    vlm_cat1_name = vlm_results_for_page.get("catalog1_name", "File1") 
                    vlm_cat2_name = vlm_results_for_page.get("catalog2_name", "File2")

                    # Check if product from catalog 1 was part of this row's comparison
                    # (i.e., not reported as "Product Missing" for catalog 1 in this specific row)
                    product1_in_row_details = row_data.get(f"{vlm_cat1_name}_details", "")
                    is_product1_present_in_row = not ("Product Missing" in product1_in_row_details if isinstance(product1_in_row_details, str) else True)

                    product2_in_row_details = row_data.get(f"{vlm_cat2_name}_details", "")
                    is_product2_present_in_row = not ("Product Missing" in product2_in_row_details if isinstance(product2_in_row_details, str) else True)

                    if comparison_status.startswith("INCORRECT"):
                        # --- START: CORRECTED LOGIC ---
                        if issue_type == "Missing Product":
                            # Check if the error message says it's missing in Catalog 2
                            if "missing in " + vlm_cat2_name in details:
                                # If Cat 2 is missing, highlight the product that exists in Cat 1.
                                issue_ranks_cat1.add(current_rank_on_page)
                            # Check if the error message says it's missing in Catalog 1
                            elif "missing in " + vlm_cat1_name in details:
                                # If Cat 1 is missing, highlight the product that exists in Cat 2.
                                issue_ranks_cat2.add(current_rank_on_page)
                        
                        # This part was already correct. It handles issues where both products exist but are different.
                        elif issue_type in ["Different Product", "Price Difference", "Multiple Issues"]:
                            issue_ranks_cat1.add(current_rank_on_page)
                            issue_ranks_cat2.add(current_rank_on_page)


                # Regenerate visualization for Catalog 1, Page `page_num`
                if pdf_results and page_num in pdf_results.get("page_level_data_catalog1", {}):
                    page_info_c1 = pdf_results["page_level_data_catalog1"][page_num]
                    
                    # Create the dictionary that the visualization function now expects
                    issue_details_c1 = {
                        idx + 1: row.get("granular_differences", [])
                        for idx, row in enumerate(comparison_rows)
                        if row.get("comparison_result") == "INCORRECT" and row.get("granular_differences")
                    }

                    if page_info_c1.get('image_pil') and page_info_c1.get('ranked_boxes'):
                        viz_filename_c1 = f"c1_p{page_num}_ranking_visualization.jpg"
                        viz_output_path_c1 = Path(page_info_c1['page_folder_path']) / viz_filename_c1
                        
                        # Use the correct parameter name: 'issue_details_per_rank'
                        create_ranking_visualization(
                            pil_img=page_info_c1['image_pil'],
                            boxes=page_info_c1['ranked_boxes'],
                            output_path=str(viz_output_path_c1),
                            issue_details_per_rank=issue_details_c1
                        )
                        logger.info(f"Updated visualization for Catalog 1 Page {page_num} with issues: {list(issue_details_c1.keys())}")

                # Regenerate visualization for Catalog 2, Page `page_num`
                if pdf_results and page_num in pdf_results.get("page_level_data_catalog2", {}):
                    page_info_c2 = pdf_results["page_level_data_catalog2"][page_num]

                    # Create the dictionary for catalog 2
                    issue_details_c2 = {
                        idx + 1: row.get("granular_differences", [])
                        for idx, row in enumerate(comparison_rows)
                        if row.get("comparison_result") == "INCORRECT" and row.get("granular_differences")
                    }

                    if page_info_c2.get('image_pil') and page_info_c2.get('ranked_boxes'):
                        viz_filename_c2 = f"c2_p{page_num}_ranking_visualization.jpg"
                        viz_output_path_c2 = Path(page_info_c2['page_folder_path']) / viz_filename_c2
                        
                        # Use the correct parameter name: 'issue_details_per_rank'
                        create_ranking_visualization(
                            pil_img=page_info_c2['image_pil'],
                            boxes=page_info_c2['ranked_boxes'],
                            output_path=str(viz_output_path_c2),
                            issue_details_per_rank=issue_details_c2
                        )
                        logger.info(f"Updated visualization for Catalog 2 Page {page_num} with issues: {list(issue_details_c2.keys())}")

        # ==============================
        # STEP 4: CONSOLIDATE RESULTS
        # ==============================
        print("\nSTEP 4: CONSOLIDATING FINAL RESULTS")
        print("-" * 50)

        # Create final summary
        total_catalog1_products = sum([
            results["results"]["catalog1_total_products"]
            for results in comparison_results.values()
            if "results" in results and "catalog1_total_products" in results["results"]
        ])

        total_catalog2_products = sum([
            results["results"]["catalog2_total_products"]
            for results in comparison_results.values()
            if "results" in results and "catalog2_total_products" in results["results"]
        ])

        total_comparisons = sum([
            len(results["results"]["comparison_rows"])
            for results in comparison_results.values()
            if "results" in results and "comparison_rows" in results["results"]
        ])

        # Create consolidated summary file
        final_output_path = os.path.join(output_directory, "FINAL_COMPARISON_SUMMARY.xlsx")

        summary_data = {
            "Metric": [
                "Pipeline Execution Mode",
                "Total Pages Processed",
                "PDF 1 Total Products",
                "PDF 2 Total Products",
                "Total Comparisons Made",
                "Template Filtering Applied",
                "VLM Analysis Applied",
                "Detection Confidence Used",
                "Similarity Threshold Used",
                "Price Tolerance Used",
                "Ranking Method Used",
                "Comparison Method Used",
                "Poppler Path Used",
                "Total Errors Encountered"
            ],
            "Value": [
                "DRY RUN" if dry_run else "EXECUTED",
                len(all_page_pairs),
                total_catalog1_products,
                total_catalog2_products,
                total_comparisons,
                "Yes" if template_results else "No",
                "Yes" if comparison_results else "No",
                f"{confidence_threshold}%",
                f"{similarity_threshold}",
                f"${price_tolerance}",
                ranking_method,
                comparison_method,
                get_poppler_path() if get_poppler_path() else "System PATH",
                len(pipeline_results["errors"])
            ]
        }

        summary_df = pd.DataFrame(summary_data)

        # Create file list
        output_files = []
        for page_results in comparison_results.values():
            if "output_path" in page_results:
                output_files.append(page_results["output_path"])

        files_data = {
            "Output File": output_files,
            "Description": [f"VLM comparison results for {Path(f).stem}" for f in output_files]
        }
        files_df = pd.DataFrame(files_data)

        # Export consolidated summary
        with pd.ExcelWriter(final_output_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Pipeline_Summary', index=False)
            files_df.to_excel(writer, sheet_name='Output_Files', index=False)

            # Add errors sheet if any
            if pipeline_results["errors"]:
                errors_df = pd.DataFrame({"Errors": pipeline_results["errors"]})
                errors_df.to_excel(writer, sheet_name='Errors', index=False)

        pipeline_results["final_output_path"] = final_output_path
        pipeline_results["pipeline_summary"] = summary_data

        print(f"Pipeline consolidation complete")
        print(f"Final summary saved to: {final_output_path}")

        # ==============================
        # FINAL PIPELINE SUMMARY
        # ==============================
        print("\n" + "=" * 80)
        print("CATALOG COMPARISON PIPELINE COMPLETE")
        print("=" * 80)
        print(f"FINAL RESULTS:")
        print(f"Pages Processed: {len(all_page_pairs)}")
        print(f"Total Products Catalog 1: {total_catalog1_products}")
        print(f"Total Products Catalog 2: {total_catalog2_products}")
        print(f"Total Comparisons: {total_comparisons}")
        print(f"Errors Encountered: {len(pipeline_results['errors'])}")
        print(f"Output Directory: {output_directory}")
        print(f"Final Summary: {final_output_path}")
        print(f"Poppler Used: {get_poppler_path() if get_poppler_path() else 'System PATH'}")

        if pipeline_results["errors"]:
            print(f"\nERRORS ENCOUNTERED:")
            for i, error in enumerate(pipeline_results["errors"], 1):
                print(f"   {i}. {error}")

        print("\nPipeline executed successfully!")
        return pipeline_results

    except Exception as e:
        error_msg = f"Pipeline failed: {e}"
        pipeline_results["errors"].append(error_msg)
        print(f"\n PIPELINE FAILED: {error_msg}")
        raise


# ========================================
# SIMPLE USAGE FUNCTION
# ========================================

def simple_catalog_comparison(
    pdf1_path: str,
    pdf2_path: str,
    template1_path: str,
    template2_path: str,
    template3_path: str,
    roboflow_api_key: str = None,
    openai_api_key: str = None,
    roboflow_project: str = "my-first-project-c49lu",
    roboflow_version: int = 1,
    poppler_path: str = None
):
    """
    Simplified function that uses environment variables for API keys if not provided.

    Args:
        pdf1_path: Path to first PDF
        pdf2_path: Path to second PDF
        template1_path: Path to first template image
        template2_path: Path to second template image
        template3_path: Path to third template image
        roboflow_api_key: Roboflow API key (optional, will use env var)
        openai_api_key: OpenAI API key (optional, will use env var)
        roboflow_project: Roboflow project name
        roboflow_version: Roboflow model version
        poppler_path: Custom poppler path (optional, will use default detection)
    """

    # Set custom poppler path if provided
    if poppler_path:
        global POPPLER_PATH
        POPPLER_PATH = poppler_path
        print(f"Using custom poppler path: {poppler_path}")

    # Get API keys from environment if not provided
    if not roboflow_api_key:
        roboflow_api_key = os.getenv('ROBOFLOW_API_KEY')
    if not openai_api_key:
        openai_api_key = os.getenv('OPENAI_API_KEY')

    if not roboflow_api_key:
        raise ValueError("Roboflow API key required. Set ROBOFLOW_API_KEY environment variable or pass as parameter.")
    if not openai_api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass as parameter.")

    return catalog_comparison_pipeline(
        pdf_path1=pdf1_path,
        pdf_path2=pdf2_path,
        template1_path=template1_path,
        template2_path=template2_path,
        template3_path=template3_path,
        roboflow_api_key=roboflow_api_key,
        roboflow_project_name=roboflow_project,
        roboflow_version=roboflow_version,
        openai_api_key=openai_api_key
    )


if __name__ == "__main__":
    print("=" * 80)
    print("STREAMLINED CATALOG COMPARISON PIPELINE")
    print("=" * 80)
    print("This script provides a complete automated pipeline for catalog comparison.")
    print()
    print("POPPLER CONFIGURATION:")
    poppler_path = get_poppler_path()
    if poppler_path:
        print(f"✓ Poppler found at: {poppler_path}")
    else:
        print("⚠ Poppler path not found. Using system PATH.")
        print("  If you encounter PDF conversion errors, please:")
        print("  1. Install poppler-utils")
        print("  2. Update POPPLER_PATH variable in the script")
        print("  3. Or add poppler to your system PATH")
    print()
    print("USAGE:")
    print("1. Set environment variables: ROBOFLOW_API_KEY and OPENAI_API_KEY")
    print("2. Call simple_catalog_comparison() with your file paths")
    print()
    print("EXAMPLE:")
    print('results = simple_catalog_comparison(')
    print('    pdf1_path="/path/to/catalog1.pdf",')
    print('    pdf2_path="/path/to/catalog2.pdf",')
    print('    template1_path="/path/to/template1.jpg",')
    print('    template2_path="/path/to/template2.jpg",')
    print('    template3_path="/path/to/template3.jpg",')
    print('    poppler_path="C:/Program Files/poppler-0.68.0/bin"  # Optional')
    print(')')
    print()
    print("Or use the full pipeline function for more control over parameters.")
    print("=" * 80)
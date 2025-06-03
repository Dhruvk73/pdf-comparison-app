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

def extract_ranked_boxes_from_image(pil_img, roboflow_model, output_folder, page_prefix="page1",
                                   filter_small_boxes=True, ranking_method="improved_grid",
                                   confidence_threshold=25):  # REDUCED from 40 to 25
    """
    Extract and rank product boxes from image with improved ranking

    Args:
        ranking_method: "improved_grid", "adaptive", "kmeans", or "reading_order"
        confidence_threshold: Detection confidence threshold (default: 25, was 40)
    """
    # Save PIL image as a temp file for Roboflow
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_img_path = tmp.name
        pil_img.save(temp_img_path, "JPEG")

    print(f"Running Roboflow detection on {page_prefix} with confidence threshold {confidence_threshold}%...")
    # CHANGED: Using configurable confidence_threshold instead of hardcoded 40
    rf_result = roboflow_model.predict(temp_img_path, confidence=confidence_threshold, overlap=30)
    boxes = rf_result.json().get("predictions", [])
    print(f"Roboflow detected {len(boxes)} boxes")

    # Extract box information and filter out small boxes
    sortable_boxes = []
    filtered_count = 0

    for pred in boxes:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        cls = pred.get("class", "unknown")

        # Filter out small boxes (banners, headers, small promotional elements)
        if is_valid_product_box(w, h, pil_img.width, pil_img.height):
            sortable_boxes.append({
                "pred": pred,
                "center_x": x,
                "center_y": y,
                "left": x - w / 2,
                "top": y - h / 2,
                "right": x + w / 2,
                "bottom": y + h / 2,
                "width": w,
                "height": h,
                "class": cls
            })
        else:
            filtered_count += 1
            print(f"Filtered out small box: {w:.0f}x{h:.0f} at ({x:.0f}, {y:.0f})")

    print(f"Kept {len(sortable_boxes)} valid product boxes, filtered out {filtered_count} small boxes")

    if not sortable_boxes:
        print("No valid boxes found!")
        os.remove(temp_img_path)
        return []

    # Apply selected ranking method
    if ranking_method == "improved_grid":
        sortable_boxes = improved_grid_ranking(sortable_boxes)
    elif ranking_method == "adaptive":
        sortable_boxes = adaptive_row_detection(sortable_boxes)
    elif ranking_method == "kmeans":
        sortable_boxes = rank_by_kmeans_rows(sortable_boxes)
    else:  # reading_order
        sortable_boxes = rank_by_reading_order(sortable_boxes)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save cropped images with ranking
    saved_files = []
    for idx, b in enumerate(sortable_boxes):
        x, y, w, h = b["pred"]["x"], b["pred"]["y"], b["pred"]["width"], b["pred"]["height"]

        # Add some padding to capture full product descriptions
        padding = 10  # pixels
        left = max(0, int(x - w / 2) - padding)
        upper = max(0, int(y - h / 2) - padding)
        right = min(pil_img.width, int(x + w / 2) + padding)
        lower = min(pil_img.height, int(y + h / 2) + padding)

        cropped = pil_img.crop((left, upper, right, lower))
        save_path = os.path.join(output_folder, f"{page_prefix}_rank_{idx+1}.jpg")
        cropped.save(save_path, "JPEG")
        saved_files.append(save_path)
        print(f"Rank {idx+1}: {save_path} (centroid: {b['center_x']:.1f}, {b['center_y']:.1f})")

    # Create visualization
    viz_path = os.path.join(output_folder, f"{page_prefix}_ranking_visualization.jpg")
    create_ranking_visualization(pil_img, sortable_boxes, viz_path)

    # Cleanup temp file
    os.remove(temp_img_path)
    return saved_files

def create_ranking_visualization(pil_img, boxes, output_path):
    """
    Create a visualization showing the ranking order on the original image
    """
    img_copy = pil_img.copy()
    draw = ImageDraw.Draw(img_copy)

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 30)
        except:
            font = ImageFont.load_default()

    # Colors for different rows
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]

    for idx, box in enumerate(boxes):
        # Determine row for coloring
        row_color = colors[idx % len(colors)]

        # Draw bounding box
        left = int(box["left"])
        top = int(box["top"])
        right = int(box["right"])
        bottom = int(box["bottom"])

        draw.rectangle([left, top, right, bottom], outline=row_color, width=3)

        # Draw ranking number with background
        rank_text = str(idx + 1)
        bbox = draw.textbbox((0, 0), rank_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position text at top-left of box with background
        text_x = left + 5
        text_y = top + 5

        # Background rectangle
        draw.rectangle([text_x-3, text_y-3, text_x + text_width + 6, text_y + text_height + 6],
                      fill="yellow", outline="black", width=1)
        draw.text((text_x, text_y), rank_text, fill="black", font=font)

        # Draw centroid
        cx, cy = int(box["center_x"]), int(box["center_y"])
        draw.ellipse([cx-5, cy-5, cx+5, cy+5], fill=row_color, outline="white", width=2)

    img_copy.save(output_path, "JPEG", quality=95)
    print(f"Ranking visualization saved to: {output_path}")

def process_dual_pdfs_for_comparison(pdf_path1, pdf_path2, output_root="catalog_comparison",
                                   ranking_method="improved_grid", filter_small_boxes=True,
                                   confidence_threshold=25):  # ADDED parameter with default 25
    """
    Process two PDFs and save ranked products to separate catalog folders

    Args:
        pdf_path1: Path to first PDF
        pdf_path2: Path to second PDF
        output_root: Base output directory
        ranking_method: Ranking algorithm to use
        filter_small_boxes: Whether to filter out small promotional boxes
        confidence_threshold: Detection confidence threshold (default: 25)

    Returns:
        Dictionary with paths to both catalogs and summary statistics
    """
    print("="*60)
    print("DUAL PDF PROCESSING FOR CATALOG COMPARISON")
    print(f"Detection Confidence Threshold: {confidence_threshold}%")  # Show threshold
    print("="*60)

    # Create output structure
    output_path = Path(output_root)
    catalog1_path = output_path / "catalog1"
    catalog2_path = output_path / "catalog2"

    # Create directories
    catalog1_path.mkdir(parents=True, exist_ok=True)
    catalog2_path.mkdir(parents=True, exist_ok=True)

    results = {
        "catalog1_path": str(catalog1_path),
        "catalog2_path": str(catalog2_path),
        "catalog1_files": [],
        "catalog2_files": [],
        "catalog1_pages": 0,
        "catalog2_pages": 0,
        "total_products_catalog1": 0,
        "total_products_catalog2": 0,
        "confidence_threshold": confidence_threshold  # Store threshold in results
    }

    # Process PDF 1
    print(f"\nPROCESSING PDF 1: {Path(pdf_path1).name}")
    print("-" * 50)
    try:
        pages1 = convert_from_path(pdf_path1, dpi=300)
        results["catalog1_pages"] = len(pages1)
        print(f"Converted PDF 1 to {len(pages1)} pages")

        for page_num, page_img in enumerate(pages1, 1):
            page_folder = catalog1_path / f"page_{page_num}"
            page_folder.mkdir(exist_ok=True)

            print(f"\nProcessing Catalog 1 - Page {page_num}...")
            saved_files = extract_ranked_boxes_from_image(
                page_img,
                roboflow_model=None,  # Will be passed from outside
                output_folder=str(page_folder),
                page_prefix=f"c1_p{page_num}",
                filter_small_boxes=filter_small_boxes,
                ranking_method=ranking_method,
                confidence_threshold=confidence_threshold  # PASS threshold parameter
            )
            results["catalog1_files"].extend(saved_files)
            results["total_products_catalog1"] += len(saved_files)
            print(f"Page {page_num}: {len(saved_files)} products extracted")

    except Exception as e:
        print(f"Error processing PDF 1: {e}")
        results["catalog1_error"] = str(e)

    # Process PDF 2
    print(f"\nPROCESSING PDF 2: {Path(pdf_path2).name}")
    print("-" * 50)
    try:
        pages2 = convert_from_path(pdf_path2, dpi=300)
        results["catalog2_pages"] = len(pages2)
        print(f"Converted PDF 2 to {len(pages2)} pages")

        for page_num, page_img in enumerate(pages2, 1):
            page_folder = catalog2_path / f"page_{page_num}"
            page_folder.mkdir(exist_ok=True)

            print(f"\nProcessing Catalog 2 - Page {page_num}...")
            saved_files = extract_ranked_boxes_from_image(
                page_img,
                roboflow_model=None,  # Will be passed from outside
                output_folder=str(page_folder),
                page_prefix=f"c2_p{page_num}",
                filter_small_boxes=filter_small_boxes,
                ranking_method=ranking_method,
                confidence_threshold=confidence_threshold  # PASS threshold parameter
            )
            results["catalog2_files"].extend(saved_files)
            results["total_products_catalog2"] += len(saved_files)
            print(f"Page {page_num}: {len(saved_files)} products extracted")

    except Exception as e:
        print(f"Error processing PDF 2: {e}")
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
def main_dual_pdf_processing(pdf_path1, pdf_path2, roboflow_model,
                            output_root="catalog_comparison", ranking_method="improved_grid",
                            confidence_threshold=25):  # ADDED parameter with default 25
    """
    Main function to process two PDFs with Roboflow model

    Args:
        pdf_path1: Path to first PDF
        pdf_path2: Path to second PDF
        roboflow_model: Initialized Roboflow model
        output_root: Output directory name
        ranking_method: Ranking algorithm to use
        confidence_threshold: Detection confidence threshold (default: 25, was 40)

    Usage:
        from roboflow import Roboflow
        rf = Roboflow(api_key="your_key")
        model = rf.project('project_name').version(1).model

        results = main_dual_pdf_processing(
            "path/to/pdf1.pdf",
            "path/to/pdf2.pdf",
            model,
            output_root="my_catalogs",
            confidence_threshold=20  # Even lower threshold
        )
    """
    # Temporarily store the model globally for the extraction function
    global GLOBAL_ROBOFLOW_MODEL
    GLOBAL_ROBOFLOW_MODEL = roboflow_model

    # Monkey patch the extract function to use the global model
    original_extract = extract_ranked_boxes_from_image

    def patched_extract(*args, **kwargs):
        if kwargs.get('roboflow_model') is None:
            kwargs['roboflow_model'] = GLOBAL_ROBOFLOW_MODEL
        return original_extract(*args, **kwargs)

    # Replace the function temporarily
    globals()['extract_ranked_boxes_from_image'] = patched_extract

    try:
        results = process_dual_pdfs_for_comparison(
            pdf_path1, pdf_path2, output_root, ranking_method,
            confidence_threshold=confidence_threshold  # PASS threshold parameter
        )
        return results
    finally:
        # Restore original function
        globals()['extract_ranked_boxes_from_image'] = original_extract

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
            system_prompt = """You are a product information extraction expert for retail catalog comparison.
            Focus ONLY on extracting the essential information needed for accurate product matching.

            EXTRACTION PRIORITIES (in order of importance):

            1. PRICES (CRITICAL):
               - Strictly Extract the Main offer price (large, prominent price - usually placed top-right)
               - Strictly Extract Regular price  (which will be given below the product with the description)
               - Format handling: "8 87" → 8.87, "887" → 8.87, "1097" → 10.97
               - Unit indicators: "c/u", "ea.", "each"

            2. BRAND IDENTIFICATION (CRITICAL):
               - Brand name from packaging, logos, or text
               - Extract exactly as shown, including variants like for e.g "Ace Simply"
               - Focus on the main brand, not minor descriptors

            3. PRODUCT BASICS:
               - Core product name/type (detergent, cleaner, etc.)
               - Size/quantity if clearly visible
               - Only extract what's clearly readable - don't guess

            4. IGNORE MINOR DETAILS:
               - Don't worry about exact wording of descriptions
               - Skip fine print unless it's price-related
               - Focus on core product identity, not marketing language

            COMPARISON FOCUS:
            The goal is to compare if these are the SAME PRODUCT at the SAME PRICE.
            Minor text differences, word order, or description variations don't matter.
            What matters: Brand match + Price match + Basic product type match.

            Return JSON with these fields:
            - "offer_price": Main price (decimal or null)
            - "regular_price": Regular price if shown (decimal or null)
            - "product_brand": Primary brand name
            - "product_type": Basic product type (detergent, cleaner, etc.)
            - "size_quantity": Size if clearly visible
            - "product_status": "Product Present" or "Product Missing"
            - "confidence_score": How confident you are in the extraction (1-10)

            Be precise with prices and brands which are present. Extract as it is. Be flexible with everything else. Strictly do not generate any information on your own"""

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
                max_tokens=1800,
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
        """Generate practical comparison focused on what matters"""
        if not catalog1_name:
            catalog1_name = Path(folder1_path).name
        if not catalog2_name:
            catalog2_name = Path(folder2_path).name

        logger.info(f"Starting practical comparison: {catalog1_name} vs {catalog2_name}")
        logger.info(f"Focus: Price differences and different products only")

        # Load and process images
        catalog1_images = self.load_ranked_images_from_folder(folder1_path)
        catalog2_images = self.load_ranked_images_from_folder(folder2_path)

        # Extract product data
        logger.info("Extracting focused product data from Catalog 1...")
        catalog1_products = {}
        for img_info in catalog1_images:
            product_data = self.extract_product_data_with_vlm(
                img_info['file_path'],
                img_info['rank'],
                catalog1_name
            )
            if "error_message" not in product_data:
                catalog1_products[img_info['rank']] = product_data

        logger.info("Extracting focused product data from Catalog 2...")
        catalog2_products = {}
        for img_info in catalog2_images:
            product_data = self.extract_product_data_with_vlm(
                img_info['file_path'],
                img_info['rank'],
                catalog2_name
            )
            if "error_message" not in product_data:
                catalog2_products[img_info['rank']] = product_data

        # Generate comparison table - position by position
        comparison_rows = []
        max_rank = max(
            max(catalog1_products.keys(), default=0),
            max(catalog2_products.keys(), default=0)
        )

        for rank in range(1, max_rank + 1):
            product1 = catalog1_products.get(rank)
            product2 = catalog2_products.get(rank)

            row = self.create_practical_comparison_row(product1, product2, rank, catalog1_name, catalog2_name)
            if row:
                comparison_rows.append(row)

        result = {
            "catalog1_name": catalog1_name,
            "catalog2_name": catalog2_name,
            "comparison_rows": comparison_rows,
            "catalog1_total_products": len(catalog1_products),
            "catalog2_total_products": len(catalog2_products),
            "catalog1_products": catalog1_products,
            "catalog2_products": catalog2_products,
            "comparison_criteria": {
                "price_tolerance": self.price_tolerance,
                "brand_similarity_threshold": 80,
                "focus": "Price differences and different products only"
            }
        }

        logger.info(f"Practical comparison complete. Generated {len(comparison_rows)} comparison rows.")
        return result

    def create_practical_comparison_row(self, product1: Dict, product2: Dict, rank: int,
                                     catalog1_name: str, catalog2_name: str) -> Dict:
        """Create practical comparison row - only flag real issues"""

        if not product1 and not product2:
            return None

        # Handle missing products
        if not product1:
            p2_info = self.format_product_display(product2)
            return {
                f"{catalog1_name}_details": "Product Missing",
                f"{catalog2_name}_details": p2_info,
                "comparison_result": "INCORRECT - Missing Product",
                "issue_type": "Missing Product",
                "details": f"Product missing in {catalog1_name}",
                "price_match": "N/A",
                "brand_match": "N/A"
            }

        if not product2:
            p1_info = self.format_product_display(product1)
            return {
                f"{catalog1_name}_details": p1_info,
                f"{catalog2_name}_details": "Product Missing",
                "comparison_result": "INCORRECT - Missing Product",
                "issue_type": "Missing Product",
                "details": f"Product missing in {catalog2_name}",
                "price_match": "N/A",
                "brand_match": "N/A"
            }

        # Both products present - practical comparison
        issues = []

        # Extract key data with safe None handling
        p1_offer = product1.get('offer_price')
        p2_offer = product2.get('offer_price')

        # Safe brand extraction
        p1_brand_raw = product1.get('product_brand')
        p2_brand_raw = product2.get('product_brand')

        p1_brand = p1_brand_raw.strip() if p1_brand_raw else ''
        p2_brand = p2_brand_raw.strip() if p2_brand_raw else ''

        # 1. CHECK BRANDS - Are these the same product?
        brand_match, brand_score = self.are_brands_same_product(p1_brand, p2_brand)

        if not brand_match and p1_brand and p2_brand:
            issues.append(f"Different Products: '{p1_brand}' vs '{p2_brand}' (similarity: {brand_score:.1f}%)")

        # 2. CHECK PRICES - Significant price differences only
        price_issue = False
        price_details = ""

        if p1_offer is not None and p2_offer is not None:
            price_diff = abs(float(p1_offer) - float(p2_offer))
            if price_diff > self.price_tolerance:
                price_issue = True
                issues.append(f"Price Difference: ${p1_offer} vs ${p2_offer} (diff: ${price_diff:.2f})")
                price_details = f"${price_diff:.2f} difference"
        elif p1_offer is not None and p2_offer is None:
            price_issue = True
            issues.append(f"Missing Price: C1=${p1_offer}, C2=No Price")
            price_details = "Missing price in catalog 2"
        elif p1_offer is None and p2_offer is not None:
            price_issue = True
            issues.append(f"Missing Price: C1=No Price, C2=${p2_offer}")
            price_details = "Missing price in catalog 1"

        # Determine overall result
        if issues:
            if not brand_match and p1_brand and p2_brand:
                result = "INCORRECT - Different Product"
                issue_type = "Different Product"
            elif price_issue:
                result = "INCORRECT - Price Issue"
                issue_type = "Price Difference"
            else:
                result = "INCORRECT - Multiple Issues"
                issue_type = "Multiple Issues"
        else:
            result = "CORRECT"
            issue_type = "Match Confirmed"
            issues.append("Same product, same price - minor text differences ignored")

        # Format displays
        p1_display = self.format_product_display(product1)
        p2_display = self.format_product_display(product2)

        return {
            f"{catalog1_name}_details": p1_display,
            f"{catalog2_name}_details": p2_display,
            "comparison_result": result,
            "issue_type": issue_type,
            "details": "; ".join(issues) if issues else "Products match",
            "price_match": "YES" if not price_issue else "NO",
            "brand_match": "YES" if brand_match else f"NO ({brand_score:.1f}%)",
            "brand_similarity": f"{brand_score:.1f}%"
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
        """Export practical comparison results"""

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
            "details",
            "price_match",
            "brand_match",
            "brand_similarity"
        ]

        # Ensure all columns exist
        for col in column_order:
            if col not in df.columns:
                df[col] = "N/A"

        df = df.reindex(columns=column_order)

        # Rename columns
        df.columns = [
            f"{catalog1_name} Details",
            f"{catalog2_name} Details",
            "Comparison Result",
            "Issue Type",
            "Details",
            "Price Match",
            "Brand Match",
            "Brand Similarity %"
        ]

        # Create practical summary
        total_rows = len(comparison_result["comparison_rows"])
        correct_matches = len([r for r in comparison_result["comparison_rows"] if r.get("comparison_result", "").startswith("CORRECT")])
        price_issues = len([r for r in comparison_result["comparison_rows"] if r.get("issue_type") == "Price Difference"])
        different_products = len([r for r in comparison_result["comparison_rows"] if r.get("issue_type") == "Different Product"])
        missing_products = len([r for r in comparison_result["comparison_rows"] if r.get("issue_type") == "Missing Product"])

        summary_data = {
            "Metric": [
                "Total Comparisons",
                "Correct Matches",
                "Price Issues",
                "Different Products",
                "Missing Products",
                "Match Rate (%)",
                "Price Tolerance Used",
                "Brand Similarity Threshold",
                "Comparison Focus"
            ],
            "Value": [
                total_rows,
                correct_matches,
                price_issues,
                different_products,
                missing_products,
                f"{(correct_matches/max(total_rows,1)*100):.1f}%",
                f"${comparison_result['comparison_criteria']['price_tolerance']}",
                f"{comparison_result['comparison_criteria']['brand_similarity_threshold']}%",
                comparison_result['comparison_criteria']['focus']
            ]
        }
        summary_df = pd.DataFrame(summary_data)

        # Export to Excel
        output_path = Path(output_path)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Practical_Comparison', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        logger.info(f"Practical comparison exported to {output_path}")
        print(f"\nPRACTICAL COMPARISON SUMMARY:")
        print(f"Total comparisons: {total_rows}")
        print(f"Correct matches: {correct_matches}")
        print(f"Price issues: {price_issues}")
        print(f"Different products: {different_products}")
        print(f"Missing products: {missing_products}")
        print(f"Match rate: {(correct_matches/max(total_rows,1)*100):.1f}%")

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

        # Export results
        comparator.export_practical_comparison(results, output_path)

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
    roboflow_version: int = 1
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
    """

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
# backend_processor_visual_enhanced.py
# This shows exactly how to modify your existing backend_processor.py

# Add these imports to your existing backend_processor.py
from PIL import ImageDraw, ImageFont
import tempfile
import base64
from pathlib import Path
import time

# ========================================
# VISUAL COMPARISON FUNCTIONS TO ADD TO YOUR BACKEND_PROCESSOR.PY
# ========================================

def create_visual_comparison_for_files(final_product_items_file1, final_product_items_file2, 
                                     comparison_report, file1_page_data, file2_page_data,
                                     output_directory: str) -> Dict:
    """
    Create visual comparison output for your backend processor.
    Add this function to your backend_processor.py
    """
    logger.info("Creating visual comparison for processed files...")
    
    # Create visual output directory
    visual_dir = Path(output_directory) / "visual_comparison"
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    visual_results = {
        "side_by_side_comparisons": [],
        "individual_highlights": [],
        "summary_dashboard": None,
        "total_files_generated": 0
    }
    
    # Get number of pages
    num_pages_file1 = len(file1_page_data["page_pils_list"])
    num_pages_file2 = len(file2_page_data["page_pils_list"])
    max_pages = max(num_pages_file1, num_pages_file2)
    
    # Create visual comparison for each page
    for page_idx in range(max_pages):
        logger.info(f"Creating visual comparison for page {page_idx + 1}")
        
        # Filter items for current page
        page1_items = [item for item in final_product_items_file1 
                      if item.get("page_idx_for_reprocessing") == page_idx]
        page2_items = [item for item in final_product_items_file2 
                      if item.get("page_idx_for_reprocessing") == page_idx]
        
        # Filter comparison report for current page
        page_comparison_items = []
        for report_item in comparison_report:
            p1_box_id = report_item.get("P1_Box_ID")
            p2_box_id = report_item.get("P2_Box_ID")
            
            page1_match = any(item.get("product_box_id") == p1_box_id for item in page1_items)
            page2_match = any(item.get("product_box_id") == p2_box_id for item in page2_items)
            
            if page1_match or page2_match:
                page_comparison_items.append(report_item)
        
        if page1_items or page2_items:
            # Get PIL images for this page
            page1_pil = file1_page_data["page_pils_list"][page_idx] if page_idx < num_pages_file1 else None
            page2_pil = file2_page_data["page_pils_list"][page_idx] if page_idx < num_pages_file2 else None
            
            # Create side-by-side comparison for this page
            comparison_path = create_side_by_side_page_comparison(
                page1_items, page2_items, page_comparison_items,
                page1_pil, page2_pil, 
                str(visual_dir / f"page_{page_idx + 1}_comparison.jpg"),
                f"File1 Page {page_idx + 1}", f"File2 Page {page_idx + 1}"
            )
            
            if comparison_path:
                visual_results["side_by_side_comparisons"].append(comparison_path)
                visual_results["total_files_generated"] += 1
            
            # Create individual highlights for items with differences
            individual_highlights = create_individual_item_highlights(
                page1_items, page2_items, page_comparison_items,
                page1_pil, page2_pil, str(visual_dir / f"page_{page_idx + 1}_highlights")
            )
            
            visual_results["individual_highlights"].extend(individual_highlights)
            visual_results["total_files_generated"] += len(individual_highlights)
    
    # Create overall summary dashboard
    summary_path = create_comparison_summary_dashboard(
        comparison_report, final_product_items_file1, final_product_items_file2,
        str(visual_dir / "summary_dashboard.jpg")
    )
    
    if summary_path:
        visual_results["summary_dashboard"] = summary_path
        visual_results["total_files_generated"] += 1
    
    logger.info(f"Visual comparison complete: {visual_results['total_files_generated']} files generated")
    return visual_results

def create_side_by_side_page_comparison(page1_items, page2_items, page_comparison_items,
                                      page1_pil, page2_pil, output_path, 
                                      page1_name, page2_name) -> str:
    """
    Create side-by-side comparison of two pages with highlighted differences.
    Similar to your draw_highlights_on_full_page_v2 but creates comparison layout.
    """
    if not page1_items and not page2_items:
        return None
    
    # Configuration
    max_items = max(len(page1_items), len(page2_items))
    item_width, item_height = 300, 400
    padding = 20
    header_height = 100
    gap_between_pages = 50
    
    # Calculate layout
    items_per_column = min(4, max_items)
    columns = (max_items + items_per_column - 1) // items_per_column
    
    # Canvas dimensions  
    page_width = item_width * columns + padding * (columns + 1)
    canvas_width = page_width * 2 + gap_between_pages
    canvas_height = header_height + item_height * items_per_column + padding * (items_per_column + 1)
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Draw header
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        subtitle_font = ImageFont.truetype("arial.ttf", 18)
    except:
        title_font = subtitle_font = ImageFont.load_default()
    
    # Main title
    title = f"Page Comparison: {page1_name} vs {page2_name}"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = bbox[2] - bbox[0]
    draw.text(((canvas_width - title_width) // 2, 10), title, fill='black', font=title_font)
    
    # Page labels
    draw.text((page_width // 4, 50), page1_name, fill='blue', font=subtitle_font)
    draw.text((page_width + gap_between_pages + page_width // 4, 50), page2_name, fill='blue', font=subtitle_font)
    
    # Divider line
    divider_x = page_width + gap_between_pages // 2
    draw.line([(divider_x, header_height), (divider_x, canvas_height)], fill='gray', width=3)
    
    # Draw items for page 1
    if page1_items and page1_pil:
        draw_page_items_with_highlights(
            canvas, page1_items, page_comparison_items, page1_pil,
            (padding, header_height, page_width - padding, canvas_height - padding),
            "file1", items_per_column
        )
    
    # Draw items for page 2
    if page2_items and page2_pil:
        draw_page_items_with_highlights(
            canvas, page2_items, page_comparison_items, page2_pil,
            (page_width + gap_between_pages + padding, header_height, 
             canvas_width - padding, canvas_height - padding),
            "file2", items_per_column
        )
    
    # Add legend
    add_visual_comparison_legend(canvas, canvas_width, canvas_height)
    
    # Save result
    canvas.save(output_path, "JPEG", quality=95)
    logger.info(f"Side-by-side comparison saved: {output_path}")
    return output_path

def draw_page_items_with_highlights(canvas, page_items, comparison_items, page_pil,
                                  area_bbox, file_type, items_per_column):
    """
    Draw items from a page with visual highlighting based on comparison results.
    Adapted from your existing draw_highlights_on_full_page_v2 function.
    """
    area_x, area_y, area_width, area_height = area_bbox
    available_width = area_width - area_x
    available_height = area_height - area_y
    
    # Calculate item layout
    item_width = min(280, available_width // max(1, len(page_items) // items_per_column + 1))
    item_height = min(350, available_height // items_per_column)
    
    # Color scheme (same as your backend_processor)
    colors = {
        'perfect_match': (0, 255, 0),      # Green
        'price_mismatch': (255, 165, 0),   # Orange
        'different_product': (255, 0, 0),  # Red
        'missing_item': (128, 128, 128),   # Gray
        'unmatched': (0, 0, 255) if file_type == "file1" else (255, 0, 255)  # Blue/Magenta
    }
    
    for idx, item in enumerate(page_items):
        # Calculate position
        row = idx % items_per_column
        col = idx // items_per_column
        
        x = area_x + col * (item_width + 10)
        y = area_y + row * (item_height + 10)
        
        # Find comparison result for this item
        comparison_status = find_item_comparison_status(item, comparison_items, file_type)
        border_color = colors.get(comparison_status['type'], colors['perfect_match'])
        
        # Get product segment from original page
        if item.get("roboflow_box_coords_pixels_center_wh"):
            segment_img = extract_product_segment_from_page(
                page_pil, item["roboflow_box_coords_pixels_center_wh"]
            )
            
            if segment_img:
                # Resize segment to fit
                segment_resized = segment_img.resize((item_width - 10, item_height - 30), 
                                                   Image.Resampling.LANCZOS)
                canvas.paste(segment_resized, (x + 5, y + 5))
        
        # Draw border based on comparison status
        draw = ImageDraw.Draw(canvas)
        border_width = 4
        for i in range(border_width):
            draw.rectangle([x + i, y + i, x + item_width - i - 1, y + item_height - i - 1], 
                          outline=border_color, fill=None)
        
        # Add status label
        status_text = comparison_status['label']
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Label background
        draw.rectangle([x, y + item_height - 25, x + item_width, y + item_height], 
                      fill='white', outline='black')
        draw.text((x + 5, y + item_height - 22), status_text, fill='black', font=font)

def find_item_comparison_status(item, comparison_items, file_type) -> Dict:
    """
    Find the comparison status for an item based on comparison results.
    """
    item_box_id = item.get("product_box_id")
    
    for comp_item in comparison_items:
        if file_type == "file1" and comp_item.get("P1_Box_ID") == item_box_id:
            return analyze_comparison_status(comp_item, "P1")
        elif file_type == "file2" and comp_item.get("P2_Box_ID") == item_box_id:
            return analyze_comparison_status(comp_item, "P2")
    
    # No comparison found - item is unmatched
    return {
        'type': 'unmatched',
        'label': f'Unmatched in {file_type.upper()}'
    }

def analyze_comparison_status(comparison_item, position) -> Dict:
    """
    Analyze a comparison item to determine visual status.
    """
    comparison_type = comparison_item.get("Comparison_Type", "")
    differences = comparison_item.get("Differences", "")
    
    if "Attributes OK" in comparison_type:
        return {'type': 'perfect_match', 'label': 'Perfect Match'}
    elif "Price" in differences:
        return {'type': 'price_mismatch', 'label': 'Price Difference'}
    elif "Different Product" in comparison_type:
        return {'type': 'different_product', 'label': 'Different Product'}
    elif "Size" in differences:
        return {'type': 'price_mismatch', 'label': 'Size Difference'}
    elif "Unmatched" in comparison_type:
        return {'type': 'unmatched', 'label': 'Unmatched'}
    
    return {'type': 'perfect_match', 'label': 'Unknown Status'}

def extract_product_segment_from_page(page_pil, box_coords):
    """
    Extract product segment from page PIL image using roboflow coordinates.
    Reuse your existing get_segment_image_bytes logic but return PIL image.
    """
    try:
        cx, cy, w, h = box_coords['x'], box_coords['y'], box_coords['width'], box_coords['height']
        
        # Add padding
        padding_factor = 0.05
        padding_x = int(w * padding_factor)
        padding_y = int(h * padding_factor)
        
        x_min = int(cx - w / 2) - padding_x
        y_min = int(cy - h / 2) - padding_y
        x_max = int(cx + w / 2) + padding_x
        y_max = int(cy + h / 2) + padding_y
        
        # Clamp to image bounds
        img_width, img_height = page_pil.size
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)
        
        if x_min >= x_max or y_min >= y_max:
            return None
        
        return page_pil.crop((x_min, y_min, x_max, y_max))
        
    except Exception as e:
        logger.error(f"Error extracting product segment: {e}")
        return None

def create_individual_item_highlights(page1_items, page2_items, page_comparison_items,
                                    page1_pil, page2_pil, highlights_dir) -> List[str]:
    """
    Create individual highlighted images for items with differences.
    """
    highlights_path = Path(highlights_dir)
    highlights_path.mkdir(parents=True, exist_ok=True)
    
    highlighted_files = []
    
    for comp_item in page_comparison_items:
        if "INCORRECT" not in comp_item.get("Comparison_Type", ""):
            continue  # Skip perfect matches
        
        differences = comp_item.get("Differences", "").split(';')
        differences = [d.strip() for d in differences if d.strip()]
        
        if not differences:
            continue
        
        # Highlight item from file 1
        p1_box_id = comp_item.get("P1_Box_ID")
        if p1_box_id:
            item1 = next((item for item in page1_items 
                         if item.get("product_box_id") == p1_box_id), None)
            if item1 and page1_pil:
                highlight_path = create_single_item_highlight(
                    item1, page1_pil, differences, comp_item,
                    str(highlights_path / f"file1_{item1.get('product_box_id', 'unknown')}.jpg")
                )
                if highlight_path:
                    highlighted_files.append(highlight_path)
        
        # Highlight item from file 2
        p2_box_id = comp_item.get("P2_Box_ID")
        if p2_box_id:
            item2 = next((item for item in page2_items 
                         if item.get("product_box_id") == p2_box_id), None)
            if item2 and page2_pil:
                highlight_path = create_single_item_highlight(
                    item2, page2_pil, differences, comp_item,
                    str(highlights_path / f"file2_{item2.get('product_box_id', 'unknown')}.jpg")
                )
                if highlight_path:
                    highlighted_files.append(highlight_path)
    
    return highlighted_files

def create_single_item_highlight(item, page_pil, differences, comparison_item, output_path) -> str:
    """
    Create a highlighted version of a single item showing differences.
    """
    try:
        # Extract product segment
        if not item.get("roboflow_box_coords_pixels_center_wh"):
            return None
        
        segment_img = extract_product_segment_from_page(
            page_pil, item["roboflow_box_coords_pixels_center_wh"]
        )
        
        if not segment_img:
            return None
        
        # Determine highlight color
        if any('Price' in diff for diff in differences):
            highlight_color = (255, 165, 0)  # Orange
        elif 'Different Product' in comparison_item.get('Comparison_Type', ''):
            highlight_color = (255, 0, 0)   # Red
        else:
            highlight_color = (255, 165, 0)  # Orange (default)
        
        # Add highlighting
        highlighted_img = add_visual_highlights_to_segment(segment_img, highlight_color, differences)
        
        # Save highlighted image
        highlighted_img.save(output_path, "JPEG", quality=95)
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating single item highlight: {e}")
        return None

def add_visual_highlights_to_segment(segment_img, border_color, differences):
    """
    Add visual highlighting to a product segment image.
    """
    highlighted = segment_img.copy()
    draw = ImageDraw.Draw(highlighted)
    
    # Add colorful border
    border_width = max(5, int(min(segment_img.width, segment_img.height) * 0.03))
    for i in range(border_width):
        draw.rectangle([i, i, segment_img.width-1-i, segment_img.height-1-i], 
                      outline=border_color, fill=None)
    
    # Add difference text overlay
    if differences:
        add_difference_text_to_image(draw, highlighted, differences, border_color)
    
    return highlighted

def add_difference_text_to_image(draw, image, differences, color):
    """
    Add text overlay showing differences on the image.
    """
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Create semi-transparent overlay
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    y_offset = 10
    for i, diff in enumerate(differences[:3]):  # Show max 3 differences
        # Truncate long text
        diff_text = diff[:35] + "..." if len(diff) > 35 else diff
        
        # Text background
        bbox = overlay_draw.textbbox((0, 0), diff_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Background rectangle with transparency
        overlay_draw.rectangle([5, y_offset-2, 5+text_width+6, y_offset+text_height+2], 
                             fill=(255, 255, 255, 200))
        overlay_draw.rectangle([5, y_offset-2, 5+text_width+6, y_offset+text_height+2], 
                             outline=color, width=2)
        
        # Text
        overlay_draw.text((7, y_offset), diff_text, fill='black', font=font)
        y_offset += text_height + 8
    
    # Composite overlay onto main image
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    result = Image.alpha_composite(image, overlay)
    return result.convert('RGB')

def create_comparison_summary_dashboard(comparison_report, final_items_file1, 
                                      final_items_file2, output_path) -> str:
    """
    Create a summary dashboard showing overall comparison statistics.
    """
    # Calculate statistics
    total_items_file1 = len(final_items_file1)
    total_items_file2 = len(final_items_file2)
    total_comparisons = len(comparison_report)
    
    perfect_matches = len([r for r in comparison_report if 'Attributes OK' in r.get('Comparison_Type', '')])
    price_mismatches = len([r for r in comparison_report if 'Price' in r.get('Differences', '')])
    different_products = len([r for r in comparison_report if 'Different Product' in r.get('Comparison_Type', '')])
    unmatched_items = len([r for r in comparison_report if 'Unmatched' in r.get('Comparison_Type', '')])
    
    # Create dashboard
    width, height = 900, 700
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    try:
        title_font = ImageFont.truetype("arial.ttf", 36)
        header_font = ImageFont.truetype("arial.ttf", 24)
        text_font = ImageFont.truetype("arial.ttf", 18)
        large_font = ImageFont.truetype("arial.ttf", 48)
    except:
        title_font = header_font = text_font = large_font = ImageFont.load_default()
    
    # Title
    title = "Catalog Comparison Dashboard"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = bbox[2] - bbox[0]
    draw.text(((width - title_width) // 2, 20), title, fill='black', font=title_font)
    
    # Subtitle
    subtitle = f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}"
    bbox_sub = draw.textbbox((0, 0), subtitle, font=text_font)
    sub_width = bbox_sub[2] - bbox_sub[0]
    draw.text(((width - sub_width) // 2, 70), subtitle, fill='gray', font=text_font)
    
    # Statistics in a grid
    stats = [
        ("Total Items File 1", total_items_file1, (70, 130, 200), 'blue'),
        ("Total Items File 2", total_items_file2, (250, 130, 200), 'blue'),
        ("Total Comparisons", total_comparisons, (450, 130, 200), 'purple'),
        ("Perfect Matches", perfect_matches, (70, 300, 200), 'green'),
        ("Price Differences", price_mismatches, (250, 300, 200), 'orange'),
        ("Different Products", different_products, (450, 300, 200), 'red'),
        ("Unmatched Items", unmatched_items, (650, 300, 200), 'gray')
    ]
    
    for label, value, (x, y, box_width), color in stats:
        box_height = 120
        
        # Draw box
        draw.rectangle([x, y, x + box_width, y + box_height], 
                      fill='white', outline=color, width=3)
        
        # Value (large)
        value_text = str(value)
        bbox_val = draw.textbbox((0, 0), value_text, font=large_font)
        val_width = bbox_val[2] - bbox_val[0]
        draw.text((x + (box_width - val_width) // 2, y + 20), value_text, 
                 fill=color, font=large_font)
        
        # Label (smaller)
        bbox_label = draw.textbbox((0, 0), label, font=text_font)
        label_width = bbox_label[2] - bbox_label[0]
        
        # Multi-line label if needed
        if label_width > box_width - 10:
            words = label.split()
            mid = len(words) // 2
            line1 = ' '.join(words[:mid])
            line2 = ' '.join(words[mid:])
            
            bbox1 = draw.textbbox((0, 0), line1, font=text_font)
            bbox2 = draw.textbbox((0, 0), line2, font=text_font)
            w1, w2 = bbox1[2] - bbox1[0], bbox2[2] - bbox2[0]
            
            draw.text((x + (box_width - w1) // 2, y + 80), line1, fill='black', font=text_font)
            draw.text((x + (box_width - w2) // 2, y + 100), line2, fill='black', font=text_font)
        else:
            draw.text((x + (box_width - label_width) // 2, y + 85), label, fill='black', font=text_font)
    
    # Match rate calculation and display
    if total_comparisons > 0:
        match_rate = (perfect_matches / total_comparisons) * 100
        match_text = f"Overall Match Rate: {match_rate:.1f}%"
        
        bbox_match = draw.textbbox((0, 0), match_text, font=header_font)
        match_width = bbox_match[2] - bbox_match[0]
        
        # Color based on match rate
        if match_rate >= 80:
            rate_color = 'green'
        elif match_rate >= 60:
            rate_color = 'orange'
        else:
            rate_color = 'red'
        
        draw.text(((width - match_width) // 2, height - 80), match_text, 
                 fill=rate_color, font=header_font)
        
        # Progress bar for match rate
        bar_width = 400
        bar_height = 20
        bar_x = (width - bar_width) // 2
        bar_y = height - 50
        
        # Background bar
        draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height], 
                      fill='lightgray', outline='black')
        
        # Progress bar
        progress_width = int((match_rate / 100) * bar_width)
        draw.rectangle([bar_x, bar_y, bar_x + progress_width, bar_y + bar_height], 
                      fill=rate_color, outline='black')
    
    canvas.save(output_path, "JPEG", quality=95)
    logger.info(f"Summary dashboard saved: {output_path}")
    return output_path

def add_visual_comparison_legend(canvas, canvas_width, canvas_height):
    """
    Add legend explaining the visual comparison colors.
    """
    legend_items = [
        ('Perfect Match', (0, 255, 0)),
        ('Price Difference', (255, 165, 0)),
        ('Different Product', (255, 0, 0)),
        ('Unmatched File 1', (0, 0, 255)),
        ('Unmatched File 2', (255, 0, 255))
    ]
    
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Position legend
    legend_width = 160
    legend_height = len(legend_items) * 18 + 25
    legend_x = canvas_width - legend_width - 15
    legend_y = canvas_height - legend_height - 15
    
    # Legend background
    draw.rectangle([legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
                  fill='white', outline='black', width=2)
    
    # Legend title
    draw.text((legend_x + 10, legend_y + 5), "Legend:", fill='black', font=font)
    
    # Legend items
    for i, (label, color) in enumerate(legend_items):
        item_y = legend_y + 22 + i * 18
        
        # Color box
        draw.rectangle([legend_x + 10, item_y, legend_x + 20, item_y + 10], 
                      fill=color, outline='black')
        
        # Label
        draw.text((legend_x + 25, item_y - 2), label, fill='black', font=font)

# ========================================
# MODIFY YOUR EXISTING process_files_for_comparison FUNCTION
# ========================================

def process_files_for_comparison_with_visual(file1_bytes, file1_name, file2_bytes, file2_name):
    """
    Enhanced version of your process_files_for_comparison function that includes visual comparison.
    Replace your existing function with this enhanced version.
    """
    request_id = f"req_{int(time.time())}"
    logger.info(f"REQUEST_ID: {request_id} - Backend processing with visual comparison started")

    # Your existing processing logic here...
    # (Keep all your existing code until the final response creation)
    
    # [ALL YOUR EXISTING CODE FROM process_files_for_comparison GOES HERE]
    # ...
    # Just before creating final_response_dict, add this:
    
    try:
        # Create visual comparison
        logger.info(f"REQUEST_ID: {request_id} - Creating visual comparison output...")
        
        visual_results = create_visual_comparison_for_files(
            final_product_items_file1,
            final_product_items_file2,
            product_centric_comparison_report,
            file1_page_data,
            file2_page_data,
            f"visual_output_{request_id}"
        )
        
        logger.info(f"REQUEST_ID: {request_id} - Visual comparison created: {visual_results['total_files_generated']} files")
        
    except Exception as e:
        logger.error(f"REQUEST_ID: {request_id} - Visual comparison failed: {e}")
        visual_results = {"error": str(e), "total_files_generated": 0}

    # Create enhanced final response that includes visual comparison
    final_response_dict = {
        "message": "Backend processing complete with visual comparison.",
        "product_items_file1_count": len(final_product_items_file1),
        "product_items_file2_count": len(final_product_items_file2),
        "product_comparison_details": product_centric_comparison_report,
        "report_csv_data": csv_data_string,
        "all_product_details_file1": final_product_items_file1,
        "all_product_details_file2": final_product_items_file2,
        
        # NEW: Visual comparison results
        "visual_comparison_results": visual_results,
        "has_visual_comparison": visual_results.get("total_files_generated", 0) > 0,
        
        # NEW: Base64 encoded visual outputs for immediate display
        "visual_comparison_base64": convert_visual_files_to_base64(visual_results)
    }
    
    return final_response_dict

def convert_visual_files_to_base64(visual_results) -> Dict:
    """
    Convert visual comparison files to base64 for immediate display in frontend.
    """
    base64_results = {
        "side_by_side_comparisons": [],
        "summary_dashboard": None,
        "total_images": 0
    }
    
    try:
        # Convert side-by-side comparisons
        for comparison_path in visual_results.get("side_by_side_comparisons", []):
            if os.path.exists(comparison_path):
                with open(comparison_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    base64_results["side_by_side_comparisons"].append({
                        "filename": Path(comparison_path).name,
                        "base64_data": img_base64,
                        "path": comparison_path
                    })
                    base64_results["total_images"] += 1
        
        # Convert summary dashboard
        summary_path = visual_results.get("summary_dashboard")
        if summary_path and os.path.exists(summary_path):
            with open(summary_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                base64_results["summary_dashboard"] = {
                    "filename": Path(summary_path).name,
                    "base64_data": img_base64,
                    "path": summary_path
                }
                base64_results["total_images"] += 1
        
        logger.info(f"Converted {base64_results['total_images']} visual files to base64")
        
    except Exception as e:
        logger.error(f"Error converting visual files to base64: {e}")
        base64_results["error"] = str(e)
    
    return base64_results

# ========================================
# STREAMLIT INTEGRATION EXAMPLE
# ========================================

def display_visual_comparison_in_streamlit(visual_comparison_base64):
    """
    Example function showing how to display visual comparison results in Streamlit.
    Add this to your Streamlit app.
    """
    import streamlit as st
    
    if not visual_comparison_base64.get("total_images", 0):
        st.warning("No visual comparison images generated.")
        return
    
    st.header("📊 Visual Comparison Results")
    
    # Display summary dashboard
    if visual_comparison_base64.get("summary_dashboard"):
        st.subheader("Summary Dashboard")
        dashboard_data = visual_comparison_base64["summary_dashboard"]["base64_data"]
        st.image(f"data:image/jpeg;base64,{dashboard_data}", 
                caption="Comparison Summary Dashboard", use_column_width=True)
        
        # Download button for dashboard
        st.download_button(
            label="Download Summary Dashboard",
            data=base64.b64decode(dashboard_data),
            file_name="comparison_summary_dashboard.jpg",
            mime="image/jpeg"
        )
    
    # Display side-by-side comparisons
    side_by_side = visual_comparison_base64.get("side_by_side_comparisons", [])
    if side_by_side:
        st.subheader("Page-by-Page Visual Comparisons")
        
        # Create tabs for each page comparison
        if len(side_by_side) > 1:
            tabs = st.tabs([f"Page {i+1}" for i in range(len(side_by_side))])
            
            for i, (tab, comparison) in enumerate(zip(tabs, side_by_side)):
                with tab:
                    st.image(f"data:image/jpeg;base64,{comparison['base64_data']}", 
                            caption=f"Page {i+1} Comparison", use_column_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label=f"Download Page {i+1} Comparison",
                            data=base64.b64decode(comparison['base64_data']),
                            file_name=comparison['filename'],
                            mime="image/jpeg",
                            key=f"download_page_{i}"
                        )
                    with col2:
                        if st.button(f"🔍 Analyze Page {i+1}", key=f"analyze_page_{i}"):
                            st.info("Detailed analysis functionality can be added here")
        else:
            # Single page comparison
            comparison = side_by_side[0]
            st.image(f"data:image/jpeg;base64,{comparison['base64_data']}", 
                    caption="Visual Comparison", use_column_width=True)
            
            st.download_button(
                label="Download Visual Comparison",
                data=base64.b64decode(comparison['base64_data']),
                file_name=comparison['filename'],
                mime="image/jpeg"
            )
    
    # Visual comparison insights
    st.subheader("🔍 Visual Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Visual Files", visual_comparison_base64["total_images"])
    with col2:
        st.metric("Page Comparisons", len(side_by_side))
    with col3:
        dashboard_available = "✅" if visual_comparison_base64.get("summary_dashboard") else "❌"
        st.metric("Summary Dashboard", dashboard_available)

# ========================================
# USAGE EXAMPLE WITH YOUR EXISTING PIPELINE
# ========================================

def example_usage_with_visual_comparison():
    """
    Example showing how to use the enhanced pipeline with visual comparison.
    """
    
    # Your existing imports and setup
    # from your_module import process_files_for_comparison_with_visual
    
    # Example usage
    with open("catalog1.pdf", "rb") as f1:
        file1_bytes = f1.read()
    
    with open("catalog2.pdf", "rb") as f2:
        file2_bytes = f2.read()
    
    # Process with visual comparison
    results = process_files_for_comparison_with_visual(
        file1_bytes, "catalog1.pdf",
        file2_bytes, "catalog2.pdf"
    )
    
    # Check if visual comparison was successful
    if results.get("has_visual_comparison"):
        print("✅ Visual comparison generated successfully!")
        
        visual_results = results["visual_comparison_results"]
        print(f"Generated files:")
        print(f"  - Side-by-side comparisons: {len(visual_results['side_by_side_comparisons'])}")
        print(f"  - Individual highlights: {len(visual_results['individual_highlights'])}")
        print(f"  - Summary dashboard: {'Yes' if visual_results['summary_dashboard'] else 'No'}")
        
        # Access base64 data for immediate display
        base64_data = results["visual_comparison_base64"]
        print(f"Base64 images ready for display: {base64_data['total_images']}")
        
        # Example: Save base64 images to HTML for viewing
        create_html_visual_report(base64_data, "visual_comparison_report.html")
        
    else:
        print("❌ Visual comparison failed or not generated")
        if "error" in results.get("visual_comparison_results", {}):
            print(f"Error: {results['visual_comparison_results']['error']}")

def create_html_visual_report(base64_data, output_path):
    """
    Create an HTML report with embedded base64 images for easy viewing.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Visual Catalog Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin: 30px 0; }}
            .comparison-image {{ max-width: 100%; border: 2px solid #ddd; border-radius: 8px; margin: 10px 0; }}
            .download-btn {{ background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 10px 5px; display: inline-block; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .stat-box {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
            .legend {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Visual Catalog Comparison Report</h1>
                <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>{base64_data['total_images']}</h3>
                    <p>Total Visual Files</p>
                </div>
                <div class="stat-box">
                    <h3>{len(base64_data.get('side_by_side_comparisons', []))}</h3>
                    <p>Page Comparisons</p>
                </div>
                <div class="stat-box">
                    <h3>{'✅' if base64_data.get('summary_dashboard') else '❌'}</h3>
                    <p>Summary Dashboard</p>
                </div>
            </div>
            
            <div class="legend">
                <h3>🎨 Color Legend</h3>
                <p><span style="color: green;">■</span> Perfect Match &nbsp;&nbsp;
                   <span style="color: orange;">■</span> Price Difference &nbsp;&nbsp;
                   <span style="color: red;">■</span> Different Product &nbsp;&nbsp;
                   <span style="color: blue;">■</span> Missing in File 1 &nbsp;&nbsp;
                   <span style="color: magenta;">■</span> Missing in File 2</p>
            </div>
    """
    
    # Add summary dashboard
    if base64_data.get("summary_dashboard"):
        dashboard = base64_data["summary_dashboard"]
        html_content += f"""
            <div class="section">
                <h2>📈 Summary Dashboard</h2>
                <img src="data:image/jpeg;base64,{dashboard['base64_data']}" 
                     class="comparison-image" alt="Summary Dashboard">
            </div>
        """
    
    # Add page comparisons
    side_by_side = base64_data.get("side_by_side_comparisons", [])
    if side_by_side:
        html_content += """
            <div class="section">
                <h2>📋 Page-by-Page Comparisons</h2>
        """
        
        for i, comparison in enumerate(side_by_side):
            html_content += f"""
                <div style="margin: 30px 0;">
                    <h3>Page {i+1} Comparison</h3>
                    <img src="data:image/jpeg;base64,{comparison['base64_data']}" 
                         class="comparison-image" alt="Page {i+1} Comparison">
                </div>
            """
        
        html_content += "</div>"
    
    html_content += """
            <div class="section">
                <h2>💡 How to Interpret the Visual Comparison</h2>
                <ul>
                    <li><strong>Green borders:</strong> Products match perfectly</li>
                    <li><strong>Orange borders:</strong> Same product with price or attribute differences</li>
                    <li><strong>Red borders:</strong> Completely different products in same position</li>
                    <li><strong>Blue/Magenta borders:</strong> Products missing from one catalog</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 HTML visual report saved to: {output_path}")

# ========================================
# FINAL INTEGRATION CHECKLIST
# ========================================

"""
INTEGRATION CHECKLIST - How to add visual comparison to your existing code:

1. ✅ Add imports to your backend_processor.py:
   - from PIL import ImageDraw, ImageFont
   - import base64
   - from pathlib import Path

2. ✅ Add all the visual comparison functions to your backend_processor.py:
   - create_visual_comparison_for_files()
   - create_side_by_side_page_comparison()
   - draw_page_items_with_highlights()
   - create_individual_item_highlights()
   - create_comparison_summary_dashboard()
   - convert_visual_files_to_base64()

3. ✅ Replace your process_files_for_comparison function with:
   - process_files_for_comparison_with_visual()

4. ✅ Update your Streamlit app to display visual results:
   - Add display_visual_comparison_in_streamlit() function
   - Call it after processing is complete

5. ✅ Test the integration:
   - Run with your existing PDF pairs
   - Check that visual files are generated
   - Verify base64 encoding works
   - Test Streamlit display

6. ✅ Optional enhancements:
   - Add more color coding options
   - Customize layout and styling
   - Add interactive features in Streamlit
   - Create animated comparisons

BENEFITS YOU'LL GET:

✨ Immediate visual feedback on catalog differences
✨ Color-coded highlighting for different types of mismatches  
✨ Side-by-side page comparisons for easy analysis
✨ Summary dashboard with overall statistics
✨ Individual highlighted images for detailed review
✨ Base64 encoding for instant display in web interfaces
✨ HTML reports for sharing with stakeholders
✨ Seamless integration with your existing pipeline

The visual comparison will make it much easier to:
- Quickly identify problem areas in catalogs
- Present results to non-technical stakeholders  
- Spot patterns in pricing discrepancies
- Validate the accuracy of your comparison algorithm
- Create compelling reports for clients
"""

if __name__ == "__main__":
    print("Visual Comparison Integration for Backend Processor")
    print("=" * 60)
    print("This integration adds powerful visual comparison capabilities")
    print("to your existing catalog comparison pipeline.")
    print()
    print("Key features added:")
    print("✅ Side-by-side visual comparisons")
    print("✅ Color-coded difference highlighting") 
    print("✅ Summary dashboard with statistics")
    print("✅ Individual item highlights")
    print("✅ Base64 encoding for web display")
    print("✅ HTML report generation")
    print()
    print("Follow the integration checklist above to add these features!")

if __name__ == "__main__":
    print("=" * 80)
    print("STREAMLINED CATALOG COMPARISON PIPELINE")
    print("=" * 80)
    print("This script provides a complete automated pipeline for catalog comparison.")
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
    print('    template3_path="/path/to/template3.jpg"')
    print(')')
    print()
    print("Or use the full pipeline function for more control over parameters.")
    print("=" * 80)
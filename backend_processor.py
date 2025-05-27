import logging
import tempfile
import os
import cv2
import base64
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from roboflow import Roboflow
from openai import OpenAI
import json
import pandas as pd
from pdf2image import convert_from_path
import time
import io

# Logging setup (matching frontend)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(levelname)s - PID:%(process)d - [%(name)s - %(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
POPPLER_BIN_PATH = os.getenv('POPPLER_PATH_OVERRIDE')

# Initialize clients
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.project(os.getenv('ROBOFLOW_PROJECT_ID'))
    roboflow_model = project.version(int(os.getenv('ROBOFLOW_VERSION_NUMBER'))).model
    logger.info("Roboflow model initialized.")
except Exception as e:
    logger.error(f"Error initializing Roboflow: {e}", exc_info=True)
    roboflow_model = None

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized.")
except Exception as e:
    logger.error(f"Error initializing OpenAI: {e}", exc_info=True)
    openai_client = None

def crop_detections(pil_image, image_filename, request_id):
    """Uses Roboflow to detect products and crop images (from main.py)."""
    if not roboflow_model:
        logger.error(f"REQUEST_ID: {request_id} - Roboflow model not initialized.")
        return []

    temp_file_path = None
    crops = []
    try:
        # Save PIL image to temporary file
        suffix = ".jpg" if image_filename.lower().endswith((".jpg", ".jpeg")) else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_img:
            pil_image.save(tmp_img.name, format="JPEG" if suffix == ".jpg" else "PNG")
            temp_file_path = tmp_img.name

        logger.info(f"REQUEST_ID: {request_id} - Saved image to {temp_file_path} for Roboflow.")

        # Roboflow inference
        result = roboflow_model.predict(temp_file_path, confidence=40, overlap=30).json()
        predictions = result.get("predictions", [])

        # Read image with OpenCV
        image = cv2.imread(temp_file_path)
        if image is None:
            logger.error(f"REQUEST_ID: {request_id} - Failed to read image: {temp_file_path}")
            return []

        # Crop each detection
        for detection in predictions:
            x_center = int(detection.get('x', 0))
            y_center = int(detection.get('y', 0))
            width = int(detection.get('width', 0))
            height = int(detection.get('height', 0))

            x_min = max(0, x_center - width // 2)
            y_min = max(0, y_center - height // 2)
            x_max = min(image.shape[1], x_center + width // 2)
            y_max = min(image.shape[0], y_center + height // 2)

            if x_min >= x_max or y_min >= y_max:
                logger.warning(f"REQUEST_ID: {request_id} - Invalid crop coordinates: ({x_min}, {y_min}, {x_max}, {y_max})")
                continue

            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            crops.append(cropped_pil)
            logger.debug(f"REQUEST_ID: {request_id} - Cropped image at: ({x_min}, {y_min}, {x_max}, {y_max})")

        return crops

    except Exception as e:
        logger.error(f"REQUEST_ID: {request_id} - Error in crop_detections: {e}", exc_info=True)
        return []
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"REQUEST_ID: {request_id} - Deleted temp file: {temp_file_path}")
            except Exception as e:
                logger.error(f"REQUEST_ID: {request_id} - Error deleting temp file: {e}")

def encode_image_pil(pil_image, item_id):
    """Encodes PIL image to base64 (adapted from main.py)."""
    try:
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"ITEM_ID: {item_id} - Error encoding image: {e}", exc_info=True)
        return None

def vlm(pil_image, request_id, item_id):
    """Extracts product details using GPT-4o (from main.py)."""
    if not openai_client:
        logger.error(f"ITEM_ID: {item_id} - OpenAI client not initialized.")
        return {"image": None, "details": {"error": "OpenAI client not initialized"}}

    try:
        base64_image = encode_image_pil(pil_image, item_id)
        if not base64_image:
            logger.error(f"ITEM_ID: {item_id} - Failed to encode image.")
            return {"image": None, "details": {"error": "Image encoding failed"}}

        prompt = (
            "Look at the provided image and provide the regular price, sale price ('sale_price' should capture the prominently highlighted sale price), "
            "title ('title' should contain the dark black text from the image), and the rest of the details about the product given in the image. "
            "Use the following JSON format, ensuring the entire output is a single valid JSON object: "  # MODIFIED
            "{'regular_price': <regular price you see>, 'sale_price': <sale price you see>, 'item_name': <Name of the product that you see>, "
            "'metadata': <Remaining details regarding the product that you see>}. "
            "Only give the answer in the specified JSON format and don't say anything extra since this JSON object will be used directly in a code block. "  # MODIFIED
            "\n\nNote:\n"
            "- Ensure that all amounts are provided in dollars ($). Convert any value in cents (e.g., 88¢) to dollars (e.g., $0.88).\n"
            "- If you encounter a price format like '7^98', interpret it as $7.98.\n"
            "- If a price is given in a format like '2x6$', interpret it as $3 per unit and not $6 total.\n"
            "- If there is no information given regarding the regular price, sale price, name of the product, or any other remaining details, "
            "ensure that the respective keys have a None value in the JSON object." # MODIFIED (optional, but good for consistency)
        )

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        details = json.loads(response.choices[0].message.content)
        logger.info(f"ITEM_ID: {item_id} - Extracted details: {details}")
        return {"image": base64_image, "details": details}

    except Exception as e:
        logger.error(f"ITEM_ID: {item_id} - Error in VLM: {e}", exc_info=True)
        return {"image": None, "details": {"error": str(e)}}

def process_pdf(file_bytes, file_name, request_id):
    """Processes a PDF, applying main.py logic to each page."""
    temp_pdf_path = None
    product_details = []
    try:
        # Save PDF to temporary file
        fd, temp_pdf_path = tempfile.mkstemp(suffix=".pdf", prefix=f"{request_id}_")
        os.close(fd)
        with open(temp_pdf_path, "wb") as f:
            f.write(file_bytes)
        logger.info(f"REQUEST_ID: {request_id} - Saved PDF to {temp_pdf_path}")

        # Convert PDF to images
        images = convert_from_path(temp_pdf_path, dpi=200, poppler_path=POPPLER_BIN_PATH, fmt='jpeg')
        logger.info(f"REQUEST_ID: {request_id} - Converted PDF {file_name} to {len(images)} images.")

        # Process each page
        for page_idx, image in enumerate(images):
            page_id = f"{request_id}-Page{page_idx}"
            logger.info(f"PAGE_ID: {page_id} - Processing page {page_idx} of {file_name}")

            # Crop detections
            cropped_images = crop_detections(image, f"page_{page_idx}.jpg", page_id)

            # Analyze each cropped image
            for crop_idx, crop in enumerate(cropped_images):
                item_id = f"{page_id}-Item{crop_idx}"
                vlm_result = vlm(crop, page_id, item_id)
                details = vlm_result["details"]
                product_details.append({
                    "item_id": item_id,
                    "item_name": details.get("item_name"),
                    "regular_price": details.get("regular_price"),
                    "sale_price": details.get("sale_price"),
                    "metadata": details.get("metadata"),
                    "error": details.get("error"),
                    "page_idx": page_idx
                })
                logger.info(f"ITEM_ID: {item_id} - Processed product: {details.get('item_name', 'N/A')}")

        return product_details

    except Exception as e:
        logger.error(f"REQUEST_ID: {request_id} - Error processing PDF {file_name}: {e}", exc_info=True)
        return [{"error": str(e)}]
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
                logger.debug(f"REQUEST_ID: {request_id} - Deleted temp PDF: {temp_pdf_path}")
            except Exception as e:
                logger.error(f"REQUEST_ID: {request_id} - Error deleting temp PDF: {e}")

def compare_products(products1, products2, file1_name, file2_name, request_id):
    """Compares products from two PDFs, generating a report for frontend."""
    comparison_details = []
    table_data = []
    csv_data = []

    # Simple comparison by item_name
    products1_dict = {p["item_name"]: p for p in products1 if p.get("item_name") and not p.get("error")}
    products2_dict = {p["item_name"]: p for p in products2 if p.get("item_name") and not p.get("error")}

    # Matches and differences
    for item_name, p1 in products1_dict.items():
        if item_name in products2_dict:
            p2 = products2_dict[item_name]
            differences = []
            if p1["regular_price"] != p2["regular_price"]:
                differences.append(f"Regular Price: {file1_name}=${p1['regular_price'] or 'N/A'} vs {file2_name}=${p2['regular_price'] or 'N/A'}")
            if p1["sale_price"] != p2["sale_price"]:
                differences.append(f"Sale Price: {file1_name}=${p1['sale_price'] or 'N/A'} vs {file2_name}=${p2['sale_price'] or 'N/A'}")
            if p1["metadata"] != p2["metadata"]:
                differences.append(f"Metadata: {file1_name}='{p1['metadata'] or 'N/A'}' vs {file2_name}='{p2['metadata'] or 'N/A'}'")

            comparison_type = "Product Match - Attribute Mismatch" if differences else "Product Match - Attributes OK"
            comparison_details.append({
                "Comparison_Type": comparison_type,
                "P1_Name": item_name,
                "P1_Regular_Price": p1["regular_price"],
                "P1_Sale_Price": p1["sale_price"],
                "P1_Metadata": p1["metadata"],
                "P2_Name": item_name,
                "P2_Regular_Price": p2["regular_price"],
                "P2_Sale_Price": p2["sale_price"],
                "P2_Metadata": p2["metadata"],
                "Differences": "; ".join(differences)
            })

            if differences:
                table_data.append({
                    "Product Name": item_name,
                    f"{file1_name} Issue": differences[0].split(" vs ")[0].split(": ")[1],
                    f"{file2_name} Issue": differences[0].split(" vs ")[1]
                })
                csv_data.append({
                    "Product Name": item_name,
                    "Issue Type": comparison_type,
                    f"{file1_name} Details": differences[0].split(" vs ")[0].split(": ")[1],
                    f"{file2_name} Details": differences[0].split(" vs ")[1],
                    "Raw Differences": "; ".join(differences)
                })

        else:
            comparison_details.append({
                "Comparison_Type": "Unmatched Product in File 1",
                "P1_Name": item_name,
                "P1_Regular_Price": p1["regular_price"],
                "P1_Sale_Price": p1["sale_price"],
                "P1_Metadata": p1["metadata"]
            })
            table_data.append({
                "Product Name": item_name,
                f"{file1_name} Issue": "Product Present",
                f"{file2_name} Issue": "Product Missing"
            })
            csv_data.append({
                "Product Name": item_name,
                "Issue Type": "Unmatched Product in File 1",
                f"{file1_name} Details": "Product Present",
                f"{file2_name} Details": "Product Missing",
                "Raw Differences": ""
            })

    # Unmatched products in File 2
    for item_name, p2 in products2_dict.items():
        if item_name not in products1_dict:
            comparison_details.append({
                "Comparison_Type": "Unmatched Product in File 2 (Extra)",
                "P2_Name": item_name,
                "P2_Regular_Price": p2["regular_price"],
                "P2_Sale_Price": p2["sale_price"],
                "P2_Metadata": p2["metadata"]
            })
            table_data.append({
                "Product Name": item_name,
                f"{file1_name} Issue": "Product Missing",
                f"{file2_name} Issue": "Product Present"
            })
            csv_data.append({
                "Product Name": item_name,
                "Issue Type": "Unmatched Product in File 2 (Extra)",
                f"{file1_name} Details": "Product Missing",
                f"{file2_name} Details": "Product Present",
                "Raw Differences": ""
            })

    # Generate CSV
    csv_df = pd.DataFrame(csv_data)
    csv_buffer = io.StringIO()
    csv_df.to_csv(csv_buffer, index=False, lineterminator='\r\n')
    csv_content = csv_buffer.getvalue().encode('utf-8')

    logger.info(f"REQUEST_ID: {request_id} - Compared {len(products1)} products from {file1_name} with {len(products2)} from {file2_name}. Found {len(comparison_details)} comparison items.")

    return {
        "message": "PDF comparison complete.",
        "product_items_file1_count": len(products1),
        "product_items_file2_count": len(products2),
        "product_comparison_details": comparison_details,
        "table_data": table_data,
        "report_csv_data": csv_content,
        "all_product_details_file1": products1,
        "all_product_details_file2": products2
    }

def process_files_for_comparison(file1_bytes, file1_name, file2_bytes, file2_name):
    """Main function to process and compare two PDFs."""
    request_id = f"req_{int(time.time())}"
    logger.info(f"REQUEST_ID: {request_id} - Comparing PDFs: {file1_name} vs {file2_name}")

    try:
        # Ensure file names are strings
        if isinstance(file1_name, bytes):
            file1_name = file1_name.decode('utf-8')
        if isinstance(file2_name, bytes):
            file2_name = file2_name.decode('utf-8')

        # Validate file types
        if not (file1_name.lower().endswith(".pdf") and file2_name.lower().endswith(".pdf")):
            logger.error(f"REQUEST_ID: {request_id} - Invalid file types: {file1_name}, {file2_name}")
            return {
                "error": "Please upload PDF files.",
                "message": "Error: Invalid file type.",
                "product_items_file1_count": 0,
                "product_items_file2_count": 0,
                "product_comparison_details": [],
                "report_csv_data": "Error,Invalid file type\n".encode('utf-8')
            }

        # Process PDFs
        products1 = process_pdf(file1_bytes, file1_name, request_id)
        products2 = process_pdf(file2_bytes, file2_name, request_id)

        # Compare products
        return compare_products(products1, products2, file1_name, file2_name, request_id)

    except Exception as e:
        logger.error(f"REQUEST_ID: {request_id} - Error in comparison: {e}", exc_info=True)
        return {
            "error": f"Internal server error: {str(e)}",
            "message": "Error: Processing failed.",
            "product_items_file1_count": 0,
            "product_items_file2_count": 0,
            "product_comparison_details": [],
            "report_csv_data": f"Error,Internal server error: {str(e)}\n".encode('utf-8')
        }
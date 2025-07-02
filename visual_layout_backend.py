# visual_layout_backend.py
import openai
import time
import logging
import tempfile
import json
import re
import boto3
from dotenv import load_dotenv
import os
import pandas as pd
import io
from io import BytesIO
from pdf2image import convert_from_path #, pdfinfo_from_path # pdfinfo_from_path not used in your main code
# from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError # Not explicitly handled, convert_from_path will raise generally
from PIL import Image
import base64 # For Vision LLM image encoding
from werkzeug.utils import secure_filename # Useful for sanitizing filenames for S3 keys if needed
from PIL import ImageDraw, ImageFont
from fuzzywuzzy import fuzz
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
import torch

logger = logging.getLogger(__name__)



# Roboflow SDK
ROBOFLOW_SDK_AVAILABLE = False
Roboflow = None
try:
    from roboflow import Roboflow
    ROBOFLOW_SDK_AVAILABLE = True
except ImportError:
    logging.warning("Could not import 'Roboflow'. Ensure 'roboflow' package is installed.")

from fuzzywuzzy import fuzz

load_dotenv()

# --- NEW: Initialize Open-Source Vision Model ---
# Load the model only once when the module is loaded to avoid reloading on every call
try:
    OPEN_SOURCE_VISION_MODEL_ID = "openai/clip-vit-base-patch32"
    vision_model = CLIPModel.from_pretrained(OPEN_SOURCE_VISION_MODEL_ID)
    vision_processor = CLIPProcessor.from_pretrained(OPEN_SOURCE_VISION_MODEL_ID)
    logger.info(f"Successfully loaded open-source vision model: {OPEN_SOURCE_VISION_MODEL_ID}")
except Exception as e:
    vision_model = None
    vision_processor = None
    logger.error(f"Failed to load open-source vision model. Photo mistake check will be disabled. Error: {e}", exc_info=True)
# --- END NEW SECTION ---

# --- Logger Setup ---
# Configure logging (you can simplify or adapt your existing setup)
logger = logging.getLogger(__name__)
if not logger.handlers: # Avoid adding handlers multiple times
    # Use a basicConfig that Streamlit can also pick up or override
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(), # Default to INFO, configurable via .env
        format='%(asctime)s - %(levelname)s - PID:%(process)d - [%(name)s - %(funcName)s:%(lineno)d] - %(message)s'
    )
    # Set log levels for verbose libraries if needed
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('pdf2image').setLevel(logging.INFO)
    logging.getLogger('PIL').setLevel(logging.INFO)
logger.info("Backend Processor Logger Initialized.")


# --- Geometric Merging Tolerances (Tune these based on logs) ---
Y_ALIGN_TOLERANCE_FACTOR = 0.7
X_SPACING_TOLERANCE_FACTOR = 1.7
CENTS_MAX_HEIGHT_FACTOR = 1.2
GEOM_MERGE_MIN_WORD_CONFIDENCE = 70

# --- Environment variables ---
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
ROBOFLOW_PROJECT_ID = os.getenv('ROBOFLOW_PROJECT_ID')
ROBOFLOW_VERSION_NUMBER = os.getenv('ROBOFLOW_VERSION_NUMBER')
POPPLER_BIN_PATH = os.getenv('POPPLER_PATH_OVERRIDE', None) # For pdf2image
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# --- Initialize clients (globally within this module) ---
s3_client, textract_client, roboflow_model_object, openai_client_instance = None, None, None, None # Renamed openai_client to avoid conflict if openai is used directly

try:
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME: # Added S3_BUCKET_NAME check
        s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_DEFAULT_REGION)
        textract_client = boto3.client('textract', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_DEFAULT_REGION)
        logger.info(f"Boto3 clients initialized for region: {AWS_DEFAULT_REGION}.")
    else:
        logger.warning("AWS credentials or S3_BUCKET_NAME not fully configured. S3/Textract operations may fail.")
except Exception as e:
    logger.error(f"Error initializing Boto3 clients: {e}", exc_info=True)

if ROBOFLOW_SDK_AVAILABLE and Roboflow and ROBOFLOW_API_KEY and ROBOFLOW_PROJECT_ID and ROBOFLOW_VERSION_NUMBER:
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.project(ROBOFLOW_PROJECT_ID)
        roboflow_model_object = project.version(int(ROBOFLOW_VERSION_NUMBER)).model
        logger.info(f"Roboflow model object initialized for project {ROBOFLOW_PROJECT_ID}, version {ROBOFLOW_VERSION_NUMBER}")
    except Exception as e:
        logger.error(f"Error initializing Roboflow model object: {e}", exc_info=True)
        roboflow_model_object = None
else:
    logger.warning("Roboflow SDK not available or configuration missing. Roboflow detection will be skipped.")
    roboflow_model_object = None

if OPENAI_API_KEY:
    try:
        openai_client_instance = openai.OpenAI(api_key=OPENAI_API_KEY) # Use the instance
        logger.info("OpenAI client configured with API key.")
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {e}", exc_info=True)
        openai_client_instance = None
else:
    logger.warning("OPENAI_API_KEY not found in environment variables. OpenAI calls will fail.")


# --- Helper Functions (Copied and adapted from your main.py) ---

def is_size_value_supported_by_text(size_value_str, size_unit_str, source_text, item_id_for_log="N/A"):
    if not size_value_str or not size_unit_str or not source_text:
        return True # Not enough info to invalidate, assume okay or handle upstream

    source_text_lower = str(source_text).lower()
    
    # Check for the numeric part of the size
    if str(size_value_str) not in source_text_lower:
        # Allow for minor variations, e.g. "6.0" vs "6"
        if isinstance(size_value_str, float) and int(size_value_str) == size_value_str: # e.g. 6.0
            if str(int(size_value_str)) not in source_text_lower: # Check for "6"
                logger.warning(f"ITEM_ID: {item_id_for_log} - Size value '{size_value_str}' (or int form) not found in source text: '{source_text_lower[:200]}...'")
                return False
        else:
            logger.warning(f"ITEM_ID: {item_id_for_log} - Size value '{size_value_str}' not found in source text: '{source_text_lower[:200]}...'")
            return False

    # Check for the unit part (can be more sophisticated with unit normalization)
    # This is a basic check; enhance with your unit_conversions if needed for robustness
    normalized_unit_variants = {
        "ct": ["ct", "count", "unidad", "unidades", "und", "un"],
        "oz": ["oz", "onzas", "onza"],
        # Add other common units and their variants from your normalization logic
    }
    
    unit_found = False
    if size_unit_str.lower() in source_text_lower:
        unit_found = True
    else:
        for canonical, variants in normalized_unit_variants.items():
            if size_unit_str.lower() == canonical:
                if any(variant in source_text_lower for variant in variants):
                    unit_found = True
                    break
    
    if not unit_found:
        logger.warning(f"ITEM_ID: {item_id_for_log} - Size unit '{size_unit_str}' (or variants) not found in source text: '{source_text_lower[:200]}...'")
        return False
        
    logger.debug(f"ITEM_ID: {item_id_for_log} - Size '{size_value_str} {size_unit_str}' appears supported by source text.")
    return True


def parse_price_string(price_str_input, item_id_for_log="N/A"):
    if price_str_input is None or price_str_input == "":
        logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Input is None or empty, returning None.")
        return None
    
    if isinstance(price_str_input, (int, float)):
        if price_str_input < 0:
            logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Input is numeric but negative ({price_str_input}), returning None.")
            return None
        if isinstance(price_str_input, int) and 100 <= price_str_input <= 99999: 
            s_price = str(price_str_input)
            if len(s_price) == 3: 
                val = float(f"{s_price[0]}.{s_price[1:]}")
                logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Input int {price_str_input} (len 3) parsed to {val}.")
                return val
            if len(s_price) == 4: 
                val = float(f"{s_price[:2]}.{s_price[2:]}")
                logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Input int {price_str_input} (len 4) parsed to {val}.")
                return val
            if len(s_price) == 5: 
                val = float(f"{s_price[:3]}.{s_price[3:]}")
                logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Input int {price_str_input} (len 5) parsed to {val}.")
                return val
        logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Input is numeric ({price_str_input}), returning as float.")
        return float(price_str_input)

    price_str = str(price_str_input).strip()
    logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Original string: '{price_str_input}', Stripped: '{price_str}'")

    geom_price_match = re.match(r'^\[GEOM_PRICE:\s*(\d{1,2})\s+(\d{2})\s*\]$', price_str)
    if geom_price_match:
        whole = geom_price_match.group(1)
        decimal_part = geom_price_match.group(2)
        val = float(f"{whole}.{decimal_part}")
        logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched GEOM_PRICE pattern '{price_str}' -> {val}.")
        return val

    space_separated_match = re.match(r'^(\d{1,2})\s+(\d{2})(?:\s*c/u)?$', price_str)
    if space_separated_match:
        whole = space_separated_match.group(1)
        decimal_part = space_separated_match.group(2)
        val = float(f"{whole}.{decimal_part}")
        logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched space-separated pattern '{price_str}' -> {val}.")
        return val

    if re.fullmatch(r'[1-9]\d{2}', price_str): 
        val = float(f"{price_str[0]}.{price_str[1:]}")
        logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched 3-digit pattern '{price_str}' -> {val}.")
        return val
    
    if re.fullmatch(r'[1-9]\d{3}', price_str): 
        val = float(f"{price_str[:2]}.{price_str[2:]}")
        logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched 4-digit pattern '{price_str}' -> {val}.")
        return val
    
    if re.fullmatch(r'[1-9]\d{4}', price_str): 
        val = float(f"{price_str[:3]}.{price_str[3:]}")
        logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched 5-digit pattern '{price_str}' -> {val}.")
        return val

    cleaned_price_str = price_str.lower()
    cleaned_price_str = re.sub(r'[$\¢₡€£¥]|regular|reg\.|oferta|esp\.|special|precio|price', '', cleaned_price_str, flags=re.IGNORECASE)
    cleaned_price_str = re.sub(r'\b(cada uno|c/u|cu|each|por)\b', '', cleaned_price_str, flags=re.IGNORECASE)
    cleaned_price_str = cleaned_price_str.strip()
    logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Cleaned for keywords: '{cleaned_price_str}'")
    
    if cleaned_price_str != price_str: 
        if re.fullmatch(r'[1-9]\d{2}', cleaned_price_str):
            val = float(f"{cleaned_price_str[0]}.{cleaned_price_str[1:]}")
            logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched 3-digit pattern on cleaned string '{cleaned_price_str}' -> {val}.")
            return val
        if re.fullmatch(r'[1-9]\d{3}', cleaned_price_str):
            val = float(f"{cleaned_price_str[:2]}.{cleaned_price_str[2:]}")
            logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched 4-digit pattern on cleaned string '{cleaned_price_str}' -> {val}.")
            return val
        if re.fullmatch(r'[1-9]\d{4}', cleaned_price_str):
            val = float(f"{cleaned_price_str[:3]}.{cleaned_price_str[3:]}")
            logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched 5-digit pattern on cleaned string '{cleaned_price_str}' -> {val}.")
            return val

    std_decimal_match_dot = re.fullmatch(r'(\d+)\.(\d{1,2})', cleaned_price_str)
    if std_decimal_match_dot:
        num_part, dec_part = std_decimal_match_dot.groups()
        if len(dec_part) == 1: dec_part += "0" 
        val = float(f"{num_part}.{dec_part}")
        logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched dot-decimal pattern '{cleaned_price_str}' -> {val}.")
        return val if val >= 0 else None

    std_decimal_match_comma = re.fullmatch(r'(\d+),(\d{1,2})', cleaned_price_str)
    if std_decimal_match_comma:
        num_part, dec_part = std_decimal_match_comma.groups()
        if len(dec_part) == 1: dec_part += "0"
        val = float(f"{num_part}.{dec_part}") 
        logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched comma-decimal pattern '{cleaned_price_str}' -> {val}.")
        return val if val >= 0 else None
        
    whole_match = re.fullmatch(r'(\d+)', cleaned_price_str)
    if whole_match:
        num = float(whole_match.group(1))
        if num == 0.0: 
            logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched whole number 0.0.")
            return 0.0
        if num >= 1 and num < 100: 
            logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched whole number pattern '{cleaned_price_str}' -> {num}.")
            return num

    logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Could not parse price string: '{price_str}' (cleaned: '{cleaned_price_str}'). Returning None.")
    return None

def detect_price_candidates(line_blocks, image_height_px, blocks_map, item_id_for_log="N/A", prepended_geom_price=None):
    candidates = []
    price_pattern_text = r"""
        (?<![\w\d.])(?:             
            \$?\s?\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{2}) |   
            \b\d{1,2}\s+\d{2}\b |                      
            \b[1-9]\d{2,4}\b |                         
            \b\d+x\d{2,3}\b |                         
            \[GEOM_PRICE:\s*\d{1,2}\s+\d{2}\s*\]       
        )(?![\d.])                                   
    """
    price_regex = re.compile(price_pattern_text, re.VERBOSE)
    
    size_unit_keywords = ['oz', 'onzas', 'lb', 'libras', 'gal', 'lt', 'ml', 'g', 'kg', 
                          'rollos', 'hojas', 'ct', 'pies', 'ft', 'metros', 'unidad', 
                          'unidades', 'gramo', 'litro', 'sheet', 'sheets', 'count', 'pk']
    
    logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Processing {len(line_blocks)} line_blocks. Prepended geom price: {prepended_geom_price}")

    if prepended_geom_price:
        parsed_geom_val = parse_price_string(prepended_geom_price, item_id_for_log=f"{item_id_for_log}-geom_cand_prep")
        if parsed_geom_val is not None:
            geom_candidate = {
                'text_content': prepended_geom_price, 
                'parsed_value': parsed_geom_val,
                'bounding_box': line_blocks[0]['Geometry']['BoundingBox'] if line_blocks else None, 
                'pixel_height': image_height_px * 0.1, # Placeholder height for geom price
                'source_block_id': 'GEOMETRIC_MERGE', 
                'full_line_text': prepended_geom_price,
                'is_regular_candidate': False, 
                'has_price_indicator': True 
            }
            candidates.append(geom_candidate)
            logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Added prepended geometric candidate: {geom_candidate}")

    for line_idx, line_block in enumerate(line_blocks):
        if line_block['BlockType'] != 'LINE':
            continue
        
        line_text_parts = []
        if 'Relationships' in line_block:
            for relationship in line_block['Relationships']:
                if relationship['Type'] == 'CHILD':
                    for child_id in relationship['Ids']:
                        word = blocks_map.get(child_id)
                        if word and word['BlockType'] == 'WORD':
                            line_text_parts.append(word['Text'])
        
        full_line_text = " ".join(line_text_parts).strip()
        if not full_line_text: # Fallback if no child words but line has text
            full_line_text = line_block.get('Text', '').strip() 
        
        if not full_line_text:
            logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Line {line_idx} is empty.")
            continue
        
        logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Line {line_idx} Text: '{full_line_text}'")

        has_price_indicator = any(indicator in full_line_text.lower() 
                                  for indicator in ['c/u', 'cada uno', '$', 'regular', 'precio', 'esp.'])
        
        for match in price_regex.finditer(full_line_text):
            raw_price_text = match.group(0).strip()
            logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Line {line_idx} - Raw regex match: '{raw_price_text}'")

            if raw_price_text.startswith("[GEOM_PRICE:"): 
                if not prepended_geom_price or raw_price_text != prepended_geom_price:
                    # This means a GEOM_PRICE was found by regex but wasn't the one prepended (unlikely if logic is correct)
                    # Or it's a new one if nothing was prepended (also unlikely here, as it's handled above)
                    pass 
                else: 
                    continue # Already handled as the prepended one

            match_start, match_end = match.span()
            context_before = full_line_text[max(0, match_start-10):match_start].lower()
            context_after = full_line_text[match_end:min(len(full_line_text), match_end+15)].lower() 
            
            is_likely_size_metric = False
            if any(re.search(r'^\s*' + re.escape(unit), context_after) for unit in size_unit_keywords):
                if not has_price_indicator: 
                    is_likely_size_metric = True
                    logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Skipping '{raw_price_text}' - looks like size (unit follows, no price indicator). Context after: '{context_after[:10]}'")
                    continue
            if any(kw in context_before for kw in ["pack of", "paquete de", "paq de"]):
                if not has_price_indicator:
                    is_likely_size_metric = True
                    logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Skipping '{raw_price_text}' - looks like size (pack of precedes, no price indicator). Context before: '{context_before[-10:]}'")
                    continue

            if re.fullmatch(r'\d{3,}', raw_price_text) and int(raw_price_text) > 100: 
                if (re.search(r'\s*(a|-|to)\s*\d+', context_after) or 
                    re.search(r'\d+\s*(a|-|to)\s*$', context_before)):  
                    if not has_price_indicator:
                        logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Skipping '{raw_price_text}' - part of size range, no price indicator.")
                        continue
            
            parsed_value = parse_price_string(raw_price_text, item_id_for_log=f"{item_id_for_log}-cand-{len(candidates)}")
            if parsed_value is not None:
                geometry = line_block['Geometry']['BoundingBox']
                is_regular = any(re.search(r'\b' + kw + r'\b', full_line_text, re.IGNORECASE) 
                                 for kw in ['regular', 'reg.', 'precio regular'])
                
                candidate_data = {
                    'text_content': raw_price_text, 
                    'parsed_value': parsed_value,
                    'bounding_box': geometry, 
                    'pixel_height': geometry['Height'] * image_height_px,
                    'source_block_id': line_block['Id'], 
                    'full_line_text': full_line_text,
                    'is_regular_candidate': is_regular,
                    'has_price_indicator': has_price_indicator
                }
                candidates.append(candidate_data)
                logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Added candidate: {candidate_data}")
            else:
                logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Line {line_idx} - Match '{raw_price_text}' did not parse to a valid price.")

    candidates.sort(key=lambda c: (
        c['source_block_id'] != 'GEOMETRIC_MERGE', # GEOMETRIC_MERGE comes first (False sorts before True)
        -c['pixel_height'], 
        not c['has_price_indicator'], 
        c['is_regular_candidate']
    ))
    logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Found {len(candidates)} sorted candidates: {json.dumps(candidates, indent=2) if candidates else '[]'}")
    return candidates

def validate_price_pair(offer_price, regular_price, item_id_for_log="N/A"):
    op, rp = offer_price, regular_price
    if op is not None: op = float(op)
    if rp is not None: rp = float(rp)

    if op is None or rp is None:
        logger.debug(f"ITEM_ID: {item_id_for_log} - validate_price_pair - One price is None (O:{op}, R:{rp}). No swap.")
        return op, rp
    
    if op > rp:
        logger.warning(f"ITEM_ID: {item_id_for_log} - validate_price_pair - Swapping prices: offer {op} > regular {rp}")
        return rp, op
    
    logger.debug(f"ITEM_ID: {item_id_for_log} - validate_price_pair - Prices validated (O:{op}, R:{rp}). No swap needed or already swapped.")
    return op, rp

# REPLACE the entire find_bbox_for_text function with this new, more robust version.

def find_bbox_for_text(text_to_find, textract_blocks, fuzz_threshold=80):
    """
    Finds a merged bounding box for a phrase by locating its constituent words
    within a list of Textract WORD blocks and merging their geometries.
    This version is more robust against LLM rephrasing.
    """
    if not text_to_find or not textract_blocks:
        return None

    # Clean and split the target phrase from the LLM into individual words
    target_words = str(text_to_find).lower().strip().split()
    if not target_words:
        return None

    word_blocks = [block for block in textract_blocks if block.get('BlockType') == 'WORD']
    found_bboxes = []
    
    # For each word the LLM provided, find the best matching word in the Textract output
    for target_word in target_words:
        best_match_block = None
        highest_score = 0
        
        for block in word_blocks:
            block_text = block.get('Text', '').lower()
            if not block_text:
                continue

            # Use token_set_ratio, which is very effective for finding a word within another
            # e.g., finding "rollos" inside "rollos_dobles"
            score = fuzz.token_set_ratio(target_word, block_text)
            
            if score > highest_score:
                highest_score = score
                best_match_block = block

        # If a sufficiently good match was found, add its bbox
        if best_match_block and highest_score >= fuzz_threshold:
            bbox = best_match_block.get('Geometry', {}).get('BoundingBox')
            if bbox:
                found_bboxes.append(bbox)
                # To prevent re-matching the same block, we can temporarily remove it
                # This helps if a word appears multiple times (e.g., "Clorox")
                word_blocks.remove(best_match_block)

    if not found_bboxes:
        logger.warning(f"find_bbox_for_text could not find any matching words for phrase: '{text_to_find}'")
        return None

    # Merge the bounding boxes of all found words into a single, encompassing box
    x_min = min(b['Left'] for b in found_bboxes)
    y_min = min(b['Top'] for b in found_bboxes)
    x_max = max(b['Left'] + b['Width'] for b in found_bboxes)
    y_max = max(b['Top'] + b['Height'] for b in found_bboxes)

    merged_bbox = {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}
    logger.debug(f"Successfully found and merged bbox for '{text_to_find}': {merged_bbox}")
    return merged_bbox

def extract_product_data_with_llm(product_snippet_text: str, item_id_for_log="N/A", llm_model: str = "gpt-4o", textract_blocks_in_segment: list = None) -> dict:
    if not openai_client_instance:
        logger.error(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - OpenAI client not initialized.")
        return {"error_message": "OpenAI client not initialized", "llm_input_snippet": product_snippet_text}
        
    logger.info(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - Sending snippet to Text LLM ({llm_model}).")
    logger.debug(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - Snippet Text:\n{product_snippet_text}")

    system_prompt = """You are an expert at extracting product information from advertisement text. Extract the following fields:

CRITICAL RULES FOR PRICE EXTRACTION:
1. If the text starts with "[GEOM_PRICE: X YZ]", prioritize this X.YZ as the offer_price. E.g., "[GEOM_PRICE: 6 97]" means offer_price is 6.97.
2. For 3-digit numbers like "897", "647", "447" presented as the main price - these represent prices like 8.97, 6.47, 4.47.
3. For 4-digit numbers like "1097" presented as the main price - this represents 10.97.
4. For prices like "8 97" (space separated), interpret as 8.97.
5. The offer_price is usually the first/most prominent price in the snippet (after any [GEOM_PRICE] marker).
6. Regular price usually appears after "Regular", "Reg.", or "Precio Regular" keyword.
7. Prices should generally NOT exceed 100.00 for these grocery/household items unless it's clearly a large appliance/furniture.
8. For "N for $X" or "N x $X" deals (e.g., "2x $5.00" or "2 for $5.00"), if this is the offer, the offer_price should be the price PER ITEM (e.g., $2.50). If there's a coupon modifying this (e.g., "2x $5.00 *Cupón... = 2x $4.50"), calculate the final price PER ITEM.

Fields to extract:
- "offer_price": The sale/promotional price PER ITEM. Return as a decimal number.
- "regular_price": The original price PER ITEM. Return as a decimal number.
- "product_brand": The brand name.
- "product_name_core": The main product name.
- "product_variant_description": The descriptive text, including size, quantity, flavor, type etc.
- "size_quantity_info": Specific size/quantity extracted (e.g., "105 a 117 onzas", "21 onzas", "6=12 Rollos").
    - CRITICALLY IMPORTANT FOR SIZE: Only extract size and quantity information that is EXPLICITLY STATED in the provided text.
    - Do NOT guess, infer, or assume any size or quantity.
    - If the size or quantity is unclear or not present in the snippet, return null or an empty string for this field.
    - Do NOT invent values like "120 ct" if it's not directly supported by the input text.
- "unit_indicator": Like "c/u", "ea." if present near a price.
- "store_specific_terms": Like "*24 por tienda", coupon details if not part of price.

IMPORTANT: Return prices as decimal numbers (e.g., 8.97), not strings. Use null if missing.
If product_variant_description contains size, also extract to size_quantity_info following the strict rules above.

Return ONLY a JSON object.
"""
    few_shot_examples = [
        {"role": "user", "content": "Text:\n[GEOM_PRICE: 6 97]\n97 c/u\nAce Simply\nDetergente Líquido 84 onzas\nRegular $7.99 c/u\n*24 por tienda\n\nReturn JSON."},
        {"role": "assistant", "content": """{
"offer_price": 6.97, "regular_price": 7.99, "product_brand": "Ace", "product_name_core": "Ace Simply",
"product_variant_description": "Detergente Líquido 84 onzas", "size_quantity_info": "84 onzas",
"unit_indicator": "c/u", "store_specific_terms": "*24 por tienda"
}"""},
        {"role": "user", "content": "Text:\n897 c/u\nAce Simply\nDetergente Líquido 105 a 117 onzas\nRegular $10.49 c/u\n*24 por tienda\n\nReturn JSON."},
        {"role": "assistant", "content": """{
"offer_price": 8.97, "regular_price": 10.49, "product_brand": "Ace", "product_name_core": "Ace Simply",
"product_variant_description": "Detergente Líquido 105 a 117 onzas", "size_quantity_info": "105 a 117 onzas",
"unit_indicator": "c/u", "store_specific_terms": "*24 por tienda"
}"""}
    ]
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(few_shot_examples)
    messages.append({"role": "user", "content": f"Text:\n{product_snippet_text}\n\nReturn JSON."})

    try:
        chat_completion = openai_client_instance.chat.completions.create(
            model=llm_model, messages=messages, response_format={"type": "json_object"}, temperature=0.1
        )
        response_content = chat_completion.choices[0].message.content
        extracted_data = json.loads(response_content)

        # Clean up prices, but do NOT look for bboxes here anymore.
        extracted_data['offer_price'] = parse_price_string(extracted_data.get('offer_price'), item_id_for_log)
        extracted_data['regular_price'] = parse_price_string(extracted_data.get('regular_price'), item_id_for_log)
        
        logger.info(f"ITEM_ID: {item_id_for_log} - LLM text extraction successful.")
        return extracted_data

    except Exception as e:
        logger.error(f"ITEM_ID: {item_id_for_log} - Error in LLM text extraction: {e}", exc_info=True)
        return {"error_message": str(e)}


def get_segment_image_bytes(page_image_pil: Image.Image, box_coords_pixels_center_wh: dict, item_id_for_log="N/A") -> BytesIO | None:
    try:
        if not all(k in box_coords_pixels_center_wh for k in ['x', 'y', 'width', 'height']):
            logger.error(f"ITEM_ID: {item_id_for_log} - get_segment_image_bytes - Invalid box_coords: {box_coords_pixels_center_wh}")
            return None

        cx, cy, w, h = (box_coords_pixels_center_wh['x'], box_coords_pixels_center_wh['y'],
                        box_coords_pixels_center_wh['width'], box_coords_pixels_center_wh['height'])
        
        padding_factor = 0.05 
        padding_x = int(w * padding_factor)
        padding_y = int(h * padding_factor)
        
        x_min = int(cx - w / 2) - padding_x
        y_min = int(cy - h / 2) - padding_y
        x_max = int(cx + w / 2) + padding_x
        y_max = int(cy + h / 2) + padding_y
        
        img_width, img_height = page_image_pil.size
        x_min_clamped = max(0, x_min)
        y_min_clamped = max(0, y_min)
        x_max_clamped = min(img_width, x_max)
        y_max_clamped = min(img_height, y_max)

        if x_min_clamped >= x_max_clamped or y_min_clamped >= y_max_clamped:
            logger.warning(f"ITEM_ID: {item_id_for_log} - get_segment_image_bytes - Invalid crop coords after clamping: ({x_min_clamped}, {y_min_clamped}, {x_max_clamped}, {y_max_clamped}). Original: ({x_min}, {y_min}, {x_max}, {y_max})")
            return None
            
        segment_image_pil = page_image_pil.crop((x_min_clamped, y_min_clamped, x_max_clamped, y_max_clamped))
        
        # Optional: Draw border for debugging saved images - remove for production if not needed
        # from PIL import ImageDraw
        # draw = ImageDraw.Draw(segment_image_pil)
        # draw.rectangle([(0, 0), (segment_image_pil.width-1, segment_image_pil.height-1)], 
        #                outline="red", width=3)
        
        img_byte_arr = BytesIO()
        segment_image_pil.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        logger.debug(f"ITEM_ID: {item_id_for_log} - get_segment_image_bytes - Successfully cropped segment image. Coords: ({x_min_clamped}, {y_min_clamped}, {x_max_clamped}, {y_max_clamped})")
        return img_byte_arr
    except Exception as e:
        logger.error(f"ITEM_ID: {item_id_for_log} - get_segment_image_bytes - Error cropping segment image: {e}", exc_info=True)
        return None

# backend_processor.py

# ... (imports and existing global initializations) ...

# Ensure PIL's ImageDraw is imported for drawing utilities
from PIL import ImageDraw # Add this import

# ... (existing helper functions like parse_price_string, detect_price_candidates, etc.) ...

# backend_processor.py

# ... (existing functions) ...

def re_extract_with_vision_llm(segment_image_bytes: BytesIO, item_id_for_log="N/A", original_item_name: str | None = None, llm_model: str = "gpt-4o") -> dict:
    if not openai_client_instance:
        logger.error(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - OpenAI client not configured for vision.")
        return {"error_message": "OpenAI client not configured for vision."}
    if not segment_image_bytes:
        logger.error(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - No segment image provided.")
        return {"error_message": "No segment image for vision."}
        
    response_content = None
    try:
        base64_image = base64.b64encode(segment_image_bytes.getvalue()).decode('utf-8')
        logger.debug(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - Base64 image snippet for Vision LLM: {base64_image[:100]}...")

        # Simplified prompt: NO REQUEST FOR BOUNDING BOXES FROM VISION LLM
        prompt_text = (
            "You are an expert product data extractor for retail flyer segments. "
            "From the provided image of a single product deal, extract the following information. "
            "Pay close attention to visually prominent numbers for prices. "
            "If a price is shown as 'XYZ' (e.g., '897'), interpret it as X.YZ dollars (e.g., $8.97). "
            "If a price is 'X YZ' (e.g., '6 47'), interpret it as X.YZ dollars (e.g., $6.47). "
            "For 'N for $M' or 'NxM' deals (e.g., '2x300' where 300 means $3.00, or '2 for $5.00'), the offer_price should be the price PER ITEM (e.g., $1.50 or $2.50). "
            "If coupon details are present and modify the price, calculate the final per-item offer_price."
            "\n\n"
            "Fields to extract:\n"
            "- \"offer_price\": The final sale/promotional price per item. Return as a decimal number. If not found, null.\n"
            "- \"regular_price\": The original price per item. Return as a decimal number.\n"
            "- \"product_brand\": The brand name.\n"
            "- \"product_name_core\": The main product name.\n"
            "- \"product_variant_description\": Detailed description including flavor, type etc.\n"
            "- \"size_quantity_info\": Specific size/quantity (e.g., '105 a 117 onzas', '21 oz', '6=12 Rollos', 'Paquete de 2').\n"
            "    - CRITICALLY IMPORTANT FOR SIZE (from image): Only extract size and quantity information that is CLEARLY AND EXPLICITLY VISIBLE in the provided image segment.\n"
            "    - Do NOT guess, infer, or assume any size or quantity if it's ambiguous or not present.\n"
            "    - If the size or quantity is unclear or not visible, return null or an empty string for this field.\n"
            "    - Pay very close attention to the actual numbers and units visible; do not hallucinate common but incorrect values like '120 ct' unless explicitly visible.\n"
            "- \"unit_indicator\": Like 'c/u', 'ea.' if present near a price.\n"
            "- \"store_specific_terms\": Like store limits or uncalculated coupon details.\n"
            "Return ONLY a JSON object with these fields. Use null for missing fields."
        )
        if original_item_name: prompt_text += f"\nThe product is likely related to: '{original_item_name}'.\n"
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
        
        logger.info(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - Sending segment image to Vision LLM ({llm_model}). Hint: '{original_item_name}'.")
        
        chat_completion = openai_client_instance.chat.completions.create(
            model=llm_model, messages=messages, response_format={"type": "json_object"}, max_tokens=1000, temperature=0.1
        )
        response_content = chat_completion.choices[0].message.content
        logger.debug(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - Vision LLM Raw Response Content: {response_content}")
        
        if response_content is None:
            logger.error(f"ITEM_ID: {item_id_for_log} - Vision LLM returned None content. Cannot parse JSON.")
            return {"error_message": "Vision LLM returned no content", "vision_llm_used": True}

        extracted_data_parsed = json.loads(response_content) # No special bbox parsing here
        
        logger.debug(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - Vision LLM Data (raw): {json.dumps(extracted_data_parsed, indent=2)}")

        extracted_data_parsed["offer_price"] = parse_price_string(extracted_data_parsed.get("offer_price"), item_id_for_log=f"{item_id_for_log}-vision_offer")
        extracted_data_parsed["regular_price"] = parse_price_string(extracted_data_parsed.get("regular_price"), item_id_for_log=f"{item_id_for_log}-vision_regular")
        
        extracted_data_parsed["vision_llm_used"] = True
        logger.info(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - Successfully extracted data from Vision LLM.")
        return extracted_data_parsed
    except json.JSONDecodeError as je:
        logger.error(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - JSONDecodeError: {je}. Response: {response_content}", exc_info=True)
        return {"error_message": f"JSONDecodeError: {je}", "vision_llm_used": True, "vision_llm_response_content": response_content}
    except Exception as e:
        logger.error(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - Error calling Vision LLM API: {e}", exc_info=True)
        return {"error_message": str(e), "vision_llm_used": True, "vision_llm_response_content": response_content}

# ... (rest of the existing helper functions) ...


def upload_to_s3(file_like_object, bucket_name, cloud_object_name):
    if not s3_client:
        logger.error("S3 client not initialized. Cannot upload.")
        return None
    try:
        # file_like_object is already BytesIO, so pass directly
        s3_client.upload_fileobj(file_like_object, bucket_name, cloud_object_name)
        logger.info(f"File '{cloud_object_name}' uploaded to S3 bucket '{bucket_name}'.")
        return cloud_object_name
    except Exception as e:
        logger.error(f"Error uploading file '{cloud_object_name}' to S3: {e}", exc_info=True)
        return None

def get_analysis_from_document_via_textract(bucket_name, document_s3_key):
    if not textract_client:
        logger.error("Textract client not initialized. Cannot analyze.")
        return None
    logger.info(f"Starting Textract Document Analysis for S3 object: s3://{bucket_name}/{document_s3_key}")
    try:
        response = textract_client.start_document_analysis(
            DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': document_s3_key}},
            FeatureTypes=['TABLES', 'FORMS', 'LAYOUT'] 
        )
        job_id = response['JobId']
        logger.info(f"Textract Analysis job started (JobId: '{job_id}') for '{document_s3_key}'.")
        
        status = 'IN_PROGRESS'
        max_retries = 90 
        retries = 0
        all_blocks = [] 
        job_status_response = None # Define for broader scope
        
        while status == 'IN_PROGRESS' and retries < max_retries:
            time.sleep(5) 
            job_status_response = textract_client.get_document_analysis(JobId=job_id)
            status = job_status_response['JobStatus']
            logger.debug(f"Textract Analysis job status for '{job_id}': {status} (Retry {retries+1}/{max_retries})")
            retries += 1
            
        if status == 'SUCCEEDED':
            nextToken = None
            # Ensure job_status_response is used for the first call if no NextToken yet
            current_response_data = job_status_response 
            while True:
                page_blocks = current_response_data.get("Blocks", [])
                logger.debug(f"Textract SUCCEEDED page fetch for '{document_s3_key}', JobId '{job_id}'. Fetched {len(page_blocks)} blocks for this page/token.")
                all_blocks.extend(page_blocks)
                nextToken = current_response_data.get('NextToken')
                if not nextToken:
                    break
                # Fetch next set of results only if nextToken exists
                current_response_data = textract_client.get_document_analysis(JobId=job_id, NextToken=nextToken)

            logger.info(f"Textract Analysis SUCCEEDED for '{document_s3_key}'. Found {len(all_blocks)} blocks in total.")
            return all_blocks
        else:
            logger.error(f"Textract Analysis job for '{document_s3_key}' status: {status}. Response: {job_status_response}")
            return None
    except Exception as e:
        logger.error(f"Error in Textract Analysis for '{document_s3_key}': {e}", exc_info=True)
        return None

def delete_from_s3(bucket_name, cloud_object_name):
    if not s3_client:
        logger.warning("S3 client not initialized. Cannot delete.") # Warning as it's cleanup
        return
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=cloud_object_name)
        logger.info(f"File '{cloud_object_name}' deleted from S3 bucket '{bucket_name}'.")
    except Exception as e:
        logger.error(f"Error deleting file '{cloud_object_name}' from S3: {e}", exc_info=True)

def clean_text(text): 
    if not text:
        return ""
    lines = text.splitlines()
    processed_lines = []
    for line_content in lines:
        stripped_line = line_content.strip()
        if stripped_line: 
            processed_lines.append(re.sub(r'\s+', ' ', stripped_line))
    return "\n".join(processed_lines)

def get_roboflow_predictions_sdk(pil_image_object, original_filename_for_temp="temp_image.jpg"):
    if not roboflow_model_object:
        logger.error("Roboflow model object is not configured/initialized. Cannot get predictions.")
        return None # Return empty list or None consistently
    
    temp_file_path = None
    # Use tempfile for Roboflow temporary images as well
    try:
        # Create a temporary file with a proper image extension
        suffix = ".jpg" if original_filename_for_temp.lower().endswith((".jpg", ".jpeg")) else ".png"
        if not original_filename_for_temp.lower().endswith((".jpg", ".jpeg", ".png")):
             original_filename_for_temp += ".jpg" # Default to JPG if no valid ext
             suffix = ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="rf_temp_") as tmp_rf_img:
            pil_image_object.save(tmp_rf_img.name, format="JPEG" if suffix == ".jpg" else "PNG")
            temp_file_path = tmp_rf_img.name
        
        logger.info(f"Saved PIL image temporarily to {temp_file_path} for Roboflow.")
        
        prediction_result_obj = roboflow_model_object.predict(temp_file_path, confidence=40, overlap=30)
        
        actual_predictions_data = []
        if hasattr(prediction_result_obj, 'json') and callable(prediction_result_obj.json):
            json_response = prediction_result_obj.json()
            actual_predictions_data = json_response.get('predictions', [])
            logger.debug(f"Roboflow raw JSON response: {json.dumps(json_response, indent=2)}")
        elif hasattr(prediction_result_obj, 'predictions'): 
            actual_predictions_data = [p.json() for p in prediction_result_obj.predictions]
            logger.debug(f"Roboflow predictions (from .predictions attribute): {json.dumps(actual_predictions_data, indent=2)}")
        elif isinstance(prediction_result_obj, list): 
            actual_predictions_data = prediction_result_obj
            logger.debug(f"Roboflow predictions (already a list): {json.dumps(actual_predictions_data, indent=2)}")
        else:
            logger.warning(f"Unexpected Roboflow prediction result format: {type(prediction_result_obj)}. Trying to iterate...")
            try: 
                actual_predictions_data = [p.json() if hasattr(p, 'json') else p for p in prediction_result_obj]
            except TypeError:
                logger.error("Could not process Roboflow prediction object.")
                return [] # Return empty list

        predictions_list = []
        for i, p_data in enumerate(actual_predictions_data):
            pred_dict = {
                'x': p_data.get('x'), 'y': p_data.get('y'),
                'width': p_data.get('width'), 'height': p_data.get('height'),
                'confidence': p_data.get('confidence'),
                'class': p_data.get('class', p_data.get('class_name', 'unknown')) 
            }
            if not all(isinstance(pred_dict[k], (int, float)) for k in ['x', 'y', 'width', 'height'] if pred_dict[k] is not None): # Check for None
                logger.warning(f"Skipping Roboflow prediction #{i} with invalid or missing coordinates: {pred_dict}")
                continue
            predictions_list.append(pred_dict)
            
        logger.info(f"Processed {len(predictions_list)} valid Roboflow predictions.")
        return predictions_list
        
    except Exception as e:
        logger.error(f"Error in get_roboflow_predictions_sdk: {e}", exc_info=True)
        return None # Or empty list
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Deleted temp Roboflow image: {temp_file_path}")
            except Exception as e_del:
                logger.error(f"Error deleting temp Roboflow image file {temp_file_path}: {e_del}")

# backend_processor.py

# ... (existing imports) ...

# In visual_layout_backend.py

# REPLACE this function in backend_processor.py

def collate_text_for_product_boxes(roboflow_boxes, textract_all_blocks, blocks_map,
                                   image_width_px, image_height_px, page_id_for_log="N/A"):
    product_texts_with_candidates = []
    if not roboflow_boxes or not textract_all_blocks or not blocks_map or \
       image_width_px is None or image_height_px is None:
        logger.warning(f"PAGE_ID: {page_id_for_log} - collate_text_for_product_boxes: Missing critical inputs.")
        return product_texts_with_candidates

    logger.info(f"PAGE_ID: {page_id_for_log} - collate_text_for_product_boxes - Starting smart collation for {len(roboflow_boxes)} Roboflow boxes.")
    
    all_lines_on_page = [block for block in textract_all_blocks if block['BlockType'] == 'LINE']

    for i, box_pred in enumerate(roboflow_boxes):
        item_id_for_log = f"{page_id_for_log}-RFBox{i}-{str(box_pred.get('class', 'UnknownClass'))}"
        
        rf_center_x_px, rf_center_y_px = box_pred.get('x'), box_pred.get('y')
        rf_width_px, rf_height_px = box_pred.get('width'), box_pred.get('height')
        
        if not all(isinstance(v, (int, float)) for v in [rf_center_x_px, rf_center_y_px, rf_width_px, rf_height_px]):
            logger.warning(f"ITEM_ID: {item_id_for_log} - Roboflow Box has invalid coordinates. Skipping.")
            continue
            
        rf_x_min_rel = (rf_center_x_px - rf_width_px / 2.0) / image_width_px
        rf_y_min_rel = (rf_center_y_px - rf_height_px / 2.0) / image_height_px
        rf_x_max_rel = (rf_center_x_px + rf_width_px / 2.0) / image_width_px
        rf_y_max_rel = (rf_center_y_px + rf_height_px / 2.0) / image_height_px
        
        # --- Geometric Price Merging has been REMOVED ---
        
        lines_in_box_objects = []
        for line_block in all_lines_on_page:
            txt_geom = line_block.get('Geometry', {}).get('BoundingBox', {})
            if not txt_geom: continue
            line_center_x_rel = txt_geom['Left'] + (txt_geom['Width'] / 2.0)
            line_center_y_rel = txt_geom['Top'] + (txt_geom['Height'] / 2.0)
            if (rf_x_min_rel <= line_center_x_rel <= rf_x_max_rel and \
                rf_y_min_rel <= line_center_y_rel <= rf_y_max_rel):
                lines_in_box_objects.append(line_block)
        
        lines_in_box_objects.sort(key=lambda line: (
            line['Geometry']['BoundingBox']['Top'],
            -line['Geometry']['BoundingBox']['Height'],
            line['Geometry']['BoundingBox']['Left']
        ))
        
        ordered_lines_text_parts = []
        detailed_blocks_in_segment = []
        for line_block in lines_in_box_objects:
            line_words_for_text = []
            if 'Relationships' in line_block:
                for relationship in line_block['Relationships']:
                    if relationship['Type'] == 'CHILD':
                        for child_id in relationship['Ids']:
                            word = blocks_map.get(child_id)
                            if word and word['BlockType'] == 'WORD':
                                line_words_for_text.append(word['Text'])
                                detailed_blocks_in_segment.append(word)
            
            line_text = " ".join(line_words_for_text).strip()
            
            if line_text:
                ordered_lines_text_parts.append(line_text)
            elif not line_text and line_block.get('Text'):
                 ordered_lines_text_parts.append(line_block.get('Text','').strip())
                 detailed_blocks_in_segment.append(line_block)

        collated_text_multiline = "\n".join(ordered_lines_text_parts)
        collated_text_cleaned = clean_text(collated_text_multiline)

        # We pass None for prepended_geom_price because it has been removed.
        price_candidates_for_segment = detect_price_candidates(lines_in_box_objects, image_height_px, blocks_map, item_id_for_log=item_id_for_log, prepended_geom_price=None)

        logger.info(f"ITEM_ID: {item_id_for_log} - Final Collated Text (cleaned) for LLM:\n{collated_text_cleaned}")

        if collated_text_cleaned:
            product_texts_with_candidates.append({
                "product_box_id": item_id_for_log,
                "roboflow_confidence": box_pred.get('confidence', 0.0),
                "class_name": str(box_pred.get('class', 'UnknownClass')),
                "collated_text": collated_text_cleaned,
                "price_candidates": price_candidates_for_segment,
                "roboflow_box_coords_pixels_center_wh": {
                    'x': rf_center_x_px, 'y': rf_center_y_px,
                    'width': rf_width_px, 'height': rf_height_px
                },
                "textract_blocks_in_segment": detailed_blocks_in_segment
            })
    
    logger.info(f"PAGE_ID: {page_id_for_log} - Collation complete. Generated {len(product_texts_with_candidates)} product text snippets.")
    return product_texts_with_candidates
# File: backend_processor.py

# In backend_processor.py
# Ensure 're' and 'json' are imported if your other logic in this function (unrelated to size) needs them.
import re 
import json # For logging, if used

def enhanced_normalize_product_data(product_data, item_id_for_log, price_candidates=None, original_collated_text=""):
    normalized_item_data = product_data.copy() # Start with a copy

    # Ensure all expected keys are present, defaulting to None if not in product_data
    keys_to_ensure = [
        'offer_price', 'regular_price', 'product_brand', 'product_name_core',
        'product_variant_description', 'size_quantity_info', 'unit_indicator',
        'store_specific_terms', 'parsed_size_details', 'size_quantity_info_normalized',
        'validation_flags'
    ]
    for key in keys_to_ensure:
        normalized_item_data.setdefault(key, None)
    
    if not isinstance(normalized_item_data.get('validation_flags'), list):
        normalized_item_data['validation_flags'] = []

    # --- For THIS TEST, we will NOT do any size normalization here ---
    # The size fields will be handled directly in post_process_and_validate_item_data
    # Just ensure the original size_quantity_info from LLM is preserved if it exists.
    sqi = normalized_item_data.get('size_quantity_info')
    if sqi is not None:
        normalized_item_data['size_quantity_info'] = str(sqi).strip()
    else:
        normalized_item_data['size_quantity_info'] = None
        
    # These will be explicitly set/overridden in the calling function for the bypass test
    normalized_item_data['parsed_size_details'] = None 
    normalized_item_data['size_quantity_info_normalized'] = None

    logger.debug(f"ITEM_ID: {item_id_for_log} - enhanced_normalize_product_data (BYPASS MODE - minimal size processing) - "
                 f"Input SQI: '{product_data.get('size_quantity_info')}', Output SQI: '{normalized_item_data.get('size_quantity_info')}'")
    
    # (Your existing logic for price candidates or other non-size normalizations can remain here if any)

    return normalized_item_data



# In visual_layout_backend.py
# REPLACE the entire post_process_and_validate_item_data function with this one.

# REPLACE the entire post_process_and_validate_item_data function with this one.

def post_process_and_validate_item_data(llm_data, price_candidates, original_collated_text, textract_blocks, item_id_for_log="N/A"):
    """
    Validates LLM data and, most importantly, LINKS each extracted value to its
    source bounding box from the Textract data.
    """
    if "error_message" in llm_data:
        return {'validation_flags': [f"LLM_ERROR: {llm_data.get('error_message')}"]}

    validated_data = {"validation_flags": []}

    # --- 1. Link Prices to their Bboxes from Candidates ---
    offer_price_val = parse_price_string(llm_data.get('offer_price'))
    regular_price_val = parse_price_string(llm_data.get('regular_price'))

    if offer_price_val is not None and price_candidates:
        # Find the best candidate from the OCR that matches the LLM's price
        best_candidate = min(price_candidates, key=lambda c: abs(c['parsed_value'] - float(offer_price_val)), default=None)
        if best_candidate and abs(best_candidate['parsed_value'] - float(offer_price_val)) < 0.02:
            bbox = best_candidate.get('bounding_box')
            validated_data['offer_price'] = {"value": offer_price_val, "bbox": bbox}

    if regular_price_val is not None and price_candidates:
        best_candidate = min(price_candidates, key=lambda c: abs(c['parsed_value'] - float(regular_price_val)), default=None)
        if best_candidate and abs(best_candidate['parsed_value'] - float(regular_price_val)) < 0.02:
            bbox = best_candidate.get('bounding_box')
            validated_data['regular_price'] = {"value": regular_price_val, "bbox": bbox}
    
    # --- 2. Link Text Fields to their Bboxes using the find_bbox_for_text function ---
    fields_to_link = ["product_brand", "product_name_core", "product_variant_description", "size_quantity_info"]
    for field in fields_to_link:
        field_text = llm_data.get(field)
        if field_text:
            # Here we use the robust search to find the bbox just once.
            bbox = find_bbox_for_text(str(field_text), textract_blocks)
            if bbox:
                validated_data[field] = {"value": field_text, "bbox": bbox}
            else:
                 # If we can't find it, we still store the value but with no bbox
                validated_data[field] = {"value": field_text, "bbox": None}
    
    # --- 3. Final Sanity Checks (Example: Swapping prices) ---
    op_data = validated_data.get('offer_price')
    rp_data = validated_data.get('regular_price')

    if op_data and rp_data and op_data['value'] > rp_data['value']:
        validated_data['offer_price'], validated_data['regular_price'] = rp_data, op_data
        validated_data['validation_flags'].append("PRICE_LOGIC_SWAP")

    # Add other fields that don't need a bbox
    for key in ["unit_indicator", "store_specific_terms", "vision_llm_used"]:
        if key in llm_data:
            validated_data[key] = llm_data[key]

    return validated_data




# REPLACE the entire compare_product_items function with this corrected version

def compare_product_items(product_items1, product_items2, similarity_threshold=70):
    logger.info(f"COMPARE_FN - Starting comparison: {len(product_items1)} items from File1 with {len(product_items2)} items from File2.")
    comparison_report = []
    matched_item2_indices = set()
    
    # Threshold for considering text fields a match (e.g., 95% similar)
    FUZZY_MATCH_THRESHOLD = 95

    def normalize_text_for_comparison(text):
        if not text: return ""
        return re.sub(r'[\s\W_]+', '', str(text)).lower()

    for idx1, item1 in enumerate(product_items1):
        item1_id_log = item1.get("product_box_id", f"File1-Item{idx1}")
        
        brand1 = str(item1.get("product_brand", {}).get("value", "")).lower().strip()
        name_core1 = str(item1.get("product_name_core", {}).get("value", "")).lower().strip()
        primary_text1 = f"{brand1} {name_core1}".strip()
        variant1 = str(item1.get("product_variant_description", {}).get("value", "")).lower().strip()
        
        if not primary_text1:
            logger.warning(f"COMPARE_FN - {item1_id_log} has no primary text for matching. Skipping.")
            comparison_report.append({"Comparison_Type": "Unmatchable Product in File 1 (No Text)", "P1_Box_ID": item1_id_log, "Differences": "Missing core product text."})
            continue

        best_match_item2 = None
        highest_similarity = 0.0
        best_match_idx = -1

        for idx2, item2 in enumerate(product_items2):
            if idx2 in matched_item2_indices: continue

            brand2 = str(item2.get("product_brand", {}).get("value", "")).lower().strip()
            name_core2 = str(item2.get("product_name_core", {}).get("value", "")).lower().strip()
            primary_text2 = f"{brand2} {name_core2}".strip()
            variant2 = str(item2.get("product_variant_description", {}).get("value", "")).lower().strip()

            if not primary_text2: continue
            
            primary_similarity = fuzz.token_set_ratio(primary_text1, primary_text2)
            secondary_similarity = fuzz.token_set_ratio(variant1, variant2) if variant1 and variant2 else 100
            current_pair_similarity = (primary_similarity * 0.8) + (secondary_similarity * 0.2)
            
            if current_pair_similarity > highest_similarity:
                highest_similarity = current_pair_similarity
                best_match_item2 = item2
                best_match_idx = idx2
        
        if best_match_item2 and highest_similarity >= similarity_threshold:
            matched_item2_indices.add(best_match_idx)
            best_match_item2_id_log = best_match_item2.get("product_box_id", f"File2-Item{best_match_idx}")
            
            diff_details = []
            price_tolerance = 0.01
            
            # Price Comparison
            op1 = item1.get("offer_price", {}).get("value")
            op2 = best_match_item2.get("offer_price", {}).get("value")
            if (op1 is not None and op2 is not None and abs(op1 - op2) > price_tolerance) or ((op1 is None) != (op2 is None)):
                diff_details.append(f"Offer Price: F1=${op1 if op1 is not None else 'N/A'} vs F2=${op2 if op2 is not None else 'N/A'}")

            rp1 = item1.get("regular_price", {}).get("value")
            rp2 = best_match_item2.get("regular_price", {}).get("value")
            if (rp1 is not None and rp2 is not None and abs(rp1 - rp2) > price_tolerance) or ((rp1 is None) != (rp2 is None)):
                diff_details.append(f"Regular Price: F1=${rp1 if rp1 is not None else 'N/A'} vs F2=${rp2 if rp2 is not None else 'N/A'}")

            # --- FUZZY LOGIC FOR TEXT FIELDS ---
            size1_val = item1.get("size_quantity_info", {}).get("value")
            size2_val = best_match_item2.get("size_quantity_info", {}).get("value")
            if fuzz.ratio(normalize_text_for_comparison(size1_val), normalize_text_for_comparison(size2_val)) < FUZZY_MATCH_THRESHOLD:
                diff_details.append(f"Size: F1='{size1_val or 'N/A'}' vs F2='{size2_val or 'N/A'}'")
            
            variant1_val = item1.get("product_variant_description", {}).get("value")
            variant2_val = best_match_item2.get("product_variant_description", {}).get("value")
            if fuzz.ratio(normalize_text_for_comparison(variant1_val), normalize_text_for_comparison(variant2_val)) < FUZZY_MATCH_THRESHOLD:
                diff_details.append(f"Variant: F1='{variant1_val or 'N/A'}' vs F2='{variant2_val or 'N/A'}'")


            base_report_item = { "P1_Box_ID": item1_id_log, "P2_Box_ID": best_match_item2_id_log, "Similarity_Percent": round(highest_similarity, 1) }

            if diff_details:
                comparison_report.append({"Comparison_Type": "Product Match - Attribute Mismatch", **base_report_item, "Differences": "; ".join(diff_details)})
            else:
                comparison_report.append({"Comparison_Type": "Product Match - Attributes OK", **base_report_item, "Differences": ""})
        else:
            comparison_report.append({"Comparison_Type": "Unmatched Product in File 1", "P1_Box_ID": item1_id_log})
    
    for idx2, item2 in enumerate(product_items2):
        if idx2 not in matched_item2_indices:
            comparison_report.append({"Comparison_Type": "Unmatched Product in File 2 (Extra)", "P2_Box_ID": item2.get("product_box_id", f"File2-Item{idx2}")})
            
    logger.info(f"COMPARE_FN - Comparison finished. Report items: {len(comparison_report)}")
    return comparison_report
# ADD this new function to backend_processor.py

def run_vision_review_for_pair(item1, item2, page_pil_file1, page_pil_file2):
    """
    Sends a pair of cropped item images to the Vision LLM to resolve a text-based discrepancy.
    """
    item1_id_log = item1.get("product_box_id")
    item2_id_log = item2.get("product_box_id")
    logger.info(f"Running Vision Review for pair: {item1_id_log} vs {item2_id_log}")

    # Get cropped images for both items
    segment_bytes1 = get_segment_image_bytes(page_pil_file1, item1["roboflow_box_coords_pixels_center_wh"], item_id_for_log=item1_id_log)
    segment_bytes2 = get_segment_image_bytes(page_pil_file2, item2["roboflow_box_coords_pixels_center_wh"], item_id_for_log=item2_id_log)

    if not segment_bytes1 or not segment_bytes2:
        logger.error("Could not generate image segments for vision review pair.")
        return None, None # Indicate failure

    base64_image1 = base64.b64encode(segment_bytes1.getvalue()).decode('utf-8')
    base64_image2 = base64.b64encode(segment_bytes2.getvalue()).decode('utf-8')

    prompt = (
        "You are a visual verification expert. You will receive two images of product deals, labeled 'Image 1' and 'Image 2'. "
        "Your task is to visually inspect both images and extract the specified text fields from each one. "
        "Be precise and extract the text exactly as you see it. "
        "Fields to extract for each image:\n"
        "- \"offer_price\": The main offer price. Return as a decimal number (e.g., 15.97).\n"
        "- \"size_and_quantity\": The full size, quantity, or count text (e.g., 'Caja de 576', '6 rollos', '250 onzas 4 in 1').\n"
        "Return ONLY a single JSON object with the structure: "
        "{ \"item1_details\": { \"offer_price\": ..., \"size_and_quantity\": \"...\" }, "
        "\"item2_details\": { \"offer_price\": ..., \"size_and_quantity\": \"...\" } }"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image1}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}", "detail": "high"}}
            ]
        }
    ]

    try:
        chat_completion = openai_client_instance.chat.completions.create(
            model="gpt-4o", messages=messages, response_format={"type": "json_object"}, max_tokens=1000, temperature=0.0
        )
        response_content = chat_completion.choices[0].message.content
        vision_data = json.loads(response_content)
        logger.info(f"Vision review successful for pair. Data: {vision_data}")
        return vision_data
    except Exception as e:
        logger.error(f"Error during vision review API call for pair {item1_id_log}/{item2_id_log}: {e}", exc_info=True)
        return None

def draw_highlights_on_full_page_v2(full_page_pil_image: Image.Image,
                                      all_items_on_this_page: list,
                                      page_comparison_report_items: list,
                                      file_type: str) -> BytesIO:
    """
    Draws highlights (borders around Roboflow boxes) on a full PIL page image.
    - Unmatched items get one border color.
    - Matched items with attribute mismatches get another border color.
    """
    draw = ImageDraw.Draw(full_page_pil_image)
    img_width, img_height = full_page_pil_image.size

    # Define colors
    # For File 1 (typically displayed on the left)
    COLOR_MISMATCH_FILE1 = (255, 0, 0)      # Red: Matched item in File 1, but has attribute differences
    COLOR_UNMATCHED_FILE1 = (255, 165, 0)   # Orange: Item present in File 1, but unmatched in File 2

    # For File 2 (typically displayed on the right)
    COLOR_MISMATCH_FILE2 = (0, 128, 0)      # Green: Matched item in File 2, but has attribute differences
    COLOR_UNMATCHED_FILE2 = (0, 0, 255)     # Blue: Item present in File 2, but unmatched in File 1 (extra in File 2)

    # Determine current context colors
    current_mismatch_color = COLOR_MISMATCH_FILE1 if file_type == "file1" else COLOR_MISMATCH_FILE2
    current_unmatched_color = COLOR_UNMATCHED_FILE1 if file_type == "file1" else COLOR_UNMATCHED_FILE2
    
    OUTLINE_WIDTH = max(2, int(min(img_width, img_height) * 0.005)) # Slightly thicker for Roboflow boxes

    logger.debug(f"DRAW_FULL_PAGE_V2 ({file_type}): Processing {len(all_items_on_this_page)} items for page. Report items for this page: {len(page_comparison_report_items)}")

    for item_data in all_items_on_this_page:
        item_product_box_id = item_data.get("product_box_id")
        if not item_product_box_id:
            logger.warning(f"DRAW_FULL_PAGE_V2 ({file_type}): Item data missing product_box_id. Skipping highlight for this item.")
            continue

        relevant_report_entry = None
        for r_entry in page_comparison_report_items:
            # Check if the current item (from all_items_on_this_page) corresponds to P1 or P2 in the report entry
            if (file_type == "file1" and r_entry.get("P1_Box_ID") == item_product_box_id) or \
               (file_type == "file2" and r_entry.get("P2_Box_ID") == item_product_box_id):
                relevant_report_entry = r_entry
                break
        
        if not relevant_report_entry:
            # This item was found on the page but is not in the filtered comparison report for this page.
            # This implies it was a "Product Match - Attributes OK" or otherwise not flagged for highlighting.
            logger.debug(f"DRAW_FULL_PAGE_V2 ({file_type}): No relevant (highlightable) report entry for {item_product_box_id}. Assuming perfect match or no action needed.")
            continue

        comparison_type = relevant_report_entry.get("Comparison_Type", "")
        differences_text = relevant_report_entry.get("Differences", "") # String of differences
        
        outline_color_to_use = None
        action_description = "None"

        # Determine if the item (in its current file context) is unmatched or mismatched
        if file_type == "file1":
            if comparison_type == "Unmatched Product in File 1" or \
               comparison_type == "Unmatchable Product in File 1 (No Text)":
                outline_color_to_use = current_unmatched_color # Orange
                action_description = "Unmatched in File 1"
            elif comparison_type == "Product Match - Attribute Mismatch" and bool(differences_text):
                outline_color_to_use = current_mismatch_color # Red
                action_description = "Attribute Mismatch in File 1 item"
        elif file_type == "file2":
            if comparison_type == "Unmatched Product in File 2 (Extra)":
                outline_color_to_use = current_unmatched_color # Blue
                action_description = "Unmatched in File 2 (Extra)"
            elif comparison_type == "Product Match - Attribute Mismatch" and bool(differences_text):
                # This condition implies that the item from file2 (item_product_box_id)
                # was part of a matched pair that had differences.
                outline_color_to_use = current_mismatch_color # Green
                action_description = "Attribute Mismatch in File 2 item"
        
        if outline_color_to_use and item_data.get("roboflow_box_coords_pixels_center_wh"):
            rf_box_coords = item_data["roboflow_box_coords_pixels_center_wh"]
            cx_px, cy_px = rf_box_coords['x'], rf_box_coords['y']
            w_px, h_px = rf_box_coords['width'], rf_box_coords['height']

            x_min_px = int(cx_px - w_px / 2.0)
            y_min_px = int(cy_px - h_px / 2.0)
            x_max_px = int(cx_px + w_px / 2.0)
            y_max_px = int(cy_px + h_px / 2.0)
            
            # Optional: Add a small padding around the box
            # pad = 2 
            # x_min_px, y_min_px = max(0, x_min_px - pad), max(0, y_min_px - pad)
            # x_max_px, y_max_px = min(img_width - 1, x_max_px + pad), min(img_height - 1, y_max_px + pad)

            if x_min_px < x_max_px and y_min_px < y_max_px: # Ensure valid box
                draw.rectangle([(x_min_px, y_min_px), (x_max_px, y_max_px)],
                               outline=outline_color_to_use, width=OUTLINE_WIDTH)
                logger.debug(f"DRAW_FULL_PAGE_V2 ({file_type}): Drew Roboflow box for {item_product_box_id} ({action_description}) with color {outline_color_to_use}. Coords: ({x_min_px},{y_min_px})-({x_max_px},{y_max_px})")
            else:
                logger.warning(f"DRAW_FULL_PAGE_V2 ({file_type}): Invalid Roboflow box coordinates for {item_product_box_id} after calculation. Original: {rf_box_coords}")

        elif outline_color_to_use: # Color was determined, but no Roboflow box coords
            logger.warning(f"DRAW_FULL_PAGE_V2 ({file_type}): Item {item_product_box_id} was flagged for highlight ({action_description}) but 'roboflow_box_coords_pixels_center_wh' is missing.")

    img_byte_arr = BytesIO()
    full_page_pil_image.save(img_byte_arr, format='JPEG', quality=90)
    img_byte_arr.seek(0)
    return img_byte_arr



# REPLACE the entire draw_detailed_highlights function with this improved version.

def draw_detailed_highlights(full_page_pil_image: Image.Image,
                             all_items_on_page: list,
                             page_comparison_report_items: list,
                             file_type: str) -> BytesIO:
    """
    Draws highlights and labels based on the new "linked" data architecture.
    - Labels are drawn for all issue categories.
    - Highlights are drawn directly from the pre-linked bounding boxes.
    - No fallbacks are used; if a bbox isn't linked, no highlight is drawn.
    """
    if full_page_pil_image.mode != 'RGBA':
        full_page_pil_image = full_page_pil_image.convert('RGBA')

    overlay = Image.new('RGBA', full_page_pil_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    img_width, img_height = full_page_pil_image.size

    LABEL_COLORS = {
        "Price Issue": (217, 48, 38),
        "Text Issue": (234, 88, 12),
        "Photo": (126, 34, 206)
    }
    HIGHLIGHT_FILL = (255, 77, 77, 95)
    COLOR_LABEL_TEXT = (255, 255, 255)
    LABEL_BG_SHADOW = (0, 0, 0, 80)

    try:
        label_font_size = max(8, int(img_height * 0.009))
        font_path = Path(__file__).resolve().parent / "assets" / "arialbd.ttf"
        label_font = ImageFont.truetype(str(font_path), label_font_size)
    except IOError:
        logging.warning("Arial Bold font not found at 'assets/arialbd.ttf'. Using default font.")
        label_font = ImageFont.load_default()

    for report_item in page_comparison_report_items:
        box_id_key = "P1_Box_ID" if file_type == "file1" else "P2_Box_ID"
        box_id = report_item.get(box_id_key)
        item_data = next((item for item in all_items_on_page if item.get("product_box_id") == box_id), None)
        
        if not item_data or not item_data.get("roboflow_box_coords_pixels_center_wh"):
            continue

        categories = report_item.get('issue_categories', [])
        if not categories:
            continue

        rf_box = item_data["roboflow_box_coords_pixels_center_wh"]
        x_min_rf, y_min_rf = int(rf_box['x'] - rf_box['width'] / 2), int(rf_box['y'] - rf_box['height'] / 2)

        # --- Step 1: Draw all relevant labels ---
        label_y_offset = y_min_rf + 4
        for category in categories:
            label_text = category.replace(" Issue", "").upper()
            label_color = LABEL_COLORS.get(category, (128, 128, 128))
            
            try:
                label_box = draw.textbbox((0,0), label_text, font=label_font)
                label_width, label_height = label_box[2] - label_box[0], label_box[3] - label_box[1]
                
                shadow_coords = [x_min_rf + 3, label_y_offset - 2, x_min_rf + 3 + label_width + 10, label_y_offset + label_height + 8]
                draw.rectangle(shadow_coords, fill=LABEL_BG_SHADOW)

                bg_coords = [x_min_rf + 4, label_y_offset - 1, x_min_rf + 4 + label_width + 8, label_y_offset + label_height + 6]
                draw.rectangle(bg_coords, fill=label_color)

                draw.text((x_min_rf + 8, label_y_offset + 2), label_text, font=label_font, fill=COLOR_LABEL_TEXT)
                
                label_y_offset += label_height + 12
            except Exception as e:
                logging.error(f"Error drawing label for {box_id}: {e}")

        # --- Step 2: Draw highlights directly from linked bboxes ---
        if "Text Issue" in categories or "Price Issue" in categories:
            differences = report_item.get("Differences", "").split(';')
            for diff in differences:
                diff_attr_name = diff.split(':')[0].strip()
                
                field_key_map = {
                    "Offer Price": "offer_price", "Regular Price": "regular_price",
                    "Size": "size_quantity_info", "Variant": "product_variant_description",
                    "Name": "product_name_core", "Brand": "product_brand"
                }
                field_key = field_key_map.get(diff_attr_name)

                if field_key and field_key in item_data and isinstance(item_data[field_key], dict) and item_data[field_key].get("bbox"):
                    bbox = item_data[field_key]["bbox"]
                    
                    if 'Left' in bbox: # It's a price bbox from price_candidates
                        px_coords = (int(bbox['Left'] * img_width), int(bbox['Top'] * img_height),
                                     int((bbox['Left'] + bbox['Width']) * img_width), int((bbox['Top'] + bbox['Height']) * img_height))
                    else: # It's a text bbox from find_bbox_for_text
                        px_coords = (int(bbox['x_min'] * img_width), int(bbox['y_min'] * img_height),
                                     int(bbox['x_max'] * img_width), int(bbox['y_max'] * img_height))
                    
                    draw.rectangle(px_coords, fill=HIGHLIGHT_FILL)

    final_image = Image.alpha_composite(full_page_pil_image, overlay)
    
    img_byte_arr = BytesIO()
    final_image.convert("RGB").save(img_byte_arr, format='JPEG', quality=95)
    img_byte_arr.seek(0)
    return img_byte_arr

def find_bbox_for_text(text_to_find, textract_blocks, fuzz_threshold=85):
    """
    Finds a merged bounding box for a phrase by locating its constituent words
    within a list of Textract WORD blocks and merging their geometries.
    """
    if not text_to_find or not textract_blocks:
        return None

    # Clean and split the target phrase into individual words
    target_words = str(text_to_find).lower().strip().split()
    if not target_words:
        return None

    found_bboxes = []
    # Create a lookup for faster access to word blocks
    word_blocks = [block for block in textract_blocks if block.get('BlockType') == 'WORD']

    # Find bounding boxes for each word in the target phrase
    for word in target_words:
        best_match_block = None
        highest_score = 0
        for block in word_blocks:
            block_text = block.get('Text', '').lower()
            # Use a high threshold to find near-exact word matches
            score = fuzz.ratio(word, block_text)
            if score > highest_score:
                highest_score = score
                best_match_block = block

        if best_match_block and highest_score >= fuzz_threshold:
            bbox = best_match_block.get('Geometry', {}).get('BoundingBox')
            if bbox:
                found_bboxes.append(bbox)

    if not found_bboxes:
        logger.warning(f"find_bbox_for_text could not find any matching words for phrase: '{text_to_find}'")
        return None

    # Merge the bounding boxes of all found words
    x_min = min(b['Left'] for b in found_bboxes)
    y_min = min(b['Top'] for b in found_bboxes)
    x_max = max(b['Left'] + b['Width'] for b in found_bboxes)
    y_max = max(b['Top'] + b['Height'] for b in found_bboxes)

    merged_bbox = {
        'x_min': x_min,
        'y_min': y_min,
        'x_max': x_max,
        'y_max': y_max
    }
    logger.debug(f"Successfully found and merged bbox for '{text_to_find}': {merged_bbox}")
    return merged_bbox

def process_files_for_comparison(file1_bytes, file1_name, file2_bytes, file2_name):
    request_id = f"req_{int(time.time())}"
    logger.info(f"REQUEST_ID: {request_id} - Backend processing started for '{file1_name}' and '{file2_name}'")

    all_files_data = []
    temp_pdf_paths_to_cleanup = []

    try:
        # --- PHASE 1: Data Extraction for Both Files ---
        for file_idx_num, (file_bytes_content, original_filename) in enumerate(
            [(file1_bytes, file1_name), (file2_bytes, file2_name)]
        ):
            file_id_log_prefix = f"{request_id}-File{file_idx_num+1}"
            s3_safe_filename_part = secure_filename(original_filename)
            logger.info(f"{file_id_log_prefix} - Processing file: {original_filename}")

            items_for_this_file = []
            page_pils_for_this_file = []

            if original_filename.lower().endswith(".pdf"):
                fd, temp_pdf_path = tempfile.mkstemp(suffix=".pdf")
                os.close(fd)
                temp_pdf_paths_to_cleanup.append(temp_pdf_path)
                with open(temp_pdf_path, "wb") as f_pdf:
                    f_pdf.write(file_bytes_content)
                page_pils_for_this_file = convert_from_path(temp_pdf_path, dpi=200, poppler_path=POPPLER_BIN_PATH, fmt='jpeg', timeout=300)
            else:
                page_pils_for_this_file = [Image.open(BytesIO(file_bytes_content))]
            
            logger.info(f"{file_id_log_prefix} - Converted to {len(page_pils_for_this_file)} page(s).")

            s3_keys_for_this_file = []
            for page_idx, page_image_pil in enumerate(page_pils_for_this_file):
                page_id_log = f"{file_id_log_prefix}-Page{page_idx}"
                image_width_px, image_height_px = page_image_pil.size

                img_byte_arr_s3 = BytesIO()
                page_image_pil.save(img_byte_arr_s3, format='JPEG', quality=90)
                img_byte_arr_s3.seek(0)
                s3_page_key = f"pages/{request_id}_{file_idx_num}_p{page_idx}.jpg"
                upload_to_s3(img_byte_arr_s3, S3_BUCKET_NAME, s3_page_key)
                s3_keys_for_this_file.append(s3_page_key)

                textract_blocks = get_analysis_from_document_via_textract(S3_BUCKET_NAME, s3_page_key)
                roboflow_preds = get_roboflow_predictions_sdk(page_image_pil, f"{s3_safe_filename_part}_p{page_idx}")

                if not textract_blocks or not roboflow_preds:
                    logger.warning(f"{page_id_log} - Missing Textract/Roboflow data. Skipping page.")
                    continue

                blocks_map = {b['Id']: b for b in textract_blocks}
                collated_snippets = collate_text_for_product_boxes(roboflow_preds, textract_blocks, blocks_map, image_width_px, image_height_px, page_id_log)

                for snippet in collated_snippets:
                    item_id_for_log = snippet.get("product_box_id")
                    
                    llm_output = extract_product_data_with_llm(
                        snippet["collated_text"],
                        item_id_for_log,
                        textract_blocks_in_segment=snippet["textract_blocks_in_segment"]
                    )
                    
                    processed_item = post_process_and_validate_item_data(
                        llm_output, 
                        snippet["price_candidates"], 
                        snippet["collated_text"], 
                        snippet["textract_blocks_in_segment"],
                        item_id_for_log
                    )
                    
                    items_for_this_file.append({
                        "page_idx_for_reprocessing": page_idx,
                        "original_filename": original_filename,
                        **snippet,
                        **processed_item
                    })

            for s3_key in s3_keys_for_this_file:
                delete_from_s3(S3_BUCKET_NAME, s3_key)

            all_files_data.append({
                "filename": original_filename,
                "items": items_for_this_file,
                "page_pils": page_pils_for_this_file
            })

        final_product_items_file1 = all_files_data[0]["items"]
        final_product_items_file2 = all_files_data[1]["items"]
        page_pils_file1 = all_files_data[0]["page_pils"]
        page_pils_file2 = all_files_data[1]["page_pils"]

        # --- PHASE 2: Comparison, Vision Review, and Data Correction ---
        logger.info(f"REQUEST_ID: {request_id} - Running preliminary comparison...")
        preliminary_comparison_report = compare_product_items(final_product_items_file1, final_product_items_file2)
        final_comparison_report = []

        
            
        for report_item in preliminary_comparison_report:
            item_data = report_item
            p1_box_id = item_data.get("P1_Box_ID")
            p2_box_id = item_data.get("P2_Box_ID")

            # --- Step 1: Handle Matched Pairs vs. Unmatched Items ---
            # If the item is not a matched pair, just add it to the final report and continue.
            if not (p1_box_id and p2_box_id):
                # You can add issue categorization for unmatched items if desired
                # For example: item_data['issue_categories'] = ["Unmatched"]
                final_comparison_report.append(item_data)
                continue

            # Find the full data for the matched pair
            item1 = next((item for item in final_product_items_file1 if item.get("product_box_id") == p1_box_id), None)
            item2 = next((item for item in final_product_items_file2 if item.get("product_box_id") == p2_box_id), None)

            if not (item1 and item2):
                final_comparison_report.append(item_data)
                continue

            # --- Step 2: Independent Vision Checks for the Matched Pair ---
            
            # --- Check 2a: Text/Price Correction (OpenAI Vision) ---
            # If the initial text comparison found a mismatch, ALWAYS try to resolve it with the OpenAI model.
            if "Mismatch" in item_data.get("Comparison_Type", ""):
                page_idx1 = item1["page_idx_for_reprocessing"]
                page_idx2 = item2["page_idx_for_reprocessing"]
                vision_data = run_vision_review_for_pair(item1, item2, page_pils_file1[page_idx1], page_pils_file2[page_idx2])
                
                if vision_data:
                    v1_details = vision_data.get("item1_details", {})
                    v2_details = vision_data.get("item2_details", {})
                    
                    v1_price = parse_price_string(v1_details.get("offer_price"))
                    v2_price = parse_price_string(v2_details.get("offer_price"))
                    v1_size = str(v1_details.get("size_and_quantity") or "")
                    v2_size = str(v2_details.get("size_and_quantity") or "")

                    # Re-compare using the new data from the vision model
                    prices_match_vision = (v1_price is not None and v2_price is not None and abs(v1_price - v2_price) < 0.02)
                    sizes_match_vision = fuzz.ratio(v1_size, v2_size) >= 95

                    if prices_match_vision and sizes_match_vision:
                        logger.warning(f"VISION CORRECTION: Mismatch for {p1_box_id} resolved by vision.")
                        item_data["Comparison_Type"] = "Product Match - Attributes OK (Corrected by Vision)"
                        item_data["Differences"] = "" # Clear previous differences
                    else:
                        logger.warning(f"VISION CONFIRMATION: Mismatch for {p1_box_id} confirmed by vision.")
                        confirmed_diffs = []
                        if not prices_match_vision:
                            confirmed_diffs.append(f"Price: F1=${v1_price or 'N/A'} vs F2=${v2_price or 'N/A'}")
                        if not sizes_match_vision:
                            confirmed_diffs.append(f"Size: F1='{v1_size or 'N/A'}' vs F2='{v2_size or 'N/A'}'")
                        item_data["Differences"] = "; ".join(confirmed_diffs)

            # --- Check 2b: Photo Similarity Sanity Check (HF CLIP Model) ---
            # This check runs independently to flag if the core products appear visually different.
            page_idx1 = item1["page_idx_for_reprocessing"]
            page_idx2 = item2["page_idx_for_reprocessing"]
            segment_bytes1 = get_segment_image_bytes(page_pils_file1[page_idx1], item1["roboflow_box_coords_pixels_center_wh"])
            segment_bytes2 = get_segment_image_bytes(page_pils_file2[page_idx2], item2["roboflow_box_coords_pixels_center_wh"])
            
            if check_for_photo_mistake(segment_bytes1, segment_bytes2, item1["product_box_id"]):
                current_diffs = item_data.get("Differences", "")
                photo_diff_str = "Photo Mistake: Core products appear different."
                if photo_diff_str not in current_diffs:
                    item_data["Differences"] = f"{current_diffs}; {photo_diff_str}".strip('; ')
                
                # Ensure the type reflects there's an issue
                if "Mismatch" not in item_data.get("Comparison_Type", ""):
                    item_data["Comparison_Type"] = "Product Match - Attribute Mismatch"

            # --- Step 3: Final Categorization for UI Highlighting ---
            # Based on the final state of "Differences", categorize the issues.
            final_differences = item_data.get("Differences", "")
            categories = []
            if "Price" in final_differences:
                categories.append("Price Issue")
            if any(term in final_differences for term in ["Size", "Variant", "Name", "Brand"]):
                categories.append("Text Issue")
            if "Photo Mistake" in final_differences:
                categories.append("Photo")
            
            item_data['issue_categories'] = list(set(categories))
            
            final_comparison_report.append(item_data)


        # --- PHASE 3: Final Report and Image Generation ---
        logger.info(f"REQUEST_ID: {request_id} - Generating final report and highlighted images...")
        highlighted_pages_file1 = []
        highlighted_pages_file2 = []
        max_pages = max(len(page_pils_file1), len(page_pils_file2))
        for page_idx in range(max_pages):
            report_items_on_page1 = [r for r in final_comparison_report if r.get("P1_Box_ID") and f"-Page{page_idx}-" in r["P1_Box_ID"]]
            report_items_on_page2 = [r for r in final_comparison_report if r.get("P2_Box_ID") and f"-Page{page_idx}-" in r["P2_Box_ID"]]
            
            if page_idx < len(page_pils_file1):
                page_pil_copy1 = page_pils_file1[page_idx].copy()
                items_on_page1 = [item for item in final_product_items_file1 if item["page_idx_for_reprocessing"] == page_idx]
                img_bytes1 = draw_detailed_highlights(page_pil_copy1, items_on_page1, report_items_on_page1, "file1")
                highlighted_pages_file1.append(base64.b64encode(img_bytes1.getvalue()).decode('utf-8'))
            
            if page_idx < len(page_pils_file2):
                page_pil_copy2 = page_pils_file2[page_idx].copy()
                items_on_page2 = [item for item in final_product_items_file2 if item["page_idx_for_reprocessing"] == page_idx]
                img_bytes2 = draw_detailed_highlights(page_pil_copy2, items_on_page2, report_items_on_page2, "file2")
                highlighted_pages_file2.append(base64.b64encode(img_bytes2.getvalue()).decode('utf-8'))

        report_df = pd.DataFrame(final_comparison_report)
        csv_buffer = io.StringIO()
        report_df.to_csv(csv_buffer, index=False)
        csv_data_string = csv_buffer.getvalue()
        
          # --- NEW: Calculate Final Summary Metrics ---
        price_mistakes = 0
        text_mistakes = 0
        photo_mistakes = 0
        
        # Use a set to count unique items with discrepancies
        items_with_discrepancies = set()

        for item in final_comparison_report:
            categories = item.get('issue_categories', [])
            if categories:
                # Add the item's ID to the set. The final count will be the size of this set.
                # We use P1_Box_ID as the unique identifier for a comparison pair.
                if item.get("P1_Box_ID"):
                    items_with_discrepancies.add(item["P1_Box_ID"])

                if "Price Issue" in categories:
                    price_mistakes += 1
                if "Text Issue" in categories:
                    text_mistakes += 1
                if "Photo Mistake" in categories:
                    photo_mistakes += 1

        detailed_summary = {
            "total_mistakes": len(items_with_discrepancies),
            "price_mistakes": price_mistakes,
            "text_mistakes": text_mistakes,
            "photo_mistakes": photo_mistakes,
        }
        # --- END OF NEW SUMMARY LOGIC ---

        return {
            "message": "Processing complete.",
            "product_comparison_details": final_comparison_report,
            "report_csv_data": csv_data_string,
            "highlighted_pages_file1": highlighted_pages_file1,
            "highlighted_pages_file2": highlighted_pages_file2,
            "all_product_details_file1": final_product_items_file1,
            "all_product_details_file2": final_product_items_file2,
            "detailed_summary": detailed_summary
        }
    finally:
        for temp_path in temp_pdf_paths_to_cleanup:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        logger.info(f"REQUEST_ID: {request_id} - process_files_for_comparison endpoint finished.")
        
        
# In visual_layout_backend.py
# ADD this entire new function to your file.

def check_for_photo_mistake(segment_bytes1: BytesIO, segment_bytes2: BytesIO, item_id_for_log="N/A") -> bool:
    """
    Uses a local, open-source CLIP model to determine if the core products in two images are different.
    Returns True if images are likely different, False otherwise.
    """
    if not vision_model or not vision_processor or not segment_bytes1 or not segment_bytes2:
        logger.warning(f"ITEM_ID: {item_id_for_log} - Vision model/processor not available. Skipping photo mistake check.")
        return False

    try:
        # Reset buffer position just in case
        segment_bytes1.seek(0)
        segment_bytes2.seek(0)
        
        image1 = Image.open(segment_bytes1)
        image2 = Image.open(segment_bytes2)

        # Process images and get their vector embeddings
        inputs = vision_processor(images=[image1, image2], return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = vision_model.get_image_features(**inputs)

        # Normalize the features and calculate cosine similarity
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        similarity = (image_features[0] @ image_features[1].T).item()
        
        # Define a threshold for what is considered a "different" image
        # You can tune this threshold. Lower means more tolerant.
        SIMILARITY_THRESHOLD = 0.7

        logger.info(f"ITEM_ID: {item_id_for_log} - Photo similarity check score: {similarity:.4f} (Threshold: {SIMILARITY_THRESHOLD})")

        # If similarity is below the threshold, they are considered different products.
        return similarity < SIMILARITY_THRESHOLD

    except Exception as e:
        logger.error(f"ITEM_ID: {item_id_for_log} - Error during local photo mistake check: {e}", exc_info=True)
        return False


# If you want to test this module directly (optional)
if __name__ == '__main__':
    logger.info("backend_processor.py is being run directly (e.g., for testing).")
  
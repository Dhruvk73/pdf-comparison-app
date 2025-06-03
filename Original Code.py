# backend_processor.py
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

def find_bbox_for_text(text_to_find, textract_blocks, fuzz_threshold=80):
    """
    Finds the bounding box for a given text string within a list of Textract blocks.
    Uses fuzzy matching for robustness. Returns normalized bbox (0-1).
    """
    if not text_to_find or not textract_blocks:
        return None

    text_to_find_lower = str(text_to_find).lower()
    best_match_bbox = None
    best_match_score = 0

    for block in textract_blocks:
        block_text = block.get('Text', '').lower()
        if block_text:
            # Use token_set_ratio for better handling of word order/extra words
            score = fuzz.token_set_ratio(text_to_find_lower, block_text)
            if score > best_match_score and score >= fuzz_threshold:
                best_match_score = score
                # Combine multiple word bounding boxes if necessary, or just use the line's bbox
                # For simplicity, we'll try to get the overall bounding box of the matched text within the segment
                # If block is a WORD, use its bbox. If a LINE, use its bbox.
                bbox = block.get('Geometry', {}).get('BoundingBox')
                if bbox:
                    best_match_bbox = {
                        'x_min': bbox['Left'],
                        'y_min': bbox['Top'],
                        'x_max': bbox['Left'] + bbox['Width'],
                        'y_max': bbox['Top'] + bbox['Height']
                    }
    logger.debug(f"Found bbox for '{text_to_find}' with score {best_match_score}: {best_match_bbox}")
    return best_match_bbox

def extract_product_data_with_llm(product_snippet_text: str, item_id_for_log="N/A", llm_model: str = "gpt-4o", textract_blocks_in_segment: list = None) -> dict: # NEW parameter
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
        logger.debug(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - Text LLM Raw Response Content: {response_content}")
        if response_content is None:
            logger.error(f"ITEM_ID: {item_id_for_log} - Text LLM returned None content.")
            return {"error_message": "Text LLM returned no content", "llm_input_snippet": product_snippet_text}

        extracted_data = json.loads(response_content)
        logger.debug(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - LLM Data (after json.loads): {json.dumps(extracted_data, indent=2)}")
        
        extracted_data['offer_price'] = parse_price_string(extracted_data.get('offer_price'), item_id_for_log=f"{item_id_for_log}-llm_offer")
        extracted_data['regular_price'] = parse_price_string(extracted_data.get('regular_price'), item_id_for_log=f"{item_id_for_log}-llm_regular")
        
        expected_fields = ["product_brand", "product_name_core", "product_variant_description", "size_quantity_info", "offer_price", "regular_price", "unit_indicator", "store_specific_terms"]
        for field in expected_fields:
            if field not in extracted_data: extracted_data[field] = None

        # NEW: Find bounding boxes using Textract blocks
        if textract_blocks_in_segment:
            for field in ["offer_price", "regular_price", "product_brand", "product_name_core", "product_variant_description", "size_quantity_info", "unit_indicator", "store_specific_terms"]:
                field_value = extracted_data.get(field)
                if field_value is not None:
                    # Convert price floats back to string for fuzzy matching (e.g. 6.97 to "6.97")
                    if field in ["offer_price", "regular_price"]:
                        field_value_str = f"{field_value:.2f}" if isinstance(field_value, (float, int)) else str(field_value)
                    else:
                        field_value_str = str(field_value)
                    
                    bbox = find_bbox_for_text(field_value_str, textract_blocks_in_segment)
                    if bbox:
                        extracted_data[f"{field}_bbox"] = bbox
                        logger.debug(f"ITEM_ID: {item_id_for_log} - Found Textract bbox for {field}: {bbox}")
                    else:
                        logger.debug(f"ITEM_ID: {item_id_for_log} - No Textract bbox found for {field} text: '{field_value_str}'")

        logger.info(f"ITEM_ID: {item_id_for_log} - Successfully extracted and parsed data from Text LLM.")
        logger.debug(f"ITEM_ID: {item_id_for_log} - Parsed LLM Data: {json.dumps(extracted_data, indent=2)}")
        return extracted_data
    except json.JSONDecodeError as je:
        logger.error(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - JSONDecodeError: {je}. Response: {response_content}", exc_info=True)
        return {"error_message": f"JSONDecodeError: {je}", "llm_input_snippet": product_snippet_text, "llm_response_content": response_content}
    except Exception as e:
        logger.error(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - Error in Text LLM processing: {e}", exc_info=True)
        return {"error_message": str(e), "llm_input_snippet": product_snippet_text, "llm_response_content": locals().get("response_content", "N/A")}


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

def collate_text_for_product_boxes(roboflow_boxes, textract_all_blocks, blocks_map,
                                   image_width_px, image_height_px, page_id_for_log="N/A"):
    product_texts_with_candidates = []
    if not roboflow_boxes or not textract_all_blocks or not blocks_map or \
       image_width_px is None or image_height_px is None:
        logger.warning(f"PAGE_ID: {page_id_for_log} - collate_text_for_product_boxes: Missing critical inputs.")
        return product_texts_with_candidates

    logger.info(f"PAGE_ID: {page_id_for_log} - collate_text_for_product_boxes - Starting smart collation for {len(roboflow_boxes)} Roboflow boxes.")
    
    all_words_on_page = [block for block in textract_all_blocks if block['BlockType'] == 'WORD']
    all_lines_on_page = [block for block in textract_all_blocks if block['BlockType'] == 'LINE']

    for i, box_pred in enumerate(roboflow_boxes):
        item_id_for_log = f"{page_id_for_log}-RFBox{i}-{str(box_pred.get('class', 'UnknownClass'))}"
        logger.debug(f"ITEM_ID: {item_id_for_log} - collate_text_for_product_boxes - Processing Roboflow Box: {json.dumps(box_pred)}")

        rf_center_x_px, rf_center_y_px = box_pred.get('x'), box_pred.get('y')
        rf_width_px, rf_height_px = box_pred.get('width'), box_pred.get('height')
        
        if not all(isinstance(v, (int, float)) for v in [rf_center_x_px, rf_center_y_px, rf_width_px, rf_height_px]):
            logger.warning(f"ITEM_ID: {item_id_for_log} - collate_text_for_product_boxes - Roboflow Box has invalid coordinates. Skipping.")
            continue
            
        rf_x_min_rel = (rf_center_x_px - rf_width_px / 2.0) / image_width_px
        rf_y_min_rel = (rf_center_y_px - rf_height_px / 2.0) / image_height_px
        rf_x_max_rel = (rf_center_x_px + rf_width_px / 2.0) / image_width_px
        rf_y_max_rel = (rf_center_y_px + rf_height_px / 2.0) / image_height_px
        logger.debug(f"ITEM_ID: {item_id_for_log} - Roboflow Box Rel Coords (xmin,ymin,xmax,ymax): ({rf_x_min_rel:.4f}, {rf_y_min_rel:.4f}, {rf_x_max_rel:.4f}, {rf_y_max_rel:.4f})")
        
        words_in_rf_box = []
        for word_block in all_words_on_page:
            geom = word_block.get('Geometry', {}).get('BoundingBox', {})
            if not geom: continue
            word_center_x = geom['Left'] + geom['Width'] / 2
            word_center_y = geom['Top'] + geom['Height'] / 2
            if rf_x_min_rel <= word_center_x <= rf_x_max_rel and \
               rf_y_min_rel <= word_center_y <= rf_y_max_rel and \
               word_block.get('Confidence', 0) >= GEOM_MERGE_MIN_WORD_CONFIDENCE:
                words_in_rf_box.append(word_block)
        
        words_in_rf_box.sort(key=lambda w: (w['Geometry']['BoundingBox']['Top'], w['Geometry']['BoundingBox']['Left']))
        
        logger.debug(f"ITEM_ID: {item_id_for_log} - Found {len(words_in_rf_box)} WORD blocks with >={GEOM_MERGE_MIN_WORD_CONFIDENCE}% conf inside this Roboflow box.")

        merged_geom_price_str = None
        used_word_ids_for_geom_price = set()

        for idx_w1, w1 in enumerate(words_in_rf_box):
            if w1['Id'] in used_word_ids_for_geom_price: continue
            w1_text = w1.get('Text', '')
            w1_geom_box = w1.get('Geometry', {}).get('BoundingBox')
            
            if not (re.fullmatch(r'[1-9]\d?', w1_text) and w1_geom_box):  # Adjusted to catch 1 or 2 digits for dollars in geometric merge
                continue
            logger.debug(f"ITEM_ID: {item_id_for_log} - GeomMerge: Potential dollar part w1='{w1_text}' (ID: {w1['Id']})")

            for idx_w2 in range(idx_w1 + 1, len(words_in_rf_box)):
                w2 = words_in_rf_box[idx_w2]
                if w2['Id'] in used_word_ids_for_geom_price: continue
                w2_text = w2.get('Text', '')
                w2_geom_box = w2.get('Geometry', {}).get('BoundingBox')

                if not (re.fullmatch(r'\d{2}', w2_text) and w2_geom_box): # Potential two cents digits
                    continue
                logger.debug(f"ITEM_ID: {item_id_for_log} - GeomMerge: Potential cents w2='{w2_text}' (ID: {w2['Id']}) for w1='{w1_text}'")
                
                w1_cy = w1_geom_box['Top'] + w1_geom_box['Height'] / 2
                w2_cy = w2_geom_box['Top'] + w2_geom_box['Height'] / 2
                y_diff_abs = abs(w1_cy - w2_cy)
                y_tolerance = w1_geom_box['Height'] * Y_ALIGN_TOLERANCE_FACTOR
                
                w1_right_edge = w1_geom_box['Left'] + w1_geom_box['Width']
                horizontal_gap = w2_geom_box['Left'] - w1_right_edge
                x_tolerance = w1_geom_box['Width'] * X_SPACING_TOLERANCE_FACTOR
                
                height_compatible = (w2_geom_box['Height'] < (w1_geom_box['Height'] * CENTS_MAX_HEIGHT_FACTOR)) and \
                                    (w2_geom_box['Height'] > (w1_geom_box['Height'] * (1/CENTS_MAX_HEIGHT_FACTOR*0.8) ))

                vertically_aligned = y_diff_abs < y_tolerance
                horizontally_close_and_ordered = 0 <= horizontal_gap < x_tolerance

                logger.debug(f"ITEM_ID: {item_id_for_log} - GeomMerge Cand: w1='{w1_text}', w2='{w2_text}'. "
                                             f"V-Align: {vertically_aligned} (y_diff:{y_diff_abs:.4f} vs tol:{y_tolerance:.4f}). "
                                             f"H-Close: {horizontally_close_and_ordered} (gap:{horizontal_gap:.4f} vs tol:{x_tolerance:.4f}). "
                                             f"H-Compat: {height_compatible} (h1:{w1_geom_box['Height']:.4f}, h2:{w2_geom_box['Height']:.4f})")

                if vertically_aligned and horizontally_close_and_ordered and height_compatible:
                    merged_geom_price_str = f"[GEOM_PRICE: {w1_text} {w2_text}]"
                    used_word_ids_for_geom_price.add(w1['Id'])
                    used_word_ids_for_geom_price.add(w2['Id'])
                    logger.info(f"ITEM_ID: {item_id_for_log} - collate_text_for_product_boxes - Geometrically merged price candidate found: '{merged_geom_price_str}' from w1:'{w1_text}' (ID:{w1['Id']}) and w2:'{w2['Id']})")
                    break
            if merged_geom_price_str:
                break

        lines_in_box_objects = []
        # Get LINE blocks that are inside the Roboflow box
        for line_block in all_lines_on_page:
            txt_geom = line_block.get('Geometry', {}).get('BoundingBox', {})
            if not txt_geom: continue
            line_center_x_rel = txt_geom['Left'] + (txt_geom['Width'] / 2.0)
            line_center_y_rel = txt_geom['Top'] + (txt_geom['Height'] / 2.0)
            if (rf_x_min_rel <= line_center_x_rel <= rf_x_max_rel and \
                rf_y_min_rel <= line_center_y_rel <= rf_y_max_rel):
                logger.debug(f"ITEM_ID: {item_id_for_log} - LineCollation: Considering LINE '{line_block.get('Text', '')}' (ID: {line_block['Id']}) Geom: {json.dumps(txt_geom)}")
                lines_in_box_objects.append(line_block)
        
        lines_in_box_objects.sort(key=lambda line: (
            line['Geometry']['BoundingBox']['Top'],
            -line['Geometry']['BoundingBox']['Height'],
            line['Geometry']['BoundingBox']['Left']
        ))
        
        ordered_lines_text_parts = []
        # Store detailed blocks for later lookup
        detailed_blocks_in_segment = [] # NEW: Store all relevant textract blocks
        for line_block in lines_in_box_objects:
            line_words_for_text = []
            if 'Relationships' in line_block:
                for relationship in line_block['Relationships']:
                    if relationship['Type'] == 'CHILD':
                        for child_id in relationship['Ids']:
                            word = blocks_map.get(child_id)
                            if word and word['BlockType'] == 'WORD' and child_id not in used_word_ids_for_geom_price:
                                line_words_for_text.append(word['Text'])
                                detailed_blocks_in_segment.append(word) # Add words that contribute to collated text
            
            line_text = " ".join(line_words_for_text).strip()
            
            if line_text:
                ordered_lines_text_parts.append(line_text)
            elif not line_words_for_text and line_block.get('Text') and \
                any(cid in used_word_ids_for_geom_price for rel in line_block.get('Relationships', []) if rel['Type'] == 'CHILD' for cid in rel.get('Ids',[])):
                logger.debug(f"ITEM_ID: {item_id_for_log} - LineCollation: Line '{line_block.get('Text')}' fully consumed by geometric merge, not adding to collated text.")
            elif not line_text and line_block.get('Text'):
                all_words_used = True
                if 'Relationships' in line_block:
                    for relationship in line_block['Relationships']:
                        if relationship['Type'] == 'CHILD':
                            if not all(child_id in used_word_ids_for_geom_price for child_id in relationship['Ids'] if blocks_map.get(child_id, {}).get('BlockType') == 'WORD'):
                                all_words_used = False
                                break
                    if not all_words_used :
                        ordered_lines_text_parts.append(line_block.get('Text','').strip())
                        detailed_blocks_in_segment.append(line_block) # Add line if it was not fully consumed by geom price
                elif line_block.get('Text','').strip():
                    ordered_lines_text_parts.append(line_block.get('Text','').strip())
                    detailed_blocks_in_segment.append(line_block) # Add line if it had text but no children

        collated_text_multiline = "\n".join(ordered_lines_text_parts)
        collated_text_cleaned = clean_text(collated_text_multiline)

        if merged_geom_price_str:
            collated_text_cleaned = f"{merged_geom_price_str}\n{collated_text_cleaned}".strip()
            logger.info(f"ITEM_ID: {item_id_for_log} - Prepended geometric price. New collated text starts with: {merged_geom_price_str}")

        price_candidates_for_segment = detect_price_candidates(lines_in_box_objects, image_height_px, blocks_map, item_id_for_log=item_id_for_log, prepended_geom_price=merged_geom_price_str if merged_geom_price_str else None)

        logger.debug(f"ITEM_ID: {item_id_for_log} - collate_text_for_product_boxes - Sorted Lines Text for Collation (after potential geom exclusion): {json.dumps(ordered_lines_text_parts)}")
        logger.info(f"ITEM_ID: {item_id_for_log} - collate_text_for_product_boxes - Final Collated Text (cleaned) for LLM:\n{collated_text_cleaned}")

        if collated_text_cleaned or merged_geom_price_str:
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
                "textract_blocks_in_segment": detailed_blocks_in_segment # NEW: Store Textract blocks
            })
    
    logger.info(f"PAGE_ID: {page_id_for_log} - collate_text_for_product_boxes - Collation complete. Generated {len(product_texts_with_candidates)} product text snippets.")
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

# In backend_processor.py

# In backend_processor.py

def post_process_and_validate_item_data(llm_data, price_candidates, original_collated_text, item_id_for_log="N/A"):
    if not isinstance(llm_data, dict):
        logger.error(f"ITEM_ID: {item_id_for_log} - Invalid llm_data input type to post_process_and_validate_item_data: {type(llm_data)}")
        return {"error_message": "Invalid input data for post-processing.", 
                "validation_flags": ["Invalid input data type for post-processing."]} # Added flag
    
    if "error_message" in llm_data:
        logger.warning(f"ITEM_ID: {item_id_for_log} - LLM data contains an error: {llm_data.get('error_message')}. Initializing empty fields.")
        # Ensure basic structure even if LLM errored
        base_keys = ['offer_price', 'regular_price', 'product_brand', 'product_name_core', 
                     'product_variant_description', 'size_quantity_info', 'unit_indicator', 
                     'store_specific_terms', 'parsed_size_details', 'size_quantity_info_normalized']
        for key in base_keys:
            llm_data.setdefault(key, None)
        llm_data.setdefault('validation_flags', []).append(f"LLM data error: {llm_data.get('error_message')}")
        return llm_data

    # Call the simplified enhanced_normalize_product_data. It will mostly ensure keys exist.
    item_data = enhanced_normalize_product_data(llm_data.copy(), item_id_for_log, price_candidates, original_collated_text)
    
    # --- TRUE BYPASS IMPLEMENTATION FOR SIZE FIELDS ---
    logger.info(f"ITEM_ID: {item_id_for_log} - Applying BYPASS for size fields. Using raw LLM 'size_quantity_info'.")
    
    # Get the size_quantity_info directly from the original LLM output (llm_data)
    raw_llm_size_info = llm_data.get('size_quantity_info')

    if isinstance(raw_llm_size_info, str):
        current_size_info = raw_llm_size_info.strip()
        item_data['size_quantity_info'] = current_size_info  # Set the primary size field
        item_data['size_quantity_info_normalized'] = current_size_info # Use raw for normalized
    elif raw_llm_size_info is None:
        item_data['size_quantity_info'] = None
        item_data['size_quantity_info_normalized'] = None
    else: # If it's some other type, convert to string
        current_size_info = str(raw_llm_size_info).strip()
        item_data['size_quantity_info'] = current_size_info
        item_data['size_quantity_info_normalized'] = current_size_info
        
    item_data['parsed_size_details'] = None # Explicitly None for bypass

    logger.debug(f"ITEM_ID: {item_id_for_log} - After BYPASS: "
                 f"SQI: '{item_data.get('size_quantity_info')}', "
                 f"Norm_SQI: '{item_data.get('size_quantity_info_normalized')}', "
                 f"ParsedDetails: {item_data.get('parsed_size_details')}")
    # --- END TRUE BYPASS IMPLEMENTATION ---

    # --- Your existing price validation, other flag settings etc. continue here ---
    # Ensure validation_flags list exists
    if not isinstance(item_data.get('validation_flags'), list):
        item_data['validation_flags'] = []

    # (Your actual price validation logic, etc.)
    # Note: The 'Size/quantity info might be missing or unparsed' flag might trigger
    # if raw_llm_size_info is empty or None. This is expected behavior with the bypass.
    if not item_data.get('size_quantity_info_normalized') and item_data.get('size_quantity_info'): # Check if LLM provided size but it ended up None
         item_data['validation_flags'].append('Size/quantity info from LLM was present but resulted in None for normalized (BYPASS).')
    elif not item_data.get('size_quantity_info'):
         item_data['validation_flags'].append('Size/quantity info was empty/None from LLM (BYPASS).')


    logger.debug(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data (SIZE BYPASSED) - Returning: "
                 f"Offer: {item_data.get('offer_price')}, Reg: {item_data.get('regular_price')}, "
                 f"Brand: '{item_data.get('product_brand')}', Name: '{item_data.get('product_name_core')}', "
                 f"SQI: '{item_data.get('size_quantity_info')}', "
                 f"Norm_SQI: '{item_data.get('size_quantity_info_normalized')}', "
                 f"Parsed: {item_data.get('parsed_size_details')}, "
                 f"Flags: {item_data.get('validation_flags')}")

    return item_data


# backend_processor.py

# Ensure these are imported at the top of your file if not already
import json # For logging complex objects if needed, and for P1/P2_Parsed_Size_Details
from fuzzywuzzy import fuzz
# import logging # Already configured

# logger = logging.getLogger(__name__) # Already configured

# backend_processor.py

import json
from fuzzywuzzy import fuzz
# import logging # Already configured elsewhere

# logger = logging.getLogger(__name__) # Already configured

def compare_product_items(product_items1, product_items2, similarity_threshold=70):
    logger.info(f"COMPARE_FN - Starting comparison: {len(product_items1)} items from File1 with {len(product_items2)} items from File2.")
    comparison_report = []
    matched_item2_indices = set()

    for idx1, item1 in enumerate(product_items1):
        item1_id_log = item1.get("product_box_id", f"File1-Item{idx1}")
        logger.debug(f"COMPARE_FN - OuterLoop: Processing item1_id: {item1_id_log}")
        
        best_match_item2 = None
        highest_similarity = 0.0
        best_match_idx = -1
        
        # --- Prepare item1 texts ---
        brand1 = str(item1.get("product_brand", "") or "").lower().strip()
        name_core1 = str(item1.get("product_name_core", "") or "").lower().strip()
        primary_text1 = f"{brand1} {name_core1}".strip()

        size1_details = item1.get("parsed_size_details", {})
        size1_str_norm = str(item1.get("size_quantity_info_normalized", item1.get("size_quantity_info", "")) or "").lower().strip()
        size1_from_details = ""
        if size1_details:
            if "value" in size1_details and "unit" in size1_details:
                size1_from_details = f"{size1_details['value']} {size1_details['unit']}"
            elif "value_min" in size1_details and "value_max" in size1_details and "unit" in size1_details:
                size1_from_details = f"{size1_details['value_min']}-{size1_details['value_max']} {size1_details['unit']}"
            elif "value_base" in size1_details and "value_equivalent" in size1_details and "unit" in size1_details:
                size1_from_details = f"{size1_details['value_base']}={size1_details['value_equivalent']} {size1_details['unit']}"
        size1_final_for_item1 = (size1_from_details if size1_from_details else size1_str_norm).lower().strip()

        variant1 = str(item1.get("product_variant_description", "") or "").lower().strip()
        secondary_text1_parts = []
        if variant1 and variant1 != primary_text1 and variant1 != size1_final_for_item1: 
            secondary_text1_parts.append(variant1)
        if size1_final_for_item1 and size1_final_for_item1 not in primary_text1 and (not variant1 or size1_final_for_item1 not in variant1) : 
            secondary_text1_parts.append(size1_final_for_item1)
        secondary_text1 = " ".join(list(dict.fromkeys(filter(None, secondary_text1_parts)))).strip()

        logger.debug(f"COMPARE_FN - Item1 ID: {item1_id_log} | Primary1: '{primary_text1}' | Secondary1: '{secondary_text1}' | Size1Final: '{size1_final_for_item1}'")

        if not primary_text1 and not secondary_text1:
            logger.warning(f"COMPARE_FN - {item1_id_log} has no text for matching. Skipping.")
            # ... (append to report for unmatchable item1 - ensure all P1_ fields are populated)
            comparison_report.append({
                "Comparison_Type": "Unmatchable Product in File 1 (No Text)",
                "P1_Brand": item1.get("product_brand"), "P1_Name_Core": item1.get("product_name_core"),
                "P1_Variant": item1.get("product_variant_description"), "P1_Size_Orig": item1.get("size_quantity_info"),
                "P1_Size_Norm": item1.get("size_quantity_info_normalized", size1_final_for_item1),
                "P1_Parsed_Size_Details": json.dumps(item1.get("parsed_size_details")) if item1.get("parsed_size_details") else None,
                "P1_Offer_Price": item1.get("offer_price"), "P1_Regular_Price": item1.get("regular_price"),
                "P1_Unit_Indicator": item1.get("unit_indicator"), "P1_Store_Terms": item1.get("store_specific_terms"),
                "P1_Val_Flags": "; ".join(item1.get("validation_flags", [])),
                "P1_Vision_Reprocessed": item1.get("reprocessed_by_vision_llm", False),
                "P1_Box_ID": item1_id_log,
            })
            continue

        for idx2, item2 in enumerate(product_items2):
            if idx2 in matched_item2_indices: continue
            item2_id_log = item2.get("product_box_id", f"File2-Item{idx2}")

            # --- Prepare item2 texts ---
            brand2 = str(item2.get("product_brand", "") or "").lower().strip()
            name_core2 = str(item2.get("product_name_core", "") or "").lower().strip()
            primary_text2 = f"{brand2} {name_core2}".strip()

            size2_details = item2.get("parsed_size_details", {})
            size2_str_norm = str(item2.get("size_quantity_info_normalized", item2.get("size_quantity_info", "")) or "").lower().strip()
            size2_from_details = ""
            if size2_details:
                if "value" in size2_details and "unit" in size2_details:
                    size2_from_details = f"{size2_details['value']} {size2_details['unit']}"
                elif "value_min" in size2_details and "value_max" in size2_details and "unit" in size2_details:
                    size2_from_details = f"{size2_details['value_min']}-{size2_details['value_max']} {size2_details['unit']}"
                elif "value_base" in size2_details and "value_equivalent" in size2_details and "unit" in size2_details:
                    size2_from_details = f"{size2_details['value_base']}={size2_details['value_equivalent']} {size2_details['unit']}"
            size2_final_for_item2 = (size2_from_details if size2_from_details else size2_str_norm).lower().strip()
            
            variant2 = str(item2.get("product_variant_description", "") or "").lower().strip()
            secondary_text2_parts = []
            if variant2 and variant2 != primary_text2 and variant2 != size2_final_for_item2: 
                secondary_text2_parts.append(variant2)
            if size2_final_for_item2 and size2_final_for_item2 not in primary_text2 and (not variant2 or size2_final_for_item2 not in variant2): 
                secondary_text2_parts.append(size2_final_for_item2)
            secondary_text2 = " ".join(list(dict.fromkeys(filter(None, secondary_text2_parts)))).strip()
            
            if not primary_text2 and not secondary_text2:
                continue
            
            primary_similarity = fuzz.token_set_ratio(primary_text1, primary_text2) if primary_text1 and primary_text2 else 0
            secondary_similarity = fuzz.token_set_ratio(secondary_text1, secondary_text2) if secondary_text1 and secondary_text2 else 0
            
            logger.debug(f"COMPARE_FN_PAIR_SCORES - Item1: {item1_id_log} (P1: '{primary_text1}') vs Item2: {item2_id_log} (P2: '{primary_text2}') ==> PrimarySim: {primary_similarity}")
            logger.debug(f"COMPARE_FN_PAIR_SCORES - Item1: {item1_id_log} (S1: '{secondary_text1}') vs Item2: {item2_id_log} (S2: '{secondary_text2}') ==> SecondarySim: {secondary_similarity}")

            current_pair_similarity = 0.0
            if primary_text1 and primary_text2 and secondary_text1 and secondary_text2:
                current_pair_similarity = (primary_similarity * 0.7) + (secondary_similarity * 0.3)
            elif primary_text1 and primary_text2:
                current_pair_similarity = float(primary_similarity)
            elif secondary_text1 and secondary_text2:
                current_pair_similarity = float(secondary_similarity)
            elif primary_text1 or primary_text2 or secondary_text1 or secondary_text2: 
                current_pair_similarity = float(max(primary_similarity, secondary_similarity))
            
            logger.debug(f"COMPARE_FN_PAIR_FINAL - Item1 ID: {item1_id_log} vs Item2 ID: {item2_id_log} ==> WeightedPairSim: {current_pair_similarity:.1f}")

            if current_pair_similarity > highest_similarity:
                highest_similarity = current_pair_similarity
                best_match_item2 = item2
                best_match_idx = idx2
                logger.debug(f"COMPARE_FN_UPDATE - New best match for {item1_id_log} is {item2_id_log} with new highest_similarity: {highest_similarity:.1f}")
        
        if best_match_item2 and highest_similarity >= similarity_threshold:
            matched_item2_indices.add(best_match_idx)
            best_match_item2_id_log = best_match_item2.get("product_box_id", f"File2-Item{best_match_idx}")
            
            # --- Correctly get size_final for best_match_item2 (P2) for reporting ---
            _p2_size_details = best_match_item2.get("parsed_size_details", {})
            _p2_size_str_norm = str(best_match_item2.get("size_quantity_info_normalized", best_match_item2.get("size_quantity_info", "")) or "").lower().strip()
            _p2_size_from_details = ""
            if _p2_size_details:
                if "value" in _p2_size_details and "unit" in _p2_size_details: _p2_size_from_details = f"{_p2_size_details['value']} {_p2_size_details['unit']}"
                elif "value_min" in _p2_size_details and "value_max" in _p2_size_details and "unit" in _p2_size_details: _p2_size_from_details = f"{_p2_size_details['value_min']}-{_p2_size_details['value_max']} {_p2_size_details['unit']}"
                elif "value_base" in _p2_size_details and "value_equivalent" in _p2_size_details and "unit" in _p2_size_details: _p2_size_from_details = f"{_p2_size_details['value_base']}={_p2_size_details['value_equivalent']} {_p2_size_details['unit']}"
            _p2_size_final_for_report = (_p2_size_from_details if _p2_size_from_details else _p2_size_str_norm).lower().strip()
            # --- End P2 size final calculation for report ---

            logger.info(f"COMPARE_FN_REPORT_DATA - For matched pair: Item1 ID: {item1_id_log}, Item2 ID: {best_match_item2_id_log}")
            logger.info(f"COMPARE_FN_REPORT_DATA - Item1 ('{item1_id_log}') P1_Size_Orig: '{item1.get('size_quantity_info')}', P1_Size_Norm_get: '{item1.get('size_quantity_info_normalized')}', P1_size_final_for_item1_calc: '{size1_final_for_item1}'")
            logger.info(f"COMPARE_FN_REPORT_DATA - Item2 ('{best_match_item2_id_log}') P2_Size_Orig: '{best_match_item2.get('size_quantity_info')}', P2_Size_Norm_get: '{best_match_item2.get('size_quantity_info_normalized')}', P2_size_final_for_report_calc: '{_p2_size_final_for_report}'")

            logger.info(f"COMPARE_FN - Match FOUND for {item1_id_log} with {best_match_item2_id_log}. Final Highest Similarity: {highest_similarity:.1f}%")
            
            diff_details = []
            offer_price1 = item1.get("offer_price")
            offer_price2 = best_match_item2.get("offer_price")
            regular_price1 = item1.get("regular_price")
            regular_price2 = best_match_item2.get("regular_price")
            price_tolerance = 0.01 

            op1_f, op2_f, rp1_f, rp2_f = None, None, None, None
            try: op1_f = float(offer_price1) if offer_price1 is not None else None
            except: pass
            try: op2_f = float(offer_price2) if offer_price2 is not None else None
            except: pass
            try: rp1_f = float(regular_price1) if regular_price1 is not None else None
            except: pass
            try: rp2_f = float(regular_price2) if regular_price2 is not None else None
            except: pass

            if (op1_f is not None and op2_f is not None and abs(op1_f - op2_f) > price_tolerance) or \
               (op1_f is None and op2_f is not None) or (op1_f is not None and op2_f is None):
                diff_details.append(f"Offer Price: F1=${offer_price1 if offer_price1 is not None else 'N/A'} vs F2=${offer_price2 if offer_price2 is not None else 'N/A'}")

            if (rp1_f is not None and rp2_f is not None and abs(rp1_f - rp2_f) > price_tolerance) or \
               (rp1_f is None and rp2_f is not None) or (rp1_f is not None and rp2_f is None):
                diff_details.append(f"Regular Price: F1=${regular_price1 if regular_price1 is not None else 'N/A'} vs F2=${regular_price2 if regular_price2 is not None else 'N/A'}")

            # Use the correctly scoped size_final values for comparison
            if size1_final_for_item1 != _p2_size_final_for_report:
                diff_details.append(f"Size: F1='{size1_final_for_item1 or 'N/A'}' vs F2='{_p2_size_final_for_report or 'N/A'}'")

            base_report_item = {
                "P1_Brand": item1.get("product_brand"), "P1_Name_Core": item1.get("product_name_core"),
                "P1_Variant": item1.get("product_variant_description"), 
                "P1_Size_Orig": item1.get("size_quantity_info"),
                "P1_Size_Norm": item1.get("size_quantity_info_normalized", size1_final_for_item1), 
                "P1_Parsed_Size_Details": json.dumps(item1.get("parsed_size_details")) if item1.get("parsed_size_details") else None, 
                "P1_Offer_Price": offer_price1, "P1_Regular_Price": regular_price1,
                "P1_Unit_Indicator": item1.get("unit_indicator"), "P1_Store_Terms": item1.get("store_specific_terms"),
                "P1_Val_Flags": "; ".join(item1.get("validation_flags", [])), 
                "P1_Vision_Reprocessed": item1.get("reprocessed_by_vision_llm", False),
                "P1_Box_ID": item1_id_log,
                "P2_Brand": best_match_item2.get("product_brand"), "P2_Name_Core": best_match_item2.get("product_name_core"),
                "P2_Variant": best_match_item2.get("product_variant_description"), 
                "P2_Size_Orig": best_match_item2.get("size_quantity_info"),
                "P2_Size_Norm": best_match_item2.get("size_quantity_info_normalized", _p2_size_final_for_report), # Use corrected fallback
                "P2_Parsed_Size_Details": json.dumps(best_match_item2.get("parsed_size_details")) if best_match_item2.get("parsed_size_details") else None,
                "P2_Offer_Price": offer_price2, "P2_Regular_Price": regular_price2,
                "P2_Unit_Indicator": best_match_item2.get("unit_indicator"), "P2_Store_Terms": best_match_item2.get("store_specific_terms"),
                "P2_Val_Flags": "; ".join(best_match_item2.get("validation_flags", [])), 
                "P2_Vision_Reprocessed": best_match_item2.get("reprocessed_by_vision_llm", False),
                "P2_Box_ID": best_match_item2_id_log,
                "Similarity_Percent": round(highest_similarity, 1),
            }
            
            if diff_details:
                comparison_report.append({"Comparison_Type": "Product Match - Attribute Mismatch", **base_report_item, "Differences": "; ".join(diff_details)})
            else:
                comparison_report.append({"Comparison_Type": "Product Match - Attributes OK", **base_report_item, "Differences": ""})
        else: # No match found for item1
            logger.info(f"COMPARE_FN - No match found for {item1_id_log} (Highest sim on this item1 after all item2 checks: {highest_similarity:.1f}%)")
            comparison_report.append({
                "Comparison_Type": "Unmatched Product in File 1",
                "P1_Brand": item1.get("product_brand"), "P1_Name_Core": item1.get("product_name_core"),
                "P1_Variant": item1.get("product_variant_description"), "P1_Size_Orig": item1.get("size_quantity_info"),
                "P1_Size_Norm": item1.get("size_quantity_info_normalized", size1_final_for_item1), # Use correctly scoped size1_final
                "P1_Parsed_Size_Details": json.dumps(item1.get("parsed_size_details")) if item1.get("parsed_size_details") else None,
                "P1_Offer_Price": item1.get("offer_price"), "P1_Regular_Price": item1.get("regular_price"),
                "P1_Unit_Indicator": item1.get("unit_indicator"), "P1_Store_Terms": item1.get("store_specific_terms"),
                "P1_Val_Flags": "; ".join(item1.get("validation_flags", [])),
                "P1_Vision_Reprocessed": item1.get("reprocessed_by_vision_llm", False),
                "P1_Box_ID": item1_id_log,
            })
    
    # Loop for items in product_items2 not matched with any item from product_items1
    for idx2, item2 in enumerate(product_items2):
        if idx2 not in matched_item2_indices:
            item2_id_log = item2.get("product_box_id", f"File2-Item{idx2}")
            
            # Correctly calculate size2_final for this unmatched item2
            _unmatched_p2_size_details = item2.get("parsed_size_details", {})
            _unmatched_p2_size_str_norm = str(item2.get("size_quantity_info_normalized", item2.get("size_quantity_info", "")) or "").lower().strip()
            _unmatched_p2_size_from_details = ""
            if _unmatched_p2_size_details:
                if "value" in _unmatched_p2_size_details and "unit" in _unmatched_p2_size_details: _unmatched_p2_size_from_details = f"{_unmatched_p2_size_details['value']} {_unmatched_p2_size_details['unit']}"
                elif "value_min" in _unmatched_p2_size_details and "value_max" in _unmatched_p2_size_details and "unit" in _unmatched_p2_size_details: _unmatched_p2_size_from_details = f"{_unmatched_p2_size_details['value_min']}-{_unmatched_p2_size_details['value_max']} {_unmatched_p2_size_details['unit']}"
                elif "value_base" in _unmatched_p2_size_details and "value_equivalent" in _unmatched_p2_size_details and "unit" in _unmatched_p2_size_details: _unmatched_p2_size_from_details = f"{_unmatched_p2_size_details['value_base']}={_unmatched_p2_size_details['value_equivalent']} {_unmatched_p2_size_details['unit']}"
            _unmatched_p2_size_final_for_report = (_unmatched_p2_size_from_details if _unmatched_p2_size_from_details else _unmatched_p2_size_str_norm).lower().strip()

            logger.info(f"COMPARE_FN - Unmatched product in File 2 (Extra): {item2_id_log}")
            comparison_report.append({
                "Comparison_Type": "Unmatched Product in File 2 (Extra)",
                "P2_Brand": item2.get("product_brand"), "P2_Name_Core": item2.get("product_name_core"),
                "P2_Variant": item2.get("product_variant_description"), 
                "P2_Size_Orig": item2.get("size_quantity_info"),
                "P2_Size_Norm": item2.get("size_quantity_info_normalized", _unmatched_p2_size_final_for_report), # Use correctly scoped fallback
                "P2_Parsed_Size_Details": json.dumps(item2.get("parsed_size_details")) if item2.get("parsed_size_details") else None,
                "P2_Offer_Price": item2.get("offer_price"), "P2_Regular_Price": item2.get("regular_price"),
                "P2_Unit_Indicator": item2.get("unit_indicator"), "P2_Store_Terms": item2.get("store_specific_terms"),
                "P2_Val_Flags": "; ".join(item2.get("validation_flags", [])),
                "P2_Vision_Reprocessed": item2.get("reprocessed_by_vision_llm", False),
                "P2_Box_ID": item2_id_log,
            })
            
    logger.info(f"COMPARE_FN - Comparison finished. Report items: {len(comparison_report)}")
    return comparison_report

# backend_processor.py

# ... (existing functions) ...

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


# ... (rest of your backend_processor.py code) ...

# --- Main Processing Function (callable by Streamlit) ---
# backend_processor.py

# ... (existing functions) ...

# backend_processor.py

# ... (existing functions) ...

def process_files_for_comparison(file1_bytes, file1_name, file2_bytes, file2_name):
    request_id = f"req_{int(time.time())}"
    logger.info(f"REQUEST_ID: {request_id} - Backend processing started for '{file1_name}' and '{file2_name}'")

    all_files_data_for_reprocessing_and_pils = []
    temp_pdf_paths_to_cleanup = []
    final_product_items_file1 = []
    final_product_items_file2 = []

    try:
        # Phase 1: Initial extraction for both files
        for file_idx_num_zero_based, (file_bytes_content, original_filename) in enumerate(
            [(file1_bytes, file1_name), (file2_bytes, file2_name)]
        ):
            file_id_log_prefix = f"{request_id}-File{file_idx_num_zero_based+1}"
            s3_safe_filename_part = secure_filename(original_filename)
            logger.info(f"{file_id_log_prefix} - Processing file: {original_filename} (S3 safe: {s3_safe_filename_part})")

            current_file_initial_items = []
            page_pil_images_for_file = []
            s3_upload_timestamp = time.time()

            # ... (PDF/image loading remains the same) ...
            if original_filename.lower().endswith(".pdf"):
                fd, temp_pdf_path = tempfile.mkstemp(suffix=".pdf", prefix=f"{request_id}_")
                os.close(fd)
                temp_pdf_paths_to_cleanup.append(temp_pdf_path)
                with open(temp_pdf_path, "wb") as f_pdf:
                    f_pdf.write(file_bytes_content)
                try:
                    logger.info(f"{file_id_log_prefix} - Converting PDF to images (DPI 200) from {temp_pdf_path}...")
                    logger.info(f"{file_id_log_prefix} - Using POPPLER_BIN_PATH: {POPPLER_BIN_PATH}")
                    page_pil_images_for_file = convert_from_path(temp_pdf_path, dpi=200, poppler_path=POPPLER_BIN_PATH, fmt='jpeg', timeout=300)
                    logger.info(f"{file_id_log_prefix} - PDF converted to {len(page_pil_images_for_file)} images.")
                except Exception as e_pdf:
                    logger.error(f"{file_id_log_prefix} - PDF conversion error for {original_filename}: {e_pdf}", exc_info=True)
                    raise
            elif original_filename.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    page_pil_images_for_file = [Image.open(BytesIO(file_bytes_content))]
                    logger.info(f"{file_id_log_prefix} - Image file loaded.")
                except Exception as e_img:
                    logger.error(f"{file_id_log_prefix} - Invalid image file {original_filename}: {str(e_img)}", exc_info=True)
                    raise
            else:
                err_msg = f"Unsupported file type for {original_filename}"
                logger.error(f"{file_id_log_prefix} - {err_msg}")
                raise ValueError(err_msg)


            s3_keys_for_this_file = []
            for page_idx, page_image_pil in enumerate(page_pil_images_for_file):
                page_id_log = f"{file_id_log_prefix}-Page{page_idx}"
                logger.info(f"{page_id_log} - Starting initial processing pass for page {page_idx} of {original_filename}...")
                image_width_px, image_height_px = page_image_pil.size
                logger.debug(f"{page_id_log} - Image dimensions: {image_width_px}x{image_height_px}")

                if not roboflow_model_object:
                    logger.warning(f"{page_id_log} - Roboflow model not available. Product box detection will be skipped.")
                    roboflow_preds = []
                else:
                    roboflow_preds = get_roboflow_predictions_sdk(page_image_pil, f"{s3_safe_filename_part}_p{page_idx}")
                
                if not roboflow_preds:
                    logger.warning(f"{page_id_log} - No Roboflow predictions or Roboflow failed. Skipping text collation for this page.")
                
                if not s3_client or not textract_client or not S3_BUCKET_NAME:
                    logger.error(f"{page_id_log} - S3/Textract clients or bucket not configured. Cannot process with Textract.")
                    raise ConnectionError("S3/Textract not configured for backend processing.")

                img_byte_arr_s3 = BytesIO()
                page_image_pil.save(img_byte_arr_s3, format='JPEG', quality=90)
                img_byte_arr_s3.seek(0)
                s3_page_key = f"pages/{request_id}_{s3_upload_timestamp}_{file_idx_num_zero_based}_p{page_idx}_{s3_safe_filename_part}.jpg"
                
                uploaded_s3_key = upload_to_s3(img_byte_arr_s3, S3_BUCKET_NAME, s3_page_key)
                if not uploaded_s3_key:
                    logger.error(f"{page_id_log} - Failed to upload page image to S3. Skipping page.")
                    continue
                s3_keys_for_this_file.append(uploaded_s3_key)

                logger.info(f"{page_id_log} - Starting Textract analysis for S3 key: {uploaded_s3_key}")
                textract_blocks = get_analysis_from_document_via_textract(S3_BUCKET_NAME, uploaded_s3_key)
                if not textract_blocks:
                    logger.error(f"{page_id_log} - Textract analysis failed or returned no blocks. Skipping page.")
                    delete_from_s3(S3_BUCKET_NAME, uploaded_s3_key)
                    continue
                logger.info(f"{page_id_log} - Textract returned {len(textract_blocks)} blocks.")
                blocks_map = {b['Id']: b for b in textract_blocks}

                collated_snippets = []
                if roboflow_preds:
                    logger.info(f"{page_id_log} - Collating text for {len(roboflow_preds)} Roboflow boxes...")
                    collated_snippets = collate_text_for_product_boxes(roboflow_preds, textract_blocks, blocks_map, image_width_px, image_height_px, page_id_for_log=page_id_log)
                else:
                    logger.warning(f"{page_id_log} - No Roboflow predictions, so no snippets to collate text for.")

                for snippet_idx, snippet in enumerate(collated_snippets):
                    item_id_for_log = snippet.get("product_box_id", f"{page_id_log}-Snip{snippet_idx}")
                    logger.info(f"ITEM_ID: {item_id_for_log} - Processing snippet with Text LLM...")
                    
                    # Pass textract_blocks_in_segment to the text LLM function
                    llm_output = extract_product_data_with_llm(
                        snippet["collated_text"], 
                        item_id_for_log=item_id_for_log, 
                        textract_blocks_in_segment=snippet.get("textract_blocks_in_segment") # NEW: Pass this
                    )
                    
                    price_candidates_for_snippet = snippet.get("price_candidates", [])
                    
                    logger.info(f"ITEM_ID: {item_id_for_log} - Post-processing and validating Text LLM output...")
                    processed_item = post_process_and_validate_item_data(llm_output, price_candidates_for_snippet, snippet["collated_text"], item_id_for_log=item_id_for_log)
                    
                    # Also carry over the Textract bbox fields from llm_output to processed_item
                    for key in llm_output.keys():
                        if key.endswith('_bbox') and key not in processed_item:
                            processed_item[key] = llm_output[key]

                    current_file_initial_items.append({
                        "product_box_id": item_id_for_log,
                        "original_filename": original_filename,
                        "page_idx_for_reprocessing": page_idx,
                        "roboflow_box_coords_pixels_center_wh": snippet.get("roboflow_box_coords_pixels_center_wh"),
                        "initial_price_candidates": price_candidates_for_snippet,
                        "original_collated_text": snippet.get("collated_text"),
                        "roboflow_class_name": snippet.get("class_name", "UnknownClass"),
                        "textract_blocks_in_segment": snippet.get("textract_blocks_in_segment"), # Also pass this for vision LLM if needed later
                        **processed_item # Unpack processed_item (includes all LLM data + validation flags + new _bbox fields)
                    })
                    logger.info(f"ITEM_ID: {item_id_for_log} - Initial processing complete. Offer: {processed_item.get('offer_price')}, Regular: {processed_item.get('regular_price')}")
            # End of page loop

            for s3_key_to_delete in s3_keys_for_this_file:
                delete_from_s3(S3_BUCKET_NAME, s3_key_to_delete)
            
            all_files_data_for_reprocessing_and_pils.append({
                "file_id_log_prefix": file_id_log_prefix,
                "filename": original_filename,
                "page_pils_list": page_pil_images_for_file,
                "items": current_file_initial_items
            })
        # End of file loop (file1, file2)

        logger.info(f"REQUEST_ID: {request_id} - Initial processing pass complete for all files.")
        logger.info(f"REQUEST_ID: {request_id} - Starting Vision LLM re-processing stage for flagged items...")

        # Phase 2: Vision LLM re-processing for flagged items
        for file_idx, file_data_obj in enumerate(all_files_data_for_reprocessing_and_pils):
            current_file_final_items_for_vision_pass = []
            file_log_prefix_vision = file_data_obj["file_id_log_prefix"]
            logger.info(f"{file_log_prefix_vision} - Vision re-processing items from: {file_data_obj['filename']}")
            
            for item_idx, item_to_evaluate_original in enumerate(file_data_obj["items"]):
                item_being_processed = item_to_evaluate_original.copy()
                item_id_for_log_vision = item_being_processed.get("product_box_id", f"{file_log_prefix_vision}-Item{item_idx}-VisionEval")

                item_being_processed.setdefault('validation_flags', [])
                item_being_processed.setdefault('reprocessed_by_vision_llm', False)

                # --- Decision logic for sending to Vision LLM (remains the same) ---
                send_to_vision = False
                vision_reason = "No specific trigger"

                flags = item_being_processed.get("validation_flags", [])
                offer_price = item_being_processed.get("offer_price")
                original_text = item_being_processed.get("original_collated_text", "")
                roboflow_class = item_being_processed.get("roboflow_class_name", "UnknownClass")
                price_candidates_for_current_item = item_being_processed.get("initial_price_candidates", [])
                
                current_item_best_offer_candidate = None
                if price_candidates_for_current_item:
                    valid_price_cands_vision = [pc for pc in price_candidates_for_current_item if pc.get('parsed_value') is not None and pc.get('parsed_value') >= 0]
                    offer_cands_vision = sorted([pc for pc in valid_price_cands_vision if not pc.get('is_regular_candidate', False)], key=lambda c: (c.get('source_block_id') == 'GEOMETRIC_MERGE', -c.get('pixel_height', 0)))
                    if offer_cands_vision: current_item_best_offer_candidate = offer_cands_vision[0]

                critical_price_error_patterns = [
                    "XYZ->0.XY error", "price seems too low, corrected to prominent candidate",
                    "Correcting LLM offer price", "Correcting LLM regular price",
                    "Multi-buy pattern.*failed", "differs from prominent visual candidate",
                    "Offer price .* populated from visually prominent candidate", 
                    "Regular price .* populated from visual candidate"
                ]
                critical_price_error_flag_found = any(any(re.search(pattern, flag, re.IGNORECASE) for pattern in critical_price_error_patterns) for flag in flags)
                
                price_suspiciously_low_text_pipeline = False
                if offer_price is not None:
                    try: price_suspiciously_low_text_pipeline = (0.01 <= float(offer_price) < 1.00)
                    except (ValueError, TypeError): pass
                
                potential_X_YZ_in_text = re.search(r'\b([1-9])\s*(\d{2})\b', original_text) or \
                                         re.search(r'\b([1-9]\d{2})\b', original_text) 

                if item_being_processed.get("error_message"):
                    send_to_vision = True; vision_reason = f"Text LLM error: {item_being_processed.get('error_message')}"
                elif critical_price_error_flag_found:
                    send_to_vision = True; vision_reason = f"Critical price error flags found: {flags}"
                elif price_suspiciously_low_text_pipeline and potential_X_YZ_in_text and not any(kw in original_text.lower() for kw in ["limit", "max"]):
                    send_to_vision = True; vision_reason = f"Suspiciously low price ${offer_price} from text pipeline, but original text has pattern '{potential_X_YZ_in_text.group(0) if potential_X_YZ_in_text else 'N/A'}'"
                elif offer_price is None and roboflow_class and roboflow_class.lower() == "product_item": 
                    send_to_vision = True; vision_reason = f"Null offer price from text pipeline for a Roboflow '{roboflow_class}' item."
                elif item_being_processed.get("product_name_core") is None and offer_price is not None: 
                    send_to_vision = True; vision_reason = "Product has price but no name from Text LLM."
                elif offer_price is not None and current_item_best_offer_candidate and \
                             current_item_best_offer_candidate.get('parsed_value') is not None and \
                             abs(float(offer_price) - float(current_item_best_offer_candidate.get('parsed_value', 0))) > 1.50 : 
                    send_to_vision = True; vision_reason = f"Large price discrepancy (>${1.50}) between Text LLM price ${offer_price} and visual candidate ${current_item_best_offer_candidate.get('parsed_value')}"
                
                if not send_to_vision and offer_price is not None: 
                    try:
                        offer_price_float = float(offer_price)
                        if 0.01 <= offer_price_float < 1.00: 
                            cents_part_str = str(int(round((offer_price_float * 100) % 100))).zfill(2)
                            if re.search(r'\b' + re.escape(cents_part_str), original_text): 
                                has_stronger_visual_candidate_for_missing_dollar = False
                                if price_candidates_for_current_item:
                                    for pc_cand_vision_trigger in price_candidates_for_current_item:
                                        pc_text_vt = pc_cand_vision_trigger.get('text_content','').strip()
                                        if re.fullmatch(r'[1-9]\s*\d{2}', pc_text_vt) or re.fullmatch(r'[1-9]\d{2}', pc_text_vt):
                                            parsed_strong_cand_vt = parse_price_string(pc_text_vt, item_id_for_log_vision + "-strongcandVT")
                                            if parsed_strong_cand_vt and parsed_strong_cand_vt >= 1.00:
                                                has_stronger_visual_candidate_for_missing_dollar = True
                                                logger.debug(f"ITEM_ID: {item_id_for_log_vision} - AGGRESSIVE Vision Trigger Check: Found stronger visual candidate '{pc_text_vt}' ({parsed_strong_cand_vt}) for low price {offer_price}")
                                                break
                                
                                if has_stronger_visual_candidate_for_missing_dollar or not any(kw in original_text.lower() for kw in ["limit", "coupon", "max", "por solo", "a solo", "menos de", "oferta", "esp."]):
                                    send_to_vision = True
                                    vision_reason = f"AGGRESSIVE: Suspiciously low offer price ${offer_price} (cents '{cents_part_str}' found in text). Potential missing dollar. Stronger visual candidate: {has_stronger_visual_candidate_for_missing_dollar}"
                                    logger.info(f"ITEM_ID: {item_id_for_log_vision} - AGGRESSIVE VISION TRIGGER activated: {vision_reason}")
                    except (ValueError, TypeError) as e_agg_vision:
                        logger.warning(f"ITEM_ID: {item_id_for_log_vision} - Error during aggressive vision trigger check for offer_price {offer_price}: {e_agg_vision}")
                # --- End decision logic for sending to Vision LLM ---


                if send_to_vision and item_being_processed.get("roboflow_box_coords_pixels_center_wh"):
                    logger.info(f"ITEM_ID: {item_id_for_log_vision} - Sending to Vision LLM. Reason: {vision_reason}. Initial Flags: {item_being_processed.get('validation_flags')}")
                    page_idx_reproc = item_being_processed["page_idx_for_reprocessing"]
                    
                    if page_idx_reproc < len(file_data_obj["page_pils_list"]):
                        page_image_pil_reproc = file_data_obj["page_pils_list"][page_idx_reproc]
                        segment_bytes = get_segment_image_bytes(page_image_pil_reproc, item_being_processed["roboflow_box_coords_pixels_center_wh"], item_id_for_log=item_id_for_log_vision)

                        if segment_bytes:
                            name_hint_vision = item_being_processed.get("product_name_core") or item_being_processed.get("product_brand")
                            # NO LONGER ASKING FOR BBOXES FROM VISION LLM
                            vision_llm_output = re_extract_with_vision_llm(segment_bytes, item_id_for_log=item_id_for_log_vision, original_item_name=name_hint_vision)
                            
                            item_being_processed['reprocessed_by_vision_llm'] = True

                            if "error_message" not in vision_llm_output:
                                parsed_op_v = parse_price_string(vision_llm_output.get("offer_price"), item_id_for_log=f"{item_id_for_log_vision}-vision_offer")
                                parsed_rp_v = parse_price_string(vision_llm_output.get("regular_price"), item_id_for_log=f"{item_id_for_log_vision}-vision_regular")
                                
                                vision_llm_output["offer_price"] = parsed_op_v
                                vision_llm_output["regular_price"] = parsed_rp_v
                                
                                vision_item_processed = post_process_and_validate_item_data(vision_llm_output, [], "", item_id_for_log=f"{item_id_for_log_vision}-vision_processed")
                                
                                # Update item_being_processed with fields from vision_item_processed
                                fields_to_update = ["offer_price", "regular_price", "product_brand", "product_name_core", 
                                                    "product_variant_description", "size_quantity_info", "unit_indicator", 
                                                    "store_specific_terms", "parsed_size_details", "size_quantity_info_normalized"]
                                
                                for key_update in fields_to_update:
                                    if key_update in vision_item_processed and vision_item_processed[key_update] is not None: 
                                        item_being_processed[key_update] = vision_item_processed[key_update]
                                # IMPORTANT: NO _bbox update here from vision_llm_output
                                # Bboxes should remain from Textract lookup in text LLM
                                        
                                item_being_processed['validation_flags'].extend(vision_item_processed.get('validation_flags',[]))
                                item_being_processed['validation_flags'] = list(set(item_being_processed['validation_flags']))
                                item_being_processed['validation_flags'].append(f"Successfully reprocessed by Vision LLM (Reason: {vision_reason})")
                                
                            else:
                                logger.warning(f"ITEM_ID: {item_id_for_log_vision} - Vision LLM re-extraction error: {vision_llm_output.get('error_message')}. Keeping pre-vision data.")
                                item_being_processed["validation_flags"].append(f"Vision LLM failed: {vision_llm_output.get('error_message')}")
                        else:
                            logger.warning(f"ITEM_ID: {item_id_for_log_vision} - Could not get segment image for Vision LLM. Keeping pre-vision data.")
                            item_being_processed["validation_flags"].append("Vision LLM skipped: Could not create segment image.")
                            item_being_processed['reprocessed_by_vision_llm'] = True
                    else:
                        logger.error(f"ITEM_ID: {item_id_for_log_vision} - page_idx_for_reprocessing {page_idx_reproc} is out of bounds for page_pils_list (len {len(file_data_obj['page_pils_list'])}).")
                        item_being_processed["validation_flags"].append("Vision LLM skipped: Page index error.")
                        item_being_processed['reprocessed_by_vision_llm'] = True
                else:
                    if not item_being_processed.get("roboflow_box_coords_pixels_center_wh") and send_to_vision :
                         item_being_processed["validation_flags"].append("Vision LLM trigger met but skipped: Missing Roboflow box coordinates.")
                    
                current_file_final_items_for_vision_pass.append(item_being_processed) 
            
            if file_idx == 0:
                final_product_items_file1 = current_file_final_items_for_vision_pass
            else:
                final_product_items_file2 = current_file_final_items_for_vision_pass
                logger.info(f"REQUEST_ID: {request_id} - Assigned {len(final_product_items_file2)} items to final_product_items_file2")
        # End of Vision LLM reprocessing loop

        logger.info(f"REQUEST_ID: {request_id} - Vision LLM re-processing stage complete.")
        logger.info(f"REQUEST_ID: {request_id} - Final items for File 1 ({len(final_product_items_file1)}):")
        # ... (logging remains the same) ...

        logger.info(f"REQUEST_ID: {request_id} - Starting final product comparison.")
        product_centric_comparison_report = compare_product_items(final_product_items_file1, final_product_items_file2)

        # --- Phase 3: Generate Full Highlighted Pages ---
        highlighted_full_pages_base64_file1 = []
        highlighted_full_pages_base64_file2 = []

        # Get page data from all_files_data_for_reprocessing_and_pils
        file1_page_data = next(f for f in all_files_data_for_reprocessing_and_pils if f["file_id_log_prefix"].endswith("-File1"))
        file2_page_data = next(f for f in all_files_data_for_reprocessing_and_pils if f["file_id_log_prefix"].endswith("-File2"))

        num_pages_file1 = len(file1_page_data["page_pils_list"])
        num_pages_file2 = len(file2_page_data["page_pils_list"])
        
        # Iterate through pages (assuming same number of pages or handle misalignment)
        max_pages = max(num_pages_file1, num_pages_file2)

        for page_idx in range(max_pages):
            page_pil_file1 = file1_page_data["page_pils_list"][page_idx] if page_idx < num_pages_file1 else None
            page_pil_file2 = file2_page_data["page_pils_list"][page_idx] if page_idx < num_pages_file2 else None

            # Filter items for current page
            items_on_current_page_file1 = [item for item in final_product_items_file1 if item["page_idx_for_reprocessing"] == page_idx]
            items_on_current_page_file2 = [item for item in final_product_items_file2 if item["page_idx_for_reprocessing"] == page_idx]
            
            # Filter comparison report for items on current page
            # This part requires careful filtering of `product_centric_comparison_report`
            # to identify which items on *this specific page* had differences or were unmatched.
            # You'll likely need to pass the `product_box_id` from items_on_current_page_file1/2
            # and match them against the 'P1_Box_ID'/'P2_Box_ID' in the report.

            # Example logic for getting differences for a specific item on a page:
            page_comparison_report_items = []
            for report_item in product_centric_comparison_report:
                p1_box_id_in_report = report_item.get("P1_Box_ID")
                p2_box_id_in_report = report_item.get("P2_Box_ID")
                
                # Check if this report item involves an item from the current page
                if any(item['product_box_id'] == p1_box_id_in_report for item in items_on_current_page_file1) or \
                any(item['product_box_id'] == p2_box_id_in_report for item in items_on_current_page_file2):
                    page_comparison_report_items.append(report_item)


            if page_pil_file1:
                # Create a copy to draw on, so original PIL list is not modified
                page_pil_file1_copy = page_pil_file1.copy() 
                # Call the drawing function (updated to draw on full page)
                # This function needs to iterate through `items_on_current_page_file1` and `page_comparison_report_items`
                # to decide what to highlight and with what color.
                highlighted_img_bytes_file1 = draw_highlights_on_full_page_v2(
                    page_pil_file1_copy,
                    items_on_current_page_file1,
                    page_comparison_report_items,
                    "file1"
                )
                highlighted_full_pages_base64_file1.append(base64.b64encode(highlighted_img_bytes_file1.getvalue()).decode('utf-8'))

            if page_pil_file2:
                page_pil_file2_copy = page_pil_file2.copy()
                highlighted_img_bytes_file2 = draw_highlights_on_full_page_v2(
                    page_pil_file2_copy,
                    items_on_current_page_file2,
                    page_comparison_report_items,
                    "file2"
                )
                highlighted_full_pages_base64_file2.append(base64.b64encode(highlighted_img_bytes_file2.getvalue()).decode('utf-8'))

        final_response_dict = {
            # ... (other existing fields) ...
            "highlighted_pages_file1": highlighted_full_pages_base64_file1,
            "highlighted_pages_file2": highlighted_full_pages_base64_file2,
        }
        return final_response_dict
        
        # --- NEW: Generate highlighted images for comparison report entries ---
        for report_item in product_centric_comparison_report:
            comparison_type = report_item.get("Comparison_Type")
            p1_box_id = report_item.get("P1_Box_ID")
            p2_box_id = report_item.get("P2_Box_ID")
            
            p1_item = next((item for item in final_product_items_file1 if item.get("product_box_id") == p1_box_id), None)
            p2_item = next((item for item in final_product_items_file2 if item.get("product_box_id") == p2_box_id), None)

            # Get differences list safely
            differences_list = report_item.get("Differences", "").split('; ')
            differences_list = [d.strip() for d in differences_list if d.strip()] # Filter out empty strings
            
            should_draw_highlights = bool(differences_list) or \
                                     (comparison_type == "Unmatched Product in File 1") or \
                                     (comparison_type == "Unmatched Product in File 2 (Extra)")

            if should_draw_highlights:
                # File 1 highlighting
                if p1_item and p1_item.get("roboflow_box_coords_pixels_center_wh"):
                    page_idx_p1 = p1_item["page_idx_for_reprocessing"]
                    file1_data_obj = next((f for f in all_files_data_for_reprocessing_and_pils if f["file_id_log_prefix"] == f"{request_id}-File1"), None)
                    if file1_data_obj and page_idx_p1 < len(file1_data_obj["page_pils_list"]):
                        original_page_pil_p1 = file1_data_obj["page_pils_list"][page_idx_p1]
                        segment_image_bytes_p1 = get_segment_image_bytes(original_page_pil_p1, p1_item["roboflow_box_coords_pixels_center_wh"], item_id_for_log=p1_box_id)
                        
                        if segment_image_bytes_p1:
                            segment_pil_p1 = Image.open(segment_image_bytes_p1)
                            highlighted_segment_bytes_p1 = draw_highlights_on_image(
                                segment_pil_p1, 
                                p1_item, # p1_item now contains Textract bboxes
                                differences_list,
                                p1_item.get("product_name_core", p1_item.get("product_box_id", "Unknown Product")), 
                                "file1"
                            )
                            report_item["P1_Highlighted_Image_Base64"] = base64.b64encode(highlighted_segment_bytes_p1.getvalue()).decode('utf-8')
                            logger.debug(f"Generated highlighted image for P1: {p1_box_id}")
                        else:
                            logger.warning(f"Could not get segment image bytes for P1: {p1_box_id} for highlighting.")

                # File 2 highlighting
                if p2_item and p2_item.get("roboflow_box_coords_pixels_center_wh"):
                    page_idx_p2 = p2_item["page_idx_for_reprocessing"]
                    file2_data_obj = next((f for f in all_files_data_for_reprocessing_and_pils if f["file_id_log_prefix"] == f"{request_id}-File2"), None)
                    if file2_data_obj and page_idx_p2 < len(file2_data_obj["page_pils_list"]):
                        original_page_pil_p2 = file2_data_obj["page_pils_list"][page_idx_p2]
                        segment_image_bytes_p2 = get_segment_image_bytes(original_page_pil_p2, p2_item["roboflow_box_coords_pixels_center_wh"], item_id_for_log=p2_box_id)

                        if segment_image_bytes_p2:
                            segment_pil_p2 = Image.open(segment_image_bytes_p2)
                            highlighted_segment_bytes_p2 = draw_highlights_on_image(
                                segment_pil_p2, 
                                p2_item, # p2_item now contains Textract bboxes
                                differences_list,
                                p2_item.get("product_name_core", p2_item.get("product_box_id", "Unknown Product")), 
                                "file2"
                            )
                            report_item["P2_Highlighted_Image_Base64"] = base64.b64encode(highlighted_segment_bytes_p2.getvalue()).decode('utf-8')
                            logger.debug(f"Generated highlighted image for P2: {p2_box_id}")
                        else:
                            logger.warning(f"Could not get segment image bytes for P2: {p2_box_id} for highlighting.")
            else:
                logger.debug(f"Skipping image highlighting for {report_item.get('Comparison_Type')}: {report_item.get('P1_Name_Core', 'N/A')}/{report_item.get('P2_Name_Core', 'N/A')}. No differences or unmatchable type that needs highlight.")
        # --- END NEW: Generate highlighted images ---

        # ... (CSV generation and final return remain the same) ...
        if product_centric_comparison_report:
            report_df = pd.DataFrame(product_centric_comparison_report)
        else:
            report_df = pd.DataFrame([{"Comparison_Type": "No items to compare or no matches found."}])

        csv_buffer = io.StringIO()
        report_df.to_csv(csv_buffer, index=False, lineterminator='\r\n')
        csv_data_string = csv_buffer.getvalue()

        final_response_dict = {
            "message": "Backend processing complete. Comparison performed.",
            "product_items_file1_count": len(final_product_items_file1),
            "product_items_file2_count": len(final_product_items_file2),
            "product_comparison_details": product_centric_comparison_report,
            "report_csv_data": csv_data_string,
            "all_product_details_file1": final_product_items_file1,
            "all_product_details_file2": final_product_items_file2
        }
        logger.info(f"REQUEST_ID: {request_id} - Backend processing finished successfully. Returning structured response.")
        return final_response_dict

    except Exception as e_global:
        logger.error(f"REQUEST_ID: {request_id} - Global error in backend processing: {str(e_global)}", exc_info=True)
        raise
    finally:
        for temp_path in temp_pdf_paths_to_cleanup:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e_del_final:
                    logger.error(f"REQUEST_ID: {request_id} - Error deleting temp PDF {temp_path} in final cleanup: {e_del_final}", exc_info=True)
        logger.info(f"REQUEST_ID: {request_id} - process_files_for_comparison endpoint finished.")
        

# If you want to test this module directly (optional)
if __name__ == '__main__':
    logger.info("backend_processor.py is being run directly (e.g., for testing).")
    # Add test code here if needed, e.g., load sample PDF bytes and call process_files_for_comparison
    # Example:
    # try:
    #     with open("path/to/your/sample1.pdf", "rb") as f1, open("path/to/your/sample2.pdf", "rb") as f2:
    #         file1_bytes_test = f1.read()
    #         file2_bytes_test = f2.read()
    #         results = process_files_for_comparison(file1_bytes_test, "sample1.pdf", file2_bytes_test, "sample2.pdf")
    #         logger.info("Test run completed. Results snippet:")
    #         logger.info(f"Message: {results.get('message')}")
    #         logger.info(f"CSV Data (first 100 chars): {results.get('report_csv_data', '')[:100]}")
    # except FileNotFoundError:
    #     logger.error("Test PDF files not found. Skipping direct run test.")
    # except Exception as e:
    #     logger.error(f"Error during direct run test: {e}", exc_info=True)
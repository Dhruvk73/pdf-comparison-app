# main.py - Phase 3 Enhanced Version with Detailed WORD Block Logging
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
from dotenv import load_dotenv
import pandas as pd
import io
from io import BytesIO
from pdf2image import convert_from_path, pdfinfo_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
from PIL import Image
import base64 # For Vision LLM image encoding

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

if 'app' not in locals():
    app = Flask(__name__)

# Ensure logger is configured for DEBUG level
if not app.logger.handlers or (hasattr(app.logger, 'hasHandlers') and not app.logger.hasHandlers()): 
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - PID:%(process)d - [%(funcName)s:%(lineno)d] - %(message)s')
    if hasattr(app, 'logger'):
        app.logger.setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers: 
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - PID:%(process)d - [%(funcName)s:%(lineno)d] - %(message)s'))
    else: 
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - PID:%(process)d - [%(funcName)s:%(lineno)d] - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
app.logger.info("Flask Logger is configured for DEBUG level.")


# --- Geometric Merging Tolerances (Tune these based on logs) ---
Y_ALIGN_TOLERANCE_FACTOR = 0.7 
X_SPACING_TOLERANCE_FACTOR = 1.7 
CENTS_MAX_HEIGHT_FACTOR = 1.2 
GEOM_MERGE_MIN_WORD_CONFIDENCE = 70 


# Environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
ROBOFLOW_PROJECT_ID = os.getenv('ROBOFLOW_PROJECT_ID')
ROBOFLOW_VERSION_NUMBER = os.getenv('ROBOFLOW_VERSION_NUMBER')
POPPLER_BIN_PATH = os.getenv('POPPLER_PATH_OVERRIDE', None)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize clients
s3_client, textract_client, roboflow_model_object, openai_client = None, None, None, None
try:
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_DEFAULT_REGION)
    textract_client = boto3.client('textract', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_DEFAULT_REGION)
    app.logger.info(f"Boto3 clients initialized for region: {AWS_DEFAULT_REGION}.")
except Exception as e:
    app.logger.error(f"Error initializing Boto3 clients: {e}")

if ROBOFLOW_SDK_AVAILABLE and Roboflow and ROBOFLOW_API_KEY and ROBOFLOW_PROJECT_ID and ROBOFLOW_VERSION_NUMBER:
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.project(ROBOFLOW_PROJECT_ID)
        roboflow_model_object = project.version(int(ROBOFLOW_VERSION_NUMBER)).model
        app.logger.info(f"Roboflow model object initialized for project {ROBOFLOW_PROJECT_ID}, version {ROBOFLOW_VERSION_NUMBER}")
    except Exception as e:
        app.logger.error(f"Error initializing Roboflow model object: {e}", exc_info=True)
        roboflow_model_object = None
else:
    app.logger.warning("Roboflow SDK not available or configuration missing. Roboflow detection will be skipped.")
    roboflow_model_object = None

if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        app.logger.info("OpenAI client configured with API key.")
    except Exception as e:
        app.logger.error(f"Error initializing OpenAI client: {e}")
        openai_client = None
else:
    app.logger.warning("OPENAI_API_KEY not found in environment variables. OpenAI calls will fail.")

# ==================== PRICE PARSING ====================
def parse_price_string(price_str_input, item_id_for_log="N/A"):
    if price_str_input is None or price_str_input == "":
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Input is None or empty, returning None.")
        return None
    
    if isinstance(price_str_input, (int, float)):
        if price_str_input < 0:
            app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Input is numeric but negative ({price_str_input}), returning None.")
            return None
        if isinstance(price_str_input, int) and 100 <= price_str_input <= 99999: 
            s_price = str(price_str_input)
            if len(s_price) == 3: 
                val = float(f"{s_price[0]}.{s_price[1:]}")
                app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Input int {price_str_input} (len 3) parsed to {val}.")
                return val
            if len(s_price) == 4: 
                val = float(f"{s_price[:2]}.{s_price[2:]}")
                app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Input int {price_str_input} (len 4) parsed to {val}.")
                return val
            if len(s_price) == 5: 
                val = float(f"{s_price[:3]}.{s_price[3:]}")
                app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Input int {price_str_input} (len 5) parsed to {val}.")
                return val
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Input is numeric ({price_str_input}), returning as float.")
        return float(price_str_input)

    price_str = str(price_str_input).strip()
    app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Original string: '{price_str_input}', Stripped: '{price_str}'")

    geom_price_match = re.match(r'^\[GEOM_PRICE:\s*(\d{1,2})\s+(\d{2})\s*\]$', price_str)
    if geom_price_match:
        whole = geom_price_match.group(1)
        decimal_part = geom_price_match.group(2)
        val = float(f"{whole}.{decimal_part}")
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched GEOM_PRICE pattern '{price_str}' -> {val}.")
        return val

    space_separated_match = re.match(r'^(\d{1,2})\s+(\d{2})(?:\s*c/u)?$', price_str)
    if space_separated_match:
        whole = space_separated_match.group(1)
        decimal_part = space_separated_match.group(2)
        val = float(f"{whole}.{decimal_part}")
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched space-separated pattern '{price_str}' -> {val}.")
        return val

    if re.fullmatch(r'[1-9]\d{2}', price_str): 
        val = float(f"{price_str[0]}.{price_str[1:]}")
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched 3-digit pattern '{price_str}' -> {val}.")
        return val
    
    if re.fullmatch(r'[1-9]\d{3}', price_str): 
        val = float(f"{price_str[:2]}.{price_str[2:]}")
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched 4-digit pattern '{price_str}' -> {val}.")
        return val
    
    if re.fullmatch(r'[1-9]\d{4}', price_str): 
        val = float(f"{price_str[:3]}.{price_str[3:]}")
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched 5-digit pattern '{price_str}' -> {val}.")
        return val

    cleaned_price_str = price_str.lower()
    cleaned_price_str = re.sub(r'[$\¢₡€£¥]|regular|reg\.|oferta|esp\.|special|precio|price', '', cleaned_price_str, flags=re.IGNORECASE)
    cleaned_price_str = re.sub(r'\b(cada uno|c/u|cu|each|por)\b', '', cleaned_price_str, flags=re.IGNORECASE)
    cleaned_price_str = cleaned_price_str.strip()
    app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Cleaned for keywords: '{cleaned_price_str}'")
    
    if cleaned_price_str != price_str: 
        if re.fullmatch(r'[1-9]\d{2}', cleaned_price_str):
            val = float(f"{cleaned_price_str[0]}.{cleaned_price_str[1:]}")
            app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched 3-digit pattern on cleaned string '{cleaned_price_str}' -> {val}.")
            return val
        if re.fullmatch(r'[1-9]\d{3}', cleaned_price_str):
            val = float(f"{cleaned_price_str[:2]}.{cleaned_price_str[2:]}")
            app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched 4-digit pattern on cleaned string '{cleaned_price_str}' -> {val}.")
            return val
        if re.fullmatch(r'[1-9]\d{4}', cleaned_price_str):
            val = float(f"{cleaned_price_str[:3]}.{cleaned_price_str[3:]}")
            app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched 5-digit pattern on cleaned string '{cleaned_price_str}' -> {val}.")
            return val

    std_decimal_match_dot = re.fullmatch(r'(\d+)\.(\d{1,2})', cleaned_price_str)
    if std_decimal_match_dot:
        num_part, dec_part = std_decimal_match_dot.groups()
        if len(dec_part) == 1: dec_part += "0" 
        val = float(f"{num_part}.{dec_part}")
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched dot-decimal pattern '{cleaned_price_str}' -> {val}.")
        return val if val >= 0 else None

    std_decimal_match_comma = re.fullmatch(r'(\d+),(\d{1,2})', cleaned_price_str)
    if std_decimal_match_comma:
        num_part, dec_part = std_decimal_match_comma.groups()
        if len(dec_part) == 1: dec_part += "0"
        val = float(f"{num_part}.{dec_part}") 
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched comma-decimal pattern '{cleaned_price_str}' -> {val}.")
        return val if val >= 0 else None
        
    whole_match = re.fullmatch(r'(\d+)', cleaned_price_str)
    if whole_match:
        num = float(whole_match.group(1))
        if num == 0.0: 
            app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched whole number 0.0.")
            return 0.0
        if num >= 1 and num < 100: 
            app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Matched whole number pattern '{cleaned_price_str}' -> {num}.")
            return num

    app.logger.debug(f"ITEM_ID: {item_id_for_log} - parse_price_string - Could not parse price string: '{price_str}' (cleaned: '{cleaned_price_str}'). Returning None.")
    return None

# ==================== PRICE CANDIDATE DETECTION ====================
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
    
    app.logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Processing {len(line_blocks)} line_blocks. Prepended geom price: {prepended_geom_price}")

    if prepended_geom_price:
        parsed_geom_val = parse_price_string(prepended_geom_price, item_id_for_log=f"{item_id_for_log}-geom_cand_prep")
        if parsed_geom_val is not None:
            geom_candidate = {
                'text_content': prepended_geom_price, 
                'parsed_value': parsed_geom_val,
                'bounding_box': line_blocks[0]['Geometry']['BoundingBox'] if line_blocks else None, 
                'pixel_height': image_height_px * 0.1, 
                'source_block_id': 'GEOMETRIC_MERGE', 
                'full_line_text': prepended_geom_price,
                'is_regular_candidate': False, 
                'has_price_indicator': True 
            }
            candidates.append(geom_candidate)
            app.logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Added prepended geometric candidate: {geom_candidate}")


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
        if not full_line_text:
            app.logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Line {line_idx} is empty.")
            continue
        
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Line {line_idx} Text: '{full_line_text}'")

        has_price_indicator = any(indicator in full_line_text.lower() 
                                  for indicator in ['c/u', 'cada uno', '$', 'regular', 'precio', 'esp.'])
        
        for match in price_regex.finditer(full_line_text):
            raw_price_text = match.group(0).strip()
            app.logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Line {line_idx} - Raw regex match: '{raw_price_text}'")

            if raw_price_text.startswith("[GEOM_PRICE:"): 
                if not prepended_geom_price or raw_price_text != prepended_geom_price:
                    pass 
                else: 
                    continue


            match_start, match_end = match.span()
            context_before = full_line_text[max(0, match_start-10):match_start].lower()
            context_after = full_line_text[match_end:min(len(full_line_text), match_end+15)].lower() 
            
            is_likely_size_metric = False
            if any(re.search(r'^\s*' + re.escape(unit), context_after) for unit in size_unit_keywords):
                if not has_price_indicator: 
                    is_likely_size_metric = True
                    app.logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Skipping '{raw_price_text}' - looks like size (unit follows, no price indicator). Context after: '{context_after[:10]}'")
                    continue
            if any(kw in context_before for kw in ["pack of", "paquete de", "paq de"]):
                 if not has_price_indicator:
                    is_likely_size_metric = True
                    app.logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Skipping '{raw_price_text}' - looks like size (pack of precedes, no price indicator). Context before: '{context_before[-10:]}'")
                    continue

            if re.fullmatch(r'\d{3,}', raw_price_text) and int(raw_price_text) > 100: 
                if (re.search(r'\s*(a|-|to)\s*\d+', context_after) or 
                    re.search(r'\d+\s*(a|-|to)\s*$', context_before)):  
                    if not has_price_indicator:
                        app.logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Skipping '{raw_price_text}' - part of size range, no price indicator.")
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
                app.logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Added candidate: {candidate_data}")
            else:
                app.logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Line {line_idx} - Match '{raw_price_text}' did not parse to a valid price.")

    candidates.sort(key=lambda c: (
        c['source_block_id'] != 'GEOMETRIC_MERGE', 
        -c['pixel_height'], 
        not c['has_price_indicator'], 
        c['is_regular_candidate']
    ))
    app.logger.debug(f"ITEM_ID: {item_id_for_log} - detect_price_candidates - Found {len(candidates)} sorted candidates: {json.dumps(candidates, indent=2)}")
    return candidates

# ==================== VALIDATE PRICE PAIR ====================
def validate_price_pair(offer_price, regular_price, item_id_for_log="N/A"):
    op, rp = offer_price, regular_price
    if op is not None: op = float(op)
    if rp is not None: rp = float(rp)

    if op is None or rp is None:
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - validate_price_pair - One price is None (O:{op}, R:{rp}). No swap.")
        return op, rp
    
    if op > rp:
        app.logger.warning(f"ITEM_ID: {item_id_for_log} - validate_price_pair - Swapping prices: offer {op} > regular {rp}")
        return rp, op
    
    app.logger.debug(f"ITEM_ID: {item_id_for_log} - validate_price_pair - Prices validated (O:{op}, R:{rp}). No swap needed or already swapped.")
    return op, rp

# ==================== TEXT LLM EXTRACTION ====================
def extract_product_data_with_llm(product_snippet_text: str, item_id_for_log="N/A", llm_model: str = "gpt-4o") -> dict:
    if not openai_client:
        app.logger.error(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - OpenAI client not initialized.")
        return {"error_message": "OpenAI client not initialized", "llm_input_snippet": product_snippet_text}
    
    app.logger.info(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - Sending snippet to Text LLM ({llm_model}).")
    app.logger.debug(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - Snippet Text:\n{product_snippet_text}")

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
- "unit_indicator": Like "c/u", "ea." if present near a price.
- "store_specific_terms": Like "*24 por tienda", coupon details if not part of price.

IMPORTANT: Return prices as decimal numbers (e.g., 8.97), not strings. Use null if missing.
If product_variant_description contains size, also extract to size_quantity_info.

Return ONLY a JSON object."""
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
        chat_completion = openai_client.chat.completions.create(
            model=llm_model, messages=messages, response_format={"type": "json_object"}, temperature=0.1
        )
        response_content = chat_completion.choices[0].message.content
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - Text LLM Raw Response: {response_content}")
        if response_content is None: # Handle case where API returns None content
            app.logger.error(f"ITEM_ID: {item_id_for_log} - Text LLM returned None content.")
            return {"error_message": "Text LLM returned no content", "llm_input_snippet": product_snippet_text}

        extracted_data = json.loads(response_content)
        
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - LLM raw offer_price: {extracted_data.get('offer_price')}, LLM raw regular_price: {extracted_data.get('regular_price')}")

        extracted_data['offer_price'] = parse_price_string(extracted_data.get('offer_price'), item_id_for_log=f"{item_id_for_log}-llm_offer")
        extracted_data['regular_price'] = parse_price_string(extracted_data.get('regular_price'), item_id_for_log=f"{item_id_for_log}-llm_regular")
        
        expected_fields = ["product_brand", "product_name_core", "product_variant_description", "size_quantity_info", "offer_price", "regular_price", "unit_indicator", "store_specific_terms"]
        for field in expected_fields:
            if field not in extracted_data: extracted_data[field] = None
        
        app.logger.info(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - Successfully extracted and parsed data from Text LLM.")
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - Parsed LLM Data: {json.dumps(extracted_data, indent=2)}")
        return extracted_data
    except json.JSONDecodeError as je:
        app.logger.error(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - JSONDecodeError: {je}. Response: {response_content}", exc_info=True)
        return {"error_message": f"JSONDecodeError: {je}", "llm_input_snippet": product_snippet_text, "llm_response_content": response_content}
    except Exception as e:
        app.logger.error(f"ITEM_ID: {item_id_for_log} - extract_product_data_with_llm - Error in Text LLM processing: {e}", exc_info=True)
        return {"error_message": str(e), "llm_input_snippet": product_snippet_text, "llm_response_content": locals().get("response_content", "N/A")}

# ==================== VISION LLM HELPER FUNCTIONS ====================
def get_segment_image_bytes(page_image_pil: Image.Image, box_coords_pixels_center_wh: dict, item_id_for_log="N/A") -> BytesIO | None:
    try:
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
            app.logger.warning(f"ITEM_ID: {item_id_for_log} - get_segment_image_bytes - Invalid crop coords after clamping: ({x_min_clamped}, {y_min_clamped}, {x_max_clamped}, {y_max_clamped}). Original: ({x_min}, {y_min}, {x_max}, {y_max})")
            return None
            
        segment_image_pil = page_image_pil.crop((x_min_clamped, y_min_clamped, x_max_clamped, y_max_clamped))
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(segment_image_pil)
        draw.rectangle([(0, 0), (segment_image_pil.width-1, segment_image_pil.height-1)], 
                       outline="red", width=3)
        
        img_byte_arr = BytesIO()
        segment_image_pil.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - get_segment_image_bytes - Successfully cropped segment image. Coords: ({x_min_clamped}, {y_min_clamped}, {x_max_clamped}, {y_max_clamped})")
        return img_byte_arr
    except Exception as e:
        app.logger.error(f"ITEM_ID: {item_id_for_log} - get_segment_image_bytes - Error cropping segment image: {e}", exc_info=True)
        return None

def re_extract_with_vision_llm(segment_image_bytes: BytesIO, item_id_for_log="N/A", original_item_name: str | None = None, llm_model: str = "gpt-4o") -> dict:
    if not openai_client:
        app.logger.error(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - OpenAI client not configured for vision.")
        return {"error_message": "OpenAI client not configured for vision."}
    if not segment_image_bytes:
        app.logger.error(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - No segment image provided.")
        return {"error_message": "No segment image for vision."}
    
    response_content = None # Initialize to handle potential errors before assignment
    try:
        base64_image = base64.b64encode(segment_image_bytes.getvalue()).decode('utf-8')
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - Base64 image snippet for Vision LLM: {base64_image[:100]}...")

        prompt_text = (
            "You are an expert product data extractor for retail flyer segments. "
            "From the provided image of a single product deal, extract the following information. "
            "Pay close attention to visually prominent numbers for prices. "
            "If a price is shown as 'XYZ' (e.g., '897'), interpret it as X.YZ dollars (e.g., $8.97). "
            "If a price is 'X YZ' (e.g., '6 47'), interpret it as X.YZ dollars (e.g., $6.47). "
            "For 'N for $M' or 'NxM' deals (e.g., '2x300' where 300 means $3.00, or '2 for $5.00'), the offer_price should be the price PER ITEM (e.g., $1.50 or $2.50). "
            "If coupon details are present and modify the price, calculate the final per-item offer_price."
        )
        if original_item_name: prompt_text += f"\nThe product is likely related to: '{original_item_name}'.\n"
        prompt_text += (
            "\nFields to extract:\n"
            "- \"offer_price\": The final sale/promotional price per item. Return as a decimal number.\n"
            "- \"regular_price\": The original price per item. Return as a decimal number.\n"
            "- \"product_brand\": The brand name.\n"
            "- \"product_name_core\": The main product name.\n"
            "- \"product_variant_description\": Detailed description including flavor, type etc.\n"
            "- \"size_quantity_info\": Specific size/quantity (e.g., '105 a 117 onzas', '21 oz', '6=12 Rollos', 'Paquete de 2').\n"
            "- \"unit_indicator\": Like 'c/u', 'ea.' if present near a price.\n"
            "- \"store_specific_terms\": Like store limits or uncalculated coupon details.\n"
            "Return ONLY a JSON object with these fields. Use null for missing fields."
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
        
        app.logger.info(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - Sending segment image to Vision LLM ({llm_model}). Hint: '{original_item_name}'.")
        
        chat_completion = openai_client.chat.completions.create(
            model=llm_model, messages=messages, response_format={"type": "json_object"}, max_tokens=1000, temperature=0.1
        )
        response_content = chat_completion.choices[0].message.content
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - Vision LLM Raw Response: {response_content}")
        
        if response_content is None:
            app.logger.error(f"ITEM_ID: {item_id_for_log} - Vision LLM returned None content. Cannot parse JSON.")
            return {"error_message": "Vision LLM returned no content", "vision_llm_used": True}

        extracted_data = json.loads(response_content)
        extracted_data["vision_llm_used"] = True
        app.logger.info(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - Successfully extracted data from Vision LLM.")
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - Vision LLM Extracted Data: {json.dumps(extracted_data, indent=2)}")
        return extracted_data
    except json.JSONDecodeError as je:
        app.logger.error(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - JSONDecodeError: {je}. Response: {response_content}", exc_info=True)
        return {"error_message": f"JSONDecodeError: {je}", "vision_llm_used": True, "vision_llm_response_content": response_content}
    except Exception as e:
        app.logger.error(f"ITEM_ID: {item_id_for_log} - re_extract_with_vision_llm - Error calling Vision LLM API: {e}", exc_info=True)
        return {"error_message": str(e), "vision_llm_used": True, "vision_llm_response_content": response_content}
        
# ==================== HELPER FUNCTIONS (S3, Textract) ====================

def upload_to_s3(file_storage_object, bucket_name, cloud_object_name=None):
    # ... (no changes) ...
    if cloud_object_name is None:
        cloud_object_name = secure_filename(file_storage_object.filename) 
    try:
        s3_client.upload_fileobj(file_storage_object, bucket_name, cloud_object_name)
        app.logger.info(f"File '{cloud_object_name}' uploaded to S3 bucket '{bucket_name}'.")
        return cloud_object_name
    except Exception as e:
        app.logger.error(f"Error uploading file '{cloud_object_name}' to S3: {e}")
        return None

def get_analysis_from_document_via_textract(bucket_name, document_s3_key):
    # ... (no changes other than adding LAYOUT earlier) ...
    app.logger.info(f"Starting Textract Document Analysis for S3 object: s3://{bucket_name}/{document_s3_key}")
    try:
        response = textract_client.start_document_analysis(
            DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': document_s3_key}},
            FeatureTypes=['TABLES', 'FORMS', 'LAYOUT'] 
        )
        job_id = response['JobId']
        app.logger.info(f"Textract Analysis job started (JobId: '{job_id}') for '{document_s3_key}'.")
        
        status = 'IN_PROGRESS'
        max_retries = 90 
        retries = 0
        all_blocks = [] 
        
        while status == 'IN_PROGRESS' and retries < max_retries:
            time.sleep(5) 
            job_status_response = textract_client.get_document_analysis(JobId=job_id)
            status = job_status_response['JobStatus']
            app.logger.debug(f"Textract Analysis job status for '{job_id}': {status} (Retry {retries+1}/{max_retries})")
            retries += 1
            
        if status == 'SUCCEEDED':
            nextToken = None
            while True:
                if nextToken:
                    current_response = textract_client.get_document_analysis(JobId=job_id, NextToken=nextToken)
                else: 
                    current_response = job_status_response 
                
                page_blocks = current_response.get("Blocks", [])
                app.logger.debug(f"Textract SUCCEEDED page fetch for '{document_s3_key}', JobId '{job_id}'. Fetched {len(page_blocks)} blocks for this page/token.")
                all_blocks.extend(page_blocks)
                nextToken = current_response.get('NextToken')
                if not nextToken:
                    break
            app.logger.info(f"Textract Analysis SUCCEEDED for '{document_s3_key}'. Found {len(all_blocks)} blocks in total.")
            return all_blocks
        else:
            app.logger.error(f"Textract Analysis job for '{document_s3_key}' status: {status}. Response: {job_status_response}")
            return None
    except Exception as e:
        app.logger.error(f"Error in Textract Analysis for '{document_s3_key}': {e}", exc_info=True)
        return None

def delete_from_s3(bucket_name, cloud_object_name):
    # ... (no changes) ...
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=cloud_object_name)
        app.logger.info(f"File '{cloud_object_name}' deleted from S3 bucket '{bucket_name}'.")
    except Exception as e:
        app.logger.error(f"Error deleting file '{cloud_object_name}' from S3: {e}")

def clean_text(text): 
    # ... (no changes) ...
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
    # ... (no changes) ...
    if not roboflow_model_object:
        app.logger.error("Roboflow model object is not configured/initialized. Cannot get predictions.")
        return None
    
    temp_file_path = None
    try:
        os.makedirs("temp_uploads", exist_ok=True)
        temp_file_path = os.path.join("temp_uploads", f"{time.time()}_{secure_filename(original_filename_for_temp)}")
        if not temp_file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            temp_file_path += ".jpg" 
        
        pil_image_object.save(temp_file_path, format="JPEG" if temp_file_path.lower().endswith((".jpg", ".jpeg")) else "PNG")
        app.logger.info(f"Saved PIL image temporarily to {temp_file_path} for Roboflow.")
        
        prediction_result_obj = roboflow_model_object.predict(temp_file_path, confidence=40, overlap=30)
        
        actual_predictions_data = []
        if hasattr(prediction_result_obj, 'json') and callable(prediction_result_obj.json):
            json_response = prediction_result_obj.json()
            actual_predictions_data = json_response.get('predictions', [])
            app.logger.debug(f"Roboflow raw JSON response: {json.dumps(json_response, indent=2)}")
        elif hasattr(prediction_result_obj, 'predictions'): 
            actual_predictions_data = [p.json() for p in prediction_result_obj.predictions]
            app.logger.debug(f"Roboflow predictions (from .predictions attribute): {json.dumps(actual_predictions_data, indent=2)}")
        elif isinstance(prediction_result_obj, list): 
            actual_predictions_data = prediction_result_obj
            app.logger.debug(f"Roboflow predictions (already a list): {json.dumps(actual_predictions_data, indent=2)}")
        else:
            app.logger.warning(f"Unexpected Roboflow prediction result format: {type(prediction_result_obj)}. Trying to iterate...")
            try: 
                actual_predictions_data = [p.json() if hasattr(p, 'json') else p for p in prediction_result_obj]
            except TypeError:
                app.logger.error("Could not process Roboflow prediction object.")
                return []

        predictions_list = []
        for i, p_data in enumerate(actual_predictions_data):
            pred_dict = {
                'x': p_data.get('x'), 'y': p_data.get('y'),
                'width': p_data.get('width'), 'height': p_data.get('height'),
                'confidence': p_data.get('confidence'),
                'class': p_data.get('class', p_data.get('class_name', 'unknown')) 
            }
            if not all(isinstance(pred_dict[k], (int, float)) for k in ['x', 'y', 'width', 'height']):
                app.logger.warning(f"Skipping Roboflow prediction #{i} with invalid coordinates: {pred_dict}")
                continue
            predictions_list.append(pred_dict)
            
        app.logger.info(f"Processed {len(predictions_list)} valid Roboflow predictions.")
        return predictions_list
        
    except Exception as e:
        app.logger.error(f"Error in get_roboflow_predictions_sdk: {e}", exc_info=True)
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e_del:
                app.logger.error(f"Error deleting temp Roboflow image file {temp_file_path}: {e_del}")

# ==================== SMART COLLATION (Enhanced with Geometric Price Merging) ====================
def collate_text_for_product_boxes(roboflow_boxes, textract_all_blocks, blocks_map,
                                   image_width_px, image_height_px, page_id_for_log="N/A"):
    product_texts_with_candidates = []
    if not roboflow_boxes or not textract_all_blocks or not blocks_map or \
       image_width_px is None or image_height_px is None:
        app.logger.warning(f"PAGE_ID: {page_id_for_log} - collate_text_for_product_boxes: Missing critical inputs.")
        return product_texts_with_candidates

    app.logger.info(f"PAGE_ID: {page_id_for_log} - collate_text_for_product_boxes - Starting smart collation for {len(roboflow_boxes)} Roboflow boxes.")
    
    # Get all WORD blocks on the page once for efficiency
    all_words_on_page = [block for block in textract_all_blocks if block['BlockType'] == 'WORD']

    for i, box_pred in enumerate(roboflow_boxes):
        item_id_for_log = f"{page_id_for_log}-RFBox{i}-{box_pred.get('class', 'UnknownClass')}" # Make sure class is a string
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - collate_text_for_product_boxes - Processing Roboflow Box: {json.dumps(box_pred)}")

        rf_center_x_px, rf_center_y_px = box_pred.get('x'), box_pred.get('y')
        rf_width_px, rf_height_px = box_pred.get('width'), box_pred.get('height')
        
        if not all(isinstance(v, (int, float)) for v in [rf_center_x_px, rf_center_y_px, rf_width_px, rf_height_px]):
            app.logger.warning(f"ITEM_ID: {item_id_for_log} - collate_text_for_product_boxes - Roboflow Box has invalid coordinates. Skipping.")
            continue
            
        rf_x_min_rel = (rf_center_x_px - rf_width_px / 2.0) / image_width_px
        rf_y_min_rel = (rf_center_y_px - rf_height_px / 2.0) / image_height_px
        rf_x_max_rel = (rf_center_x_px + rf_width_px / 2.0) / image_width_px
        rf_y_max_rel = (rf_center_y_px + rf_height_px / 2.0) / image_height_px
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - Roboflow Box Rel Coords (xmin,ymin,xmax,ymax): ({rf_x_min_rel:.4f}, {rf_y_min_rel:.4f}, {rf_x_max_rel:.4f}, {rf_y_max_rel:.4f})")
        
        # Collect WORDs within the current Roboflow box
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
        
        # Sort words by top, then left for sequential processing
        words_in_rf_box.sort(key=lambda w: (w['Geometry']['BoundingBox']['Top'], w['Geometry']['BoundingBox']['Left']))
        
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - Found {len(words_in_rf_box)} WORD blocks with >={GEOM_MERGE_MIN_WORD_CONFIDENCE}% conf inside this Roboflow box.")
        for word_idx, wb_debug in enumerate(words_in_rf_box):
             app.logger.debug(f"ITEM_ID: {item_id_for_log} - WordInBox {word_idx}: '{wb_debug.get('Text')}' Geom: {json.dumps(wb_debug.get('Geometry',{}).get('BoundingBox'))} Conf: {wb_debug.get('Confidence')}")


        # --- Attempt Geometric Price Fragment Merging ---
        merged_geom_price_str = None
        used_word_ids_for_geom_price = set()

        # Iterate through sorted words_in_rf_box to find geometric price pairs
        for idx_w1, w1 in enumerate(words_in_rf_box):
            if w1['Id'] in used_word_ids_for_geom_price: continue
            w1_text = w1.get('Text', '')
            w1_geom_box = w1.get('Geometry', {}).get('BoundingBox')
            
            if not (re.fullmatch(r'[1-9]', w1_text) and w1_geom_box): # Potential single dollar digit
                continue
            app.logger.debug(f"ITEM_ID: {item_id_for_log} - GeomMerge: Potential dollar digit w1='{w1_text}' (ID: {w1['Id']})")

            for idx_w2 in range(idx_w1 + 1, len(words_in_rf_box)):
                w2 = words_in_rf_box[idx_w2]
                if w2['Id'] in used_word_ids_for_geom_price: continue
                w2_text = w2.get('Text', '')
                w2_geom_box = w2.get('Geometry', {}).get('BoundingBox')

                if not (re.fullmatch(r'\d{2}', w2_text) and w2_geom_box): # Potential two cents digits
                    continue
                app.logger.debug(f"ITEM_ID: {item_id_for_log} - GeomMerge: Potential cents w2='{w2_text}' (ID: {w2['Id']}) for w1='{w1_text}'")
                
                # Geometric checks
                w1_cy = w1_geom_box['Top'] + w1_geom_box['Height'] / 2
                w2_cy = w2_geom_box['Top'] + w2_geom_box['Height'] / 2
                y_diff_abs = abs(w1_cy - w2_cy)
                y_tolerance = w1_geom_box['Height'] * Y_ALIGN_TOLERANCE_FACTOR
                
                w1_right_edge = w1_geom_box['Left'] + w1_geom_box['Width']
                horizontal_gap = w2_geom_box['Left'] - w1_right_edge
                x_tolerance = w1_geom_box['Width'] * X_SPACING_TOLERANCE_FACTOR
                
                # Cents height check (relative to dollar digit)
                # Allow cents to be somewhat smaller or slightly larger, but not drastically different in height
                # This assumes dollar digit (w1) is generally not smaller than cents (w2)
                height_compatible = (w2_geom_box['Height'] < (w1_geom_box['Height'] * CENTS_MAX_HEIGHT_FACTOR)) and \
                                    (w2_geom_box['Height'] > (w1_geom_box['Height'] * (1/CENTS_MAX_HEIGHT_FACTOR*0.8) )) # Cents not too small either


                vertically_aligned = y_diff_abs < y_tolerance
                horizontally_close_and_ordered = 0 <= horizontal_gap < x_tolerance # w2 must be to the right

                app.logger.debug(f"ITEM_ID: {item_id_for_log} - GeomMerge Cand: w1='{w1_text}', w2='{w2_text}'. "
                                 f"V-Align: {vertically_aligned} (y_diff:{y_diff_abs:.4f} vs tol:{y_tolerance:.4f}). "
                                 f"H-Close: {horizontally_close_and_ordered} (gap:{horizontal_gap:.4f} vs tol:{x_tolerance:.4f}). "
                                 f"H-Compat: {height_compatible} (h1:{w1_geom_box['Height']:.4f}, h2:{w2_geom_box['Height']:.4f})")

                if vertically_aligned and horizontally_close_and_ordered and height_compatible:
                    merged_geom_price_str = f"[GEOM_PRICE: {w1_text} {w2_text}]"
                    used_word_ids_for_geom_price.add(w1['Id'])
                    used_word_ids_for_geom_price.add(w2['Id'])
                    app.logger.info(f"ITEM_ID: {item_id_for_log} - collate_text_for_product_boxes - Geometrically merged price candidate found: '{merged_geom_price_str}' from w1:'{w1_text}' (ID:{w1['Id']}) and w2:'{w2_text}' (ID:{w2['Id']})")
                    break 
            if merged_geom_price_str:
                break 


        # --- Existing Line-based Collation (now respects used_word_ids_for_geom_price) ---
        lines_in_box_objects = [] 
        for block_id, block in blocks_map.items():
            if block.get('BlockType') == 'LINE':
                txt_geom = block.get('Geometry', {}).get('BoundingBox', {})
                if not txt_geom: continue
                line_center_x_rel = txt_geom['Left'] + (txt_geom['Width'] / 2.0)
                line_center_y_rel = txt_geom['Top'] + (txt_geom['Height'] / 2.0)
                if (rf_x_min_rel <= line_center_x_rel <= rf_x_max_rel and \
                    rf_y_min_rel <= line_center_y_rel <= rf_y_max_rel):
                    # Log the LINE block that's being considered for line-based collation
                    app.logger.debug(f"ITEM_ID: {item_id_for_log} - LineCollation: Considering LINE '{block.get('Text', '')}' (ID: {block_id}) Geom: {json.dumps(txt_geom)}")
                    lines_in_box_objects.append(block)
        
        lines_in_box_objects.sort(key=lambda line: (
            line['Geometry']['BoundingBox']['Top'],
            -line['Geometry']['BoundingBox']['Height'], 
            line['Geometry']['BoundingBox']['Left']
        ))
        
        ordered_lines_text_parts = []
        for line_block in lines_in_box_objects:
            line_actual_text_parts = []
            words_in_this_line_for_text = []
            if 'Relationships' in line_block:
                for relationship in line_block['Relationships']:
                    if relationship['Type'] == 'CHILD':
                        for child_id in relationship['Ids']:
                            word = blocks_map.get(child_id)
                            if word and word['BlockType'] == 'WORD' and child_id not in used_word_ids_for_geom_price:
                                words_in_this_line_for_text.append(word['Text'])
            
            # If all words in the line were used by geometric merge, this line might be empty or just remnants.
            # Only add if it still has content.
            line_text = " ".join(words_in_this_line_for_text).strip()
            
            if line_text: 
                ordered_lines_text_parts.append(line_text)
            elif not words_in_this_line_for_text and line_block.get('Text'): # Line had text, but all its words were used for geom_price
                 app.logger.debug(f"ITEM_ID: {item_id_for_log} - LineCollation: Line '{line_block.get('Text')}' fully consumed by geometric merge, not adding to collated text.")
            elif not line_text and line_block.get('Text'): # Fallback for lines that Textract gives text for but no children
                 app.logger.debug(f"ITEM_ID: {item_id_for_log} - LineCollation: Line '{line_block.get('Text')}' has no child words not used by geom_price, using raw line text.")
                 ordered_lines_text_parts.append(line_block.get('Text','').strip())


        collated_text_multiline = "\n".join(ordered_lines_text_parts)
        collated_text_cleaned = clean_text(collated_text_multiline) 

        if merged_geom_price_str:
            # Prepend, ensuring it's on its own line for clarity for the LLM
            collated_text_cleaned = f"{merged_geom_price_str}\n{collated_text_cleaned}".strip()
            app.logger.info(f"ITEM_ID: {item_id_for_log} - Prepended geometric price. New collated text starts with: {merged_geom_price_str}")

        # Price candidates are now detected based on the potentially modified (prepended) collated text
        # However, detect_price_candidates works on line_blocks.
        # For now, the prepended_geom_price is passed to detect_price_candidates to be added directly.
        price_candidates_for_segment = detect_price_candidates(lines_in_box_objects, image_height_px, blocks_map, item_id_for_log=item_id_for_log, prepended_geom_price=merged_geom_price_str if merged_geom_price_str else None)


        app.logger.debug(f"ITEM_ID: {item_id_for_log} - collate_text_for_product_boxes - Sorted Lines Text for Collation (after potential geom exclusion): {json.dumps(ordered_lines_text_parts)}")
        app.logger.info(f"ITEM_ID: {item_id_for_log} - collate_text_for_product_boxes - Final Collated Text (cleaned) for LLM:\n{collated_text_cleaned}")
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - collate_text_for_product_boxes - Price candidates for this segment: {json.dumps(price_candidates_for_segment, indent=2)}")


        if collated_text_cleaned or merged_geom_price_str: 
            product_texts_with_candidates.append({
                "product_box_id": item_id_for_log, 
                "roboflow_confidence": box_pred.get('confidence', 0.0),
                "class_name": str(box_pred.get('class', 'UnknownClass')), # Ensure class is string
                "collated_text": collated_text_cleaned,
                "price_candidates": price_candidates_for_segment, 
                "roboflow_box_coords_pixels_center_wh": {
                    'x': rf_center_x_px, 'y': rf_center_y_px,
                    'width': rf_width_px, 'height': rf_height_px
                }
            })
    
    app.logger.info(f"PAGE_ID: {page_id_for_log} - collate_text_for_product_boxes - Collation complete. Generated {len(product_texts_with_candidates)} product text snippets.")
    return product_texts_with_candidates


# ==================== ENHANCED NORMALIZATION & POST-PROCESSING ====================
def enhanced_normalize_product_data(product_data, original_collated_text="", item_id_for_log="N/A"):
    # ... (no changes) ...
    if not product_data or not isinstance(product_data, dict):
        app.logger.warning(f"ITEM_ID: {item_id_for_log} - enhanced_normalize_product_data - Input product_data is invalid or empty.")
        return product_data if isinstance(product_data, dict) else {}

    app.logger.debug(f"ITEM_ID: {item_id_for_log} - enhanced_normalize_product_data - Starting normalization. Input data: {json.dumps(product_data, indent=2)}")
    normalized = product_data.copy() 

    brand_corrections = {
        'downy': 'Downy', 'gain': 'Gain', 'tide': 'Tide', 'glad': 'Glad', 'scott': 'Scott',
        'raid': 'Raid', 'ace': 'Ace', 'purex': 'Purex', 'clorox': 'Clorox', 'bounty': 'Bounty',
        'lysol': 'Lysol', 'glade': 'Glade', 'airwick': 'Air Wick', 'air wick': 'Air Wick',
        'reynolds': 'Reynolds', 'lestoil': 'Lestoil', 'ensueño': 'Ensueño', 'ensueno': 'Ensueño',
        'woolite': 'Woolite', 'suavitel': 'Suavitel', 'rocio': 'Rocio', 'real kill': 'Real Kill',
        'scrubbing bubbles': 'Scrubbing Bubbles', 'oxi clean': 'OxiClean', 'oxiclean': 'OxiClean',
        'reynolds wrap': 'Reynolds Wrap' 
    }
    if normalized.get('product_brand'):
        brand_l = str(normalized['product_brand']).lower().strip()
        normalized['product_brand'] = brand_corrections.get(brand_l, normalized['product_brand'])
    elif normalized.get('product_name_core'): 
        name_l = str(normalized['product_name_core']).lower()
        for known_brand_l, corrected_brand in brand_corrections.items():
            if known_brand_l in name_l:
                normalized['product_brand'] = corrected_brand
                break
    
    size_text = normalized.get('size_quantity_info', '')
    if not size_text and normalized.get('product_variant_description'): 
        size_text = normalized.get('product_variant_description', '')
    
    parsed_size_details = {}
    size_text_lower_normalized = "" 
    if size_text:
        size_text_lower = str(size_text).lower()
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - enhanced_normalize_product_data - Original size text for normalization: '{size_text_lower}'")
        
        unit_conversions = {
            'onzas': 'oz', 'onza': 'oz', 'ozs': 'oz',
            'libras': 'lb', 'libra': 'lb', 'lbs': 'lb',
            'galones': 'gal', 'galon': 'gal',
            'litro': 'L', 'litros': 'L', 'lt': 'L', 'lts': 'L',
            'mililitros': 'mL', 'ml': 'mL',
            'gramos': 'g', 'gramo': 'g', 'gr': 'g',
            'kilo': 'kg', 'kilos': 'kg',
            'pies': 'ft', 'pie': 'ft', 
            'metros': 'm', 'metro': 'm',
            'hojas': 'sheets', 'hoja': 'sheet',
            'rollos': 'rolls', 'rollo': 'roll', 'rll': 'rolls',
            'ct': 'ct', 'count': 'ct', 'unidades': 'ct', 'unidad': 'ct', 'u': 'ct', 'und': 'ct', 'un': 'ct',
            'docena': '12 ct', 
            'paquete de': 'pk of', 'pq': 'pk', 'pack': 'pk'
        }

        size_text_lower_normalized = size_text_lower
        for spa, eng in unit_conversions.items():
            size_text_lower_normalized = re.sub(r'\b' + re.escape(spa) + r'\b', eng, size_text_lower_normalized)
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - enhanced_normalize_product_data - Size text after unit keyword normalization: '{size_text_lower_normalized}'")

        range_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:a|-|to)\s*(\d+(?:[.,]\d+)?)\s*([a-z]+(?:\s[a-z]+)?)', size_text_lower_normalized) 
        if range_match:
            try:
                parsed_size_details['value_min'] = float(range_match.group(1).replace(',', '.'))
                parsed_size_details['value_max'] = float(range_match.group(2).replace(',', '.'))
                parsed_size_details['unit'] = range_match.group(3).strip()
            except ValueError: app.logger.warning(f"ITEM_ID: {item_id_for_log} - enhanced_normalize_product_data - ValueError parsing range match: {range_match.groups()}")
        else:
            single_value_match = re.search(r'(\d+(?:[.,]\d+)?)\s*([a-z]+(?:\s[a-z]+)?)(?:\s|$)', size_text_lower_normalized)
            if single_value_match:
                try:
                    parsed_size_details['value'] = float(single_value_match.group(1).replace(',', '.'))
                    parsed_size_details['unit'] = single_value_match.group(2).strip()
                except ValueError: app.logger.warning(f"ITEM_ID: {item_id_for_log} - enhanced_normalize_product_data - ValueError parsing single value match: {single_value_match.groups()}")
            else:
                equals_match = re.search(r'(\d+)\s*=\s*(\d+)\s*([a-z]+(?:\s[a-z]+)?)', size_text_lower_normalized)
                if equals_match:
                    try:
                        parsed_size_details['value_base'] = int(equals_match.group(1))
                        parsed_size_details['value_equivalent'] = int(equals_match.group(2))
                        parsed_size_details['unit'] = equals_match.group(3).strip()
                        parsed_size_details['value'] = parsed_size_details['value_base'] if 'value_base' in parsed_size_details else parsed_size_details['value_equivalent']

                    except ValueError: app.logger.warning(f"ITEM_ID: {item_id_for_log} - enhanced_normalize_product_data - ValueError parsing equals match: {equals_match.groups()}")
                else: 
                    count_match = re.search(r'(?:pk of\s*)?(\d+)\s*(ct|pk|refill)?', size_text_lower_normalized)
                    if count_match:
                        try:
                            parsed_size_details['value'] = int(count_match.group(1))
                            parsed_size_details['unit'] = count_match.group(2) if count_match.group(2) else 'ct' 
                        except ValueError: app.logger.warning(f"ITEM_ID: {item_id_for_log} - enhanced_normalize_product_data - ValueError parsing count match: {count_match.groups()}")
        
        fraction_match = re.search(r'(\d+)/(\d+)\s*([a-z]+)', size_text_lower_normalized)
        if fraction_match and not parsed_size_details: 
            try:
                val = int(fraction_match.group(1)) / int(fraction_match.group(2))
                parsed_size_details['value'] = val
                parsed_size_details['unit'] = fraction_match.group(3)
            except ValueError: pass 

        normalized['size_quantity_info_normalized'] = size_text_lower_normalized.strip() 
        if parsed_size_details:
            normalized['parsed_size_details'] = parsed_size_details
            app.logger.debug(f"ITEM_ID: {item_id_for_log} - enhanced_normalize_product_data - Parsed size details: {json.dumps(parsed_size_details)}")
        else:
            app.logger.debug(f"ITEM_ID: {item_id_for_log} - enhanced_normalize_product_data - No structured size details parsed from '{size_text_lower_normalized}'.")
            
    if 'size_quantity_info' not in normalized and product_data.get('size_quantity_info'):
        normalized['size_quantity_info'] = product_data.get('size_quantity_info')

    if normalized.get('unit_indicator'):
        ui_lower = str(normalized['unit_indicator']).lower().strip()
        if ui_lower in ['c/u', 'cu', 'cada uno']: normalized['unit_indicator'] = 'c/u'
        elif ui_lower in ['ea', 'each']: normalized['unit_indicator'] = 'ea'
    
    app.logger.debug(f"ITEM_ID: {item_id_for_log} - enhanced_normalize_product_data - Normalization complete. Output data: {json.dumps(normalized, indent=2)}")
    return normalized


# ==================== POST-PROCESSING AND VALIDATION (Enhanced with Debugging) ====================
def post_process_and_validate_item_data(llm_data, price_candidates, original_collated_text, item_id_for_log="N/A"):
    # ... (no changes) ...
    if not isinstance(llm_data, dict) or "error_message" in llm_data:
        app.logger.error(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Invalid or error in LLM data: {llm_data}")
        return llm_data

    item_data = llm_data.copy()
    item_data['validation_flags'] = item_data.get('validation_flags', [])
    app.logger.debug(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Starting. Initial LLM Data: {json.dumps(llm_data, indent=2)}")
    app.logger.debug(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Price Candidates: {json.dumps(price_candidates, indent=2)}")
    app.logger.debug(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Original Collated Text:\n{original_collated_text}")

    llm_offer_price = item_data.get('offer_price') 
    llm_regular_price = item_data.get('regular_price')

    multibuy_patterns = [
        (r'(\d+)\s*x\s*(\d{3})(?!\d)', lambda m: (int(m.group(1)), float(f"{m.group(2)[0]}.{m.group(2)[1:]}"))),
        (r'(\d+)\s*(?:for|x)\s*\$?\s*(\d+(?:\.\d{2})?)', lambda m: (int(m.group(1)), float(m.group(2)))),
        (r'(\d+)\s*x\s*(\d{2})(?!\d)', lambda m: (int(m.group(1)), float(f"0.{m.group(2)}"))),
    ]
    
    for pattern_idx, (pattern, parser) in enumerate(multibuy_patterns):
        multibuy_match = re.search(pattern, original_collated_text, re.IGNORECASE)
        if multibuy_match:
            app.logger.debug(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Multi-buy pattern #{pattern_idx} matched: '{multibuy_match.group(0)}'")
            try:
                quantity, total_price = parser(multibuy_match)
                if quantity > 0 and total_price > 0:
                    per_item_price = round(total_price / quantity, 2)
                    if 0.01 <= per_item_price <= 100.00: 
                        if llm_offer_price is None or abs(llm_offer_price - per_item_price) > 0.05: 
                            item_data['offer_price'] = per_item_price
                            flag_msg = f"Multi-buy pattern '{multibuy_match.group(0)}' -> ${per_item_price}/item. LLM price was {llm_offer_price}."
                            item_data['validation_flags'].append(flag_msg)
                            app.logger.info(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - {flag_msg}")
                            llm_offer_price = per_item_price 
                            break 
            except (ValueError, IndexError, TypeError) as e:
                app.logger.warning(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Error parsing multi-buy pattern '{multibuy_match.group(0)}': {e}")

    coupon_pattern = r'(\d+)\s*x\s*\$?\s*(\d+(?:\.\d{2})?)\s*.*(?:coupon|cup[oó]n).*=\s*\d+\s*x\s*\$?\s*(\d+(?:\.\d{2})?)'
    coupon_match = re.search(coupon_pattern, original_collated_text, re.IGNORECASE)
    if coupon_match:
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Coupon pattern matched: '{coupon_match.group(0)}'")
        try:
            quantity = int(coupon_match.group(1))
            final_total_after_coupon = float(coupon_match.group(3))
            if quantity > 0:
                per_item_price_coupon = round(final_total_after_coupon / quantity, 2)
                if 0.01 <= per_item_price_coupon <= 100.00:
                    if llm_offer_price is None or abs(llm_offer_price - per_item_price_coupon) > 0.05:
                        item_data['offer_price'] = per_item_price_coupon
                        flag_msg = f"Coupon-adjusted multi-buy: {quantity} for ${final_total_after_coupon} -> ${per_item_price_coupon}/item. LLM price was {llm_offer_price}."
                        item_data['validation_flags'].append(flag_msg)
                        app.logger.info(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - {flag_msg}")
                        llm_offer_price = per_item_price_coupon
        except (ValueError, IndexError, TypeError) as e:
            app.logger.warning(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Error parsing coupon pattern '{coupon_match.group(0)}': {e}")

    best_offer_candidate = None
    best_regular_candidate = None
    if price_candidates:
        valid_price_cands = [pc for pc in price_candidates if pc.get('parsed_value') is not None and pc['parsed_value'] >= 0]
        offer_cands = sorted([pc for pc in valid_price_cands if not pc.get('is_regular_candidate')], key=lambda c: (c.get('source_block_id') == 'GEOMETRIC_MERGE', -c.get('pixel_height',0), not c.get('has_price_indicator', False))) 
        regular_cands = sorted([pc for pc in valid_price_cands if pc.get('is_regular_candidate')], key=lambda c: (-c.get('pixel_height',0), not c.get('has_price_indicator', False)))
        if offer_cands: best_offer_candidate = offer_cands[0]
        if regular_cands: best_regular_candidate = regular_cands[0]
        app.logger.debug(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Best Offer Cand: {json.dumps(best_offer_candidate, indent=2)}, Best Regular Cand: {json.dumps(best_regular_candidate, indent=2)}")

    if best_offer_candidate:
        cand_val = best_offer_candidate['parsed_value']
        cand_text_raw = best_offer_candidate['text_content'].strip()
        is_geom_candidate = best_offer_candidate['source_block_id'] == 'GEOMETRIC_MERGE'

        if llm_offer_price is None and 0.01 <= cand_val < 150.00 : 
            item_data['offer_price'] = cand_val
            flag_msg = f"Offer price '{cand_val}' populated from {'geometric ' if is_geom_candidate else 'visual '}candidate '{cand_text_raw}' (LLM missed)."
            item_data['validation_flags'].append(flag_msg)
            app.logger.info(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - {flag_msg}")
            llm_offer_price = cand_val
        elif llm_offer_price is not None and abs(llm_offer_price - cand_val) > 0.01 and 0.01 <= cand_val < 150.00:
            if is_geom_candidate and (0.01 <= llm_offer_price < 1.00 and cand_val >= 1.00):
                flag_msg = f"LLM offer price {llm_offer_price} corrected by geometric candidate '{cand_text_raw}' ({cand_val})."
                item_data['offer_price'] = cand_val
                item_data['validation_flags'].append(flag_msg)
                app.logger.info(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - {flag_msg}")
                llm_offer_price = cand_val
            elif (0.01 <= llm_offer_price < 1.00 and cand_val >= 1.00) or \
                 (llm_offer_price >= 1.00 and cand_val >= 1.00 and abs(llm_offer_price - cand_val) > 0.50): 
                flag_msg = f"LLM offer price {llm_offer_price} differs from prominent visual candidate '{cand_text_raw}' ({cand_val}). Corrected to candidate."
                item_data['offer_price'] = cand_val
                item_data['validation_flags'].append(flag_msg)
                app.logger.info(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - {flag_msg}")
                llm_offer_price = cand_val

    if best_regular_candidate:
        cand_val = best_regular_candidate['parsed_value']
        cand_text_raw = best_regular_candidate['text_content'].strip()
        if llm_regular_price is None and 0.01 <= cand_val < 200.00: 
            item_data['regular_price'] = cand_val
            flag_msg = f"Regular price '{cand_val}' populated from visual candidate '{cand_text_raw}' (LLM missed)."
            item_data['validation_flags'].append(flag_msg)
            app.logger.info(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - {flag_msg}")
        elif llm_regular_price is not None and abs(llm_regular_price - cand_val) > 0.01 and 0.01 <= cand_val < 200.00:
            if (0.01 <= llm_regular_price < 1.00 and cand_val >= 1.00) or \
               (llm_regular_price >=1.00 and cand_val >= 1.00 and abs(llm_regular_price - cand_val) > 0.50):
                flag_msg = f"LLM regular price {llm_regular_price} differs from visual regular candidate '{cand_text_raw}' ({cand_val}). Corrected."
                item_data['regular_price'] = cand_val
                item_data['validation_flags'].append(flag_msg)
                app.logger.info(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - {flag_msg}")

    current_op = item_data.get('offer_price')
    current_rp = item_data.get('regular_price')
    
    if current_op is not None:
        try:
            op_float = float(current_op)
            # Adjusted range for offers to be slightly wider to accommodate furniture etc.
            if not (0.01 <= op_float <= 500.00): # Example: increased upper limit
                item_data['validation_flags'].append(f"Final offer price {op_float} is out of typical range ($0.01-$500.00).")
                app.logger.warning(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Offer price {op_float} out of range.")
        except (ValueError, TypeError):
            item_data['validation_flags'].append(f"Final offer price '{current_op}' is not a valid number.")
            item_data['offer_price'] = None
            app.logger.warning(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Invalid offer price '{current_op}'.")
            
    if current_rp is not None:
        try:
            rp_float = float(current_rp)
            if not (0.01 <= rp_float <= 600.00): # Example: increased upper limit
                item_data['validation_flags'].append(f"Final regular price {rp_float} is out of typical range ($0.01-$600.00).")
                app.logger.warning(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Regular price {rp_float} out of range.")
        except (ValueError, TypeError):
            item_data['validation_flags'].append(f"Final regular price '{current_rp}' is not a valid number.")
            item_data['regular_price'] = None
            app.logger.warning(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Invalid regular price '{current_rp}'.")

    final_op_before_swap = item_data.get('offer_price')
    final_rp_before_swap = item_data.get('regular_price')
    if final_op_before_swap is not None and final_rp_before_swap is not None:
        op_corr, rp_corr = validate_price_pair(final_op_before_swap, final_rp_before_swap, item_id_for_log=item_id_for_log)
        if op_corr != final_op_before_swap or rp_corr != final_rp_before_swap: 
            item_data['offer_price'] = op_corr
            item_data['regular_price'] = rp_corr
    
    item_data = enhanced_normalize_product_data(item_data, original_collated_text, item_id_for_log=item_id_for_log)

    if not item_data.get('product_name_core') and not item_data.get('product_brand'):
        item_data['validation_flags'].append("Missing product name/brand after all processing.")
    
    if item_data.get('offer_price') is None and item_data.get('regular_price') is None:
        if not price_candidates or all(pc.get('parsed_value') is None for pc in price_candidates):
            item_data['validation_flags'].append("Missing both offer and regular price, and no usable visual price candidates were found.")
        elif any(pc.get('parsed_value') is not None for pc in price_candidates):
             item_data['validation_flags'].append("Missing both offer and regular price, despite some visual price candidates being found.")
    
    size_keywords = ['oz', 'onzas', 'lb', 'libras', 'gal', 'lt', 'ml', 'g', 'kg', 'rollos', 'hojas', 'ct', 'pies', 'ft', 'unidad']
    variant_desc_lower = str(item_data.get('product_variant_description', "")).lower()
    collated_lower = str(original_collated_text).lower()
    variant_has_keyword = any(kw in variant_desc_lower for kw in size_keywords)
    collated_has_keyword = any(kw in collated_lower for kw in size_keywords)

    if (variant_has_keyword or collated_has_keyword):
        if not item_data.get('size_quantity_info') and not item_data.get('parsed_size_details'):
            item_data['validation_flags'].append("Size/quantity info might be missing or unparsed (keywords found in text but no structured parse).")
    
    app.logger.info(f"ITEM_ID: {item_id_for_log} - post_process_and_validate_item_data - Finished. Final Item Data: {json.dumps(item_data, indent=2)}")
    return item_data

# ==================== ENHANCED COMPARISON FUNCTION ====================
def compare_product_items(product_items1, product_items2, similarity_threshold=70):
    # ... (no changes) ...
    app.logger.info(f"COMPARE_FN - Starting comparison: {len(product_items1)} items from File1 with {len(product_items2)} items from File2.")
    comparison_report = []
    matched_item2_indices = set()

    for idx1, item1 in enumerate(product_items1):
        item1_id_log = item1.get("product_box_id", f"File1-Item{idx1}")
        app.logger.debug(f"COMPARE_FN - Comparing {item1_id_log}: {item1.get('product_brand')} {item1.get('product_name_core')}")
        best_match_item2 = None
        highest_similarity = 0
        best_match_idx = -1
        
        brand1 = str(item1.get("product_brand", "") or "").lower().strip()
        name_core1 = str(item1.get("product_name_core", "") or "").lower().strip()
        size1_details = item1.get("parsed_size_details", {})
        size1_str_norm = item1.get("size_quantity_info_normalized", item1.get("size_quantity_info", ""))
        size1 = str(size1_str_norm or "").lower().strip()
        
        if size1_details:
            if "value" in size1_details and "unit" in size1_details:
                size1 = f"{size1_details['value']} {size1_details['unit']}"
            elif "value_min" in size1_details and "value_max" in size1_details and "unit" in size1_details:
                size1 = f"{size1_details['value_min']}-{size1_details['value_max']} {size1_details['unit']}"
            elif "value_base" in size1_details: 
                size1 = f"{size1_details['value_base']}={size1_details.get('value_equivalent','')} {size1_details.get('unit','')}"


        variant1 = str(item1.get("product_variant_description", "") or "").lower().strip()
        primary_text1 = f"{brand1} {name_core1}".strip()
        secondary_text1 = variant1 if variant1 != size1 else "" 
        secondary_text1 = f"{secondary_text1} {size1}".strip() 

        if not primary_text1 and not secondary_text1:
            app.logger.warning(f"COMPARE_FN - {item1_id_log} has no text for matching. Skipping.")
            continue

        for idx2, item2 in enumerate(product_items2):
            if idx2 in matched_item2_indices: continue
            item2_id_log = item2.get("product_box_id", f"File2-Item{idx2}")

            brand2 = str(item2.get("product_brand", "") or "").lower().strip()
            name_core2 = str(item2.get("product_name_core", "") or "").lower().strip()
            size2_details = item2.get("parsed_size_details", {})
            size2_str_norm = item2.get("size_quantity_info_normalized", item2.get("size_quantity_info", ""))
            size2 = str(size2_str_norm or "").lower().strip()

            if size2_details:
                if "value" in size2_details and "unit" in size2_details:
                    size2 = f"{size2_details['value']} {size2_details['unit']}"
                elif "value_min" in size2_details and "value_max" in size2_details and "unit" in size2_details:
                    size2 = f"{size2_details['value_min']}-{size2_details['value_max']} {size2_details['unit']}"
                elif "value_base" in size2_details:
                    size2 = f"{size2_details['value_base']}={size2_details.get('value_equivalent','')} {size2_details.get('unit','')}"


            variant2 = str(item2.get("product_variant_description", "") or "").lower().strip()
            primary_text2 = f"{brand2} {name_core2}".strip()
            secondary_text2 = variant2 if variant2 != size2 else ""
            secondary_text2 = f"{secondary_text2} {size2}".strip()

            if not primary_text2 and not secondary_text2: continue
            
            primary_similarity = fuzz.token_set_ratio(primary_text1, primary_text2) if primary_text1 and primary_text2 else 0
            secondary_similarity = fuzz.token_set_ratio(secondary_text1, secondary_text2) if secondary_text1 and secondary_text2 else 0
            
            similarity = (primary_similarity * 0.7) + (secondary_similarity * 0.3)
            if not (primary_text1 and primary_text2): similarity = secondary_similarity 
            if not (secondary_text1 and secondary_text2): similarity = primary_similarity 
            if not (primary_text1 and primary_text2 and secondary_text1 and secondary_text2): 
                 similarity = max(primary_similarity, secondary_similarity) if (primary_text1 and primary_text2) or \
                                                                               (secondary_text1 and secondary_text2) \
                                                                          else (primary_similarity + secondary_similarity) / 2.0


            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_item2 = item2
                best_match_idx = idx2
        
        if best_match_item2 and highest_similarity >= similarity_threshold:
            matched_item2_indices.add(best_match_idx)
            best_match_item2_id_log = best_match_item2.get("product_box_id", f"File2-Item{best_match_idx}")
            app.logger.info(f"COMPARE_FN - Match found for {item1_id_log} with {best_match_item2_id_log}. Similarity: {highest_similarity:.1f}%")
            diff_details = []
            
            offer_price1 = item1.get("offer_price")
            offer_price2 = best_match_item2.get("offer_price")
            regular_price1 = item1.get("regular_price")
            regular_price2 = best_match_item2.get("regular_price")
            price_tolerance = 0.01 

            if (offer_price1 is not None and offer_price2 is not None and abs(float(offer_price1) - float(offer_price2)) > price_tolerance) or \
               (offer_price1 is None and offer_price2 is not None) or (offer_price1 is not None and offer_price2 is None):
                diff_details.append(f"Offer Price: F1=${offer_price1 if offer_price1 is not None else 'N/A'} vs F2=${offer_price2 if offer_price2 is not None else 'N/A'}")

            if (regular_price1 is not None and regular_price2 is not None and abs(float(regular_price1) - float(regular_price2)) > price_tolerance) or \
               (regular_price1 is None and regular_price2 is not None) or (regular_price1 is not None and regular_price2 is None):
                diff_details.append(f"Regular Price: F1=${regular_price1 if regular_price1 is not None else 'N/A'} vs F2=${regular_price2 if regular_price2 is not None else 'N/A'}")

            size_norm1 = item1.get("size_quantity_info_normalized", item1.get("size_quantity_info"))
            size_norm2 = best_match_item2.get("size_quantity_info_normalized", best_match_item2.get("size_quantity_info"))
            if str(size_norm1 or '').strip().lower() != str(size_norm2 or '').strip().lower():
                diff_details.append(f"Size: F1='{size_norm1 or 'N/A'}' vs F2='{size_norm2 or 'N/A'}'")

            base_report_item = {
                "product1_brand": item1.get("product_brand"), "product1_name_core": item1.get("product_name_core"),
                "product1_variant": item1.get("product_variant_description"), "product1_size": item1.get("size_quantity_info"),
                "product1_size_normalized": item1.get("size_quantity_info_normalized"), 
                "product1_parsed_size_details": item1.get("parsed_size_details"), 
                "offer_price1": offer_price1, "regular_price1": regular_price1,
                "unit_indicator1": item1.get("unit_indicator"), "store_terms1": item1.get("store_specific_terms"),
                "validation_flags1": item1.get("validation_flags", []), 

                "product2_brand": best_match_item2.get("product_brand"), "product2_name_core": best_match_item2.get("product_name_core"),
                "product2_variant": best_match_item2.get("product_variant_description"), "product2_size": best_match_item2.get("size_quantity_info"),
                "product2_size_normalized": best_match_item2.get("size_quantity_info_normalized"), 
                "product2_parsed_size_details": best_match_item2.get("parsed_size_details"), 
                "offer_price2": offer_price2, "regular_price2": regular_price2,
                "unit_indicator2": best_match_item2.get("unit_indicator"), "store_terms2": best_match_item2.get("store_specific_terms"),
                "validation_flags2": best_match_item2.get("validation_flags", []), 
                "text_similarity_percent": round(highest_similarity, 1),
            }
            
            if diff_details:
                comparison_report.append({"type": "Product Match - Attribute Mismatch", **base_report_item, "differences": "; ".join(diff_details)})
            else:
                comparison_report.append({"type": "Product Match - Attributes OK", **base_report_item})
        else:
            app.logger.info(f"COMPARE_FN - No match found for {item1_id_log} (Highest sim: {highest_similarity:.1f}%)")
            comparison_report.append({
                "type": "Unmatched Product in File 1",
                "product_brand": item1.get("product_brand"), "product_name_core": item1.get("product_name_core"),
                "product_variant": item1.get("product_variant_description"), "product_size": item1.get("size_quantity_info"),
                "product_size_normalized": item1.get("size_quantity_info_normalized"),
                "parsed_size_details": item1.get("parsed_size_details"),
                "offer_price": item1.get("offer_price"), "regular_price": item1.get("regular_price"),
                "unit_indicator": item1.get("unit_indicator"), "store_terms": item1.get("store_specific_terms"),
                "validation_flags": item1.get("validation_flags", []),
                "reprocessed_by_vision_llm": item1.get("reprocessed_by_vision_llm", False) 
            })
    
    for idx2, item2 in enumerate(product_items2):
        if idx2 not in matched_item2_indices:
            item2_id_log = item2.get("product_box_id", f"File2-Item{idx2}")
            app.logger.info(f"COMPARE_FN - Unmatched product in File 2: {item2_id_log}")
            comparison_report.append({
                "type": "Unmatched Product in File 2 (Extra)",
                "product_brand": item2.get("product_brand"), "product_name_core": item2.get("product_name_core"),
                "product_variant": item2.get("product_variant_description"), "product_size": item2.get("size_quantity_info"),
                "product_size_normalized": item2.get("size_quantity_info_normalized"),
                "parsed_size_details": item2.get("parsed_size_details"),
                "offer_price": item2.get("offer_price"), "regular_price": item2.get("regular_price"),
                "unit_indicator": item2.get("unit_indicator"), "store_terms": item2.get("store_specific_terms"),
                "validation_flags": item2.get("validation_flags", []),
                "reprocessed_by_vision_llm": item2.get("reprocessed_by_vision_llm", False)
            })
    app.logger.info(f"COMPARE_FN - Comparison finished. Report items: {len(comparison_report)}")
    return comparison_report

# ==================== MAIN API ENDPOINT ====================
@app.route("/upload", methods=["POST"])
def process_uploaded_files_route():
    request_id = f"req_{int(time.time())}" 
    app.logger.info(f"REQUEST_ID: {request_id} - Received request at /upload endpoint.")
    if 'file1' not in request.files or 'file2' not in request.files:
        app.logger.error(f"REQUEST_ID: {request_id} - Missing file1 or file2.")
        return jsonify({"error": "Two files ('file1' and 'file2') are required."}), 400
    
    s3_upload_timestamp = time.time()
    temp_pdf_paths_to_cleanup = []
    all_files_data_for_reprocessing_and_pils = [] 

    try:
        for file_idx_num_zero_based, file_storage_key in enumerate(['file1', 'file2']):
            file_id_log_prefix = f"{request_id}-File{file_idx_num_zero_based+1}"
            file_storage = request.files[file_storage_key]
            filename = secure_filename(file_storage.filename)
            app.logger.info(f"{file_id_log_prefix} - Processing file: {filename}")
            file_bytes = file_storage.read()
            file_storage.seek(0) 
            
            current_file_initial_items = []
            page_pil_images_for_file = []

            if filename.lower().endswith(".pdf"):
                os.makedirs("temp_uploads", exist_ok=True)
                temp_pdf_path = os.path.join("temp_uploads", f"{s3_upload_timestamp}_{file_idx_num_zero_based}_{filename}")
                with open(temp_pdf_path, "wb") as f_pdf: f_pdf.write(file_bytes)
                try:
                    app.logger.info(f"{file_id_log_prefix} - Converting PDF to images (DPI 200)...")
                    page_pil_images_for_file = convert_from_path(temp_pdf_path, dpi=200, poppler_path=POPPLER_BIN_PATH, fmt='jpeg', timeout=300)
                    app.logger.info(f"{file_id_log_prefix} - PDF converted to {len(page_pil_images_for_file)} images.")
                except Exception as e_pdf:
                    app.logger.error(f"{file_id_log_prefix} - PDF conversion error for {filename}: {e_pdf}", exc_info=True)
                    if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
                    return jsonify({"error": f"PDF processing error for {filename}. Details: {str(e_pdf)}"}), 500
                temp_pdf_paths_to_cleanup.append(temp_pdf_path)
            elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
                try: 
                    page_pil_images_for_file = [Image.open(BytesIO(file_bytes))]
                    app.logger.info(f"{file_id_log_prefix} - Image file loaded.")
                except Exception as e_img: 
                    app.logger.error(f"{file_id_log_prefix} - Invalid image file {filename}: {str(e_img)}")
                    return jsonify({"error": f"Invalid image file {filename}: {str(e_img)}"}), 400
            else: 
                app.logger.error(f"{file_id_log_prefix} - Unsupported file type for {filename}")
                return jsonify({"error": f"Unsupported file type for {filename}"}), 400

            s3_keys_for_this_file = []
            for page_idx, page_image_pil in enumerate(page_pil_images_for_file):
                page_id_log = f"{file_id_log_prefix}-Page{page_idx}"
                app.logger.info(f"{page_id_log} - Starting initial processing pass...")
                image_width_px, image_height_px = page_image_pil.size
                app.logger.debug(f"{page_id_log} - Image dimensions: {image_width_px}x{image_height_px}")
                
                app.logger.info(f"{page_id_log} - Getting Roboflow predictions...")
                roboflow_preds = get_roboflow_predictions_sdk(page_image_pil, f"{filename}_p{page_idx}")
                if not roboflow_preds: 
                    app.logger.warning(f"{page_id_log} - No Roboflow predictions. Skipping page.")
                    continue
                app.logger.info(f"{page_id_log} - Roboflow returned {len(roboflow_preds)} predictions.")

                img_byte_arr_s3 = BytesIO(); page_image_pil.save(img_byte_arr_s3, format='JPEG')
                s3_page_key = f"pages/{s3_upload_timestamp}_{file_idx_num_zero_based}_p{page_idx}_{secure_filename(filename)}.jpg"
                uploaded_s3_key = upload_to_s3(BytesIO(img_byte_arr_s3.getvalue()), S3_BUCKET_NAME, s3_page_key)
                if not uploaded_s3_key: 
                    app.logger.error(f"{page_id_log} - Failed to upload page image to S3. Skipping page.")
                    continue
                s3_keys_for_this_file.append(uploaded_s3_key)

                app.logger.info(f"{page_id_log} - Starting Textract analysis for S3 key: {uploaded_s3_key}")
                textract_blocks = get_analysis_from_document_via_textract(S3_BUCKET_NAME, uploaded_s3_key)
                if not textract_blocks: 
                    app.logger.error(f"{page_id_log} - Textract analysis failed or returned no blocks. Skipping page.")
                    delete_from_s3(S3_BUCKET_NAME, uploaded_s3_key) 
                    continue
                app.logger.info(f"{page_id_log} - Textract returned {len(textract_blocks)} blocks.")
                blocks_map = {b['Id']: b for b in textract_blocks}

                app.logger.info(f"{page_id_log} - Collating text for {len(roboflow_preds)} Roboflow boxes...")
                collated_snippets = collate_text_for_product_boxes(roboflow_preds, textract_blocks, blocks_map, image_width_px, image_height_px, page_id_for_log=page_id_log)
                
                for snippet_idx, snippet in enumerate(collated_snippets):
                    item_id_for_log = snippet.get("product_box_id", f"{page_id_log}-Snip{snippet_idx}") 
                    app.logger.info(f"ITEM_ID: {item_id_for_log} - Processing snippet with Text LLM...")
                    llm_output = extract_product_data_with_llm(snippet["collated_text"], item_id_for_log=item_id_for_log)
                    price_candidates_for_snippet = snippet.get("price_candidates", [])
                    
                    app.logger.info(f"ITEM_ID: {item_id_for_log} - Post-processing and validating Text LLM output...")
                    processed_item = post_process_and_validate_item_data(llm_output, price_candidates_for_snippet, snippet["collated_text"], item_id_for_log=item_id_for_log)
                    
                    current_file_initial_items.append({
                        "product_box_id": item_id_for_log, 
                        "original_filename": filename, 
                        "page_idx_for_reprocessing": page_idx,
                        "roboflow_box_coords_pixels_center_wh": snippet.get("roboflow_box_coords_pixels_center_wh"),
                        "initial_price_candidates": price_candidates_for_snippet, 
                        "original_collated_text": snippet.get("collated_text"),
                        "roboflow_class_name": snippet.get("class_name", "UnknownClass"), 
                        **processed_item 
                    })
                    app.logger.info(f"ITEM_ID: {item_id_for_log} - Initial processing complete. Offer: {processed_item.get('offer_price')}, Regular: {processed_item.get('regular_price')}")
            
            for s3_key in s3_keys_for_this_file: 
                delete_from_s3(S3_BUCKET_NAME, s3_key)
            
            all_files_data_for_reprocessing_and_pils.append({
                "file_id_log_prefix": file_id_log_prefix, 
                "filename": filename, 
                "page_pils_list": page_pil_images_for_file, 
                "items": current_file_initial_items
            })
        
        app.logger.info(f"REQUEST_ID: {request_id} - Initial processing pass complete for all files.")
        app.logger.info(f"REQUEST_ID: {request_id} - Starting Vision LLM re-processing stage for flagged items...")
        final_product_items_file1, final_product_items_file2 = [], []

        for file_idx, file_data_obj in enumerate(all_files_data_for_reprocessing_and_pils):
            current_file_final_items = []
            file_log_prefix = file_data_obj["file_id_log_prefix"]
            app.logger.info(f"{file_log_prefix} - Vision re-processing items from: {file_data_obj['filename']}")
            
            for item_idx, item_to_evaluate in enumerate(file_data_obj["items"]):
                item_id_for_log = item_to_evaluate.get("product_box_id", f"{file_log_prefix}-Item{item_idx}-VisionEval")
                app.logger.debug(f"ITEM_ID: {item_id_for_log} - Evaluating for Vision LLM. Current Data: {json.dumps(item_to_evaluate, indent=2, ensure_ascii=False)}")

                flags = item_to_evaluate.get("validation_flags", [])
                offer_price = item_to_evaluate.get("offer_price")
                original_text = item_to_evaluate.get("original_collated_text", "")
                roboflow_class = item_to_evaluate.get("roboflow_class_name", "UnknownClass") 
                
                price_candidates_for_current_item = item_to_evaluate.get("initial_price_candidates", [])
                current_item_best_offer_candidate = None
                if price_candidates_for_current_item:
                    valid_price_cands = [pc for pc in price_candidates_for_current_item if pc.get('parsed_value') is not None and pc.get('parsed_value') >= 0]
                    offer_cands = sorted([pc for pc in valid_price_cands if not pc.get('is_regular_candidate', False)], key=lambda c: (c.get('source_block_id') == 'GEOMETRIC_MERGE', -c.get('pixel_height', 0)))
                    if offer_cands: current_item_best_offer_candidate = offer_cands[0]
                
                send_to_vision = False
                vision_reason = "No specific trigger"

                critical_price_error_patterns = [
                    "XYZ->0.XY error", "price seems too low, corrected to prominent candidate",
                    "Correcting LLM offer price", "Correcting LLM regular price",
                    "Multi-buy pattern.*failed", "differs from prominent visual candidate",
                    "Offer price .* populated from visually prominent candidate", 
                    "Regular price .* populated from visual candidate"
                ]
                critical_price_error_flag_found = any(any(pattern in flag for pattern in critical_price_error_patterns) for flag in flags)
                
                price_suspiciously_low_text_pipeline = False
                if offer_price is not None:
                    try: price_suspiciously_low_text_pipeline = (0.01 <= float(offer_price) < 1.00)
                    except (ValueError, TypeError): pass
                
                potential_X_YZ_in_text = re.search(r'\b([1-9])\s*(\d{2})\b', original_text) or \
                                         re.search(r'\b([1-9]\d{2})\b', original_text) 

                if item_to_evaluate.get("error_message"):
                    send_to_vision = True; vision_reason = f"Text LLM error: {item_to_evaluate.get('error_message')}"
                elif critical_price_error_flag_found:
                    send_to_vision = True; vision_reason = f"Critical price error flags found: {flags}"
                elif price_suspiciously_low_text_pipeline and potential_X_YZ_in_text and not any(kw in original_text.lower() for kw in ["limit", "max"]):
                    send_to_vision = True; vision_reason = f"Suspiciously low price ${offer_price} from text pipeline, but original text has pattern '{potential_X_YZ_in_text.group(0) if potential_X_YZ_in_text else 'N/A'}'"
                
                # Vision Trigger for NULL price
                elif offer_price is None and roboflow_class.lower() == "product_item": 
                     send_to_vision = True; vision_reason = f"Null offer price from text pipeline for a Roboflow '{roboflow_class}' item."
                
                elif item_to_evaluate.get("product_name_core") is None and offer_price is not None: 
                     send_to_vision = True; vision_reason = "Product has price but no name from Text LLM."
                elif offer_price is not None and current_item_best_offer_candidate and \
                     current_item_best_offer_candidate.get('parsed_value') is not None and \
                     abs(float(offer_price) - float(current_item_best_offer_candidate.get('parsed_value', 0))) > 1.50 : 
                     send_to_vision = True; vision_reason = f"Large price discrepancy (>${1.50}) between Text LLM price ${offer_price} and visual candidate ${current_item_best_offer_candidate.get('parsed_value')}"
                
                # AGGRESSIVE VISION TRIGGER for "missing dollar digit"
                if not send_to_vision and offer_price is not None: 
                    try:
                        offer_price_float = float(offer_price)
                        if 0.01 <= offer_price_float < 1.00: 
                            cents_part_str = str(int(round((offer_price_float * 100) % 100))).zfill(2)
                            if re.match(r'\s*' + re.escape(cents_part_str), original_text): 
                                has_stronger_visual_candidate_for_missing_dollar = False
                                if price_candidates_for_current_item:
                                    for pc_cand_vision_trigger in price_candidates_for_current_item:
                                        pc_text_vt = pc_cand_vision_trigger.get('text_content','').strip()
                                        if re.fullmatch(r'[1-9]\s*\d{2}', pc_text_vt) or re.fullmatch(r'[1-9]\d{2}', pc_text_vt):
                                            parsed_strong_cand_vt = parse_price_string(pc_text_vt, item_id_for_log + "-strongcandVT")
                                            if parsed_strong_cand_vt and parsed_strong_cand_vt >= 1.00:
                                                has_stronger_visual_candidate_for_missing_dollar = True
                                                app.logger.debug(f"ITEM_ID: {item_id_for_log} - AGGRESSIVE Vision Trigger Check: Found stronger visual candidate '{pc_text_vt}' ({parsed_strong_cand_vt}) for low price {offer_price}")
                                                break
                                
                                if has_stronger_visual_candidate_for_missing_dollar or not any(kw in original_text.lower() for kw in ["limit", "coupon", "max", "por solo", "a solo", "menos de"]):
                                    send_to_vision = True
                                    vision_reason = f"AGGRESSIVE: Suspiciously low offer price ${offer_price} (collated text starts with '{cents_part_str}'). Potential missing dollar digit. Stronger visual candidate indicating missing dollar: {has_stronger_visual_candidate_for_missing_dollar}"
                                    app.logger.info(f"ITEM_ID: {item_id_for_log} - AGGRESSIVE VISION TRIGGER activated: {vision_reason}")
                    except (ValueError, TypeError) as e_agg_vision:
                        app.logger.warning(f"ITEM_ID: {item_id_for_log} - Error during aggressive vision trigger check for offer_price {offer_price}: {e_agg_vision}")


                if send_to_vision and item_to_evaluate.get("roboflow_box_coords_pixels_center_wh"):
                    app.logger.info(f"ITEM_ID: {item_id_for_log} - Sending to Vision LLM. Reason: {vision_reason}. Initial Flags: {flags}")
                    page_idx_reproc = item_to_evaluate["page_idx_for_reprocessing"]
                    page_image_pil_reproc = file_data_obj["page_pils_list"][page_idx_reproc]
                    segment_bytes = get_segment_image_bytes(page_image_pil_reproc, item_to_evaluate["roboflow_box_coords_pixels_center_wh"], item_id_for_log=item_id_for_log)

                    if segment_bytes:
                        name_hint_vision = item_to_evaluate.get("product_name_core") or item_to_evaluate.get("product_brand")
                        vision_llm_output = re_extract_with_vision_llm(segment_bytes, item_id_for_log=item_id_for_log, original_item_name=name_hint_vision)
                        
                        if "error_message" not in vision_llm_output:
                            parsed_op_v = parse_price_string(vision_llm_output.get("offer_price"), item_id_for_log=f"{item_id_for_log}-vision_offer")
                            parsed_rp_v = parse_price_string(vision_llm_output.get("regular_price"), item_id_for_log=f"{item_id_for_log}-vision_regular")
                            vision_llm_output["offer_price"] = parsed_op_v
                            vision_llm_output["regular_price"] = parsed_rp_v
                            
                            vision_item_processed = post_process_and_validate_item_data(vision_llm_output, [], "", item_id_for_log=f"{item_id_for_log}-vision_processed")
                            
                            final_item_after_vision = item_to_evaluate.copy() 
                            for key in ["offer_price", "regular_price", "product_brand", "product_name_core", 
                                        "product_variant_description", "size_quantity_info", "unit_indicator", 
                                        "store_specific_terms", "parsed_size_details", "size_quantity_info_normalized"]:
                                if key in vision_item_processed and vision_item_processed[key] is not None: 
                                    final_item_after_vision[key] = vision_item_processed[key]
                            
                            final_item_after_vision["validation_flags"] = vision_item_processed.get("validation_flags", []) 
                            final_item_after_vision["validation_flags"].append("Reprocessed with Vision LLM.")
                            final_item_after_vision["reprocessed_by_vision_llm"] = True 
                            current_file_final_items.append(final_item_after_vision)
                            app.logger.info(f"ITEM_ID: {item_id_for_log} - Successfully re-processed by Vision LLM. New Offer: {final_item_after_vision.get('offer_price')}, New Regular: {final_item_after_vision.get('regular_price')}")
                            continue 
                        else:
                            app.logger.warning(f"ITEM_ID: {item_id_for_log} - Vision LLM re-extraction error: {vision_llm_output.get('error_message')}. Keeping text-pipeline data.")
                            item_to_evaluate["validation_flags"].append(f"Vision LLM failed: {vision_llm_output.get('error_message')}")
                    else:
                        app.logger.warning(f"ITEM_ID: {item_id_for_log} - Could not get segment image for Vision LLM. Keeping text-pipeline data.")
                        item_to_evaluate["validation_flags"].append("Vision LLM skipped: Could not create segment image.")
                else:
                     app.logger.debug(f"ITEM_ID: {item_id_for_log} - Not sent to Vision LLM. Reason: {vision_reason if send_to_vision else 'No trigger met'}.")


                current_file_final_items.append(item_to_evaluate) 
            
            if file_idx == 0: final_product_items_file1 = current_file_final_items
            else: final_product_items_file2 = current_file_final_items
        
        app.logger.info(f"REQUEST_ID: {request_id} - Vision LLM re-processing stage complete.")
        app.logger.info(f"REQUEST_ID: {request_id} - Final items for File 1 ({len(final_product_items_file1)}):")
        for idx, item_debug in enumerate(final_product_items_file1):
            app.logger.debug(f"REQUEST_ID: {request_id} - File1 Final Item {idx} ({item_debug.get('product_box_id')}): {json.dumps(item_debug, indent=2, ensure_ascii=False)}")
        
        app.logger.info(f"REQUEST_ID: {request_id} - Final items for File 2 ({len(final_product_items_file2)}):")
        for idx, item_debug in enumerate(final_product_items_file2):
            app.logger.debug(f"REQUEST_ID: {request_id} - File2 Final Item {idx} ({item_debug.get('product_box_id')}): {json.dumps(item_debug, indent=2, ensure_ascii=False)}")


        app.logger.info(f"REQUEST_ID: {request_id} - Starting final product comparison.")
        product_centric_comparison_report = compare_product_items(final_product_items_file1, final_product_items_file2)
        
        final_response = {
            "message": "Phase 3 with Vision LLM re-processing and detailed logging. Comparison performed.",
            "product_items_file1_count": len(final_product_items_file1),
            "product_items_file2_count": len(final_product_items_file2),
            "product_comparison_details": product_centric_comparison_report,
        }
        
        report_lines_for_csv = []
        for diff_item in product_centric_comparison_report:
            p1_reprocessed = False
            val_flags1 = diff_item.get("validation_flags1", diff_item.get("validation_flags", [])) 
            if any("Reprocessed with Vision LLM" in f for f in val_flags1):
                p1_reprocessed = True
            
            p2_reprocessed = False
            val_flags2 = diff_item.get("validation_flags2", [])
            if any("Reprocessed with Vision LLM" in f for f in val_flags2):
                p2_reprocessed = True
            if diff_item.get("type") == "Unmatched Product in File 2 (Extra)" and diff_item.get("reprocessed_by_vision_llm"):
                 p2_reprocessed = True


            line = {
                "Comparison_Type": diff_item.get("type", "N/A"),
                "P1_Brand": diff_item.get("product1_brand", diff_item.get("product_brand", "")),
                "P1_Name_Core": diff_item.get("product1_name_core", diff_item.get("product_name_core", "")),
                "P1_Offer_Price": str(diff_item.get("offer_price1", diff_item.get("offer_price", ""))),
                "P1_Regular_Price": str(diff_item.get("regular_price1", diff_item.get("regular_price", ""))),
                "P1_Size_Norm": diff_item.get("product1_size_normalized", diff_item.get("product_size_normalized", "")),
                "P1_Val_Flags": "; ".join(val_flags1),
                "P1_Vision_Reprocessed": p1_reprocessed,

                "P2_Brand": diff_item.get("product2_brand", ""),
                "P2_Name_Core": diff_item.get("product2_name_core", ""),
                "P2_Offer_Price": str(diff_item.get("offer_price2", "")),
                "P2_Regular_Price": str(diff_item.get("regular_price2", "")),
                "P2_Size_Norm": diff_item.get("product2_size_normalized", ""),
                "P2_Val_Flags": "; ".join(val_flags2),
                "P2_Vision_Reprocessed": p2_reprocessed,

                "Similarity_Percent": diff_item.get("text_similarity_percent", ""),
                "Differences": diff_item.get("differences", "")
            }
            report_lines_for_csv.append(line)
        
        report_df = pd.DataFrame(report_lines_for_csv if report_lines_for_csv else [{"Comparison_Type": "No items to compare."}])
        csv_buffer = io.StringIO(); report_df.to_csv(csv_buffer, index=False, lineterminator='\r\n') 
        final_response["report_csv_data"] = csv_buffer.getvalue()
        
        app.logger.info(f"REQUEST_ID: {request_id} - Processing complete. Returning response.")
        return jsonify(final_response), 200
        
    except Exception as e_global:
        app.logger.error(f"REQUEST_ID: {request_id} - Global error in /upload: {str(e_global)}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during processing. Check logs for request ID: " + request_id}), 500
    finally:
        for temp_path in temp_pdf_paths_to_cleanup:
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except Exception as e_del_final: app.logger.error(f"REQUEST_ID: {request_id} - Error deleting temp PDF {temp_path} in final cleanup: {e_del_final}")
        app.logger.info(f"REQUEST_ID: {request_id} - /upload endpoint finished.")

# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    # debug_env_val = os.getenv('FLASK_DEBUG', '0').lower() # Temporarily override for testing
    # debug_mode = debug_env_val in ['1', 'true', 'on', 'yes']
    debug_mode = True # <--- FORCE DEBUG MODE FOR THIS TEST

    if debug_mode:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO 
    
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - PID:%(process)d - [%(funcName)s:%(lineno)d] - %(message)s')
    
    app.logger.setLevel(log_level) # Ensure app's logger is set
    # Also critical: ensure handlers are also set to DEBUG
    if app.logger.handlers: # If Flask already added handlers
        for handler in app.logger.handlers:
            handler.setLevel(log_level)
            handler.setFormatter(log_formatter) # Optional: ensure format consistency
    else: # If no handlers, add one (e.g. when not using Flask's default run)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(log_formatter)
        app.logger.addHandler(stream_handler)

    # Ensure root logger doesn't suppress Flask's logger if it's more restrictive
    # For other libraries, keep them at INFO or WARNING if too verbose
    logging.getLogger().setLevel(logging.DEBUG) # Set root logger to DEBUG to be permissive
    logging.getLogger('werkzeug').setLevel(logging.INFO) # Werkzeug can be verbose
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('pdf2image').setLevel(logging.INFO)

    app.logger.info(f"Flask app starting. DEBUG MODE FORCED. Effective Log level for app: {logging.getLevelName(app.logger.getEffectiveLevel())}")
    os.makedirs("temp_uploads", exist_ok=True)
    app.run(debug=debug_mode, host='0.0.0.0', port=port, use_reloader=debug_mode)
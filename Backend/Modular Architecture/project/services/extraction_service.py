# project/services/extraction_service.py
import logging
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List

# Assuming utils are in a directory relative to the project root
from utils.price_utils import parse_price_string
from utils.validation import validate_price_pair

logger = logging.getLogger(__name__)

class ExtractionServiceInterface(ABC):
    @abstractmethod
    def extract_product_data_with_llm(self, product_snippet_text: str, llm_model: str = "gpt-4o") -> Dict[str, Any]:
        pass

    @abstractmethod
    def normalize_product_data(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

class ExtractionService(ExtractionServiceInterface):
    def __init__(self, openai_client):
        self.openai_client = openai_client
        if not self.openai_client:
            logger.error("OpenAI client not initialized for ExtractionService.")
            # Depending on how critical this is, you might raise an error or allow graceful failure
            # For now, methods will check for self.openai_client
        logger.info("ExtractionService initialized.")

    def extract_product_data_with_llm(self, product_snippet_text: str, llm_model: str = "gpt-4o") -> Dict[str, Any]:
        if not self.openai_client:
            logger.error("OpenAI client is not initialized. Cannot make LLM call for product data extraction.")
            return {"error_message": "OpenAI client not initialized", "llm_call_failed": True}

        if not product_snippet_text or not product_snippet_text.strip():
            logger.warning("Empty product snippet text received for LLM extraction.")
            return {"error_message": "Empty product snippet", "llm_call_failed": True}


        logger.info(f"Sending snippet to LLM ({llm_model}) for product data extraction:\n---\n{product_snippet_text}\n---")

        system_prompt = """You are an expert at extracting product information from advertisement text. Extract the following fields:

CRITICAL RULES FOR PRICE EXTRACTION:
1. For 3-digit numbers like "897", "647", "447" - these represent prices like 8.97, 6.47, 4.47 respectively.
2. For 4-digit numbers like "1097", "1249" - these represent prices like 10.97, 12.49 respectively.
3. The "offer_price" is usually the most prominent price, often appearing first or largest in the snippet.
4. The "regular_price" usually appears after a keyword like "Regular", "Reg.", "Antes", "Normal".
5. Prices should generally NOT exceed 100.00 for typical grocery or household items. If you see a number like "2999", it's more likely $29.99 than $2,999.00 unless context strongly suggests otherwise (e.g., electronics). Assume grocery context.
6. If a price is formatted as "2x$5.00" or "2/$5.00", the unit offer price is $2.50. Extract the per-unit price. If there's a coupon like "*Cupón al Instante: - $0.50 = 2x $4.50", the final per unit offer price is $2.25. Calculate this final per-unit price.

Fields to extract:
- "offer_price": The sale/promotional price (main large price shown), per unit. (Numeric, e.g., 8.97)
- "regular_price": The original price, per unit. (Numeric, e.g., 10.49)
- "product_brand": The brand name of the product. (String)
- "product_name_core": The main name of the product, often the second line of text. (String)
- "product_variant_description": Descriptive text for the product variant, e.g., flavor, type, specific model. Often the third line. (String)
- "size_quantity_info": Information about size, weight, volume, or count (e.g., "105 a 117 onzas", "21 onzas", "Pack of 6"). (String)
- "unit_indicator": Any unit indicators like "c/u", "ea", "lb", "kg", if present near the price. (String)
- "store_specific_terms": Any store-specific terms, purchase limits, or coupon details (e.g., "*24 por tienda", "*Cupón: 2x $5.00 - $0.50 = 2x $4.50"). (String)

IMPORTANT:
- Convert 3-digit prices correctly: e.g., 897 -> 8.97.
- Convert 4-digit prices correctly: e.g., 1097 -> 10.97.
- If you see prices over 100, double-check your interpretation. They are rare for these items.
- Return prices as decimal numbers (e.g., 8.97, not "8.97").
- Return ONLY a JSON object with these fields. Use null for fields that are not found or not applicable. Do not add any explanatory text outside the JSON.
"""
        few_shot_examples = [
            {"role": "user", "content": "Text:\n897 c/u\nAce Simply\nDetergente Líquido 105 a 117 onzas\nRegular $10.49 c/u\n*24 por tienda\n\nReturn the extracted information strictly as a JSON object."},
            {"role": "assistant", "content": """{
"offer_price": 8.97,
"regular_price": 10.49,
"product_brand": "Ace",
"product_name_core": "Ace Simply",
"product_variant_description": "Detergente Líquido 105 a 117 onzas",
"size_quantity_info": "105 a 117 onzas",
"unit_indicator": "c/u",
"store_specific_terms": "*24 por tienda"
}"""},
            {"role": "user", "content": "Text:\n647 c/u\nPurex Crystals\nVariedad 21 onzas\nRegular $7.29 c/u\n*24 por tienda\n\nReturn the extracted information strictly as a JSON object."},
            {"role": "assistant", "content": """{
"offer_price": 6.47,
"regular_price": 7.29,
"product_brand": "Purex",
"product_name_core": "Purex Crystals",
"product_variant_description": "Variedad 21 onzas",
"size_quantity_info": "21 onzas",
"unit_indicator": "c/u",
"store_specific_terms": "*24 por tienda"
}"""},
            {"role": "user", "content": "Text:\n1097 c/u\nWoolite\nDetergente Líquido 50 onzas\nRegular $11.99 c/u\n*24 por tienda\n\nReturn the extracted information strictly as a JSON object."},
            {"role": "assistant", "content": """{
"offer_price": 10.97,
"regular_price": 11.99,
"product_brand": "Woolite",
"product_name_core": "Woolite",
"product_variant_description": "Detergente Líquido 50 onzas",
"size_quantity_info": "50 onzas",
"unit_indicator": "c/u",
"store_specific_terms": "*24 por tienda"
}"""},
            {"role": "user", "content": "Text:\n2x $5.00 *Cupón al Instante: - $0.50 = 2x $4.50\nRaid\nMata Mosquitos 11 onzas\nRegular $2.99 c/u\n*24 por tienda\n\nReturn the extracted information strictly as a JSON object."},
            {"role": "assistant", "content": """{
"offer_price": 2.25,
"regular_price": 2.99,
"product_brand": "Raid",
"product_name_core": "Raid",
"product_variant_description": "Mata Mosquitos 11 onzas",
"size_quantity_info": "11 onzas",
"unit_indicator": "c/u",
"store_specific_terms": "*24 por tienda *Cupón al Instante: - $0.50 = 2x $4.50"
}"""},
            {"role": "user", "content": "Text:\nRegular $3.19 c/u a 3.39 c/u\n*24 por tienda\n\nReturn the extracted information strictly as a JSON object."},
            {"role": "assistant", "content": """{
"offer_price": null,
"regular_price": 3.19,
"product_brand": null,
"product_name_core": null,
"product_variant_description": null,
"size_quantity_info": null,
"unit_indicator": "c/u",
"store_specific_terms": "*24 por tienda"
}"""},
            {"role": "user", "content": "Text:\nDowny\nSuavizante Unstopables\nVariedad 14.8oz\n2/$12\nReg $7.99ea\n\nReturn the extracted information strictly as a JSON object."},
            {"role": "assistant", "content": """{
"offer_price": 6.00,
"regular_price": 7.99,
"product_brand": "Downy",
"product_name_core": "Downy Unstopables",
"product_variant_description": "Suavizante Variedad 14.8oz",
"size_quantity_info": "14.8oz",
"unit_indicator": "ea",
"store_specific_terms": "2/$12"
}"""}
        ]

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(few_shot_examples)
        messages.append({
            "role": "user",
            "content": f"Text:\n{product_snippet_text}\n\nReturn the extracted information strictly as a JSON object."
        })

        max_retries = 3
        for attempt in range(max_retries):
            try:
                chat_completion = self.openai_client.chat.completions.create(
                    model=llm_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1, # Low temperature for deterministic output
                    # max_tokens=500 # Optional: to control response length
                )
                response_content = chat_completion.choices[0].message.content
                logger.debug(f"LLM Raw Response (Attempt {attempt+1}):\n{response_content}")

                try:
                    extracted_data = json.loads(response_content)
                except json.JSONDecodeError as json_e:
                    logger.error(f"LLM response (Attempt {attempt+1}) not valid JSON: {json_e}. Response: '{response_content}'")
                    if attempt == max_retries - 1:
                        return {"error_message": f"LLM response JSON decode error after {max_retries} attempts.", "llm_response": response_content, "llm_call_failed": True}
                    time.sleep(1) # Wait before retrying
                    continue # Retry LLM call

                # Post-process prices using the robust utility
                offer_price_raw = extracted_data.get('offer_price')
                regular_price_raw = extracted_data.get('regular_price')

                extracted_data['offer_price'] = parse_price_string(offer_price_raw)
                extracted_data['regular_price'] = parse_price_string(regular_price_raw)
                
                logger.debug(f"Prices after parsing: Offer='{extracted_data['offer_price']}' (Raw='{offer_price_raw}'), Regular='{extracted_data['regular_price']}' (Raw='{regular_price_raw}')")


                # Validate price relationship
                if extracted_data.get('offer_price') is not None and extracted_data.get('regular_price') is not None:
                    op, rp = validate_price_pair(
                        extracted_data['offer_price'],
                        extracted_data['regular_price']
                    )
                    extracted_data['offer_price'] = op
                    extracted_data['regular_price'] = rp
                    logger.debug(f"Prices after validation: Offer='{op}', Regular='{rp}'")


                # Ensure all expected fields exist, fill with None if missing
                expected_fields = [
                    "product_brand", "product_name_core", "product_variant_description",
                    "size_quantity_info", "offer_price", "regular_price",
                    "unit_indicator", "store_specific_terms"
                ]
                for field in expected_fields:
                    if field not in extracted_data:
                        extracted_data[field] = None
                
                extracted_data["llm_call_succeeded"] = True # Mark success
                return extracted_data

            except Exception as e: # Catch OpenAI API errors or other unexpected issues
                logger.error(f"Error calling LLM API (Attempt {attempt+1}): {e}", exc_info=True)
                if attempt == max_retries - 1:
                    return {"error_message": str(e), "llm_call_failed": True}
                time.sleep(1) # Wait before retrying
        
        # Should not be reached if retries are handled correctly, but as a fallback:
        return {"error_message": "LLM extraction failed after multiple retries.", "llm_call_failed": True}


    def normalize_product_data(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic normalization for product data, especially size and brand."""
        if not product_data or product_data.get("llm_call_failed"): # Don't normalize if extraction failed
            return product_data

        normalized = product_data.copy()
        logger.debug(f"Normalizing data: {product_data}")

        # Normalize size/quantity info
        size_info = normalized.get('size_quantity_info')
        if size_info and isinstance(size_info, str):
            s_info_lower = size_info.lower()
            # Basic replacements (expand as needed)
            replacements = {
                ' onzas': 'oz', ' onza': 'oz', 'onzas': 'oz', 'onza': 'oz', # Spanish to common
                ' libras': 'lb', ' libra': 'lb', 'libras': 'lb', 'libra': 'lb',
                ' galones': 'gal', ' galon': 'gal', 'galones': 'gal', 'galon': 'gal',
                'gramos': 'g', 'gramo': 'g',
                'litros': 'L', 'litro': 'L',
                # Add more common variations
            }
            for old, new in replacements.items():
                s_info_lower = s_info_lower.replace(old, new)
            
            # Remove spaces around units for consistency e.g. "14.8 oz" -> "14.8oz"
            s_info_lower = re.sub(r'\s*([a-zA-Z]+)\s*$', r'\1', s_info_lower) # Trim trailing spaces from unit
            s_info_lower = re.sub(r'(\d+)\s+([a-zA-Z]+)', r'\1\2', s_info_lower) # Remove space between number and unit

            normalized['size_quantity_info_normalized'] = s_info_lower.strip()
        else:
            normalized['size_quantity_info_normalized'] = size_info # Keep as is if not string or None

        # Normalize brand names (example, can be expanded or use a more sophisticated method)
        brand = normalized.get('product_brand')
        if brand and isinstance(brand, str):
            brand_lower = brand.lower().strip()
            brand_corrections = {
                'downy': 'Downy', 'gain': 'Gain', 'tide': 'Tide', 'glad': 'Glad',
                'scott': 'Scott', 'raid': 'Raid', 'ace': 'Ace', 'purex': 'Purex',
                'clorox': 'Clorox', 'bounty': 'Bounty', 'lysol': 'Lysol',
                'glade': 'Glade', 'airwick': 'Air Wick', 'air wick': 'Air Wick',
                'reynolds': 'Reynolds', 'lestoil': 'Lestoil', 'ensueño': 'Ensueño',
                'ensueno': 'Ensueño', 'woolite': 'Woolite', 'suavitel': 'Suavitel',
                'rocio': 'Rocio', 'real kill': 'Real Kill',
                # Add more common misspellings or variations
            }
            # Use corrected version if found, otherwise capitalize the original
            normalized['product_brand'] = brand_corrections.get(brand_lower, brand.title())
        
        logger.debug(f"Data after normalization: {normalized}")
        return normalized

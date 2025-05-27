# project/utils/price_utils.py
import re
import logging

logger = logging.getLogger(__name__)

def parse_price_string(price_str):
    """Enhanced price parsing with better handling of various formats"""
    if price_str is None or price_str == "":
        return None

    if isinstance(price_str, (int, float)):
        return float(price_str)

    price_str = str(price_str).strip()
    price_str = re.sub(r'[$¢€£¥]', '', price_str) # More currency symbols
    price_str = re.sub(r'\b(cada uno|c/u|cu|each|ea)\b', '', price_str, flags=re.IGNORECASE)
    price_str = price_str.strip()

    if not price_str: # Check if string became empty after cleaning
        return None

    # Handle 3-digit prices (897 -> 8.97)
    if re.fullmatch(r'^\d{3}$', price_str):
        logger.debug(f"Parsing 3-digit price: {price_str} -> {price_str[0]}.{price_str[1:3]}")
        return float(f"{price_str[0]}.{price_str[1:3]}")

    # Handle 4-digit prices (1049 -> 10.49)
    if re.fullmatch(r'^\d{4}$', price_str):
        logger.debug(f"Parsing 4-digit price: {price_str} -> {price_str[:2]}.{price_str[2:4]}")
        return float(f"{price_str[:2]}.{price_str[2:4]}")

    # Handle prices with spaces (8 97 -> 8.97)
    space_match = re.fullmatch(r'^(\d+)\s+(\d{2})$', price_str)
    if space_match:
        parsed_val = float(f"{space_match.group(1)}.{space_match.group(2)}")
        logger.debug(f"Parsing space-separated price: {price_str} -> {parsed_val}")
        return parsed_val

    # Handle normal decimal prices (e.g., 8.97, 10,49)
    # Allow comma as decimal separator as well
    decimal_match = re.search(r'(\d+)[.,](\d{2})\b', price_str) # \b to avoid matching parts of larger numbers
    if decimal_match:
        parsed_val = float(f"{decimal_match.group(1)}.{decimal_match.group(2)}")
        logger.debug(f"Parsing decimal price: {price_str} -> {parsed_val}")
        return parsed_val

    # Handle whole numbers (e.g., 8, 10)
    whole_match = re.fullmatch(r'^(\d+)$', price_str)
    if whole_match:
        num = int(whole_match.group(1))
        # If it's a reasonable price (under 100), return as is, otherwise it might be a misparsed 3/4 digit number
        if num < 100:
            logger.debug(f"Parsing whole number price: {price_str} -> {float(num)}")
            return float(num)
        elif 100 <= num <= 999: # Treat as 3-digit if it's in the hundreds. e.g. 897
             logger.debug(f"Re-parsing whole number {num} as 3-digit price: {str(num)[0]}.{str(num)[1:3]}")
             return float(f"{str(num)[0]}.{str(num)[1:3]}")
        elif 1000 <= num <= 9999: # Treat as 4-digit if it's in the thousands. e.g. 1049
             logger.debug(f"Re-parsing whole number {num} as 4-digit price: {str(num)[:2]}.{str(num)[2:4]}")
             return float(f"{str(num)[:2]}.{str(num)[2:4]}")


    logger.warning(f"Could not parse price from string: '{price_str}'")
    return None


def extract_price_from_text(text: str, is_offer_price: bool = True):
    """Extract price from text with context awareness"""
    if not text:
        return None

    # For offer prices, look for leading numbers
    if is_offer_price:
        # Check for patterns like "897 c/u" or "8.97 c/u" or "897" or "1049"
        match = re.search(r'^(\d{3,4}|\d+[\.,]\d{1,2}|\d+)\s*(c/u|cu|ea)?', text.strip(), re.IGNORECASE)
        if match:
            return parse_price_string(match.group(1))

    # For regular prices, look after "Regular" keyword
    else:
        match = re.search(r'Regular\s*\$?\s*(\d{3,4}|\d+[\.,]\d{1,2}|\d+)', text, re.IGNORECASE)
        if match:
            return parse_price_string(match.group(1))

    # Fallback to general price extraction if specific patterns fail
    # This is a bit aggressive and might pick up other numbers. Context is key.
    price_matches = re.findall(r'(\d{3,4}|\d+[\.,]\d{2})', text) # Prefer 3-4 digit numbers or explicit decimals
    if not price_matches:
        price_matches = re.findall(r'(\d+)', text) # Simpler numbers if above fails

    if price_matches:
        # Try parsing the first found match. This might need more sophisticated logic
        # depending on how noisy the text is.
        for p_match in price_matches:
            parsed = parse_price_string(p_match)
            if parsed is not None:
                return parsed
    return None
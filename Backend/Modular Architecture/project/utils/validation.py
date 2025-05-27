# project/utils/validation.py
import logging

logger = logging.getLogger(__name__)

def validate_price_pair(offer_price, regular_price):
    """Validate that offer price is less than regular price. Swaps if necessary."""
    if offer_price is None or regular_price is None:
        return offer_price, regular_price

    try:
        offer_p = float(offer_price)
        regular_p = float(regular_price)

        if offer_p > regular_p:
            logger.warning(f"Swapping prices: offer {offer_p} > regular {regular_p}. Original: offer='{offer_price}', regular='{regular_price}'")
            return regular_price, offer_price # Return original format/type
        
        return offer_price, regular_price
    except (ValueError, TypeError) as e:
        logger.error(f"Could not validate price pair due to type error: offer='{offer_price}', regular='{regular_price}'. Error: {e}")
        return offer_price, regular_price # Return as is if conversion fails
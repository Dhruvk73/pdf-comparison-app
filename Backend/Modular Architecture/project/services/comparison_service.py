# project/services/comparison_service.py
import logging
from abc import ABC, abstractmethod
from fuzzywuzzy import fuzz # Keep using fuzzywuzzy for this phase
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ComparisonServiceInterface(ABC):
    @abstractmethod
    def compare_product_items(
        self,
        product_items1: List[Dict[str, Any]],
        product_items2: List[Dict[str, Any]],
        similarity_threshold: int = 70 # Default threshold
    ) -> List[Dict[str, Any]]:
        pass

class ComparisonService(ComparisonServiceInterface):
    def __init__(self):
        logger.info("ComparisonService initialized.")

    def compare_product_items(
        self,
        product_items1: List[Dict[str, Any]],
        product_items2: List[Dict[str, Any]],
        similarity_threshold: int = 70
    ) -> List[Dict[str, Any]]:
        logger.info(f"Comparing {len(product_items1)} items from File1 with {len(product_items2)} items from File2. Threshold: {similarity_threshold}%")
        comparison_report = []
        matched_item2_indices = set() # To track items from list2 already matched

        for item1_idx, item1 in enumerate(product_items1):
            best_match_item2 = None
            highest_similarity = 0
            best_match_idx_in_item2_list = -1

            # Prepare item1's text for matching
            # Use normalized fields if available, otherwise fallback
            brand1 = str(item1.get("product_brand", "") or "").lower().strip()
            name_core1 = str(item1.get("product_name_core", "") or "").lower().strip()
            # Variant and size are crucial for differentiation
            variant1 = str(item1.get("product_variant_description", "") or "").lower().strip()
            size1 = str(item1.get("size_quantity_info_normalized", item1.get("size_quantity_info", "")) or "").lower().strip()
            
            # Construct structured text for matching. Prioritize specific fields.
            # Stronger signal from brand + name, then variant + size
            primary_text1 = f"{brand1} {name_core1}".strip()
            secondary_text1 = f"{variant1} {size1}".strip() # Concatenate variant and size for richer secondary text
            
            # Fallback to original collated text if structured fields are sparse
            full_collated_text1 = str(item1.get("original_collated_text", "")).lower().strip()

            if not primary_text1 and not secondary_text1 and not full_collated_text1:
                logger.warning(f"Item1 (ID: {item1.get('roboflow_box_id', f'internal_idx_{item1_idx}')}) has insufficient text for matching. Skipping.")
                # Add as unmatched immediately
                comparison_report.append(self._format_unmatched_item(item1, 1))
                continue

            for item2_idx, item2 in enumerate(product_items2):
                if item2_idx in matched_item2_indices:
                    continue # Skip if item2 already matched

                brand2 = str(item2.get("product_brand", "") or "").lower().strip()
                name_core2 = str(item2.get("product_name_core", "") or "").lower().strip()
                variant2 = str(item2.get("product_variant_description", "") or "").lower().strip()
                size2 = str(item2.get("size_quantity_info_normalized", item2.get("size_quantity_info", "")) or "").lower().strip()
                
                primary_text2 = f"{brand2} {name_core2}".strip()
                secondary_text2 = f"{variant2} {size2}".strip()
                full_collated_text2 = str(item2.get("original_collated_text", "")).lower().strip()

                if not primary_text2 and not secondary_text2 and not full_collated_text2:
                    logger.debug(f"Item2 (ID: {item2.get('roboflow_box_id', f'internal_idx_{item2_idx}')}) has insufficient text for matching against Item1. Skipping this pair.")
                    continue

                # Calculate similarity:
                # Weighted average: 60% for primary (brand+name), 40% for secondary (variant+size)
                # If primary texts are missing, rely more on secondary or full text.
                current_similarity = 0
                
                sim_primary = 0
                if primary_text1 and primary_text2:
                    sim_primary = fuzz.token_set_ratio(primary_text1, primary_text2)
                
                sim_secondary = 0
                if secondary_text1 and secondary_text2:
                    sim_secondary = fuzz.token_set_ratio(secondary_text1, secondary_text2)
                
                if primary_text1 and primary_text2 and secondary_text1 and secondary_text2:
                    current_similarity = (sim_primary * 0.6) + (sim_secondary * 0.4)
                elif primary_text1 and primary_text2: # Only primary available for both
                    current_similarity = sim_primary
                elif secondary_text1 and secondary_text2: # Only secondary available for both
                    current_similarity = sim_secondary
                else: # Fallback to full collated text if structured fields are sparse
                    if full_collated_text1 and full_collated_text2:
                        current_similarity = fuzz.token_set_ratio(full_collated_text1, full_collated_text2)
                    else: # Cannot compare if no text available
                        current_similarity = 0
                
                logger.debug(f"Comparing '{primary_text1} | {secondary_text1}' (Item1) VS '{primary_text2} | {secondary_text2}' (Item2) -> Sim_Primary: {sim_primary}, Sim_Secondary: {sim_secondary}, Combined_Sim: {current_similarity:.1f}")

                if current_similarity > highest_similarity:
                    highest_similarity = current_similarity
                    best_match_item2 = item2
                    best_match_idx_in_item2_list = item2_idx
            
            # Process the match
            if best_match_item2 and highest_similarity >= similarity_threshold:
                matched_item2_indices.add(best_match_idx_in_item2_list)
                comparison_report.append(self._format_matched_item(item1, best_match_item2, highest_similarity))
                logger.info(f"Matched Item1 (idx {item1_idx}) with Item2 (idx {best_match_idx_in_item2_list}) - Similarity: {highest_similarity:.1f}%")
            else:
                # Item1 is unmatched
                comparison_report.append(self._format_unmatched_item(item1, 1))
                logger.info(f"Item1 (idx {item1_idx}) is unmatched. Best attempt had similarity: {highest_similarity:.1f}% (Threshold: {similarity_threshold}%)")


        # Add any remaining unmatched items from list2
        for item2_idx, item2 in enumerate(product_items2):
            if item2_idx not in matched_item2_indices:
                comparison_report.append(self._format_unmatched_item(item2, 2, extra=True))
                logger.info(f"Item2 (idx {item2_idx}) from File 2 is unmatched (extra item).")
        
        return comparison_report

    def _format_matched_item(self, item1: Dict, item2: Dict, similarity: float) -> Dict:
        diff_details = []
        price_tolerance = 0.01 # For float comparisons

        # Offer Price
        op1 = item1.get("offer_price")
        op2 = item2.get("offer_price")
        if op1 is not None and op2 is not None:
            try:
                if abs(float(op1) - float(op2)) > price_tolerance:
                    diff_details.append(f"Offer Price: F1=${op1} vs F2=${op2}")
            except (ValueError, TypeError):
                 if str(op1) != str(op2): # Fallback to string comparison if float conversion fails
                    diff_details.append(f"Offer Price (type mismatch or invalid): F1='{op1}' vs F2='{op2}'")
        elif op1 != op2: # One is None, the other is not
            diff_details.append(f"Offer Price: F1={op1} vs F2={op2}")

        # Regular Price
        rp1 = item1.get("regular_price")
        rp2 = item2.get("regular_price")
        if rp1 is not None and rp2 is not None:
            try:
                if abs(float(rp1) - float(rp2)) > price_tolerance:
                    diff_details.append(f"Regular Price: F1=${rp1} vs F2=${rp2}")
            except (ValueError, TypeError):
                if str(rp1) != str(rp2):
                    diff_details.append(f"Regular Price (type mismatch or invalid): F1='{rp1}' vs F2='{rp2}'")
        elif rp1 != rp2:
            diff_details.append(f"Regular Price: F1={rp1} vs F2={rp2}")
        
        # Other attribute comparisons can be added here (e.g. size, unit indicator)
        # For now, focusing on price as per original logic

        report_item_type = "Product Match - Attributes OK"
        if diff_details:
            report_item_type = "Product Match - Attribute Mismatch"

        return {
            "type": report_item_type,
            "product1_brand": item1.get("product_brand"),
            "product1_name_core": item1.get("product_name_core"),
            "product1_variant": item1.get("product_variant_description"),
            "product1_size": item1.get("size_quantity_info_normalized", item1.get("size_quantity_info")),
            "offer_price1": op1,
            "regular_price1": rp1,
            "unit_indicator1": item1.get("unit_indicator"),
            "store_terms1": item1.get("store_specific_terms"),
            "product2_brand": item2.get("product_brand"),
            "product2_name_core": item2.get("product_name_core"),
            "product2_variant": item2.get("product_variant_description"),
            "product2_size": item2.get("size_quantity_info_normalized", item2.get("size_quantity_info")),
            "offer_price2": op2,
            "regular_price2": rp2,
            "unit_indicator2": item2.get("unit_indicator"),
            "store_terms2": item2.get("store_specific_terms"),
            "text_similarity_percent": round(similarity, 1),
            "differences": "; ".join(diff_details) if diff_details else None,
            # Include original text for review if needed
            # "original_text1": item1.get("original_collated_text"),
            # "original_text2": item2.get("original_collated_text"),
        }

    def _format_unmatched_item(self, item: Dict, file_number: int, extra: bool = False) -> Dict:
        type_str = f"Unmatched Product in File {file_number}"
        if extra:
            type_str += " (Extra)"
        
        return {
            "type": type_str,
            "product_brand": item.get("product_brand"),
            "product_name_core": item.get("product_name_core"),
            "product_variant": item.get("product_variant_description"),
            "product_size": item.get("size_quantity_info_normalized", item.get("size_quantity_info")),
            "offer_price": item.get("offer_price"),
            "regular_price": item.get("regular_price"),
            "unit_indicator": item.get("unit_indicator"),
            "store_terms": item.get("store_specific_terms"),
            # "original_text": item.get("original_collated_text"),
        }

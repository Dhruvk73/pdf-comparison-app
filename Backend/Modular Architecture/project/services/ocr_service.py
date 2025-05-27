# project/services/ocr_service.py
import logging
import time
from abc import ABC, abstractmethod
from botocore.exceptions import ClientError
from typing import List, Dict, Any

# Assuming utils.text_processing is in the same parent directory for utils
from utils.text_processing import clean_text # Corrected import path

logger = logging.getLogger(__name__)

class OCRServiceInterface(ABC):
    @abstractmethod
    def analyze_document_from_s3(self, document_s3_key: str) -> List[Dict[str, Any]] | None:
        pass

    @abstractmethod
    def collate_text_for_product_boxes(
        self,
        roboflow_boxes: List[Dict[str, Any]],
        textract_all_blocks: List[Dict[str, Any]],
        blocks_map: Dict[str, Dict[str, Any]],
        image_width_px: int,
        image_height_px: int
    ) -> List[Dict[str, Any]]:
        pass

class OCRService(OCRServiceInterface):
    def __init__(self, textract_client, s3_bucket_name: str):
        self.textract_client = textract_client
        self.s3_bucket_name = s3_bucket_name # Bucket where Textract will find the doc
        if not self.textract_client:
            logger.error("Textract client not initialized for OCRService.")
            raise ValueError("Textract client is required for OCRService.")
        logger.info("OCRService initialized.")

    def analyze_document_from_s3(self, document_s3_key: str) -> List[Dict[str, Any]] | None:
        logger.info(f"Starting Textract Document Analysis for S3 object: s3://{self.s3_bucket_name}/{document_s3_key}")
        try:
            response = self.textract_client.start_document_analysis(
                DocumentLocation={'S3Object': {'Bucket': self.s3_bucket_name, 'Name': document_s3_key}},
                FeatureTypes=['TABLES', 'FORMS', 'LAYOUT'] # Added LAYOUT for potentially better structure
            )
            job_id = response['JobId']
            logger.info(f"Textract Analysis job started (JobId: '{job_id}') for '{document_s3_key}'.")

            status = 'IN_PROGRESS'
            max_retries = 60  # 5 minutes
            retries = 0
            job_status_response = {}

            while status == 'IN_PROGRESS' and retries < max_retries:
                time.sleep(5)
                job_status_response = self.textract_client.get_document_analysis(JobId=job_id)
                status = job_status_response['JobStatus']
                logger.debug(f"Textract Analysis job status for '{job_id}': {status}")
                retries += 1

            if status == 'SUCCEEDED':
                all_blocks = []
                nextToken = None
                current_response = job_status_response # Initial response after loop

                # Fetch all blocks if paginated
                while True:
                    if nextToken: # If there was a nextToken in the previous iteration's response
                        current_response = self.textract_client.get_document_analysis(JobId=job_id, NextToken=nextToken)

                    all_blocks.extend(current_response.get("Blocks", []))
                    nextToken = current_response.get('NextToken')

                    if not nextToken:
                        break
                
                logger.info(f"Textract Analysis SUCCEEDED for '{document_s3_key}'. Found {len(all_blocks)} blocks.")
                return all_blocks
            else:
                logger.error(f"Textract Analysis job for '{document_s3_key}' status: {status}. Response: {job_status_response}")
                return None
        except ClientError as e:
            logger.error(f"ClientError in Textract Analysis for '{document_s3_key}': {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Textract Analysis for '{document_s3_key}': {e}", exc_info=True)
            return None

    def collate_text_for_product_boxes(
        self,
        roboflow_boxes: List[Dict[str, Any]],
        textract_all_blocks: List[Dict[str, Any]], # This is now directly passed
        blocks_map: Dict[str, Dict[str, Any]],    # This is now directly passed
        image_width_px: int,
        image_height_px: int
    ) -> List[Dict[str, Any]]:
        product_texts = []
        if not all([roboflow_boxes, textract_all_blocks, blocks_map, image_width_px, image_height_px]):
            logger.warning("collate_text_for_product_boxes: Missing critical inputs. Roboflow boxes: %s, Textract blocks count: %s, Blocks map count: %s, Img W: %s, Img H: %s",
                           "Present" if roboflow_boxes else "Missing",
                           len(textract_all_blocks) if textract_all_blocks else "Missing",
                           len(blocks_map) if blocks_map else "Missing",
                           image_width_px, image_height_px)
            return product_texts

        logger.info(f"Starting collation for {len(roboflow_boxes)} Roboflow boxes.")

        for i, box_pred in enumerate(roboflow_boxes):
            rf_center_x_px = box_pred.get('x')
            rf_center_y_px = box_pred.get('y')
            rf_width_px = box_pred.get('width')
            rf_height_px = box_pred.get('height')
            rf_class = box_pred.get('class', 'UnknownClass')
            rf_confidence = box_pred.get('confidence', 0.0)

            logger.debug(f"Collating for Roboflow Box #{i}: Class='{rf_class}', Conf={rf_confidence:.2f}, Center=({rf_center_x_px},{rf_center_y_px}), Size=({rf_width_px}x{rf_height_px})")


            if not all(isinstance(v, (int, float)) for v in [rf_center_x_px, rf_center_y_px, rf_width_px, rf_height_px]):
                logger.warning(f"Roboflow Box #{i} has invalid or missing coordinates: {box_pred}. Skipping.")
                continue

            rf_x_min_rel = (rf_center_x_px - rf_width_px / 2.0) / image_width_px
            rf_y_min_rel = (rf_center_y_px - rf_height_px / 2.0) / image_height_px
            rf_x_max_rel = (rf_center_x_px + rf_width_px / 2.0) / image_width_px
            rf_y_max_rel = (rf_center_y_px + rf_height_px / 2.0) / image_height_px

            lines_in_box_objects = []
            for block_id, block in blocks_map.items():
                if block.get('BlockType') == 'LINE':
                    txt_geom = block.get('Geometry')
                    if not txt_geom or 'BoundingBox' not in txt_geom:
                        continue

                    txt_bb = txt_geom['BoundingBox']
                    # Calculate center of the Textract line block
                    line_center_x_rel = txt_bb['Left'] + (txt_bb['Width'] / 2.0)
                    line_center_y_rel = txt_bb['Top'] + (txt_bb['Height'] / 2.0)

                    # Check if the center of the Textract line is within the Roboflow box
                    if (rf_x_min_rel <= line_center_x_rel <= rf_x_max_rel and \
                        rf_y_min_rel <= line_center_y_rel <= rf_y_max_rel):
                        lines_in_box_objects.append(block)
            
            # Sort lines by their vertical position primarily, then horizontal
            lines_in_box_objects.sort(key=lambda line_block: (
                line_block['Geometry']['BoundingBox']['Top'],
                line_block['Geometry']['BoundingBox']['Left']
            ))

            ordered_lines_text_parts = []
            for line_block in lines_in_box_objects:
                line_actual_text = ""
                current_line_words = []
                if 'Relationships' in line_block:
                    for relationship in line_block['Relationships']:
                        if relationship['Type'] == 'CHILD':
                            for child_id in relationship['Ids']:
                                word = blocks_map.get(child_id)
                                if word and word['BlockType'] == 'WORD' and 'Text' in word:
                                    current_line_words.append(word['Text'])
                
                if current_line_words: # Text assembled from WORD blocks
                    line_actual_text = " ".join(current_line_words)
                elif line_block.get('Text') and not current_line_words: # Fallback to LINE's Text if no WORDs (rare)
                    line_actual_text = line_block.get('Text')
                
                if line_actual_text.strip():
                    ordered_lines_text_parts.append(line_actual_text.strip())

            collated_text_multiline = "\n".join(ordered_lines_text_parts)
            collated_text_cleaned = clean_text(collated_text_multiline) # Use the utility function

            logger.debug(f"Box #{i} - Candidate Lines: {len(lines_in_box_objects)}, Assembled Text Lines: {ordered_lines_text_parts}")
            logger.info(f"Box #{i} - Final Collated Text (len {len(collated_text_cleaned)}):\n{collated_text_cleaned if collated_text_cleaned else '<<EMPTY>>'}")

            if collated_text_cleaned:
                product_texts.append({
                    "product_box_id": f"roboflow_box_{i}_{rf_class}",
                    "roboflow_confidence": rf_confidence,
                    "class_name": rf_class,
                    "collated_text": collated_text_cleaned,
                    "roboflow_box_coords_pixels_center_wh": {
                        'x': rf_center_x_px, 'y': rf_center_y_px,
                        'width': rf_width_px, 'height': rf_height_px
                    }
                })
        
        logger.info(f"Collation complete. Generated {len(product_texts)} product text snippets.")
        return product_texts

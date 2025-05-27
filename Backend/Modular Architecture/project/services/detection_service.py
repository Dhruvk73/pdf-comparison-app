# project/services/detection_service.py
import logging
import os
import time
from abc import ABC, abstractmethod
from PIL import Image
from werkzeug.utils import secure_filename
from typing import List, Dict, Any
# Roboflow SDK import will be conditional in main.py, service assumes model object is passed
# from roboflow import Roboflow # This line would be here if SDK always available

logger = logging.getLogger(__name__)

class DetectionServiceInterface(ABC):
    @abstractmethod
    def get_roboflow_predictions(
        self,
        pil_image_object: Image.Image,
        temp_uploads_dir: str,
        original_filename_for_temp: str = "temp_image.jpg"
    ) -> List[Dict[str, Any]] | None:
        pass

class DetectionService(DetectionServiceInterface):
    def __init__(self, roboflow_model_object, roboflow_sdk_available: bool):
        self.roboflow_model = roboflow_model_object
        self.roboflow_sdk_available = roboflow_sdk_available
        if not self.roboflow_sdk_available:
            logger.warning("Roboflow SDK is not available. DetectionService will not function.")
        elif not self.roboflow_model:
            logger.error("Roboflow model object not provided to DetectionService, but SDK is available.")
            # Depending on strictness, you might raise ValueError here
        logger.info("DetectionService initialized.")

    def get_roboflow_predictions(
        self,
        pil_image_object: Image.Image,
        temp_uploads_dir: str, # Pass the configured temp_uploads_dir
        original_filename_for_temp: str = "temp_image.jpg"
    ) -> List[Dict[str, Any]] | None:
        if not self.roboflow_sdk_available or not self.roboflow_model:
            logger.error("Roboflow model not configured/initialized or SDK not available.")
            return None

        temp_file_path = None
        try:
            os.makedirs(temp_uploads_dir, exist_ok=True)
            # Sanitize filename further for temp file
            base_name, ext = os.path.splitext(original_filename_for_temp)
            safe_base_name = secure_filename(base_name if base_name else "image")
            timestamp = int(time.time() * 1000) # milliseconds for uniqueness
            temp_filename = f"{timestamp}_{safe_base_name}.jpg" # Force JPEG for Roboflow
            temp_file_path = os.path.join(temp_uploads_dir, temp_filename)

            pil_image_object.save(temp_file_path, format="JPEG")
            logger.info(f"Saved PIL image temporarily to {temp_file_path} for Roboflow prediction.")

            # Confidence and overlap should ideally be configurable
            prediction_result_obj = self.roboflow_model.predict(temp_file_path, confidence=40, overlap=30)
            
            predictions_list = []
            actual_predictions = []

            # Handle various possible structures of prediction_result_obj
            if isinstance(prediction_result_obj, list):
                actual_predictions = prediction_result_obj
            elif hasattr(prediction_result_obj, 'predictions') and isinstance(getattr(prediction_result_obj, 'predictions'), list):
                actual_predictions = prediction_result_obj.predictions
            elif isinstance(prediction_result_obj, dict) and 'predictions' in prediction_result_obj:
                actual_predictions = prediction_result_obj.get('predictions', [])
            else:
                logger.warning(f"Unexpected Roboflow prediction result format: {type(prediction_result_obj)}")
                # Attempt to iterate if it's an iterable but not a list (e.g. some custom collection)
                try:
                    actual_predictions = [item for item in prediction_result_obj]
                except TypeError:
                    logger.error("Cannot iterate over prediction_result_obj. Prediction failed or format is unknown.")
                    return [] # Return empty list on failure to parse

            logger.info(f"Roboflow raw prediction count: {len(actual_predictions)}")

            for p_model_item in actual_predictions:
                data_dict_source = None
                # Try to get dict from .json() method if available
                if hasattr(p_model_item, 'json') and callable(getattr(p_model_item, 'json')):
                    try:
                        data_dict_source = p_model_item.json()
                        if not isinstance(data_dict_source, dict):
                            data_dict_source = None # Reset if .json() didn't return a dict
                    except Exception as e_json:
                        logger.warning(f"Error calling .json() on prediction item: {e_json}")
                        data_dict_source = None
                
                # If .json() failed or not available, try direct dict access or attribute access
                if data_dict_source is None:
                    if isinstance(p_model_item, dict):
                        data_dict_source = p_model_item
                    else: # Fallback to attribute access for objects not conforming to dict or .json()
                        data_dict_source = {
                            'x': getattr(p_model_item, 'x', None),
                            'y': getattr(p_model_item, 'y', None),
                            'width': getattr(p_model_item, 'width', None),
                            'height': getattr(p_model_item, 'height', None),
                            'confidence': getattr(p_model_item, 'confidence', None),
                            'class': getattr(p_model_item, 'class_name', getattr(p_model_item, 'class', 'unknown')) # 'class_name' is often used
                        }
                
                if not isinstance(data_dict_source, dict):
                    logger.error(f"Could not obtain a usable data dictionary from p_model_item of type {type(p_model_item)}. Item: {p_model_item}")
                    continue

                pred_dict = {
                    'x': data_dict_source.get('x'),
                    'y': data_dict_source.get('y'),
                    'width': data_dict_source.get('width'),
                    'height': data_dict_source.get('height'),
                    'confidence': data_dict_source.get('confidence'),
                    'class': data_dict_source.get('class', data_dict_source.get('class_name', 'unknown'))
                }

                required_keys = ['x', 'y', 'width', 'height']
                valid_prediction = True
                for key in required_keys:
                    val = pred_dict.get(key)
                    if val is None:
                        logger.warning(f"Missing required key '{key}' in prediction item: {pred_dict}")
                        valid_prediction = False
                        break
                    try:
                        pred_dict[key] = float(val) # Ensure numeric types for coordinates
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid non-numeric value for key '{key}' in prediction item: {pred_dict}")
                        valid_prediction = False
                        break
                
                if valid_prediction:
                    predictions_list.append(pred_dict)
                else:
                    logger.warning(f"Skipping invalid or incomplete Roboflow prediction: {pred_dict}")

            logger.info(f"Processed {len(predictions_list)} valid Roboflow predictions.")
            return predictions_list

        except Exception as e:
            logger.error(f"Error in get_roboflow_predictions: {e}", exc_info=True)
            return None
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Deleted temporary Roboflow image: {temp_file_path}")
                except Exception as e_del:
                    logger.error(f"Error deleting temp file {temp_file_path}: {e_del}")
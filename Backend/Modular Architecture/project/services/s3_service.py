# project/services/s3_service.py
import logging
import boto3
from botocore.exceptions import ClientError
from werkzeug.utils import secure_filename
from abc import ABC, abstractmethod
from io import BytesIO

logger = logging.getLogger(__name__)

class S3ServiceInterface(ABC):
    @abstractmethod
    def upload_fileobj(self, file_storage_object: BytesIO, cloud_object_name: str = None) -> str | None:
        pass

    @abstractmethod
    def delete_object(self, cloud_object_name: str) -> bool:
        pass

class S3Service(S3ServiceInterface):
    def __init__(self, s3_client, bucket_name: str):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        if not self.s3_client:
            logger.error("S3 client not initialized for S3Service.")
            raise ValueError("S3 client is required for S3Service.")
        if not self.bucket_name:
            logger.error("S3 bucket name not provided for S3Service.")
            raise ValueError("S3 bucket name is required for S3Service.")
        logger.info(f"S3Service initialized for bucket '{self.bucket_name}'.")

    def upload_fileobj(self, file_storage_object: BytesIO, cloud_object_name: str = None) -> str | None:
        if cloud_object_name is None:
            # This part is tricky if file_storage_object is BytesIO without a filename attribute
            # For simplicity, we'll require cloud_object_name if BytesIO is passed directly without a source filename
            logger.error("cloud_object_name must be provided for BytesIO uploads if not derived from a file.")
            return None
            # If file_storage_object were a FileStorage from Flask:
            # cloud_object_name = secure_filename(file_storage_object.filename)

        try:
            self.s3_client.upload_fileobj(file_storage_object, self.bucket_name, cloud_object_name)
            logger.info(f"File '{cloud_object_name}' uploaded to S3 bucket '{self.bucket_name}'.")
            return cloud_object_name
        except ClientError as e:
            logger.error(f"ClientError uploading file '{cloud_object_name}' to S3: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading file '{cloud_object_name}' to S3: {e}", exc_info=True)
            return None

    def delete_object(self, cloud_object_name: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=cloud_object_name)
            logger.info(f"File '{cloud_object_name}' deleted from S3 bucket '{self.bucket_name}'.")
            return True
        except ClientError as e:
            logger.error(f"ClientError deleting file '{cloud_object_name}' from S3: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting file '{cloud_object_name}' from S3: {e}", exc_info=True)
            return False

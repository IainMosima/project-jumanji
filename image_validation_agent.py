import os
import logging
from PIL import Image
import numpy as np
import cv2
import requests
from typing import Dict, List, Tuple, Union, Optional
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageValidationError(Exception):
    """Custom exception for image validation errors."""
    pass

class ImageValidationAgent:
    """Agent for validating and classifying images."""
    
    def __init__(self, notification_url: Optional[str] = None, min_dimensions: Tuple[int, int] = (100, 100), 
                 max_file_size_mb: float = 10.0, allowed_formats: List[str] = None):
        """
        Initialize the image validation agent.
        
        Args:
            notification_url: URL to send notifications about invalid images
            min_dimensions: Minimum width and height for valid images
            max_file_size_mb: Maximum file size in MB
            allowed_formats: List of allowed image formats (e.g., ['jpg', 'png'])
        """
        self.notification_url = notification_url
        self.min_dimensions = min_dimensions
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.allowed_formats = allowed_formats or ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
        logger.info("Image validation agent initialized")
    
    def validate_image_path(self, image_path: str) -> Dict:
        """
        Validate an image from a file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with validation results
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return self._create_invalid_result("File does not exist")
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size > self.max_file_size_bytes:
                return self._create_invalid_result(f"File size exceeds limit: {file_size} bytes")
            
            # Check file extension
            file_ext = os.path.splitext(image_path)[1].lower().replace('.', '')
            if file_ext not in self.allowed_formats:
                return self._create_invalid_result(f"Format not allowed: {file_ext}")
            
            # Open and validate the image
            return self._validate_image_object(Image.open(image_path), image_path)
            
        except Exception as e:
            logger.error(f"Error validating image {image_path}: {str(e)}")
            return self._create_invalid_result(f"Validation error: {str(e)}")
    
    def validate_image_bytes(self, image_bytes: bytes, source_identifier: str = "unknown") -> Dict:
        """
        Validate an image from bytes data.
        
        Args:
            image_bytes: Raw image data
            source_identifier: Identifier for the image source
            
        Returns:
            Dict with validation results
        """
        try:
            # Check file size
            if len(image_bytes) > self.max_file_size_bytes:
                return self._create_invalid_result(f"File size exceeds limit: {len(image_bytes)} bytes")
            
            # Open and validate the image
            image = Image.open(BytesIO(image_bytes))
            return self._validate_image_object(image, source_identifier)
            
        except Exception as e:
            logger.error(f"Error validating image bytes from {source_identifier}: {str(e)}")
            return self._create_invalid_result(f"Validation error: {str(e)}")
    
    def _validate_image_object(self, image: Image.Image, identifier: str) -> Dict:
        """
        Validate an image object.
        
        Args:
            image: PIL Image object
            identifier: Image identifier (path or source)
            
        Returns:
            Dict with validation results
        """
        # Check dimensions
        width, height = image.size
        if width < self.min_dimensions[0] or height < self.min_dimensions[1]:
            return self._create_invalid_result(f"Image dimensions too small: {width}x{height}")
        
        # Check if image is corrupted by attempting to load it
        try:
            image_array = np.array(image)
            if len(image_array.shape) < 2:
                return self._create_invalid_result("Invalid image data structure")
        except Exception as e:
            return self._create_invalid_result(f"Corrupted image data: {str(e)}")
        
        # Add more checks as needed (e.g., blurriness, content verification)
        
        # Image is valid
        return {
            "valid": True,
            "identifier": identifier,
            "dimensions": (width, height),
            "format": image.format.lower() if image.format else "unknown"
        }
    
    def _create_invalid_result(self, reason: str) -> Dict:
        """Create a result dict for an invalid image."""
        result = {
            "valid": False,
            "reason": reason,
        }
        # Send notification about invalid image
        if self.notification_url:
            self._send_notification(result)
        return result
    
    def _send_notification(self, result: Dict) -> None:
        """
        Send notification about invalid image.
        
        Args:
            result: Validation result dictionary
        """
        try:
            if self.notification_url:
                response = requests.post(
                    self.notification_url,
                    json={
                        "event": "invalid_image_detected",
                        "data": result
                    }
                )
                if response.status_code != 200:
                    logger.warning(f"Failed to send notification: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")

    def classify_invalid_image(self, image_path_or_bytes: Union[str, bytes]) -> Dict:
        """
        Classify why an image is invalid.
        
        Args:
            image_path_or_bytes: Path to image file or image bytes
            
        Returns:
            Dict with classification results
        """
        # Validate the image first
        if isinstance(image_path_or_bytes, str):
            result = self.validate_image_path(image_path_or_bytes)
        else:
            result = self.validate_image_bytes(image_path_or_bytes)
        
        # If the image is valid, no need to classify
        if result["valid"]:
            return {"classification": "valid", "details": "Image is valid"}
        
        # Classify the issue based on the reason
        reason = result.get("reason", "unknown")
        
        if "file size" in reason.lower():
            classification = "oversized"
        elif "dimensions" in reason.lower():
            classification = "undersized"
        elif "format" in reason.lower():
            classification = "invalid_format"
        elif "corrupt" in reason.lower():
            classification = "corrupted"
        elif "does not exist" in reason.lower():
            classification = "missing"
        else:
            classification = "other_issue"
        
        return {
            "classification": classification,
            "reason": reason,
            "valid": False
        }


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = ImageValidationAgent(
        notification_url="http://example.com/notifications",
        min_dimensions=(200, 200),
        max_file_size_mb=5.0
    )
    
    # Example validation
    test_image_path = "path/to/test/image.jpg"
    if os.path.exists(test_image_path):
        result = agent.validate_image_path(test_image_path)
        print(f"Validation result: {result}")
        
        if not result["valid"]:
            classification = agent.classify_invalid_image(test_image_path)
            print(f"Classification: {classification}")

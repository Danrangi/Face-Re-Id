import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import config

class FaceDetector:
    def __init__(self):
        # Initialize MTCNN for face detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(
            keep_all=False,
            device=self.device,
            thresholds=[config.MTCNN_CONFIDENCE_THRESHOLD, 0.7, 0.7]
        )
    
    def detect_face(self, image):
        """
        Detect face in image and return cropped face
        Args:
            image: PIL Image or numpy array
        Returns:
            cropped_face: PIL Image of detected face or None
            bbox: Bounding box coordinates or None
        """
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect face
        face, prob = self.mtcnn(image, return_prob=True)
        
        if face is not None and prob > config.MTCNN_CONFIDENCE_THRESHOLD:
            # Get bounding box
            boxes, _ = self.mtcnn.detect(image)
            if boxes is not None and len(boxes) > 0:
                bbox = boxes[0].astype(int)
                # Crop face from original image
                x1, y1, x2, y2 = bbox
                cropped_face = image.crop((x1, y1, x2, y2))
                return cropped_face, bbox
        
        return None, None
    
    def draw_bbox(self, image, bbox, label=""):
        """Draw bounding box on image"""
        if bbox is None:
            return image
        
        # Convert PIL to OpenCV format if needed
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label if provided
        if label:
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return image

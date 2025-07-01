import numpy as np
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import torch
from PIL import Image
import config

class FaceRecognizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load pre-trained FaceNet model
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.embedding_size = config.FACE_EMBEDDING_SIZE
    
    def get_embedding(self, face_image):
        """
        Extract face embedding from image
        Args:
            face_image: PIL Image of cropped face
        Returns:
            embedding: numpy array of face features
        """
        # Resize and normalize
        face_image = face_image.resize((160, 160))
        
        # Convert to tensor
        face_tensor = self._preprocess_image(face_image)
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model(face_tensor)
        
        return embedding.cpu().numpy().flatten()
    
    def _preprocess_image(self, image):
        """Preprocess image for FaceNet"""
        # Convert to numpy array
        image_np = np.array(image)
        
        # Normalize to [-1, 1]
        image_np = (image_np - 127.5) / 128.0
        
        # Convert to tensor and transpose
        image_tensor = torch.FloatTensor(image_np).permute(2, 0, 1)
        
        return image_tensor
    
    def find_match(self, query_embedding, database_embeddings, threshold=None):
        """
        Find matching face in database
        Args:
            query_embedding: embedding of face to search
            database_embeddings: list of (matric_number, name, department, embedding) tuples
            threshold: similarity threshold
        Returns:
            match_info: dict with match details or None
        """
        if threshold is None:
            threshold = config.FACE_RECOGNITION_THRESHOLD
        
        if len(database_embeddings) == 0:
            return None
        
        # Extract embeddings for comparison
        db_embeddings = np.array([emb[3] for emb in database_embeddings])
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, db_embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= threshold:
            match = database_embeddings[best_idx]
            return {
                'matric_number': match[0],
                'name': match[1],
                'department': match[2],
                'similarity': float(best_similarity),
                'confidence': f"{best_similarity*100:.1f}%"
            }
        
        return None

#!/usr/bin/env python3
"""
CORRECTED Urine Test Classifier - Fixed class mapping
"""

from pathlib import Path
from ultralytics import YOLO
import cv2

class UrineTestClassifierCorrected:
    def __init__(self):
        """Initialize the corrected classifier"""
        # Load model
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        self.model_path = parent_dir / "yolo_project" / "models" / "yolo_classification_20251122_194056" / "weights" / "best.pt"
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        
        # CORRECTED CLASS MAPPING - Based on alphabetical order from training
        # YOLO assigned classes based on directory names alphabetically:
        # Index 0: Negative (alphabetically first)
        # Index 1: Positive (alphabetically second) 
        # Index 2: Uncertain (alphabetically third)
        self.class_names = ['Negative', 'Positive', 'Uncertain']  # Corrected order!
        
        print("🔬 Urine Test Classifier (CORRECTED) Initialized")
        print(f"📁 Model: {self.model_path.name}")
        print(f"🎯 Classes: {self.class_names}")
        print("✅ Ready for classification")
    
    def classify_image(self, image_path):
        """
        Classify a single urine test image with corrected class mapping
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Classification results with corrected classes
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Run prediction
        results = self.model(str(image_path), verbose=False)
        result = results[0]
        
        if hasattr(result, 'probs'):
            probs = result.probs.data.cpu().numpy()
            predicted_idx = probs.argmax()
            confidence = probs[predicted_idx]
            predicted_class = self.class_names[predicted_idx]
            
            return {
                'image': str(image_path),
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'probabilities': {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(probs)
                },
                'raw_probabilities': probs.tolist()
            }
        else:
            raise ValueError("Unexpected model output format")
    
    def get_medical_interpretation(self, classification_result):
        """
        Provide medical interpretation of the classification result
        
        Args:
            classification_result: Output from classify_image()
            
        Returns:
            dict: Medical interpretation
        """
        predicted_class = classification_result['predicted_class']
        confidence = classification_result['confidence']
        
        interpretations = {
            'Positive': {
                'finding': 'POSITIVE for abnormal substances',
                'recommendation': 'Medical evaluation recommended',
                'urgency': 'High' if confidence > 0.8 else 'Medium',
                'description': 'Abnormal substances detected in urine sample'
            },
            'Negative': {
                'finding': 'NEGATIVE for abnormal substances', 
                'recommendation': 'Normal result - routine monitoring',
                'urgency': 'Low',
                'description': 'No significant abnormal substances detected'
            },
            'Uncertain': {
                'finding': 'UNCERTAIN result',
                'recommendation': 'Repeat test or manual review recommended',
                'urgency': 'Medium',
                'description': 'Results are inconclusive and require further analysis'
            }
        }
        
        interpretation = interpretations[predicted_class]
        interpretation['confidence_level'] = 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'
        
        return interpretation

def main():
    """Test the corrected classifier"""
    classifier = UrineTestClassifierCorrected()
    
    # Test images
    test_images = ['P1.jpg', 'N1.jpg', 'U1.jpg']
    expected_classes = ['Positive', 'Negative', 'Uncertain']
    
    print(f"\n🧪 TESTING CORRECTED CLASSIFIER")
    print("=" * 50)
    
    for img, expected in zip(test_images, expected_classes):
        if Path(img).exists():
            print(f"\n📊 Testing {img} (Expected: {expected})")
            
            try:
                result = classifier.classify_image(img)
                interpretation = classifier.get_medical_interpretation(result)
                
                predicted = result['predicted_class']
                confidence = result['confidence']
                is_correct = predicted == expected
                
                print(f"   🎯 Predicted: {predicted} ({confidence:.3f})")
                print(f"   ✅ Correct: {'YES' if is_correct else 'NO'}")
                print(f"   📋 Probabilities:")
                for class_name, prob in result['probabilities'].items():
                    print(f"      {class_name}: {prob:.4f}")
                
                print(f"   🏥 Medical Interpretation:")
                print(f"      Finding: {interpretation['finding']}")
                print(f"      Confidence: {interpretation['confidence_level']}")
                print(f"      Recommendation: {interpretation['recommendation']}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"\n❌ Image not found: {img}")

if __name__ == "__main__":
    main()
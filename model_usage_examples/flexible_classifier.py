#!/usr/bin/env python3
"""
Flexible Urine Test Classifier - Test any image file
Usage: python flexible_classifier.py <image_path>
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import cv2

class UrineTestClassifierFlexible:
    def __init__(self):
        """Initialize the flexible classifier"""
        # Load model
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        self.model_path = parent_dir / "yolo_project" / "models" / "yolo_classification_20251122_194056" / "weights" / "best.pt"
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        
        # CORRECTED CLASS MAPPING - Based on alphabetical order from training
        self.class_names = ['Negative', 'Positive', 'Uncertain']
        
        print("🔬 Flexible Urine Test Classifier Initialized")
        print(f"📁 Model: {self.model_path.name}")
        print(f"🎯 Classes: {self.class_names}")
        print("✅ Ready for classification")
    
    def classify_image(self, image_path):
        """
        Classify any urine test image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Classification results
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Check if it's a valid image file
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        if image_path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        
        print(f"📸 Processing image: {image_path.name}")
        
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
                'image_name': image_path.name,
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
        """
        predicted_class = classification_result['predicted_class']
        confidence = classification_result['confidence']
        
        interpretations = {
            'Positive': {
                'finding': 'POSITIVE for abnormal substances',
                'recommendation': 'Medical evaluation recommended - abnormal substances detected',
                'urgency': 'High' if confidence > 0.8 else 'Medium',
                'description': 'The urine sample shows presence of abnormal substances that may indicate kidney disease or other health conditions',
                'action': '🚨 Consult healthcare provider immediately'
            },
            'Negative': {
                'finding': 'NEGATIVE for abnormal substances', 
                'recommendation': 'Normal result - continue routine monitoring',
                'urgency': 'Low',
                'description': 'No significant abnormal substances detected in the urine sample',
                'action': '✅ Maintain healthy lifestyle and regular checkups'
            },
            'Uncertain': {
                'finding': 'UNCERTAIN result - inconclusive',
                'recommendation': 'Repeat test recommended or manual laboratory review',
                'urgency': 'Medium',
                'description': 'The analysis is inconclusive and requires further testing for accurate diagnosis',
                'action': '🔄 Repeat test or get professional lab analysis'
            }
        }
        
        interpretation = interpretations[predicted_class]
        interpretation['confidence_level'] = 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'
        
        return interpretation

def main():
    """Main function to handle command line usage"""
    if len(sys.argv) != 2:
        print("🔬 Flexible Urine Test Classifier")
        print("=" * 50)
        print("Usage: python flexible_classifier.py <image_path>")
        print("\nExamples:")
        print("  python flexible_classifier.py P1.jpg")
        print("  python flexible_classifier.py C:\\path\\to\\urine_sample.jpg")
        print("  python flexible_classifier.py ../test_images/sample.png")
        print("\nSupported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
        return
    
    image_path = sys.argv[1]
    
    try:
        # Initialize classifier
        classifier = UrineTestClassifierFlexible()
        print()
        
        # Classify the image
        result = classifier.classify_image(image_path)
        interpretation = classifier.get_medical_interpretation(result)
        
        # Display results
        print("🎯 CLASSIFICATION RESULTS")
        print("=" * 50)
        print(f"📁 Image: {result['image_name']}")
        print(f"🔬 Prediction: {result['predicted_class']}")
        print(f"📊 Confidence: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)")
        print(f"💪 Confidence Level: {interpretation['confidence_level']}")
        
        print(f"\n📋 DETAILED PROBABILITIES:")
        for class_name, prob in result['probabilities'].items():
            bar_length = int(prob * 20)  # Scale to 20 characters
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {class_name:>9}: {prob:6.4f} [{bar}] {prob*100:5.1f}%")
        
        print(f"\n🏥 MEDICAL INTERPRETATION:")
        print(f"   Finding: {interpretation['finding']}")
        print(f"   Description: {interpretation['description']}")
        print(f"   Urgency: {interpretation['urgency']}")
        print(f"   Recommendation: {interpretation['recommendation']}")
        print(f"   Action: {interpretation['action']}")
        
        # Risk assessment
        print(f"\n⚠️ RISK ASSESSMENT:")
        if result['predicted_class'] == 'Positive':
            print(f"   🔴 HIGH PRIORITY - Abnormal substances detected")
        elif result['predicted_class'] == 'Negative':
            print(f"   🟢 LOW RISK - Normal urine sample")
        else:
            print(f"   🟡 MEDIUM PRIORITY - Requires additional testing")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the image file exists and the path is correct")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("💡 Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Please check your image file and try again")

if __name__ == "__main__":
    main()
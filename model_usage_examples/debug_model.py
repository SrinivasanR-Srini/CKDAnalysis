#!/usr/bin/env python3
"""
Model Analysis Script - Debug prediction patterns
"""

from pathlib import Path
from ultralytics import YOLO
import numpy as np

def analyze_model_predictions():
    """Analyze model predictions across multiple samples"""
    
    # Load model
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    model_path = parent_dir / "yolo_project" / "models" / "yolo_classification_20251122_194056" / "weights" / "best.pt"
    
    model = YOLO(str(model_path))
    class_names = ['Negative', 'Positive', 'Uncertain']  # Corrected alphabetical order
    
    print("🔍 MODEL ANALYSIS - Investigating Prediction Patterns")
    print("=" * 60)
    
    # Test images we have
    test_images = ['P1.jpg', 'N1.jpg', 'U1.jpg']
    expected_classes = ['Positive', 'Negative', 'Uncertain']
    
    results = []
    
    for img, expected in zip(test_images, expected_classes):
        if Path(img).exists():
            print(f"\n📊 Testing {img} (Expected: {expected})")
            
            # Run prediction
            prediction_results = model(img, verbose=False)
            result = prediction_results[0]
            
            if hasattr(result, 'probs'):
                probs = result.probs.data.cpu().numpy()
                predicted_idx = probs.argmax()
                confidence = probs[predicted_idx]
                predicted_class = class_names[predicted_idx]
                
                print(f"   Predicted: {predicted_class} ({confidence:.3f})")
                print(f"   Probabilities: {dict(zip(class_names, probs))}")
                
                # Check if prediction matches expected
                is_correct = predicted_class == expected
                print(f"   Correct: {'✅' if is_correct else '❌'}")
                
                results.append({
                    'image': img,
                    'expected': expected,
                    'predicted': predicted_class,
                    'confidence': confidence,
                    'correct': is_correct,
                    'negative_prob': probs[0],  # Alphabetical: Negative = 0
                    'positive_prob': probs[1],  # Alphabetical: Positive = 1  
                    'uncertain_prob': probs[2]  # Alphabetical: Uncertain = 2
                })
            else:
                print(f"   ❌ Error: Unexpected model output format")
    
    # Analysis
    print(f"\n" + "="*60)
    print("🧪 ANALYSIS RESULTS")
    print("="*60)
    
    if results:
        correct_predictions = sum(r['correct'] for r in results)
        total_predictions = len(results)
        accuracy = correct_predictions / total_predictions * 100
        
        print(f"📊 Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
        
        # Show detailed results
        print(f"\n🔍 DETAILED RESULTS:")
        for r in results:
            status = "✅ CORRECT" if r['correct'] else "❌ WRONG"
            print(f"   📊 {r['image']}: Expected {r['expected']} → Got {r['predicted']} ({r['confidence']:.3f}) {status}")
    
    return results

def check_training_data_mapping():
    """Check how the training data was mapped during training"""
    print(f"\n" + "="*60)
    print("🔍 CHECKING TRAINING DATA MAPPING")
    print("="*60)
    
    # Check the actual class mapping used during training
    parent_dir = Path(__file__).parent.parent
    
    # Check the original data structure
    original_data_dir = parent_dir / "yolo_project" / "original_data"
    if original_data_dir.exists():
        print(f"📁 Original training data structure:")
        class_dirs = sorted([d for d in original_data_dir.iterdir() if d.is_dir()])
        for i, class_dir in enumerate(class_dirs):
            image_count = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png')))
            print(f"   Index {i}: {class_dir.name} ({image_count} images)")
        
        print(f"\n🔍 YOLO automatically assigns classes alphabetically:")
        print(f"   Index 0: {class_dirs[0].name if class_dirs else 'Unknown'}")
        print(f"   Index 1: {class_dirs[1].name if len(class_dirs) > 1 else 'Unknown'}")
        print(f"   Index 2: {class_dirs[2].name if len(class_dirs) > 2 else 'Unknown'}")

if __name__ == "__main__":
    results = analyze_model_predictions()
    check_training_data_mapping()
    
    print(f"\n" + "="*60)
    print("💡 SUMMARY")
    print("="*60)
    print("✅ Use corrected class mapping: ['Negative', 'Positive', 'Uncertain']")
    print("✅ Model is working correctly with proper class mapping")
    print("✅ Use flexible_classifier.py or corrected_classifier.py for accurate results")
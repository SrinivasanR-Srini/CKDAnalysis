#!/usr/bin/env python3
"""
Batch Urine Test Processor - Process multiple images at once
Usage: python batch_processor.py <image_directory>
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime

class BatchUrineTestProcessor:
    def __init__(self):
        """Initialize the batch processor"""
        # Load model
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        self.model_path = parent_dir / "yolo_project" / "models" / "yolo_classification_20251122_194056" / "weights" / "best.pt"
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        
        # CORRECTED CLASS MAPPING - Alphabetical order
        self.class_names = ['Negative', 'Positive', 'Uncertain']
        
        print("🔬 Batch Urine Test Processor Initialized")
        print(f"📁 Model: {self.model_path.name}")
        print(f"🎯 Classes: {self.class_names}")
        print("✅ Ready for batch processing")
    
    def process_directory(self, image_dir, output_file=None):
        """
        Process all images in a directory
        
        Args:
            image_dir: Directory containing images to process
            output_file: Optional output file for results (JSON format)
        
        Returns:
            list: Results for all processed images
        """
        image_dir = Path(image_dir)
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ No image files found in {image_dir}")
            return []
        
        print(f"\n📊 Found {len(image_files)} images to process")
        print("=" * 50)
        
        results = []
        stats = {'Negative': 0, 'Positive': 0, 'Uncertain': 0}
        
        for i, img_file in enumerate(image_files, 1):
            print(f"\n📸 Processing {i}/{len(image_files)}: {img_file.name}")
            
            try:
                # Run prediction
                prediction_results = self.model(str(img_file), verbose=False)
                result = prediction_results[0]
                
                if hasattr(result, 'probs'):
                    probs = result.probs.data.cpu().numpy()
                    predicted_idx = probs.argmax()
                    confidence = probs[predicted_idx]
                    predicted_class = self.class_names[predicted_idx]
                    
                    # Create result record
                    result_record = {
                        'image_name': img_file.name,
                        'image_path': str(img_file),
                        'predicted_class': predicted_class,
                        'confidence': float(confidence),
                        'probabilities': {
                            self.class_names[j]: float(prob) 
                            for j, prob in enumerate(probs)
                        },
                        'timestamp': datetime.now().isoformat(),
                        'processing_order': i
                    }
                    
                    results.append(result_record)
                    stats[predicted_class] += 1
                    
                    # Show result
                    print(f"   🎯 Result: {predicted_class} ({confidence:.3f})")
                    confidence_emoji = "🔴" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🟠"
                    print(f"   📊 Confidence: {confidence_emoji} {confidence*100:.1f}%")
                    
                else:
                    print(f"   ❌ Error: Unexpected model output format")
                    
            except Exception as e:
                print(f"   ❌ Error processing {img_file.name}: {e}")
        
        # Summary
        print(f"\n" + "=" * 50)
        print("📊 BATCH PROCESSING SUMMARY")
        print("=" * 50)
        print(f"🔬 Total Images Processed: {len(results)}")
        print(f"📈 Results Distribution:")
        
        for class_name, count in stats.items():
            percentage = (count / len(results) * 100) if results else 0
            bar_length = int(percentage / 5)  # Scale to 20 chars max
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {class_name:>9}: {count:3d} [{bar}] {percentage:5.1f}%")
        
        # Risk Assessment
        print(f"\n⚠️ RISK ASSESSMENT:")
        positive_count = stats['Positive']
        uncertain_count = stats['Uncertain']
        
        if positive_count > 0:
            print(f"   🔴 HIGH RISK: {positive_count} positive samples detected")
        if uncertain_count > 0:
            print(f"   🟡 MEDIUM RISK: {uncertain_count} uncertain samples need review")
        if stats['Negative'] == len(results):
            print(f"   🟢 LOW RISK: All samples are negative")
        
        # Save results if output file specified
        if output_file and results:
            output_path = Path(output_file)
            
            batch_summary = {
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'model_used': str(self.model_path),
                    'total_images': len(results),
                    'source_directory': str(image_dir)
                },
                'statistics': stats,
                'results': results
            }
            
            with open(output_path, 'w') as f:
                json.dump(batch_summary, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_path}")
        
        return results
    
    def get_summary_report(self, results):
        """Generate a summary report from batch results"""
        if not results:
            return "No results to summarize"
        
        stats = {'Negative': 0, 'Positive': 0, 'Uncertain': 0}
        high_confidence_count = 0
        
        for result in results:
            stats[result['predicted_class']] += 1
            if result['confidence'] > 0.8:
                high_confidence_count += 1
        
        total = len(results)
        
        report = f"""
📋 BATCH PROCESSING REPORT
{"="*40}
📊 Total Images: {total}
🎯 High Confidence (>80%): {high_confidence_count} ({high_confidence_count/total*100:.1f}%)

📈 Classification Results:
   🔴 Positive: {stats['Positive']} ({stats['Positive']/total*100:.1f}%)
   🟢 Negative: {stats['Negative']} ({stats['Negative']/total*100:.1f}%)
   🟡 Uncertain: {stats['Uncertain']} ({stats['Uncertain']/total*100:.1f}%)

⚠️ Medical Recommendations:
   • {stats['Positive']} samples require immediate medical attention
   • {stats['Uncertain']} samples need repeat testing or manual review
   • {stats['Negative']} samples show normal results
        """
        
        return report

def main():
    """Main function for command line usage"""
    if len(sys.argv) != 2:
        print("🔬 Batch Urine Test Processor")
        print("=" * 50)
        print("Usage: python batch_processor.py <image_directory>")
        print("\nExamples:")
        print("  python batch_processor.py ../test_images/")
        print("  python batch_processor.py C:\\path\\to\\urine_samples\\")
        print("\nSupported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
        return
    
    image_directory = sys.argv[1]
    
    try:
        # Initialize processor
        processor = BatchUrineTestProcessor()
        
        # Process directory
        results = processor.process_directory(
            image_directory,
            output_file=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if results:
            # Generate and show summary report
            report = processor.get_summary_report(results)
            print(report)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the directory exists and contains image files")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Please check your input and try again")

if __name__ == "__main__":
    main()
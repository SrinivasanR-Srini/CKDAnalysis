#!/usr/bin/env python3
"""
Quick Setup Script - Restore YOLO Training Environment
"""

import sys
from pathlib import Path
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = ['ultralytics', 'opencv-python', 'numpy', 'matplotlib', 'seaborn', 'pillow']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                from PIL import Image
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"   ✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to install {package}")
                return False
    
    return True

def check_dataset():
    """Check if dataset is available"""
    print(f"\n🔍 Checking for dataset...")
    
    # Common dataset paths
    dataset_paths = [
        Path("../Clinical Urine Test Strips"),
        Path("Clinical Urine Test Strips"),
        Path("../Clinical Urine Test Strips/Clinical Urine Test Strips"),
        Path("./Clinical Urine Test Strips/Clinical Urine Test Strips")
    ]
    
    for path in dataset_paths:
        if path.exists():
            class_dirs = [d for d in path.iterdir() if d.is_dir()]
            if class_dirs:
                print(f"   ✅ Dataset found at: {path}")
                print(f"   📁 Classes found: {[d.name for d in class_dirs]}")
                return True
    
    print(f"   ❌ Dataset not found in common locations")
    print(f"   💡 Expected: Clinical urine test dataset with class directories")
    return False

def check_model():
    """Check if trained model exists"""
    print(f"\n🔍 Checking for trained model...")
    
    models_dir = Path("../yolo_project/models")
    if models_dir.exists():
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and 'yolo_classification' in d.name]
        if model_dirs:
            for model_dir in model_dirs:
                best_pt = model_dir / "weights" / "best.pt"
                if best_pt.exists():
                    print(f"   ✅ Model found: {best_pt}")
                    print(f"   📊 Size: {best_pt.stat().st_size / 1024 / 1024:.1f} MB")
                    return True
    
    print(f"   ❌ No trained model found")
    print(f"   💡 Need to run training notebook")
    return False

def main():
    """Main setup check"""
    print("🚀 YOLO Training Environment Setup Check")
    print("=" * 50)
    
    # Check requirements
    req_ok = check_requirements()
    
    # Check dataset  
    dataset_ok = check_dataset()
    
    # Check model
    model_ok = check_model()
    
    print(f"\n" + "=" * 50)
    print("📋 SETUP SUMMARY")
    print("=" * 50)
    
    print(f"📦 Requirements: {'✅ Ready' if req_ok else '❌ Missing packages'}")
    print(f"📁 Dataset: {'✅ Available' if dataset_ok else '❌ Not found'}")
    print(f"🤖 Model: {'✅ Trained' if model_ok else '❌ Need training'}")
    
    if req_ok and dataset_ok and model_ok:
        print(f"\n🎉 Environment is ready!")
        print(f"✅ You can use the classifier tools:")
        print(f"   python flexible_classifier.py <image_path>")
        print(f"   python corrected_classifier.py")
        print(f"   python batch_processor.py <directory>")
    
    elif req_ok and dataset_ok:
        print(f"\n🔄 Ready for training!")
        print(f"✅ Requirements installed")
        print(f"✅ Dataset available") 
        print(f"📝 Next step: Run yolo_object_detection_mendeley.ipynb")
        
    elif req_ok:
        print(f"\n⚠️ Partial setup")
        print(f"✅ Requirements installed")
        print(f"❌ Dataset not found - please provide clinical urine test dataset")
        
    else:
        print(f"\n❌ Setup incomplete")
        print(f"💡 Install missing packages and provide dataset")

if __name__ == "__main__":
    main()
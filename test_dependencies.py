#!/usr/bin/env python3
"""
Test script to verify backend dependencies and functionality
"""
import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing Python dependencies...\n")
    
    tests = [
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Pillow", "PIL"),
        ("Pydantic", "pydantic"),
        ("Mediapipe", "mediapipe"),
    ]
    
    failed = []
    
    for name, module in tests:
        try:
            __import__(module)
            print(f"✓ {name:20} OK")
        except ImportError as e:
            print(f"✗ {name:20} FAILED: {e}")
            failed.append(name)
    
    print("\n" + "="*50)
    
    if failed:
        print(f"\n⚠ {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"  - {pkg}")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed successfully!")
        return True

def test_mediapipe():
    """Test Mediapipe specifically"""
    print("\n" + "="*50)
    print("Testing Mediapipe segmentation...\n")
    
    try:
        import mediapipe as mp
        import numpy as np
        
        # Try to initialize segmentation
        mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # Create a dummy image
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Try to process
        results = mp_selfie.process(dummy_img)
        
        if results.segmentation_mask is not None:
            print("✓ Mediapipe segmentation working!")
            print("  Model loaded and can process images")
            return True
        else:
            print("⚠ Mediapipe loaded but segmentation returned None")
            return False
            
    except Exception as e:
        print(f"✗ Mediapipe test failed: {e}")
        print("\nNote: Backend will fall back to GrabCut (slower but works)")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    print("\n" + "="*50)
    print("Testing OpenCV...\n")
    
    try:
        import cv2
        import numpy as np
        
        # Create test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test basic operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        print("✓ OpenCV working!")
        print(f"  Version: {cv2.__version__}")
        return True
        
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def main():
    print("="*50)
    print("Image Editor Backend - Dependency Test")
    print("="*50)
    print()
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Some dependencies are missing!")
        sys.exit(1)
    
    # Test OpenCV
    opencv_ok = test_opencv()
    
    # Test Mediapipe (optional but recommended)
    mediapipe_ok = test_mediapipe()
    
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    print(f"Dependencies: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"OpenCV:       {'✓ PASS' if opencv_ok else '✗ FAIL'}")
    print(f"Mediapipe:    {'✓ PASS' if mediapipe_ok else '⚠ OPTIONAL'}")
    
    if imports_ok and opencv_ok:
        print("\n✓ Backend is ready to run!")
        print("\nStart the server with:")
        print("  uvicorn app.main:app --reload")
        sys.exit(0)
    else:
        print("\n❌ Backend has issues that need to be fixed")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

print("Testing minimal import...")

try:
    import sys
    sys.path.append('backend')
    print("✅ sys.path updated")
    
    from models.cross_validation import CrossValidator
    print("✅ CrossValidator imported")
    
    from models.hyperparameter_tuning import HyperparameterTuner
    print("✅ HyperparameterTuner imported")
    
    print("🎉 All imports successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

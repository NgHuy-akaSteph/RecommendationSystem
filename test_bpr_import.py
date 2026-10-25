"""Quick test to verify BPR implementation can be imported"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Testing BPR import...")

try:
    from src.recommenders import BPRRecommender
    print("✓ BPRRecommender imported successfully")
    
    # Check class attributes
    bpr = BPRRecommender(factors=32, iterations=10)
    print(f"✓ BPRRecommender instantiated with factors={bpr.factors}, iterations={bpr.iterations}")
    
    # Check methods exist
    assert hasattr(bpr, 'fit'), "Missing fit() method"
    assert hasattr(bpr, 'recommend'), "Missing recommend() method"
    assert hasattr(bpr, 'save'), "Missing save() method"
    assert hasattr(bpr, 'load'), "Missing load() method"
    print("✓ All required methods present")
    
    print("\n✅ BPR implementation verified successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

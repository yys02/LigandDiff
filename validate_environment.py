#!/usr/bin/env python3
"""
Environment validation script for LigandDiff.
This script checks if the environment is properly configured for running the model.
"""

import os
import sys
import subprocess
import importlib

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is not compatible (requires 3.9+)")
        return False

def check_cuda_availability():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  CUDA is not available (will use CPU)")
            return True
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_critical_imports():
    """Check critical imports for the model."""
    critical_imports = [
        ("torch", "PyTorch"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("torch_geometric", "PyTorch Geometric"),
        ("rdkit", "RDKit"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
    ]
    
    all_imported = True
    for module, name in critical_imports:
        try:
            importlib.import_module(module)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
            all_imported = False
    
    return all_imported

def check_model_files():
    """Check if essential model files exist."""
    essential_files = [
        "train.py",
        "generate.py", 
        "config.yml",
        "src/lightning.py",
        "src/const.py",
        "src/utils.py"
    ]
    
    all_exist = True
    for file in essential_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False
    
    return all_exist

def test_basic_functionality():
    """Test basic functionality of the model components."""
    try:
        # Test config loading
        import yaml
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Config file loads successfully")
        
        # Test constants import
        from src import const
        print("✓ Constants module imports successfully")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all validation checks."""
    print("LigandDiff Environment Validation")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("CUDA Availability", check_cuda_availability),
        ("Critical Imports", check_critical_imports),
        ("Model Files", check_model_files),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 20)
        if check_func():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 Environment is ready for LigandDiff!")
        print("\nNext steps:")
        print("1. Download the dataset from the provided link")
        print("2. Run training: python train.py --config config.yml")
        print("3. Generate ligands: python generate.py --model model.ckpt --complex complex.xyz --outdir output/")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the correct conda environment: conda activate liganddiff")
        print("2. Check ENVIRONMENT_SETUP.md for detailed setup instructions")
        print("3. Run: python test_environment.py for detailed dependency checks")
        return 1

if __name__ == "__main__":
    sys.exit(main())
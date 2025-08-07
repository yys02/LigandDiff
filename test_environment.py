#!/usr/bin/env python3
"""
Test script to validate LigandDiff environment dependencies.
This script checks if all required packages can be imported successfully.
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported successfully."""
    try:
        if package_name:
            module = importlib.import_module(module_name, package=package_name)
        else:
            module = importlib.import_module(module_name)
        print(f"✓ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {module_name} import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ {module_name} error: {e}")
        return False

def main():
    """Test all required dependencies for LigandDiff."""
    print("Testing LigandDiff environment dependencies...")
    print("=" * 50)
    
    # Core Python packages
    core_packages = [
        "os", "sys", "argparse", "tempfile", "datetime", "warnings",
        "typing", "math", "glob", "random", "pathlib"
    ]
    
    # Scientific computing
    scientific_packages = [
        "numpy", "pandas", "scipy", "sklearn"
    ]
    
    # PyTorch ecosystem
    pytorch_packages = [
        "torch", "torchvision", "torchaudio", "pytorch_lightning"
    ]
    
    # PyTorch Geometric
    geometric_packages = [
        "torch_geometric", "torch_scatter", "torch_sparse", 
        "torch_cluster", "torch_spline_conv"
    ]
    
    # Chemistry packages
    chemistry_packages = [
        "rdkit", "openbabel"
    ]
    
    # Visualization
    viz_packages = [
        "matplotlib", "imageio"
    ]
    
    # Utilities
    util_packages = [
        "yaml", "wandb"
    ]
    
    # Test all package categories
    all_packages = {
        "Core Python": core_packages,
        "Scientific Computing": scientific_packages,
        "PyTorch Ecosystem": pytorch_packages,
        "PyTorch Geometric": geometric_packages,
        "Chemistry": chemistry_packages,
        "Visualization": viz_packages,
        "Utilities": util_packages
    }
    
    total_tests = 0
    passed_tests = 0
    
    for category, packages in all_packages.items():
        print(f"\n{category}:")
        print("-" * 30)
        
        for package in packages:
            total_tests += 1
            if test_import(package):
                passed_tests += 1
    
    # Test specific imports that might have different names
    print(f"\nSpecial Cases:")
    print("-" * 30)
    
    # Test sklearn import
    total_tests += 1
    if test_import("sklearn"):
        passed_tests += 1
    
    # Test rdkit specific import
    total_tests += 1
    if test_import("rdkit", "Chem"):
        passed_tests += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed_tests}/{total_tests} packages imported successfully")
    
    if passed_tests == total_tests:
        print("🎉 All dependencies are properly installed!")
        return 0
    else:
        print("⚠️  Some dependencies failed to import. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
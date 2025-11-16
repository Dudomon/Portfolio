#!/usr/bin/env python3
"""
Clear all Python caches more aggressively
"""

import sys
import os
import shutil
import importlib

def clear_python_cache():
    """Clear all Python caches"""
    
    print("LIMPANDO CACHE PYTHON...")
    
    # 1. Clear import caches
    if hasattr(importlib, 'invalidate_caches'):
        importlib.invalidate_caches()
        print("Import caches cleared")
    
    # 2. Clear module cache
    modules_to_remove = []
    for module_name in sys.modules:
        if 'trading_framework' in module_name or 'two_head_v7' in module_name:
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]
            print(f"Removed {module_name} from sys.modules")
    
    # 3. Clear __pycache__ directories
    cache_dirs_removed = 0
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs[:]:  # Copy list to safely modify during iteration
            if dir_name == '__pycache__':
                cache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_path)
                    cache_dirs_removed += 1
                    dirs.remove(dir_name)  # Don't recurse into removed directory
                except Exception as e:
                    print(f"Could not remove {cache_path}: {e}")
    
    print(f"Removed {cache_dirs_removed} __pycache__ directories")
    
    # 4. Clear .pyc files
    pyc_files_removed = 0
    for root, dirs, files in os.walk('.'):
        for file_name in files:
            if file_name.endswith('.pyc'):
                pyc_path = os.path.join(root, file_name)
                try:
                    os.remove(pyc_path)
                    pyc_files_removed += 1
                except Exception as e:
                    print(f"Could not remove {pyc_path}: {e}")
    
    print(f"Removed {pyc_files_removed} .pyc files")
    
    print("CACHE PYTHON COMPLETAMENTE LIMPO!")
    print("Proximos passos:")
    print("   1. Restart o processo Python")
    print("   2. Re-import a policy")
    print("   3. Verificar se usa MLP critic")

if __name__ == "__main__":
    clear_python_cache()
#!/usr/bin/env python3
"""
🔥 Fix V7 Intuition - Forçar restart para aplicar mudanças
"""

import sys
import importlib
import torch

def force_reload_v7():
    """Force reload of V7 Intuition module"""
    
    # Remove from cache
    modules_to_remove = []
    for module_name in sys.modules:
        if 'two_head_v7_intuition' in module_name:
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        print(f"🔄 Removing {module_name} from cache")
        del sys.modules[module_name]
    
    # Clear torch.jit cache
    torch.jit.clear_cache()
    
    print("✅ V7 Intuition cache cleared - restart Python para aplicar mudanças")

if __name__ == "__main__":
    force_reload_v7()
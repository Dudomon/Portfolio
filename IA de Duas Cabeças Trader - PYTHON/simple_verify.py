#!/usr/bin/env python3
"""
Verify critic architecture without unicode
"""

import sys
sys.path.append('D:/Projeto')

# Test direct import
try:
    from trading_framework.policies.two_head_v7_simple import TwoHeadV7Simple
    print("SUCCESS: TwoHeadV7Simple imported")
    
    # Check if class has MLP
    if hasattr(TwoHeadV7Simple, '__init__'):
        print("SUCCESS: Class has __init__")
        
    # Test creating simple instance - just check the source
    import inspect
    source = inspect.getsource(TwoHeadV7Simple.forward_critic)
    
    if 'v7_critic_mlp' in source:
        print("SUCCESS: forward_critic uses v7_critic_mlp")
    else:
        print("FAILED: forward_critic does NOT use v7_critic_mlp")
        
    if 'critic_memory_buffer' in source:
        print("SUCCESS: forward_critic uses memory buffer")
    else:
        print("FAILED: forward_critic does NOT use memory buffer")
        
    if 'v7_critic_lstm' in source:
        print("WARNING: forward_critic still references LSTM")
    else:
        print("SUCCESS: forward_critic does NOT use LSTM")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
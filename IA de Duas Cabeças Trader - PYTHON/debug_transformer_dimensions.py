"""
🔍 DEBUG: Transformer dimensions para V8Elegance
"""

import torch
import sys
sys.path.append(r'D:\Projeto')

from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor
from gym import spaces
import numpy as np

def debug_transformer_dimensions():
    """Debug das dimensões do transformer"""
    
    print("🔍 DEBUGGING TRANSFORMER DIMENSIONS")
    print("="*50)
    
    # Create observation space
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2580,), dtype=np.float32)
    
    # Create transformer
    transformer = TradingTransformerFeatureExtractor(observation_space, features_dim=256)
    
    # Test input
    batch_size = 2
    observations = torch.randn(batch_size, 2580)
    
    print(f"Input shape: {observations.shape}")
    
    # Forward pass
    output = transformer(observations)
    
    print(f"Output shape: {output.shape}")
    print(f"Output type: {type(output)}")
    
    # Analyze dimensions
    if len(output.shape) == 3:
        print(f"   Batch: {output.shape[0]}")
        print(f"   Sequence: {output.shape[1]}")  
        print(f"   Features: {output.shape[2]}")
    elif len(output.shape) == 2:
        print(f"   Batch: {output.shape[0]}")
        print(f"   Features: {output.shape[1]}")
    
    return output

if __name__ == "__main__":
    result = debug_transformer_dimensions()
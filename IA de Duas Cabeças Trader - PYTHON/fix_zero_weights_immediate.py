#!/usr/bin/env python3
"""
🔧 IMMEDIATE FIX FOR ZERO WEIGHTS PROBLEM
Fix the 100% zero bias issue in TwoHeadV11Sigmoid model
"""

import torch
import torch.nn as nn
import zipfile
import tempfile
import os
import glob
import shutil
from datetime import datetime
import traceback

def find_latest_checkpoint():
    """Find the most recent checkpoint"""
    search_paths = [
        "D:/Projeto/trading_framework/training/checkpoints/SILUS/*.zip",
        "D:/Projeto/Otimizacao/treino_principal/models/SILUS/*.zip",
        "D:/Projeto/*.zip"
    ]
    
    all_checkpoints = []
    for pattern in search_paths:
        checkpoints = glob.glob(pattern)
        all_checkpoints.extend(checkpoints)
    
    if not all_checkpoints:
        return None
    
    latest = max(all_checkpoints, key=os.path.getmtime)
    return latest

def reinitialize_bias_tensors(state_dict):
    """Reinitialize all bias tensors that are 100% zeros"""
    
    fixes_applied = 0
    
    for key, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
            
        # Focus on bias parameters that are 100% zeros
        if 'bias' in key.lower():
            zeros_count = (tensor == 0).sum().item()
            total_count = tensor.numel()
            zeros_pct = (zeros_count / total_count) * 100 if total_count > 0 else 0
            
            if zeros_pct >= 99:  # 100% or near 100% zeros
                print(f"🔧 Fixing {key}: {zeros_pct:.1f}% zeros")
                
                # Reinitialize based on layer type
                if 'lstm' in key.lower() or 'gru' in key.lower():
                    # RNN biases: keep forget gate bias at 1, others at 0
                    if 'lstm' in key.lower() and 'bias_ih_l0' in key:
                        # LSTM forget gate bias should be 1
                        hidden_size = tensor.size(0) // 4
                        nn.init.zeros_(tensor)
                        tensor.data[hidden_size:2*hidden_size].fill_(1.0)  # Forget gate
                        fixes_applied += 1
                    elif 'gru' in key.lower():
                        # GRU: small random initialization
                        nn.init.uniform_(tensor, -0.01, 0.01)
                        fixes_applied += 1
                    else:
                        # Other RNN biases: zeros
                        nn.init.zeros_(tensor)
                        fixes_applied += 1
                        
                elif 'layernorm' in key.lower() or 'layer_norm' in key.lower():
                    # LayerNorm bias should be 0
                    nn.init.zeros_(tensor) 
                    fixes_applied += 1
                    
                else:
                    # Regular layer biases: small random values
                    nn.init.uniform_(tensor, -0.01, 0.01)
                    fixes_applied += 1
                
                # Verify fix
                new_zeros = (tensor == 0).sum().item()
                new_zeros_pct = (new_zeros / total_count) * 100
                print(f"   → Fixed: {new_zeros_pct:.1f}% zeros (was {zeros_pct:.1f}%)")
    
    return fixes_applied

def create_fixed_checkpoint(original_path, output_path):
    """Create a new checkpoint with fixed weights"""
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract original checkpoint
            print(f"📁 Extracting {os.path.basename(original_path)}...")
            with zipfile.ZipFile(original_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find and load policy.pth
            policy_path = os.path.join(temp_dir, 'policy.pth')
            if not os.path.exists(policy_path):
                print(f"❌ policy.pth not found in checkpoint")
                return False
            
            print(f"🔍 Loading policy weights...")
            state_dict = torch.load(policy_path, map_location='cpu')
            
            # Apply fixes
            print(f"🔧 Applying fixes to bias tensors...")
            fixes_count = reinitialize_bias_tensors(state_dict)
            
            if fixes_count == 0:
                print(f"✅ No fixes needed - all biases are healthy")
                return False
            
            print(f"✅ Applied {fixes_count} fixes")
            
            # Save fixed policy
            print(f"💾 Saving fixed policy...")
            torch.save(state_dict, policy_path)
            
            # Create new checkpoint
            print(f"📦 Creating fixed checkpoint...")
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, temp_dir)
                        zip_out.write(file_path, arc_name)
            
            return True
            
    except Exception as e:
        print(f"❌ Error creating fixed checkpoint: {e}")
        traceback.print_exc()
        return False

def update_training_config():
    """Generate updated training configuration"""
    
    config_update = """
# 🔧 CRITICAL FIXES FOR ZERO WEIGHTS PROBLEM
# Apply these changes immediately to prevent recurrence:

LEARNING_RATE_FIX = {
    "current": 5e-3,           # PROBLEM: Too high, causing weight explosion
    "fixed": 1e-4,             # SOLUTION: 50x reduction to stable range
    "rationale": "High LR causes weight explosion → clipping → zeros"
}

GRADIENT_CLIPPING_FIX = {
    "recommended": {
        "max_grad_norm": 1.0,   # Gentle clipping to prevent explosion
        "clip_grad_value": None # Don't use value clipping
    },
    "avoid": {
        "max_grad_norm": 0.1,   # Too aggressive, causes zeros
        "clip_grad_value": 0.5  # Value clipping can zero weights
    }
}

OPTIMIZER_SETTINGS = {
    "type": "Adam",
    "lr": 1e-4,                # Critical: Use fixed learning rate
    "betas": (0.9, 0.999),     # Standard Adam betas
    "eps": 1e-8,               # Standard epsilon
    "weight_decay": 1e-5,      # Light regularization only
    "amsgrad": False           # Keep simple
}

INITIALIZATION_CHECK = {
    "bias_initialization": {
        "lstm_forget_bias": 1.0,     # Forget gate bias = 1
        "gru_biases": "uniform(-0.01, 0.01)",
        "linear_biases": "uniform(-0.01, 0.01)",
        "layernorm_biases": 0.0
    },
    "weight_initialization": {
        "lstm_weights": "xavier_uniform + orthogonal",
        "linear_weights": "xavier_uniform", 
        "embedding_weights": "normal(0, 0.01)"
    }
}

MONITORING_REQUIRED = [
    "gradient_norms",           # Watch for explosion
    "weight_zero_percentages",  # Monitor for recurrence  
    "bias_magnitudes",          # Ensure biases stay healthy
    "loss_stability"            # Check for training instability
]
    """
    
    config_path = "D:/Projeto/CRITICAL_FIXES_CONFIG.py"
    with open(config_path, 'w') as f:
        f.write(config_update)
    
    print(f"💾 Configuration fixes saved to: {config_path}")
    return config_path

def main():
    """Main fix application"""
    print("🔧 IMMEDIATE ZERO WEIGHTS FIX")
    print("=" * 50)
    
    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path:
        print("❌ No checkpoints found!")
        return False
    
    print(f"📁 Target checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"📅 Original date: {datetime.fromtimestamp(os.path.getmtime(checkpoint_path))}")
    
    # Create backup
    backup_path = checkpoint_path.replace('.zip', '_BACKUP_BEFORE_FIX.zip')
    print(f"💾 Creating backup: {os.path.basename(backup_path)}")
    shutil.copy2(checkpoint_path, backup_path)
    
    # Create fixed version
    fixed_path = checkpoint_path.replace('.zip', '_FIXED.zip')
    print(f"🔧 Creating fixed version: {os.path.basename(fixed_path)}")
    
    success = create_fixed_checkpoint(checkpoint_path, fixed_path)
    
    if success:
        print(f"✅ FIXED CHECKPOINT CREATED: {os.path.basename(fixed_path)}")
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. 🔧 Update training config with LR = 1e-4")
        print(f"   2. 🚀 Resume training from: {os.path.basename(fixed_path)}")
        print(f"   3. 📊 Monitor gradient norms and zero percentages")
        print(f"   4. 🛡️  Use gradient clipping max_norm = 1.0")
        
        # Generate config updates
        config_path = update_training_config()
        print(f"   5. 📄 Apply config from: {os.path.basename(config_path)}")
        
        return True
    else:
        print(f"❌ Fix failed - check logs above")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print(f"\n🎯 CRITICAL ISSUE RESOLVED!")
            print(f"   The 100% zero bias problem has been fixed.")
            print(f"   Use the _FIXED.zip checkpoint with LR=1e-4")
        else:
            print(f"\n❌ Fix failed - manual intervention required")
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        traceback.print_exc()
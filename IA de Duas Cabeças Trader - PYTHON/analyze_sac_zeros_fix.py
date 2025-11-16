#!/usr/bin/env python3
"""
🔍 SAC ZERO WEIGHTS ANALYZER - FIXED VERSION
Analyze SAC checkpoints for zero weight patterns and root causes
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from stable_baselines3 import SAC
import glob
from datetime import datetime

def find_latest_sac_checkpoint():
    """Find the most recent SAC checkpoint"""
    # Search in multiple locations
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
    
    # Sort by modification time
    latest = max(all_checkpoints, key=os.path.getmtime)
    return latest

def analyze_layer_zeros(layer_name, weight_tensor, detailed=True):
    """Analyze zero patterns in a weight tensor"""
    if weight_tensor is None or weight_tensor.numel() == 0:
        return None
    
    weight_flat = weight_tensor.flatten()
    zeros_mask = (weight_flat == 0)
    total_weights = weight_flat.numel()
    zeros_count = zeros_mask.sum().item()
    zeros_pct = (zeros_count / total_weights) * 100 if total_weights > 0 else 0
    
    analysis = {
        'layer_name': layer_name,
        'shape': weight_tensor.shape,
        'total_weights': total_weights,
        'zeros_count': zeros_count,
        'zeros_percentage': zeros_pct,
        'non_zeros_count': total_weights - zeros_count
    }
    
    if detailed and zeros_count > 0:
        # Analyze distribution of zeros
        if len(weight_tensor.shape) == 2:  # Linear layer
            zeros_per_neuron = (weight_tensor == 0).sum(dim=1)
            zeros_per_input = (weight_tensor == 0).sum(dim=0)
            
            analysis['zeros_per_neuron'] = {
                'max': zeros_per_neuron.max().item(),
                'min': zeros_per_neuron.min().item(),
                'mean': zeros_per_neuron.float().mean().item(),
                'std': zeros_per_neuron.float().std().item()
            }
            
            analysis['dead_neurons'] = (zeros_per_neuron == weight_tensor.shape[1]).sum().item()
            analysis['dead_inputs'] = (zeros_per_input == weight_tensor.shape[0]).sum().item()
    
    if zeros_count < total_weights:  # If there are non-zero weights
        non_zeros_data = weight_flat[~zeros_mask]
        analysis['non_zeros_stats'] = {
            'min': non_zeros_data.min().item(),
            'max': non_zeros_data.max().item(),
            'mean': non_zeros_data.mean().item(),
            'std': non_zeros_data.std().item(),
            'median': non_zeros_data.median().item()
        }
    
    return analysis

def test_initialization_methods(input_dim, output_dim):
    """Test different initialization methods to compare with actual weights"""
    
    results = {}
    
    # Test different initialization methods
    methods = {
        'xavier_uniform': lambda w: nn.init.xavier_uniform_(w),
        'xavier_normal': lambda w: nn.init.xavier_normal_(w),
        'kaiming_uniform': lambda w: nn.init.kaiming_uniform_(w, nonlinearity='leaky_relu'),
        'kaiming_normal': lambda w: nn.init.kaiming_normal_(w, nonlinearity='leaky_relu'),
        'normal_0.01': lambda w: nn.init.normal_(w, mean=0, std=0.01),
        'normal_0.1': lambda w: nn.init.normal_(w, mean=0, std=0.1),
        'uniform': lambda w: nn.init.uniform_(w, -0.1, 0.1)
    }
    
    for method_name, init_func in methods.items():
        # Create test layer
        test_layer = nn.Linear(input_dim, output_dim, bias=False)
        init_func(test_layer.weight)
        
        # Count zeros
        zeros_count = (test_layer.weight == 0).sum().item()
        total_count = test_layer.weight.numel()
        zeros_pct = (zeros_count / total_count) * 100 if total_count > 0 else 0
        
        results[method_name] = {
            'zeros_count': zeros_count,
            'zeros_percentage': zeros_pct,
            'weight_range': [test_layer.weight.min().item(), test_layer.weight.max().item()],
            'weight_mean': test_layer.weight.mean().item(),
            'weight_std': test_layer.weight.std().item()
        }
    
    return results

def analyze_sac_model():
    """Main analysis function"""
    print("🔍 SAC ZERO WEIGHTS ANALYSIS")
    print("=" * 60)
    
    # Find latest checkpoint
    checkpoint_path = find_latest_sac_checkpoint()
    if not checkpoint_path:
        print("❌ No SAC checkpoints found!")
        return False
    
    print(f"📁 Loading checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"📅 Modified: {datetime.fromtimestamp(os.path.getmtime(checkpoint_path))}")
    
    try:
        # Load SAC model
        model = SAC.load(checkpoint_path, verbose=0)
        print("✅ SAC model loaded successfully")
        
        # Get policy networks
        actor = model.policy.actor
        critic1 = model.policy.critic
        critic2 = model.policy.critic_target if hasattr(model.policy, 'critic_target') else None
        
        print(f"\n🏗️  MODEL ARCHITECTURE:")
        print(f"   Actor: {actor}")
        print(f"   Critic 1: {critic1}")
        if critic2:
            print(f"   Critic 2: {critic2}")
        
        # Analyze actor networks
        print(f"\n🎭 ACTOR NETWORK ANALYSIS:")
        print("-" * 40)
        
        actor_analysis = []
        
        # Analyze actor layers
        for name, module in actor.named_modules():
            if isinstance(module, nn.Linear):
                layer_analysis = analyze_layer_zeros(f"actor.{name}", module.weight.data, detailed=True)
                if layer_analysis:
                    actor_analysis.append(layer_analysis)
                    
                    print(f"\n🔍 Layer: {layer_analysis['layer_name']}")
                    print(f"   Shape: {layer_analysis['shape']}")
                    print(f"   Zeros: {layer_analysis['zeros_count']} / {layer_analysis['total_weights']} ({layer_analysis['zeros_percentage']:.1f}%)")
                    
                    if layer_analysis['zeros_percentage'] > 50:
                        print(f"   🚨 CRITICAL: {layer_analysis['zeros_percentage']:.1f}% zeros is abnormal!")
                        if 'dead_neurons' in layer_analysis:
                            print(f"   💀 Dead neurons: {layer_analysis['dead_neurons']}")
                        if 'dead_inputs' in layer_analysis:
                            print(f"   💀 Dead inputs: {layer_analysis['dead_inputs']}")
                    elif layer_analysis['zeros_percentage'] > 10:
                        print(f"   ⚠️  WARNING: {layer_analysis['zeros_percentage']:.1f}% zeros is suspicious")
                    else:
                        print(f"   ✅ Normal zero percentage")
                    
                    if 'non_zeros_stats' in layer_analysis:
                        stats = layer_analysis['non_zeros_stats']
                        print(f"   📊 Non-zero range: [{stats['min']:.6f}, {stats['max']:.6f}]")
                        print(f"   📊 Non-zero mean±std: {stats['mean']:.6f}±{stats['std']:.6f}")
        
        # Analyze critic networks  
        print(f"\n🎯 CRITIC NETWORK ANALYSIS:")
        print("-" * 40)
        
        for critic_name, critic_net in [("critic1", critic1), ("critic2", critic2)]:
            if critic_net is None:
                continue
                
            print(f"\n🔍 {critic_name.upper()}:")
            
            for name, module in critic_net.named_modules():
                if isinstance(module, nn.Linear):
                    layer_analysis = analyze_layer_zeros(f"{critic_name}.{name}", module.weight.data, detailed=False)
                    if layer_analysis:
                        print(f"   {name}: {layer_analysis['zeros_count']} / {layer_analysis['total_weights']} ({layer_analysis['zeros_percentage']:.1f}% zeros)")
                        
                        if layer_analysis['zeros_percentage'] > 50:
                            print(f"      🚨 CRITICAL: Abnormally high zeros!")
        
        # Compare with proper initialization
        print(f"\n🧪 INITIALIZATION COMPARISON:")
        print("-" * 40)
        
        if actor_analysis:
            first_layer = actor_analysis[0]
            if len(first_layer['shape']) == 2:
                input_dim, output_dim = first_layer['shape'][1], first_layer['shape'][0]
                
                print(f"Testing initialization methods for shape {first_layer['shape']}:")
                init_results = test_initialization_methods(input_dim, output_dim)
                
                for method, results in init_results.items():
                    print(f"   {method}: {results['zeros_percentage']:.1f}% zeros")
                
                print(f"\n🔍 DIAGNOSIS:")
                max_normal_zeros = max(r['zeros_percentage'] for r in init_results.values())
                actual_zeros = first_layer['zeros_percentage']
                
                if actual_zeros > max_normal_zeros * 10:  # 10x more than any normal initialization
                    print(f"   🚨 CRITICAL PROBLEM CONFIRMED:")
                    print(f"      Actual: {actual_zeros:.1f}% zeros")
                    print(f"      Expected: <{max_normal_zeros:.1f}% zeros")
                    print(f"      Ratio: {actual_zeros/max_normal_zeros:.1f}x higher than normal")
                    
                    print(f"\n   🎯 PROBABLE CAUSES:")
                    print(f"      1. 🔥 Learning rate too high (current: 5e-3)")
                    print(f"         → Weights exploding then getting clipped to zero")
                    print(f"      2. 📉 Gradient clipping too aggressive")
                    print(f"         → Large gradients being clipped to zero")
                    print(f"      3. 💀 Dead ReLU cascading effect")  
                    print(f"         → Neurons dying and staying dead")
                    print(f"      4. 🎛️  Optimizer state corruption")
                    print(f"         → Adam moments causing weight decay to zero")
                    print(f"      5. 📏 Regularization too strong")
                    print(f"         → L1/L2 penalty zeroing small weights")
                    
                    print(f"\n   💡 IMMEDIATE SOLUTIONS:")
                    print(f"      1. 🔧 REDUCE LEARNING RATE: 5e-3 → 1e-4 (20x lower)")
                    print(f"      2. 🔄 REINITIALIZE WEIGHTS: Apply proper Kaiming init")
                    print(f"      3. 🛡️  CHECK GRADIENT CLIPPING: Ensure reasonable values")
                    print(f"      4. 🧹 RESET OPTIMIZER STATE: Clear Adam momentum")
                else:
                    print(f"   ✅ Zero percentage within normal range")
        
        # Check optimizer configuration
        print(f"\n🔧 OPTIMIZER CONFIGURATION:")
        print("-" * 40)
        
        if hasattr(model.policy, 'optimizer'):
            optimizer = model.policy.optimizer
            print(f"   Type: {type(optimizer).__name__}")
            
            for i, param_group in enumerate(optimizer.param_groups):
                print(f"   Group {i}:")
                print(f"      Learning rate: {param_group['lr']}")
                print(f"      Weight decay: {param_group.get('weight_decay', 'N/A')}")
                
                # Check if learning rate is too high
                if param_group['lr'] > 1e-3:
                    print(f"      🚨 WARNING: Learning rate {param_group['lr']} is very high for SAC!")
                    print(f"      📝 Recommended: 1e-4 to 3e-4")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = analyze_sac_model()
        if success:
            print(f"\n✅ Analysis completed successfully")
            print(f"\n📋 SUMMARY RECOMMENDATIONS:")
            print(f"   1. Lower learning rate from 5e-3 to 1e-4")
            print(f"   2. Reinitialize actor network with proper Kaiming init")
            print(f"   3. Monitor gradient norms during training")
            print(f"   4. Consider using LeakyReLU instead of ReLU")
        else:
            print(f"\n❌ Analysis failed - check error messages above")
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
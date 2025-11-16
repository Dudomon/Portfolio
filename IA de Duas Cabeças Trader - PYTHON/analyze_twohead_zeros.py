#!/usr/bin/env python3
"""
🔍 TWOHEAD V11 ZERO WEIGHTS ANALYZER
Analyze TwoHeadV11Sigmoid policy for zero weight patterns and root causes
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import glob
from datetime import datetime
from stable_baselines3 import SAC
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

def analyze_layer_zeros_detailed(layer_name, weight_tensor):
    """Detailed analysis of zero patterns in a weight tensor"""
    if weight_tensor is None or weight_tensor.numel() == 0:
        return None
    
    # Basic statistics
    weight_flat = weight_tensor.flatten()
    zeros_mask = (weight_flat == 0)
    total_weights = weight_flat.numel()
    zeros_count = zeros_mask.sum().item()
    zeros_pct = (zeros_count / total_weights) * 100 if total_weights > 0 else 0
    
    analysis = {
        'layer_name': layer_name,
        'shape': weight_tensor.shape,
        'device': str(weight_tensor.device),
        'dtype': str(weight_tensor.dtype),
        'total_weights': total_weights,
        'zeros_count': zeros_count,
        'zeros_percentage': zeros_pct,
        'non_zeros_count': total_weights - zeros_count,
    }
    
    if zeros_count < total_weights:  # Non-zero stats
        non_zeros = weight_flat[~zeros_mask]
        analysis['non_zeros_stats'] = {
            'min': non_zeros.min().item(),
            'max': non_zeros.max().item(), 
            'mean': non_zeros.mean().item(),
            'std': non_zeros.std().item(),
            'median': non_zeros.median().item(),
            'abs_mean': non_zeros.abs().mean().item()
        }
        
        # Check for magnitude distribution
        tiny_weights = (non_zeros.abs() < 1e-6).sum().item()
        small_weights = (non_zeros.abs() < 1e-4).sum().item()
        analysis['magnitude_distribution'] = {
            'tiny_weights_1e-6': tiny_weights,
            'small_weights_1e-4': small_weights,
            'tiny_pct': (tiny_weights / non_zeros.numel()) * 100,
            'small_pct': (small_weights / non_zeros.numel()) * 100
        }
    
    # Pattern analysis for linear layers
    if len(weight_tensor.shape) == 2:
        # Analyze dead neurons (all zeros in output dimension)
        zeros_per_neuron = (weight_tensor == 0).sum(dim=1)
        dead_neurons = (zeros_per_neuron == weight_tensor.shape[1]).sum().item()
        
        # Analyze dead inputs (all zeros in input dimension)
        zeros_per_input = (weight_tensor == 0).sum(dim=0)
        dead_inputs = (zeros_per_input == weight_tensor.shape[0]).sum().item()
        
        analysis['pattern_analysis'] = {
            'dead_neurons': dead_neurons,
            'dead_inputs': dead_inputs,
            'dead_neuron_pct': (dead_neurons / weight_tensor.shape[0]) * 100,
            'dead_input_pct': (dead_inputs / weight_tensor.shape[1]) * 100,
            'max_zeros_per_neuron': zeros_per_neuron.max().item(),
            'min_zeros_per_neuron': zeros_per_neuron.min().item(),
            'avg_zeros_per_neuron': zeros_per_neuron.float().mean().item()
        }
        
        # Check for systematic patterns
        if zeros_count > 0:
            # Row-wise patterns
            fully_dead_rows = (zeros_per_neuron == weight_tensor.shape[1]).sum().item()
            mostly_dead_rows = (zeros_per_neuron > weight_tensor.shape[1] * 0.8).sum().item()
            
            # Column-wise patterns  
            fully_dead_cols = (zeros_per_input == weight_tensor.shape[0]).sum().item()
            mostly_dead_cols = (zeros_per_input > weight_tensor.shape[0] * 0.8).sum().item()
            
            analysis['systematic_patterns'] = {
                'fully_dead_rows': fully_dead_rows,
                'mostly_dead_rows': mostly_dead_rows,
                'fully_dead_cols': fully_dead_cols,
                'mostly_dead_cols': mostly_dead_cols
            }
    
    return analysis

def test_initialization_comparison(input_dim, output_dim):
    """Test how different initializations compare to actual weights"""
    methods = {
        'xavier_uniform': lambda w: nn.init.xavier_uniform_(w, gain=1.0),
        'kaiming_uniform_leaky': lambda w: nn.init.kaiming_uniform_(w, a=0.01, nonlinearity='leaky_relu'),
        'normal_001': lambda w: nn.init.normal_(w, mean=0, std=0.01),
        'normal_01': lambda w: nn.init.normal_(w, mean=0, std=0.1),
        'uniform_small': lambda w: nn.init.uniform_(w, -0.01, 0.01),
        'zeros': lambda w: nn.init.zeros_(w),
    }
    
    results = {}
    for method_name, init_func in methods.items():
        test_layer = nn.Linear(input_dim, output_dim, bias=False)
        init_func(test_layer.weight)
        
        zeros_count = (test_layer.weight == 0).sum().item()
        total_count = test_layer.weight.numel()
        zeros_pct = (zeros_count / total_count) * 100
        
        results[method_name] = {
            'zeros_count': zeros_count,
            'zeros_percentage': zeros_pct,
            'mean': test_layer.weight.mean().item(),
            'std': test_layer.weight.std().item(),
            'range': [test_layer.weight.min().item(), test_layer.weight.max().item()]
        }
    
    return results

def diagnose_zero_causes(analysis):
    """Diagnose likely causes of zero weights based on analysis"""
    zeros_pct = analysis['zeros_percentage']
    
    causes = []
    confidence = []
    
    if zeros_pct > 60:
        causes.append("🚨 CRITICAL: Weight explosion followed by clipping")
        confidence.append(0.9)
        
        causes.append("🔥 Learning rate too high (5e-3 is 25x higher than typical)")
        confidence.append(0.95)
    
    if zeros_pct > 40:
        causes.append("📉 Aggressive gradient clipping zeroing weights")
        confidence.append(0.7)
        
        causes.append("💀 Dead ReLU cascading effect")
        confidence.append(0.6)
    
    if 'pattern_analysis' in analysis:
        dead_neurons = analysis['pattern_analysis']['dead_neurons']
        total_neurons = analysis['shape'][0]
        
        if dead_neurons > total_neurons * 0.3:
            causes.append("🧠 Systematic neuron death (>30% dead neurons)")
            confidence.append(0.8)
    
    if 'magnitude_distribution' in analysis:
        tiny_pct = analysis['magnitude_distribution']['tiny_pct']
        if tiny_pct > 20:
            causes.append("🔬 Many weights near zero (regularization effect)")
            confidence.append(0.6)
    
    # Sort by confidence
    sorted_causes = sorted(zip(causes, confidence), key=lambda x: x[1], reverse=True)
    
    return [cause for cause, conf in sorted_causes[:5]]

def analyze_twohead_model():
    """Main analysis function for TwoHeadV11Sigmoid model"""
    print("🔍 TWOHEAD V11 SIGMOID ZERO WEIGHTS ANALYSIS")
    print("=" * 70)
    
    # Find checkpoint
    checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path:
        print("❌ No checkpoints found!")
        return False
    
    print(f"📁 Loading: {os.path.basename(checkpoint_path)}")
    print(f"📅 Modified: {datetime.fromtimestamp(os.path.getmtime(checkpoint_path))}")
    print(f"📏 Size: {os.path.getsize(checkpoint_path) / (1024*1024):.1f} MB")
    
    try:
        # Load model
        model = SAC.load(checkpoint_path, verbose=0)
        print("✅ Model loaded successfully")
        print(f"📋 Policy type: {type(model.policy).__name__}")
        
        # Analyze TwoHeadV11Sigmoid components
        policy = model.policy
        
        print(f"\n🏗️  TWOHEAD V11 ARCHITECTURE ANALYSIS:")
        print("-" * 50)
        
        # Critical layers to analyze
        critical_layers = []
        
        # 1. LSTM layers
        if hasattr(policy, 'v8_shared_lstm'):
            lstm = policy.v8_shared_lstm
            for name, param in lstm.named_parameters():
                if 'weight' in name:
                    layer_name = f"v8_shared_lstm.{name}"
                    analysis = analyze_layer_zeros_detailed(layer_name, param.data)
                    if analysis:
                        critical_layers.append(analysis)
        
        # 2. GRU layers (V11 hybrid)
        if hasattr(policy, 'v11_shared_gru'):
            gru = policy.v11_shared_gru
            for name, param in gru.named_parameters():
                if 'weight' in name:
                    layer_name = f"v11_shared_gru.{name}"
                    analysis = analyze_layer_zeros_detailed(layer_name, param.data)
                    if analysis:
                        critical_layers.append(analysis)
        
        # 3. Entry Head
        if hasattr(policy, 'entry_head'):
            entry_head = policy.entry_head
            for name, module in entry_head.named_modules():
                if isinstance(module, nn.Linear):
                    layer_name = f"entry_head.{name}"
                    analysis = analyze_layer_zeros_detailed(layer_name, module.weight.data)
                    if analysis:
                        critical_layers.append(analysis)
        
        # 4. Management Head
        if hasattr(policy, 'management_head'):
            mgmt_head = policy.management_head
            for name, module in mgmt_head.named_modules():
                if isinstance(module, nn.Linear):
                    layer_name = f"management_head.{name}"
                    analysis = analyze_layer_zeros_detailed(layer_name, module.weight.data)
                    if analysis:
                        critical_layers.append(analysis)
        
        # 5. Critic
        if hasattr(policy, 'v8_critic'):
            critic = policy.v8_critic
            for i, module in enumerate(critic):
                if isinstance(module, nn.Linear):
                    layer_name = f"v8_critic.{i}"
                    analysis = analyze_layer_zeros_detailed(layer_name, module.weight.data)
                    if analysis:
                        critical_layers.append(analysis)
        
        # 6. Market Context and Fusion layers
        if hasattr(policy, 'market_context'):
            mc = policy.market_context
            for name, module in mc.named_modules():
                if isinstance(module, nn.Linear):
                    layer_name = f"market_context.{name}"
                    analysis = analyze_layer_zeros_detailed(layer_name, module.weight.data)
                    if analysis:
                        critical_layers.append(analysis)
        
        if hasattr(policy, 'hybrid_fusion'):
            fusion = policy.hybrid_fusion
            for i, module in enumerate(fusion):
                if isinstance(module, nn.Linear):
                    layer_name = f"hybrid_fusion.{i}"
                    analysis = analyze_layer_zeros_detailed(layer_name, module.weight.data)
                    if analysis:
                        critical_layers.append(analysis)
        
        # Sort by zeros percentage (worst first)
        critical_layers.sort(key=lambda x: x['zeros_percentage'], reverse=True)
        
        print(f"\n🎯 ZERO ANALYSIS RESULTS ({len(critical_layers)} layers):")
        print("-" * 50)
        
        max_zeros_layer = None
        total_zeros = 0
        total_weights = 0
        
        for analysis in critical_layers:
            zeros_pct = analysis['zeros_percentage']
            layer_name = analysis['layer_name']
            shape = analysis['shape']
            
            total_zeros += analysis['zeros_count']
            total_weights += analysis['total_weights']
            
            if max_zeros_layer is None:
                max_zeros_layer = analysis
            
            # Status indicator
            if zeros_pct > 50:
                status = "🚨 CRITICAL"
            elif zeros_pct > 20:
                status = "⚠️  WARNING"
            elif zeros_pct > 5:
                status = "🔸 ELEVATED"
            else:
                status = "✅ NORMAL"
            
            print(f"\n{status} {layer_name}")
            print(f"   Shape: {shape}")
            print(f"   Zeros: {analysis['zeros_count']:,} / {analysis['total_weights']:,} ({zeros_pct:.1f}%)")
            
            if zeros_pct > 30:
                if 'pattern_analysis' in analysis:
                    pa = analysis['pattern_analysis']
                    print(f"   💀 Dead neurons: {pa['dead_neurons']} ({pa['dead_neuron_pct']:.1f}%)")
                    print(f"   💀 Dead inputs: {pa['dead_inputs']} ({pa['dead_input_pct']:.1f}%)")
                
                if 'non_zeros_stats' in analysis:
                    nz = analysis['non_zeros_stats']
                    print(f"   📊 Non-zero range: [{nz['min']:.6f}, {nz['max']:.6f}]")
                    print(f"   📊 Non-zero mean±std: {nz['mean']:.6f}±{nz['std']:.6f}")
        
        # Overall statistics
        overall_zeros_pct = (total_zeros / total_weights) * 100 if total_weights > 0 else 0
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total weights: {total_weights:,}")
        print(f"   Total zeros: {total_zeros:,} ({overall_zeros_pct:.1f}%)")
        
        # Worst layer analysis
        if max_zeros_layer and max_zeros_layer['zeros_percentage'] > 30:
            print(f"\n🔥 WORST LAYER ANALYSIS: {max_zeros_layer['layer_name']}")
            print("-" * 40)
            
            # Test initialization comparison
            if len(max_zeros_layer['shape']) == 2:
                input_dim, output_dim = max_zeros_layer['shape'][1], max_zeros_layer['shape'][0]
                print(f"Testing initialization methods for shape {max_zeros_layer['shape']}:")
                
                init_results = test_initialization_comparison(input_dim, output_dim)
                for method, results in init_results.items():
                    print(f"   {method}: {results['zeros_percentage']:.3f}% zeros")
                
                # Compare with actual
                actual_zeros = max_zeros_layer['zeros_percentage']
                expected_zeros = max(r['zeros_percentage'] for r in init_results.values() if r['zeros_percentage'] > 0)
                if expected_zeros == 0:
                    expected_zeros = 0.001  # Practically zero
                
                print(f"\n   Actual vs Expected:")
                print(f"   🎯 Actual zeros: {actual_zeros:.1f}%")
                print(f"   📏 Expected zeros: {expected_zeros:.3f}%")
                print(f"   📈 Ratio: {actual_zeros/expected_zeros:.0f}x higher than normal")
            
            # Diagnose causes
            causes = diagnose_zero_causes(max_zeros_layer)
            print(f"\n   🔍 PROBABLE CAUSES (ranked by likelihood):")
            for i, cause in enumerate(causes[:3], 1):
                print(f"      {i}. {cause}")
        
        # Specific recommendations
        print(f"\n💡 IMMEDIATE SOLUTIONS:")
        print("-" * 30)
        
        if max_zeros_layer and max_zeros_layer['zeros_percentage'] > 50:
            print(f"   🚨 CRITICAL ISSUE DETECTED!")
            print(f"   ")
            print(f"   1. 🔧 REDUCE LEARNING RATE:")
            print(f"      Current: 5e-3 (extremely high)")
            print(f"      → Change to: 1e-4 or 3e-4")
            print(f"      → This is 12-50x lower, preventing weight explosion")
            print(f"   ")
            print(f"   2. 🔄 REINITIALIZE WEIGHTS:")
            print(f"      → Apply Kaiming uniform initialization")
            print(f"      → Focus on: {max_zeros_layer['layer_name']}")
            print(f"   ")
            print(f"   3. 🛡️  CHECK GRADIENT CLIPPING:")
            print(f"      → Ensure max_grad_norm is reasonable (0.5-2.0)")
            print(f"      → Avoid aggressive clipping that zeros weights")
            print(f"   ")
            print(f"   4. 🧹 RESET OPTIMIZER STATE:")
            print(f"      → Clear Adam momentum buffers")
            print(f"      → Start fresh optimization")
            
        elif overall_zeros_pct > 10:
            print(f"   ⚠️  MODERATE ISSUE:")
            print(f"   → Monitor gradient norms")  
            print(f"   → Consider slight learning rate reduction")
            print(f"   → Check for regularization effects")
        else:
            print(f"   ✅ No critical issues detected")
            print(f"   → Monitor for trends over time")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = analyze_twohead_model()
        if success:
            print(f"\n✅ ANALYSIS COMPLETED")
            print(f"\n📋 KEY RECOMMENDATIONS:")
            print(f"   1. Lower learning rate from 5e-3 to 1e-4")
            print(f"   2. Reinitialize layers with >50% zeros")
            print(f"   3. Monitor gradient norms during training")
            print(f"   4. Consider using gradient clipping with max_norm=1.0")
        else:
            print(f"\n❌ Analysis failed")
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        traceback.print_exc()
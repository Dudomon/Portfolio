#!/usr/bin/env python3
"""
🔍 DIRECT CHECKPOINT ANALYZER
Directly analyze checkpoint weights without loading through SAC
"""

import torch
import torch.nn as nn
import zipfile
import tempfile
import os
import glob
import pickle
import numpy as np
from datetime import datetime

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

def extract_policy_weights(checkpoint_path):
    """Extract policy weights directly from checkpoint"""
    try:
        # Extract checkpoint to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(checkpoint_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find policy weights file
            policy_files = glob.glob(os.path.join(temp_dir, "policy*.pth"))
            if not policy_files:
                # Try alternative names
                policy_files = glob.glob(os.path.join(temp_dir, "*policy*"))
                if not policy_files:
                    # Look for any .pth files
                    policy_files = glob.glob(os.path.join(temp_dir, "*.pth"))
            
            if not policy_files:
                print(f"❌ No policy weights found in checkpoint")
                return None
            
            # Load the policy weights
            policy_path = policy_files[0]
            print(f"📁 Loading policy from: {os.path.basename(policy_path)}")
            
            # Load with CPU mapping to avoid GPU issues
            state_dict = torch.load(policy_path, map_location='cpu')
            
            return state_dict
            
    except Exception as e:
        print(f"❌ Error extracting weights: {e}")
        return None

def analyze_weight_tensor(name, tensor):
    """Analyze a single weight tensor"""
    if tensor is None or tensor.numel() == 0:
        return None
    
    # Basic stats
    total_weights = tensor.numel()
    zeros_mask = (tensor == 0)
    zeros_count = zeros_mask.sum().item()
    zeros_pct = (zeros_count / total_weights) * 100
    
    analysis = {
        'name': name,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'total_weights': total_weights,
        'zeros_count': zeros_count,
        'zeros_percentage': zeros_pct,
        'device': str(tensor.device)
    }
    
    # Non-zero statistics
    if zeros_count < total_weights:
        non_zeros = tensor[~zeros_mask]
        analysis['non_zero_stats'] = {
            'count': non_zeros.numel(),
            'min': non_zeros.min().item(),
            'max': non_zeros.max().item(),
            'mean': non_zeros.mean().item(),
            'std': non_zeros.std().item(),
            'abs_mean': non_zeros.abs().mean().item()
        }
        
        # Magnitude analysis
        tiny_count = (non_zeros.abs() < 1e-6).sum().item()
        small_count = (non_zeros.abs() < 1e-4).sum().item()
        
        analysis['magnitude_analysis'] = {
            'tiny_weights': tiny_count,
            'small_weights': small_count,
            'tiny_pct': (tiny_count / non_zeros.numel()) * 100,
            'small_pct': (small_count / non_zeros.numel()) * 100
        }
    
    # Pattern analysis for 2D tensors (Linear layers)
    if len(tensor.shape) == 2:
        rows, cols = tensor.shape
        
        # Dead neurons (all zeros in a row/output)
        zeros_per_row = zeros_mask.sum(dim=1)
        dead_rows = (zeros_per_row == cols).sum().item()
        
        # Dead inputs (all zeros in a column/input)
        zeros_per_col = zeros_mask.sum(dim=0)  
        dead_cols = (zeros_per_col == rows).sum().item()
        
        analysis['pattern_analysis'] = {
            'dead_rows': dead_rows,
            'dead_cols': dead_cols,
            'dead_row_pct': (dead_rows / rows) * 100,
            'dead_col_pct': (dead_cols / cols) * 100,
            'max_zeros_per_row': zeros_per_row.max().item(),
            'max_zeros_per_col': zeros_per_col.max().item(),
            'avg_zeros_per_row': zeros_per_row.float().mean().item(),
            'avg_zeros_per_col': zeros_per_col.float().mean().item()
        }
    
    return analysis

def diagnose_causes(analysis_results):
    """Diagnose probable causes based on analysis results"""
    # Find worst layers
    linear_layers = [a for a in analysis_results if 'pattern_analysis' in a]
    linear_layers.sort(key=lambda x: x['zeros_percentage'], reverse=True)
    
    if not linear_layers:
        return []
    
    worst_layer = linear_layers[0]
    zeros_pct = worst_layer['zeros_percentage']
    
    causes = []
    
    # High learning rate indicators
    if zeros_pct > 60:
        causes.append("🔥 PRIMARY CAUSE: Learning rate too high (5e-3)")
        causes.append("   → Weights exploding then getting clipped/zeroed")
        causes.append("   → Immediate fix: Reduce LR to 1e-4")
    
    # Dead neuron patterns
    if 'pattern_analysis' in worst_layer:
        pa = worst_layer['pattern_analysis']
        if pa['dead_row_pct'] > 30:
            causes.append("💀 Systematic neuron death pattern detected")
            causes.append(f"   → {pa['dead_row_pct']:.1f}% of neurons completely dead")
    
    # Magnitude distribution issues
    if zeros_pct > 40:
        causes.append("📉 Gradient clipping/optimization issues")
        causes.append("   → Check gradient norms and clipping settings")
    
    # Pattern-based diagnosis
    high_zero_layers = [a for a in linear_layers if a['zeros_percentage'] > 30]
    if len(high_zero_layers) > 3:
        causes.append("🌊 Systematic problem across multiple layers")
        causes.append("   → Not isolated to single layer initialization")
        causes.append("   → Points to training dynamics issue")
    
    return causes

def test_normal_initialization(shape):
    """Test what normal initialization should look like"""
    if len(shape) != 2:
        return None
    
    rows, cols = shape
    
    # Test different initialization methods
    methods = {
        'xavier_uniform': lambda: nn.init.xavier_uniform_(torch.empty(rows, cols)),
        'kaiming_uniform': lambda: nn.init.kaiming_uniform_(torch.empty(rows, cols), a=0.01, nonlinearity='leaky_relu'),
        'normal_0.01': lambda: nn.init.normal_(torch.empty(rows, cols), 0, 0.01),
    }
    
    results = {}
    for name, init_func in methods.items():
        tensor = init_func()
        zeros = (tensor == 0).sum().item()
        total = tensor.numel()
        zeros_pct = (zeros / total) * 100
        
        results[name] = {
            'zeros_count': zeros,
            'zeros_percentage': zeros_pct,
            'mean': tensor.mean().item(),
            'std': tensor.std().item()
        }
    
    return results

def analyze_checkpoint_directly():
    """Main analysis function"""
    print("🔍 DIRECT CHECKPOINT WEIGHT ANALYSIS")
    print("=" * 60)
    
    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path:
        print("❌ No checkpoints found!")
        return False
    
    print(f"📁 Analyzing: {os.path.basename(checkpoint_path)}")
    print(f"📅 Modified: {datetime.fromtimestamp(os.path.getmtime(checkpoint_path))}")
    print(f"📏 Size: {os.path.getsize(checkpoint_path) / (1024*1024):.1f} MB")
    
    # Extract weights
    state_dict = extract_policy_weights(checkpoint_path)
    if state_dict is None:
        return False
    
    print(f"✅ Policy weights extracted")
    print(f"📊 Found {len(state_dict)} parameter tensors")
    
    # Analyze all weight tensors
    print(f"\n🔍 ANALYZING WEIGHT TENSORS:")
    print("-" * 40)
    
    all_analyses = []
    
    for name, tensor in state_dict.items():
        if tensor is None:
            continue
            
        # Skip non-weight parameters (like biases, buffers)
        if not name.endswith('.weight'):
            continue
            
        # Skip very small tensors
        if tensor.numel() < 10:
            continue
        
        analysis = analyze_weight_tensor(name, tensor)
        if analysis:
            all_analyses.append(analysis)
    
    # Sort by zeros percentage
    all_analyses.sort(key=lambda x: x['zeros_percentage'], reverse=True)
    
    print(f"\n📊 WEIGHT ANALYSIS RESULTS ({len(all_analyses)} layers):")
    print("-" * 50)
    
    total_weights_all = sum(a['total_weights'] for a in all_analyses)
    total_zeros_all = sum(a['zeros_count'] for a in all_analyses)
    overall_zeros_pct = (total_zeros_all / total_weights_all) * 100 if total_weights_all > 0 else 0
    
    print(f"\n🎯 OVERALL STATISTICS:")
    print(f"   Total parameters analyzed: {total_weights_all:,}")
    print(f"   Total zeros: {total_zeros_all:,} ({overall_zeros_pct:.1f}%)")
    
    # Show top problematic layers
    print(f"\n🚨 TOP PROBLEMATIC LAYERS:")
    print("-" * 30)
    
    for i, analysis in enumerate(all_analyses[:10]):  # Top 10
        zeros_pct = analysis['zeros_percentage']
        
        if zeros_pct > 50:
            status = "🚨 CRITICAL"
        elif zeros_pct > 20:
            status = "⚠️  WARNING"
        elif zeros_pct > 5:
            status = "🔸 ELEVATED"
        else:
            status = "✅ NORMAL"
            if i > 3:  # Skip normal layers after top 3
                continue
        
        print(f"\n{i+1}. {status} {analysis['name']}")
        print(f"    Shape: {analysis['shape']}")
        print(f"    Zeros: {analysis['zeros_count']:,} / {analysis['total_weights']:,} ({zeros_pct:.1f}%)")
        
        if zeros_pct > 30 and 'pattern_analysis' in analysis:
            pa = analysis['pattern_analysis']
            print(f"    💀 Dead rows: {pa['dead_rows']} ({pa['dead_row_pct']:.1f}%)")
            print(f"    💀 Dead cols: {pa['dead_cols']} ({pa['dead_col_pct']:.1f}%)")
        
        if zeros_pct > 30 and 'non_zero_stats' in analysis:
            nz = analysis['non_zero_stats']
            print(f"    📊 Non-zero range: [{nz['min']:.6f}, {nz['max']:.6f}]")
        
        # Show initialization comparison for worst layer
        if i == 0 and zeros_pct > 30 and len(analysis['shape']) == 2:
            print(f"\n    🧪 INITIALIZATION COMPARISON:")
            init_results = test_normal_initialization(analysis['shape'])
            if init_results:
                for method, result in init_results.items():
                    print(f"       {method}: {result['zeros_percentage']:.3f}% zeros (normal)")
                
                max_normal = max(r['zeros_percentage'] for r in init_results.values())
                if max_normal == 0:
                    max_normal = 0.001
                ratio = zeros_pct / max_normal
                print(f"       → Actual vs Normal: {ratio:.0f}x higher zeros!")
    
    # Diagnosis
    print(f"\n🔍 ROOT CAUSE DIAGNOSIS:")
    print("-" * 25)
    
    causes = diagnose_causes(all_analyses)
    for cause in causes:
        print(f"   {cause}")
    
    # Critical threshold check
    worst_zeros = all_analyses[0]['zeros_percentage'] if all_analyses else 0
    if worst_zeros > 50:
        print(f"\n🚨 CRITICAL PROBLEM CONFIRMED!")
        print(f"   Worst layer: {all_analyses[0]['name']}")
        print(f"   Zero percentage: {worst_zeros:.1f}%")
        print(f"\n   IMMEDIATE ACTIONS REQUIRED:")
        print(f"   1. 🔧 REDUCE LEARNING RATE: 5e-3 → 1e-4")
        print(f"   2. 🔄 REINITIALIZE WORST LAYERS")
        print(f"   3. 🛡️  CHECK GRADIENT CLIPPING")
        print(f"   4. 🧹 RESET OPTIMIZER STATE")
        
    elif worst_zeros > 20:
        print(f"\n⚠️  MODERATE PROBLEM:")
        print(f"   Monitor training closely")
        print(f"   Consider learning rate reduction")
        
    else:
        print(f"\n✅ NO CRITICAL ISSUES")
        print(f"   Zero percentages within normal range")
    
    return True

if __name__ == "__main__":
    try:
        success = analyze_checkpoint_directly()
        if success:
            print(f"\n✅ DIRECT ANALYSIS COMPLETED")
        else:
            print(f"\n❌ Analysis failed")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
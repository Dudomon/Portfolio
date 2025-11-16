#!/usr/bin/env python3
"""
🔍 COMPLETE CHECKPOINT ANALYZER
Extract and analyze all components of SAC checkpoint
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
import json

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

def explore_checkpoint_contents(checkpoint_path):
    """Explore the complete contents of checkpoint"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(checkpoint_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            print(f"📁 CHECKPOINT CONTENTS:")
            all_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, temp_dir)
                    file_size = os.path.getsize(full_path)
                    all_files.append((rel_path, file_size))
            
            # Sort by size
            all_files.sort(key=lambda x: x[1], reverse=True)
            
            for rel_path, file_size in all_files:
                size_mb = file_size / (1024 * 1024)
                print(f"   {rel_path} ({size_mb:.1f} MB)")
            
            # Try to load each .pth file
            weight_data = {}
            
            for rel_path, file_size in all_files:
                if rel_path.endswith('.pth'):
                    full_path = os.path.join(temp_dir, rel_path)
                    try:
                        print(f"\n🔍 Loading {rel_path}...")
                        data = torch.load(full_path, map_location='cpu')
                        
                        if isinstance(data, dict):
                            print(f"   Dictionary with {len(data)} keys:")
                            for key in list(data.keys())[:10]:  # First 10 keys
                                value = data[key]
                                if torch.is_tensor(value):
                                    print(f"      {key}: tensor {value.shape} ({value.dtype})")
                                else:
                                    print(f"      {key}: {type(value)}")
                            if len(data) > 10:
                                print(f"      ... and {len(data) - 10} more keys")
                            
                            # Store weight data
                            weight_data[rel_path] = data
                            
                        elif torch.is_tensor(data):
                            print(f"   Single tensor: {data.shape} ({data.dtype})")
                            weight_data[rel_path] = {'tensor': data}
                        else:
                            print(f"   Type: {type(data)}")
                            
                    except Exception as e:
                        print(f"   ❌ Error loading {rel_path}: {e}")
            
            return weight_data
            
    except Exception as e:
        print(f"❌ Error exploring checkpoint: {e}")
        return None

def analyze_weight_tensor(name, tensor):
    """Analyze a single weight tensor for zeros"""
    if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
        return None
    
    # Convert to CPU if needed
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    
    # Basic stats
    total_weights = tensor.numel()
    zeros_mask = (tensor == 0.0)
    zeros_count = zeros_mask.sum().item()
    zeros_pct = (zeros_count / total_weights) * 100
    
    analysis = {
        'name': name,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'total_weights': total_weights,
        'zeros_count': zeros_count,
        'zeros_percentage': zeros_pct
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
        
        # Check for very small weights
        tiny_threshold = 1e-6
        small_threshold = 1e-4
        tiny_count = (non_zeros.abs() < tiny_threshold).sum().item()
        small_count = (non_zeros.abs() < small_threshold).sum().item()
        
        analysis['magnitude_analysis'] = {
            'tiny_weights': tiny_count,
            'small_weights': small_count,
            'tiny_pct': (tiny_count / non_zeros.numel()) * 100,
            'small_pct': (small_count / non_zeros.numel()) * 100
        }
    
    # Pattern analysis for 2D tensors
    if len(tensor.shape) == 2:
        rows, cols = tensor.shape
        
        # Count zeros per row and column
        zeros_per_row = zeros_mask.sum(dim=1)
        zeros_per_col = zeros_mask.sum(dim=0)
        
        # Dead neurons/inputs
        dead_rows = (zeros_per_row == cols).sum().item()
        dead_cols = (zeros_per_col == rows).sum().item()
        
        analysis['pattern_analysis'] = {
            'dead_rows': dead_rows,
            'dead_cols': dead_cols, 
            'dead_row_pct': (dead_rows / rows) * 100,
            'dead_col_pct': (dead_cols / cols) * 100,
            'max_zeros_per_row': zeros_per_row.max().item(),
            'avg_zeros_per_row': zeros_per_row.float().mean().item()
        }
    
    return analysis

def analyze_all_weights():
    """Main analysis function"""
    print("🔍 COMPLETE SAC CHECKPOINT ANALYSIS")
    print("=" * 60)
    
    # Find checkpoint
    checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path:
        print("❌ No checkpoints found!")
        return False
    
    print(f"📁 Analyzing: {os.path.basename(checkpoint_path)}")
    print(f"📅 Modified: {datetime.fromtimestamp(os.path.getmtime(checkpoint_path))}")
    print(f"📏 Size: {os.path.getsize(checkpoint_path) / (1024*1024):.1f} MB")
    
    # Explore contents
    weight_data = explore_checkpoint_contents(checkpoint_path)
    if not weight_data:
        return False
    
    # Analyze all tensors
    print(f"\n🔍 ANALYZING ALL WEIGHT TENSORS:")
    print("-" * 40)
    
    all_analyses = []
    
    for file_path, data in weight_data.items():
        print(f"\n📄 Processing {file_path}:")
        
        if isinstance(data, dict):
            for key, value in data.items():
                if torch.is_tensor(value) and value.numel() > 0:
                    # Focus on weight parameters
                    if 'weight' in key.lower() or value.numel() > 100:  # Bias and small tensors
                        full_name = f"{file_path}::{key}"
                        analysis = analyze_weight_tensor(full_name, value)
                        if analysis:
                            all_analyses.append(analysis)
                            
                            # Quick preview
                            zeros_pct = analysis['zeros_percentage']
                            if zeros_pct > 30:
                                status = "🚨 CRITICAL"
                            elif zeros_pct > 10:
                                status = "⚠️  WARNING"
                            else:
                                status = "✅ OK"
                            
                            print(f"      {status} {key}: {analysis['shape']} -> {zeros_pct:.1f}% zeros")
    
    # Sort by zero percentage
    all_analyses.sort(key=lambda x: x['zeros_percentage'], reverse=True)
    
    if not all_analyses:
        print("❌ No weight tensors found!")
        return False
    
    # Statistics
    total_weights = sum(a['total_weights'] for a in all_analyses)
    total_zeros = sum(a['zeros_count'] for a in all_analyses)
    overall_zeros_pct = (total_zeros / total_weights) * 100 if total_weights > 0 else 0
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Layers analyzed: {len(all_analyses)}")
    print(f"   Total weights: {total_weights:,}")
    print(f"   Total zeros: {total_zeros:,} ({overall_zeros_pct:.1f}%)")
    
    # Top problematic layers
    print(f"\n🚨 TOP 10 LAYERS BY ZERO PERCENTAGE:")
    print("-" * 45)
    
    for i, analysis in enumerate(all_analyses[:10]):
        zeros_pct = analysis['zeros_percentage']
        
        if zeros_pct > 50:
            status = "🚨 CRITICAL"
        elif zeros_pct > 20:
            status = "⚠️  WARNING" 
        elif zeros_pct > 5:
            status = "🔸 ELEVATED"
        else:
            status = "✅ NORMAL"
        
        print(f"\n{i+1:2d}. {status} {analysis['name']}")
        print(f"     Shape: {analysis['shape']}")
        print(f"     Zeros: {analysis['zeros_count']:,} / {analysis['total_weights']:,} ({zeros_pct:.1f}%)")
        
        if 'pattern_analysis' in analysis and zeros_pct > 20:
            pa = analysis['pattern_analysis']
            print(f"     💀 Dead rows/neurons: {pa['dead_rows']} ({pa['dead_row_pct']:.1f}%)")
            print(f"     💀 Dead cols/inputs: {pa['dead_cols']} ({pa['dead_col_pct']:.1f}%)")
        
        if 'non_zero_stats' in analysis and zeros_pct > 20:
            nz = analysis['non_zero_stats']
            print(f"     📊 Non-zero range: [{nz['min']:.6f}, {nz['max']:.6f}]")
            print(f"     📊 Mean±std: {nz['mean']:.6f}±{nz['std']:.6f}")
        
        if 'magnitude_analysis' in analysis and zeros_pct > 20:
            ma = analysis['magnitude_analysis'] 
            if ma['tiny_pct'] > 5:
                print(f"     🔬 Tiny weights (<1e-6): {ma['tiny_pct']:.1f}%")
            if ma['small_pct'] > 10:
                print(f"     🔬 Small weights (<1e-4): {ma['small_pct']:.1f}%")
    
    # Critical analysis
    if all_analyses:
        worst = all_analyses[0]
        worst_zeros = worst['zeros_percentage']
        
        print(f"\n🔍 CRITICAL ANALYSIS:")
        print("-" * 25)
        
        if worst_zeros > 60:
            print(f"🚨 SEVERE PROBLEM DETECTED!")
            print(f"   Worst layer: {worst['name']}")
            print(f"   Zero rate: {worst_zeros:.1f}%")
            print(f"\n   PRIMARY CAUSE: Learning rate too high (5e-3)")
            print(f"   → Weight explosion → clipping → zeros")
            print(f"\n   IMMEDIATE FIXES:")
            print(f"   1. Reduce LR: 5e-3 → 1e-4 (25x reduction)")
            print(f"   2. Reinitialize layers with >50% zeros")
            print(f"   3. Check gradient clipping settings")
            print(f"   4. Monitor gradient norms")
            
        elif worst_zeros > 30:
            print(f"⚠️  MODERATE PROBLEM:")
            print(f"   Worst layer: {worst['name']} ({worst_zeros:.1f}% zeros)")
            print(f"   → Consider learning rate reduction")
            print(f"   → Monitor training stability")
            
        else:
            print(f"✅ No critical issues detected")
            print(f"   Maximum zeros: {worst_zeros:.1f}%")
        
        # High zero layer count
        high_zero_layers = [a for a in all_analyses if a['zeros_percentage'] > 30]
        if len(high_zero_layers) > 3:
            print(f"\n🌊 SYSTEMATIC ISSUE:")
            print(f"   {len(high_zero_layers)} layers have >30% zeros")
            print(f"   → Training dynamics problem, not just initialization")
    
    return True

if __name__ == "__main__":
    try:
        success = analyze_all_weights()
        if success:
            print(f"\n✅ COMPLETE ANALYSIS FINISHED")
            print(f"\n📋 SUMMARY:")
            print(f"   - If >50% zeros found: CRITICAL - reduce LR immediately")
            print(f"   - If >20% zeros found: WARNING - monitor closely")
            print(f"   - If <10% zeros: NORMAL - continue training")
        else:
            print(f"\n❌ Analysis failed")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
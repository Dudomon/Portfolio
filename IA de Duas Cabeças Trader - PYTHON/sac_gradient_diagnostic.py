#!/usr/bin/env python3
"""
🔍 SAC v2 Zero Gradient Diagnostic Tool

This script helps diagnose and monitor the zero gradient problem in SAC v2 networks.
Use this to verify that the fixes are working correctly.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import time

class SACGradientDiagnostic:
    """Diagnostic tool for SAC gradient analysis"""
    
    def __init__(self, model, threshold_warning=0.3, threshold_critical=0.7):
        self.model = model
        self.threshold_warning = threshold_warning
        self.threshold_critical = threshold_critical
        
    def analyze_network_gradients(self) -> Dict:
        """
        Analyze gradients across all SAC networks
        
        Returns:
            Dict with gradient analysis for actor, critic, features_extractor
        """
        analysis = {
            'timestamp': time.time(),
            'total_zeros': 0,
            'total_params': 0,
            'networks': {}
        }
        
        if not hasattr(self.model, 'policy'):
            return {'error': 'Model has no policy attribute'}
        
        policy = self.model.policy
        
        # Analyze Features Extractor
        if hasattr(policy, 'features_extractor'):
            fe_analysis = self._analyze_module_gradients(
                policy.features_extractor, 'features_extractor'
            )
            analysis['networks']['features_extractor'] = fe_analysis
        
        # Analyze Actor Network
        if hasattr(policy, 'actor'):
            actor_analysis = self._analyze_module_gradients(
                policy.actor, 'actor'
            )
            analysis['networks']['actor'] = actor_analysis
        
        # Analyze Critic Networks (Twin Q-networks)
        if hasattr(policy, 'critic'):
            critic_analysis = self._analyze_module_gradients(
                policy.critic, 'critic'
            )
            analysis['networks']['critic'] = critic_analysis
        
        # Calculate total statistics
        total_zeros = sum(net['total_zeros'] for net in analysis['networks'].values())
        total_params = sum(net['total_params'] for net in analysis['networks'].values())
        
        analysis['total_zeros'] = total_zeros
        analysis['total_params'] = total_params
        analysis['zero_ratio'] = total_zeros / total_params if total_params > 0 else 0.0
        analysis['health_status'] = self._get_health_status(analysis['zero_ratio'])
        
        return analysis
    
    def _analyze_module_gradients(self, module, module_name: str) -> Dict:
        """Analyze gradients for a specific module"""
        result = {
            'name': module_name,
            'total_zeros': 0,
            'total_params': 0,
            'layers': [],
            'problematic_layers': []
        }
        
        for name, param in module.named_parameters():
            if param.grad is not None:
                grad_array = param.grad.detach().cpu().numpy()
                
                # Count zeros
                zero_count = np.sum(np.abs(grad_array) < 1e-8)
                total_count = grad_array.size
                zero_ratio = zero_count / total_count
                
                layer_analysis = {
                    'name': name,
                    'shape': list(grad_array.shape),
                    'zero_count': int(zero_count),
                    'total_count': int(total_count),
                    'zero_ratio': float(zero_ratio),
                    'grad_norm': float(np.linalg.norm(grad_array)),
                    'grad_mean': float(np.mean(grad_array)),
                    'grad_std': float(np.std(grad_array)),
                }
                
                result['layers'].append(layer_analysis)
                result['total_zeros'] += zero_count
                result['total_params'] += total_count
                
                # Flag problematic layers
                if zero_ratio > self.threshold_critical:
                    layer_analysis['problem_level'] = 'critical'
                    result['problematic_layers'].append(layer_analysis)
                elif zero_ratio > self.threshold_warning:
                    layer_analysis['problem_level'] = 'warning'
                    result['problematic_layers'].append(layer_analysis)
                else:
                    layer_analysis['problem_level'] = 'healthy'
        
        # Calculate module statistics
        if result['total_params'] > 0:
            result['zero_ratio'] = result['total_zeros'] / result['total_params']
            result['health_status'] = self._get_health_status(result['zero_ratio'])
        else:
            result['zero_ratio'] = 0.0
            result['health_status'] = 'no_gradients'
        
        return result
    
    def _get_health_status(self, zero_ratio: float) -> str:
        """Get health status based on zero ratio"""
        if zero_ratio > self.threshold_critical:
            return 'critical'
        elif zero_ratio > self.threshold_warning:
            return 'warning'
        else:
            return 'healthy'
    
    def print_detailed_report(self, analysis: Dict):
        """Print detailed analysis report"""
        print("\n" + "="*80)
        print("🔍 SAC v2 ZERO GRADIENT DIAGNOSTIC REPORT")
        print("="*80)
        
        if 'error' in analysis:
            print(f"❌ ERROR: {analysis['error']}")
            return
        
        # Overall statistics
        overall_ratio = analysis['zero_ratio']
        health_status = analysis['health_status']
        
        status_icon = {
            'healthy': '✅',
            'warning': '⚠️',
            'critical': '🚨',
            'no_gradients': '❓'
        }.get(health_status, '❓')
        
        print(f"\n📊 OVERALL GRADIENT HEALTH: {status_icon} {health_status.upper()}")
        print(f"   Zero Gradient Ratio: {overall_ratio*100:.2f}%")
        print(f"   Total Parameters: {analysis['total_params']:,}")
        print(f"   Zero Gradients: {analysis['total_zeros']:,}")
        
        # Network-specific analysis
        for network_name, network_data in analysis['networks'].items():
            self._print_network_analysis(network_name, network_data)
        
        # Recommendations
        self._print_recommendations(analysis)
        
        print("="*80)
    
    def _print_network_analysis(self, network_name: str, data: Dict):
        """Print analysis for a specific network"""
        status_icon = {
            'healthy': '✅',
            'warning': '⚠️', 
            'critical': '🚨',
            'no_gradients': '❓'
        }.get(data['health_status'], '❓')
        
        print(f"\n🧠 {network_name.upper()} NETWORK: {status_icon} {data['health_status'].upper()}")
        print(f"   Zero Ratio: {data['zero_ratio']*100:.2f}%")
        print(f"   Parameters: {data['total_params']:,}")
        
        # Show problematic layers
        if data['problematic_layers']:
            print(f"   🚨 Problematic Layers ({len(data['problematic_layers'])}):")
            for layer in data['problematic_layers']:
                level_icon = '🚨' if layer['problem_level'] == 'critical' else '⚠️'
                print(f"      {level_icon} {layer['name']}: {layer['zero_ratio']*100:.1f}% zeros")
                print(f"         Shape: {layer['shape']}, Norm: {layer['grad_norm']:.6f}")
        else:
            print("   ✅ No problematic layers detected")
    
    def _print_recommendations(self, analysis: Dict):
        """Print recommendations based on analysis"""
        print(f"\n🎯 RECOMMENDATIONS:")
        
        overall_ratio = analysis['zero_ratio']
        
        if overall_ratio > 0.8:
            print("   🚨 CRITICAL - Immediate action required:")
            print("      • Check network initialization (use Xavier/He initialization)")
            print("      • Verify activation functions (avoid ReLU, use ELU/Swish)")
            print("      • Reduce learning rate (try 1e-5 to 5e-5)")
            print("      • Add gradient clipping (max_norm=1.0)")
            print("      • Use LayerNorm instead of BatchNorm")
            
        elif overall_ratio > 0.5:
            print("   ⚠️ WARNING - Monitoring and adjustments needed:")
            print("      • Monitor gradient flow during training")
            print("      • Consider reducing learning rate")
            print("      • Add LayerNorm to problematic layers")
            print("      • Check for vanishing gradients in deep layers")
            
        elif overall_ratio > 0.2:
            print("   ✅ GOOD - Minor optimizations possible:")
            print("      • Current gradient flow is acceptable")
            print("      • Monitor for any degradation during training")
            print("      • Consider small learning rate adjustments if needed")
            
        else:
            print("   🎉 EXCELLENT - Gradients are healthy!")
            print("      • Continue with current configuration")
            print("      • Monitor periodically to ensure stability")
        
        # Network-specific recommendations
        for network_name, network_data in analysis['networks'].items():
            if network_data['health_status'] == 'critical':
                print(f"\n   🎯 {network_name.upper()} specific fixes:")
                if network_name == 'features_extractor':
                    print("      • Re-initialize features extractor with Xavier normal")
                    print("      • Replace ReLU with ELU activation")
                    print("      • Add LayerNorm after each linear layer")
                elif network_name == 'actor':
                    print("      • Lower actor learning rate specifically")
                    print("      • Check log_std initialization (-3.0 recommended)")
                    print("      • Ensure proper policy network initialization")
                elif network_name == 'critic':
                    print("      • Re-initialize critic networks independently")
                    print("      • Check target network synchronization")
                    print("      • Verify twin Q-network architecture")

def run_diagnostic(model, save_report=True):
    """
    Run complete diagnostic on SAC model
    
    Args:
        model: SAC model instance
        save_report: Whether to save report to file
    """
    diagnostic = SACGradientDiagnostic(model)
    
    print("🔍 Running SAC gradient diagnostic...")
    analysis = diagnostic.analyze_network_gradients()
    
    diagnostic.print_detailed_report(analysis)
    
    if save_report:
        timestamp = int(time.time())
        report_file = f"sac_gradient_diagnostic_{timestamp}.json"
        
        import json
        with open(report_file, 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Recursively convert the analysis dict
            def clean_for_json(d):
                if isinstance(d, dict):
                    return {k: clean_for_json(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [clean_for_json(i) for i in d]
                else:
                    return convert_numpy(d)
            
            json.dump(clean_for_json(analysis), f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_file}")
    
    return analysis

if __name__ == "__main__":
    print("🔍 SAC v2 Zero Gradient Diagnostic Tool")
    print("Import this module and call run_diagnostic(your_sac_model)")
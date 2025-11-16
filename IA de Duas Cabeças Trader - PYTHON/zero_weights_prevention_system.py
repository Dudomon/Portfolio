#!/usr/bin/env python3
"""
🛡️ ZERO WEIGHTS PREVENTION SYSTEM
Real-time monitoring and correction during training to prevent zero weight recurrence
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
from stable_baselines3.common.callbacks import BaseCallback
import warnings

class ZeroWeightsPreventionCallback(BaseCallback):
    """
    🛡️ Prevention callback to monitor and fix zero weights during training
    Prevents the recurrence of the critical zero weights problem
    """
    
    def __init__(
        self,
        check_frequency: int = 1000,
        warning_threshold: float = 10.0,
        critical_threshold: float = 30.0,
        auto_fix: bool = True,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.check_frequency = check_frequency
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.auto_fix = auto_fix
        
        # Tracking
        self.zero_history = []
        self.last_check_step = 0
        self.fixes_applied = 0
        self.alerts_sent = 0
        
    def _on_step(self) -> bool:
        """Check weights every N steps"""
        
        if self.num_timesteps - self.last_check_step >= self.check_frequency:
            self.last_check_step = self.num_timesteps
            
            # Check for zero weights problem
            issues = self._check_zero_weights()
            
            if issues:
                self._handle_zero_weights_issues(issues)
                
        return True
    
    def _check_zero_weights(self) -> List[Dict]:
        """Check all model weights for zero patterns"""
        
        issues = []
        
        if hasattr(self.training_env, 'get_attr'):
            # Get the model from environment
            model = getattr(self.locals.get('self', None), 'policy', None)
            if model is None:
                return issues
        else:
            # Fallback: try to get from callback locals
            model = self.locals.get('policy', None)
            if model is None:
                model = self.model.policy if hasattr(self.model, 'policy') else None
                if model is None:
                    return issues
        
        # Check all named parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            tensor = param.data
            if tensor.numel() == 0:
                continue
            
            # Calculate zero percentage
            zeros_count = (tensor == 0).sum().item()
            total_count = tensor.numel()
            zeros_pct = (zeros_count / total_count) * 100
            
            # Check for problematic patterns
            if zeros_pct > self.critical_threshold:
                issues.append({
                    'name': name,
                    'zeros_percentage': zeros_pct,
                    'shape': tuple(tensor.shape),
                    'severity': 'CRITICAL',
                    'tensor_ref': param
                })
            elif zeros_pct > self.warning_threshold:
                issues.append({
                    'name': name,
                    'zeros_percentage': zeros_pct,
                    'shape': tuple(tensor.shape),
                    'severity': 'WARNING',
                    'tensor_ref': param
                })
        
        # Store history
        if issues:
            self.zero_history.append({
                'step': self.num_timesteps,
                'issues_count': len(issues),
                'worst_zeros_pct': max(issue['zeros_percentage'] for issue in issues)
            })
        
        return issues
    
    def _handle_zero_weights_issues(self, issues: List[Dict]):
        """Handle detected zero weight issues"""
        
        critical_issues = [i for i in issues if i['severity'] == 'CRITICAL']
        warning_issues = [i for i in issues if i['severity'] == 'WARNING']
        
        # Log issues
        if critical_issues:
            print(f"\n🚨 CRITICAL ZERO WEIGHTS DETECTED at step {self.num_timesteps}:")
            for issue in critical_issues[:3]:  # Top 3 worst
                print(f"   {issue['name']}: {issue['zeros_percentage']:.1f}% zeros")
        
        if warning_issues:
            print(f"\n⚠️  Warning: {len(warning_issues)} layers with elevated zeros")
        
        # Apply fixes if enabled
        if self.auto_fix and critical_issues:
            self._apply_emergency_fixes(critical_issues)
        
        # Update tracking
        self.alerts_sent += len(issues)
    
    def _apply_emergency_fixes(self, critical_issues: List[Dict]):
        """Apply emergency fixes to critical zero weight issues"""
        
        fixes_applied = 0
        
        for issue in critical_issues:
            name = issue['name']
            tensor_ref = issue['tensor_ref']
            tensor = tensor_ref.data
            
            try:
                # Focus on bias parameters (most common issue)
                if 'bias' in name.lower():
                    print(f"   🔧 Emergency fix: {name}")
                    
                    if 'lstm' in name.lower() and 'bias_ih_l0' in name:
                        # LSTM forget gate fix
                        hidden_size = tensor.size(0) // 4
                        nn.init.zeros_(tensor)
                        tensor.data[hidden_size:2*hidden_size].fill_(1.0)
                        fixes_applied += 1
                        
                    elif 'gru' in name.lower():
                        # GRU bias fix
                        nn.init.uniform_(tensor, -0.01, 0.01)
                        fixes_applied += 1
                        
                    else:
                        # General bias fix
                        nn.init.uniform_(tensor, -0.01, 0.01)
                        fixes_applied += 1
                
                # Fix weight parameters if needed
                elif 'weight' in name.lower() and issue['zeros_percentage'] > 80:
                    print(f"   🔧 Emergency weight fix: {name}")
                    
                    if len(tensor.shape) == 2:
                        # Linear layer
                        nn.init.xavier_uniform_(tensor)
                        fixes_applied += 1
                    elif len(tensor.shape) == 1:
                        # 1D weights (LayerNorm, etc.)
                        nn.init.ones_(tensor)
                        fixes_applied += 1
                
            except Exception as e:
                print(f"   ❌ Fix failed for {name}: {e}")
        
        if fixes_applied > 0:
            print(f"   ✅ Applied {fixes_applied} emergency fixes")
            self.fixes_applied += fixes_applied
            
            # Force optimizer state reset for fixed parameters
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                print(f"   🧹 Resetting optimizer state for fixed parameters")
                optimizer = self.model.policy.optimizer
                
                # Clear optimizer state for fixed parameters
                for issue in critical_issues:
                    param = issue['tensor_ref']
                    param_id = id(param)
                    if param_id in optimizer.state:
                        del optimizer.state[param_id]
    
    def _on_training_end(self) -> None:
        """Report final statistics"""
        
        print(f"\n🛡️ ZERO WEIGHTS PREVENTION SUMMARY:")
        print(f"   Alerts sent: {self.alerts_sent}")
        print(f"   Emergency fixes applied: {self.fixes_applied}")
        print(f"   History entries: {len(self.zero_history)}")
        
        if self.zero_history:
            worst_episode = max(self.zero_history, key=lambda x: x['worst_zeros_pct'])
            print(f"   Worst zeros episode: {worst_episode['worst_zeros_pct']:.1f}% at step {worst_episode['step']}")

class LearningRateValidator:
    """
    🔍 Validate learning rate settings to prevent zero weights
    """
    
    @staticmethod
    def validate_learning_rate(lr: float, algorithm: str = "SAC") -> Dict[str, Any]:
        """Validate if learning rate is appropriate"""
        
        # Safe learning rate ranges by algorithm
        safe_ranges = {
            "SAC": (1e-5, 3e-4),
            "PPO": (1e-5, 1e-3),
            "A2C": (1e-4, 1e-2)
        }
        
        min_safe, max_safe = safe_ranges.get(algorithm, (1e-5, 1e-3))
        
        result = {
            'lr': lr,
            'algorithm': algorithm,
            'safe_range': safe_ranges[algorithm],
            'is_safe': min_safe <= lr <= max_safe,
            'recommendations': []
        }
        
        if lr > max_safe:
            danger_level = lr / max_safe
            result['danger_level'] = danger_level
            result['recommendations'].append(f"🚨 CRITICAL: LR {lr} is {danger_level:.1f}x too high!")
            result['recommendations'].append(f"   → Reduce to {max_safe} immediately")
            result['recommendations'].append(f"   → High LR causes weight explosion → zeros")
            
        elif lr < min_safe:
            result['recommendations'].append(f"⚠️  LR {lr} may be too low for effective learning")
            result['recommendations'].append(f"   → Consider increasing to {min_safe}")
            
        else:
            result['recommendations'].append(f"✅ LR {lr} is within safe range")
        
        return result

def create_zero_prevention_system(
    check_frequency: int = 1000,
    auto_fix: bool = True,
    verbose: int = 1
) -> ZeroWeightsPreventionCallback:
    """
    Create a complete zero weights prevention system
    
    Args:
        check_frequency: How often to check weights (steps)
        auto_fix: Whether to automatically fix critical issues
        verbose: Verbosity level
    
    Returns:
        Prevention callback ready for training
    """
    
    callback = ZeroWeightsPreventionCallback(
        check_frequency=check_frequency,
        warning_threshold=10.0,
        critical_threshold=30.0,
        auto_fix=auto_fix,
        verbose=verbose
    )
    
    return callback

def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate complete training configuration for zero weights prevention
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Validation results with recommendations
    """
    
    results = {
        'valid': True,
        'issues': [],
        'recommendations': []
    }
    
    # Check learning rate
    lr = config.get('learning_rate', None)
    if lr is not None:
        lr_result = LearningRateValidator.validate_learning_rate(lr, "SAC")
        if not lr_result['is_safe']:
            results['valid'] = False
            results['issues'].extend(lr_result['recommendations'])
    
    # Check gradient clipping
    max_grad_norm = config.get('max_grad_norm', None)
    if max_grad_norm is not None:
        if max_grad_norm < 0.5:
            results['issues'].append(f"⚠️  max_grad_norm {max_grad_norm} may be too aggressive")
            results['recommendations'].append(f"   → Consider increasing to 1.0")
        elif max_grad_norm > 5.0:
            results['issues'].append(f"⚠️  max_grad_norm {max_grad_norm} may be too lenient")
            results['recommendations'].append(f"   → Consider reducing to 1.0-2.0")
    
    # Check other settings
    if config.get('clip_grad_value') is not None:
        results['issues'].append(f"⚠️  clip_grad_value can cause zero weights")
        results['recommendations'].append(f"   → Use max_grad_norm instead")
    
    return results

if __name__ == "__main__":
    print("🛡️ ZERO WEIGHTS PREVENTION SYSTEM")
    print("=" * 50)
    
    # Test learning rate validation
    test_config = {
        'learning_rate': 5e-3,  # The problematic LR
        'max_grad_norm': 1.0
    }
    
    validation = validate_training_config(test_config)
    
    print(f"Configuration validation:")
    print(f"Valid: {validation['valid']}")
    
    for issue in validation['issues']:
        print(f"  {issue}")
    
    for rec in validation['recommendations']:
        print(f"  {rec}")
    
    print(f"\n✅ Prevention system ready for deployment")
    print(f"   → Use create_zero_prevention_system() in training callbacks")
    print(f"   → Monitor will check every 1000 steps")
    print(f"   → Auto-fix will repair critical issues immediately")
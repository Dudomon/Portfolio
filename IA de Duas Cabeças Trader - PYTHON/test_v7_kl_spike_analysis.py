#!/usr/bin/env python3
"""
🔍 V7 INTUITION KL SPIKE DIAGNOSTIC TEST

Análise arquitetural completa do V7 Intuition buscando causas de KL divergence spikes.
Testa todos os componentes sistematicamente sem fazer alterações.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Imports necessários
from trading_framework.policies.two_head_v7_intuition import TwoHeadV7Intuition, get_v7_intuition_kwargs
from trading_framework.policies.two_head_v7_simple import EnhancedFeaturesExtractor
from gym import spaces
import traceback

class V7IntuitionKLDiagnostic:
    """🔍 Sistema de diagnóstico completo para KL spikes no V7 Intuition"""
    
    def __init__(self):
        self.results = {
            'architectural_issues': [],
            'potential_kl_causes': [],
            'gradient_flow_problems': [],
            'initialization_issues': [],
            'numerical_instabilities': [],
            'recommendations': []
        }
        
        # Create test environment
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(2580,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(8,), dtype=np.float32
        )
        
    def analyze_architecture(self) -> Dict[str, Any]:
        """🏗️ Análise arquitetural completa"""
        print("🔍 INICIANDO ANÁLISE ARQUITETURAL V7 INTUITION")
        print("=" * 80)
        
        try:
            # 1. Test policy creation
            self._test_policy_creation()
            
            # 2. Analyze unified backbone
            self._analyze_unified_backbone()
            
            # 3. Test gradient flow
            self._test_gradient_flow()
            
            # 4. Check activation saturation
            self._check_activation_saturation()
            
            # 5. Analyze learning rate conflicts
            self._analyze_lr_conflicts()
            
            # 6. Test numerical stability
            self._test_numerical_stability()
            
            # 7. Check initialization patterns
            self._check_initialization_patterns()
            
            # 8. Analyze action space mapping
            self._analyze_action_space_mapping()
            
            # 9. Test memory components
            self._test_memory_components()
            
            # 10. Overall assessment
            self._generate_final_assessment()
            
        except Exception as e:
            self.results['architectural_issues'].append(f"CRITICAL ERROR: {e}")
            traceback.print_exc()
        
        return self.results
    
    def _test_policy_creation(self):
        """🧪 Test basic policy creation"""
        print("1. 🧪 Testing Policy Creation...")
        
        try:
            kwargs = get_v7_intuition_kwargs()
            kwargs['critic_learning_rate'] = 2e-5  # From BEST_PARAMS
            
            policy = TwoHeadV7Intuition(
                observation_space=self.observation_space,
                action_space=self.action_space,
                lr_schedule=lambda x: 2e-5,
                **kwargs
            )
            
            print("   ✅ Policy created successfully")
            print(f"   📊 Action space: {self.action_space.shape}")
            print(f"   📊 Observation space: {self.observation_space.shape}")
            
            # Store policy for later tests
            self.test_policy = policy
            
            # Check critical components
            critical_components = [
                'unified_backbone', 'actor_lstm', 'critic_lstm', 
                'entry_head', 'management_head', 'enhanced_memory'
            ]
            
            missing_components = []
            for component in critical_components:
                if not hasattr(policy, component):
                    missing_components.append(component)
            
            if missing_components:
                self.results['architectural_issues'].append(
                    f"Missing critical components: {missing_components}"
                )
            else:
                print("   ✅ All critical components present")
                
        except Exception as e:
            self.results['architectural_issues'].append(f"Policy creation failed: {e}")
            raise
    
    def _analyze_unified_backbone(self):
        """🧠 Analyze unified backbone architecture"""
        print("2. 🧠 Analyzing Unified Backbone...")
        
        backbone = self.test_policy.unified_backbone
        
        # Test forward pass
        try:
            test_input = torch.randn(2, 256)  # batch_size=2, features=256
            actor_features, critic_features, regime_id, info = backbone(test_input)
            
            print(f"   📊 Actor features shape: {actor_features.shape}")
            print(f"   📊 Critic features shape: {critic_features.shape}")
            print(f"   📊 Regime detected: {info['regime_name']}")
            print(f"   📊 Specialization divergence: {info['specialization_divergence']:.4f}")
            
            # Check for potential issues
            if info['specialization_divergence'] < 0.1:
                self.results['potential_kl_causes'].append(
                    "LOW SPECIALIZATION: Actor/Critic too similar (backbone interference)"
                )
            
            if info['actor_attention_mean'] < 0.1 or info['critic_attention_mean'] < 0.1:
                self.results['potential_kl_causes'].append(
                    "ATTENTION COLLAPSE: Low attention values detected"
                )
            
            # Check gradient gates (potential sigmoid saturation)
            self._check_gradient_gates(backbone)
            
        except Exception as e:
            self.results['architectural_issues'].append(f"Backbone forward pass failed: {e}")
    
    def _check_gradient_gates(self, backbone):
        """🚪 Check gradient gates for saturation"""
        print("   🚪 Checking Gradient Gates...")
        
        # Test gates with various inputs
        test_inputs = [
            torch.randn(2, 512),  # Normal
            torch.randn(2, 512) * 3,  # Large values
            torch.randn(2, 512) * 0.1,  # Small values
            torch.zeros(2, 512),  # Zeros
            torch.ones(2, 512) * 5,  # Large positive
            torch.ones(2, 512) * -5,  # Large negative
        ]
        
        saturation_issues = []
        
        for i, test_input in enumerate(test_inputs):
            try:
                # Actor gate
                actor_gate_raw = backbone.actor_gate(test_input)
                actor_gate_final = (actor_gate_raw + 1.0) / 2.0
                
                # Critic gate
                critic_gate_raw = backbone.critic_gate(test_input)
                critic_gate_final = (critic_gate_raw + 1.0) / 2.0
                
                # Check saturation
                actor_saturated = (actor_gate_final < 0.01).sum().item() + (actor_gate_final > 0.99).sum().item()
                critic_saturated = (critic_gate_final < 0.01).sum().item() + (critic_gate_final > 0.99).sum().item()
                
                total_elements = actor_gate_final.numel()
                actor_sat_pct = (actor_saturated / total_elements) * 100
                critic_sat_pct = (critic_saturated / total_elements) * 100
                
                if actor_sat_pct > 20:  # More than 20% saturated
                    saturation_issues.append(f"Actor gate {actor_sat_pct:.1f}% saturated (input {i})")
                
                if critic_sat_pct > 20:
                    saturation_issues.append(f"Critic gate {critic_sat_pct:.1f}% saturated (input {i})")
                    
            except Exception as e:
                saturation_issues.append(f"Gate test {i} failed: {e}")
        
        if saturation_issues:
            self.results['potential_kl_causes'].append(
                f"GATE SATURATION: {'; '.join(saturation_issues)}"
            )
            print(f"   ⚠️ Gate saturation detected: {len(saturation_issues)} issues")
        else:
            print("   ✅ Gates appear healthy")
    
    def _test_gradient_flow(self):
        """🌊 Test gradient flow through network"""
        print("3. 🌊 Testing Gradient Flow...")
        
        policy = self.test_policy
        
        # Create test batch with correct dimensions (2580 = 129 features × 20 bars)
        test_obs = torch.randn(4, 2580, requires_grad=True)  # batch_size=4
        episode_starts = torch.zeros(4, dtype=torch.bool)
        
        try:
            # Test actor forward
            features = policy.extract_features(test_obs)
            actions, lstm_states, gate_info = policy.forward_actor(
                features, None, episode_starts
            )
            
            # Test critic forward  
            values, _ = policy.forward_critic(features, None, episode_starts)
            
            print(f"   📊 Actions shape: {actions.shape}")
            print(f"   📊 Values shape: {values.shape}")
            
            # Test gradients
            actor_loss = actions.sum()
            critic_loss = values.sum()
            
            # Backward pass
            actor_loss.backward(retain_graph=True)
            critic_loss.backward(retain_graph=True)
            
            # Check gradient magnitudes
            grad_stats = self._analyze_gradients(policy)
            
            if grad_stats['zero_grad_params'] > 10:
                self.results['gradient_flow_problems'].append(
                    f"HIGH ZERO GRADIENTS: {grad_stats['zero_grad_params']} parameters with zero gradients"
                )
            
            if grad_stats['exploding_grads'] > 0:
                self.results['gradient_flow_problems'].append(
                    f"EXPLODING GRADIENTS: {grad_stats['exploding_grads']} parameters with large gradients"
                )
            
            print(f"   📊 Zero grad params: {grad_stats['zero_grad_params']}")
            print(f"   📊 Mean grad magnitude: {grad_stats['mean_grad_mag']:.6f}")
            print(f"   📊 Max grad magnitude: {grad_stats['max_grad_mag']:.6f}")
            
        except Exception as e:
            self.results['gradient_flow_problems'].append(f"Gradient flow test failed: {e}")
    
    def _analyze_gradients(self, policy) -> Dict[str, Any]:
        """📊 Analyze gradient statistics"""
        grad_mags = []
        zero_count = 0
        exploding_count = 0
        
        for name, param in policy.named_parameters():
            if param.grad is not None:
                grad_mag = param.grad.abs().mean().item()
                grad_mags.append(grad_mag)
                
                if grad_mag < 1e-8:
                    zero_count += 1
                elif grad_mag > 10.0:
                    exploding_count += 1
        
        return {
            'zero_grad_params': zero_count,
            'exploding_grads': exploding_count,
            'mean_grad_mag': np.mean(grad_mags) if grad_mags else 0,
            'max_grad_mag': np.max(grad_mags) if grad_mags else 0,
            'total_params_with_grads': len(grad_mags)
        }
    
    def _check_activation_saturation(self):
        """⚡ Check for activation function saturation"""
        print("4. ⚡ Checking Activation Saturation...")
        
        policy = self.test_policy
        
        # Test with extreme inputs (correct 2580 dimensions for transformer)
        extreme_inputs = [
            torch.randn(2, 2580) * 10,    # Large values
            torch.randn(2, 2580) * 0.01,  # Small values  
            torch.ones(2, 2580) * 5,      # Positive extreme
            torch.ones(2, 2580) * -5,     # Negative extreme
        ]
        
        saturation_results = []
        
        for i, test_input in enumerate(extreme_inputs):
            try:
                features = policy.extract_features(test_input)
                actions, _, _ = policy.forward_actor(features, None, torch.zeros(2, dtype=torch.bool))
                
                # Check action ranges
                action_stats = {
                    'min': actions.min().item(),
                    'max': actions.max().item(),
                    'mean': actions.mean().item(),
                    'std': actions.std().item()
                }
                
                # Check for saturation indicators
                if action_stats['std'] < 0.1:  # Very low variance
                    saturation_results.append(f"Input {i}: Low variance (std={action_stats['std']:.4f})")
                
                if abs(action_stats['mean']) > 3:  # Mean too far from center
                    saturation_results.append(f"Input {i}: Extreme mean (mean={action_stats['mean']:.4f})")
                
            except Exception as e:
                saturation_results.append(f"Input {i}: Forward failed - {e}")
        
        if saturation_results:
            self.results['potential_kl_causes'].append(
                f"ACTIVATION SATURATION: {'; '.join(saturation_results)}"
            )
            print(f"   ⚠️ Activation issues detected: {len(saturation_results)}")
        else:
            print("   ✅ Activations appear healthy")
    
    def _analyze_lr_conflicts(self):
        """⚔️ Analyze learning rate conflicts"""
        print("5. ⚔️ Analyzing Learning Rate Conflicts...")
        
        policy = self.test_policy
        
        # Check if separate optimizers are configured
        try:
            actor_opt, critic_opt = policy.get_actor_critic_optimizers()
            
            actor_lr = actor_opt.param_groups[0]['lr']
            critic_lr = critic_opt.param_groups[0]['lr']
            
            print(f"   📊 Actor LR: {actor_lr:.2e}")
            print(f"   📊 Critic LR: {critic_lr:.2e}")
            
            lr_ratio = critic_lr / actor_lr if actor_lr > 0 else float('inf')
            
            if lr_ratio > 10 or lr_ratio < 0.1:
                self.results['potential_kl_causes'].append(
                    f"EXTREME LR RATIO: Critic/Actor LR ratio = {lr_ratio:.2f} (may cause instability)"
                )
            
            # Check for very small LRs
            if actor_lr < 1e-6 or critic_lr < 1e-6:
                self.results['potential_kl_causes'].append(
                    f"VERY SMALL LRs: Actor={actor_lr:.2e}, Critic={critic_lr:.2e} (may cause slow adaptation)"
                )
            
            # Check for very large LRs
            if actor_lr > 1e-3 or critic_lr > 1e-3:
                self.results['potential_kl_causes'].append(
                    f"LARGE LRs: Actor={actor_lr:.2e}, Critic={critic_lr:.2e} (may cause instability)"
                )
            
        except Exception as e:
            self.results['potential_kl_causes'].append(f"LR analysis failed: {e}")
    
    def _test_numerical_stability(self):
        """🔢 Test numerical stability"""
        print("6. 🔢 Testing Numerical Stability...")
        
        policy = self.test_policy
        
        # Test with edge case inputs - more conservative ranges for stability
        edge_cases = [
            torch.zeros(2, 2580),                    # All zeros
            torch.ones(2, 2580) * 1e3,               # Large (more conservative)
            torch.ones(2, 2580) * 1e-6,              # Very small (more conservative)
            torch.ones(2, 2580) * 1e4,               # Maximum safe value
            torch.randn(2, 2580) * 100,              # Controlled high variance
        ]
        
        stability_issues = []
        
        for i, test_input in enumerate(edge_cases):
            try:
                # Replace inf/nan with large numbers for testability
                if torch.isinf(test_input).any() or torch.isnan(test_input).any():
                    test_input = torch.clamp(test_input, -1e4, 1e4)  # More conservative clamp
                    test_input = torch.nan_to_num(test_input, nan=0.0, posinf=1e4, neginf=-1e4)
                
                features = policy.extract_features(test_input)
                actions, _, _ = policy.forward_actor(features, None, torch.zeros(2, dtype=torch.bool))
                values, _ = policy.forward_critic(features, None, torch.zeros(2, dtype=torch.bool))
                
                # Check for NaN/Inf outputs
                if torch.isnan(actions).any():
                    stability_issues.append(f"Input {i}: Actions contain NaN")
                if torch.isinf(actions).any():
                    stability_issues.append(f"Input {i}: Actions contain Inf")
                if torch.isnan(values).any():
                    stability_issues.append(f"Input {i}: Values contain NaN")
                if torch.isinf(values).any():
                    stability_issues.append(f"Input {i}: Values contain Inf")
                
            except Exception as e:
                stability_issues.append(f"Input {i}: Forward failed - {str(e)[:100]}")
        
        if stability_issues:
            self.results['numerical_instabilities'].extend(stability_issues)
            print(f"   ⚠️ Numerical instabilities: {len(stability_issues)}")
        else:
            print("   ✅ Numerical stability looks good")
    
    def _check_initialization_patterns(self):
        """🎯 Check weight initialization patterns"""
        print("7. 🎯 Checking Initialization Patterns...")
        
        policy = self.test_policy
        
        init_issues = []
        
        # Check key layers
        key_components = {
            'actor_head': policy.actor_head,
            'critic_head': policy.critic_head,
            'unified_backbone': policy.unified_backbone,
        }
        
        for comp_name, component in key_components.items():
            for name, param in component.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    # Check weight statistics
                    weight_mean = param.data.mean().item()
                    weight_std = param.data.std().item()
                    weight_max = param.data.abs().max().item()
                    
                    # Check for problematic patterns
                    if abs(weight_mean) > 0.5:
                        init_issues.append(f"{comp_name}.{name}: High mean ({weight_mean:.3f})")
                    
                    if weight_std < 0.001:
                        init_issues.append(f"{comp_name}.{name}: Very low std ({weight_std:.6f})")
                    
                    if weight_std > 2.0:
                        init_issues.append(f"{comp_name}.{name}: Very high std ({weight_std:.3f})")
                    
                    if weight_max > 10.0:
                        init_issues.append(f"{comp_name}.{name}: Extreme weights (max={weight_max:.3f})")
        
        if init_issues:
            self.results['initialization_issues'].extend(init_issues)
            print(f"   ⚠️ Initialization issues: {len(init_issues)}")
        else:
            print("   ✅ Initialization patterns look reasonable")
    
    def _analyze_action_space_mapping(self):
        """🎯 Analyze action space mapping for KL issues"""
        print("8. 🎯 Analyzing Action Space Mapping...")
        
        policy = self.test_policy
        
        # Test action generation with various inputs
        test_inputs = [
            torch.randn(10, 2580),       # Random normal
            torch.zeros(10, 2580),       # Zeros
            torch.ones(10, 2580),        # Ones
            torch.randn(10, 2580) * 3,   # High variance
        ]
        
        action_issues = []
        
        for i, test_input in enumerate(test_inputs):
            try:
                features = policy.extract_features(test_input)
                actions, _, _ = policy.forward_actor(features, None, torch.zeros(10, dtype=torch.bool))
                
                # Analyze each action dimension
                for dim in range(actions.shape[1]):
                    dim_actions = actions[:, dim]
                    
                    dim_mean = dim_actions.mean().item()
                    dim_std = dim_actions.std().item()
                    dim_min = dim_actions.min().item()
                    dim_max = dim_actions.max().item()
                    
                    # Check for problematic patterns
                    if dim_std < 0.01:  # Very low variation
                        action_issues.append(f"Input {i}, Action {dim}: Low variation (std={dim_std:.6f})")
                    
                    # Check action 1 (entry_confidence) - should be [0,1]
                    if dim == 1 and (dim_min < -0.1 or dim_max > 1.1):
                        action_issues.append(f"Input {i}, Action {dim}: Out of range [0,1] ({dim_min:.3f}, {dim_max:.3f})")
                    
                    # Check actions 2-7 (adjustments) - should be [-3,3]
                    if dim >= 2 and (dim_min < -4 or dim_max > 4):
                        action_issues.append(f"Input {i}, Action {dim}: Out of range [-3,3] ({dim_min:.3f}, {dim_max:.3f})")
                
            except Exception as e:
                action_issues.append(f"Input {i}: Action generation failed - {e}")
        
        if action_issues:
            self.results['potential_kl_causes'].extend(action_issues)
            print(f"   ⚠️ Action space issues: {len(action_issues)}")
        else:
            print("   ✅ Action space mapping looks good")
    
    def _test_memory_components(self):
        """🧠 Test memory components"""
        print("9. 🧠 Testing Memory Components...")
        
        policy = self.test_policy
        
        memory_issues = []
        
        # Test enhanced memory
        try:
            memory = policy.enhanced_memory
            
            # Test memory storage
            dummy_state = np.random.randn(256)
            dummy_action = np.random.randn(3)
            
            memory.store_memory(
                regime=1,
                state=dummy_state,
                action=dummy_action,
                reward=0.5,
                next_state=dummy_state,
                done=False
            )
            
            # Test memory retrieval
            context = memory.get_regime_context(1, dummy_state)
            
            if len(context) == 0:
                memory_issues.append("Enhanced memory returns empty context")
            
            print(f"   📊 Memory context size: {len(context)}")
            
        except Exception as e:
            memory_issues.append(f"Enhanced memory test failed: {e}")
        
        # Test LSTM memory buffers
        try:
            # Test critic memory buffer
            if hasattr(policy, 'critic_memory_buffer'):
                print("   📊 Critic memory buffer exists")
            else:
                memory_issues.append("Critic memory buffer not initialized")
        
        except Exception as e:
            memory_issues.append(f"LSTM memory test failed: {e}")
        
        if memory_issues:
            self.results['architectural_issues'].extend(memory_issues)
            print(f"   ⚠️ Memory issues: {len(memory_issues)}")
        else:
            print("   ✅ Memory components working")
    
    def _generate_final_assessment(self):
        """📋 Generate final assessment and recommendations"""
        print("10. 📋 Generating Final Assessment...")
        
        # Count issues by category
        issue_counts = {
            'architectural': len(self.results['architectural_issues']),
            'kl_causes': len(self.results['potential_kl_causes']),
            'gradient_flow': len(self.results['gradient_flow_problems']),
            'initialization': len(self.results['initialization_issues']),
            'numerical': len(self.results['numerical_instabilities']),
        }
        
        total_issues = sum(issue_counts.values())
        
        print(f"   📊 Total issues found: {total_issues}")
        for category, count in issue_counts.items():
            if count > 0:
                print(f"   📊 {category.title()}: {count} issues")
        
        # Generate recommendations based on findings
        recommendations = []
        
        if issue_counts['kl_causes'] > 0:
            recommendations.append("🎯 PRIORITY: Address KL divergence causes first")
        
        if issue_counts['gradient_flow'] > 0:
            recommendations.append("🌊 Fix gradient flow issues - may be root cause")
        
        if issue_counts['numerical'] > 0:
            recommendations.append("🔢 Improve numerical stability - use gradient clipping")
        
        if issue_counts['architectural'] > 0:
            recommendations.append("🏗️ Review architecture - fundamental issues detected")
        
        # Specific recommendations based on common patterns
        all_issues = (
            self.results['architectural_issues'] + 
            self.results['potential_kl_causes'] + 
            self.results['gradient_flow_problems'] + 
            self.results['initialization_issues'] + 
            self.results['numerical_instabilities']
        )
        
        issue_text = ' '.join(all_issues).lower()
        
        if 'saturation' in issue_text:
            recommendations.append("⚡ Replace saturating activations (sigmoid/tanh) with ReLU variants")
        
        if 'gate' in issue_text:
            recommendations.append("🚪 Review gate mechanisms - may cause gradient blocking")
        
        if 'lr' in issue_text or 'learning rate' in issue_text:
            recommendations.append("📚 Tune learning rates - current rates may be suboptimal")
        
        if 'zero' in issue_text and 'gradient' in issue_text:
            recommendations.append("🔄 Check dead neurons - may need weight initialization fix")
        
        if not recommendations:
            recommendations.append("✅ Architecture appears healthy - KL spikes may be data/environment related")
        
        self.results['recommendations'] = recommendations
        
        print("   📋 Assessment complete!")
    
    def print_detailed_report(self):
        """📄 Print detailed diagnostic report"""
        print("\n" + "="*80)
        print("🔍 V7 INTUITION KL SPIKE DIAGNOSTIC REPORT")
        print("="*80)
        
        categories = [
            ('🏗️ ARCHITECTURAL ISSUES', 'architectural_issues'),
            ('🎯 POTENTIAL KL CAUSES', 'potential_kl_causes'),
            ('🌊 GRADIENT FLOW PROBLEMS', 'gradient_flow_problems'),
            ('🎯 INITIALIZATION ISSUES', 'initialization_issues'),
            ('🔢 NUMERICAL INSTABILITIES', 'numerical_instabilities'),
            ('💡 RECOMMENDATIONS', 'recommendations')
        ]
        
        for title, key in categories:
            issues = self.results[key]
            print(f"\n{title}:")
            if issues:
                for i, issue in enumerate(issues, 1):
                    print(f"   {i:2d}. {issue}")
            else:
                print("   ✅ None detected")
        
        print("\n" + "="*80)
        print("🎯 SUMMARY")
        print("="*80)
        
        total_issues = sum(len(self.results[key]) for key in self.results if key != 'recommendations')
        
        if total_issues == 0:
            print("✅ No critical architectural issues detected.")
            print("   KL spikes may be due to:")
            print("   • Data distribution changes")
            print("   • Environment reward scaling")
            print("   • PPO hyperparameter tuning needed")
            print("   • Learning rate scheduling")
        else:
            print(f"⚠️ {total_issues} potential issues detected.")
            print("   Address issues in order of priority:")
            print("   1. KL divergence causes (most critical)")
            print("   2. Gradient flow problems")
            print("   3. Numerical instabilities")
            print("   4. Initialization issues")
            print("   5. Architectural issues")
        
        print("\n" + "="*80)


def main():
    """🚀 Run V7 Intuition KL Spike Diagnostic"""
    print("🔍 V7 INTUITION KL SPIKE DIAGNOSTIC TEST")
    print("Analyzing architecture for potential KL divergence spike causes...")
    print()
    
    diagnostic = V7IntuitionKLDiagnostic()
    results = diagnostic.analyze_architecture()
    diagnostic.print_detailed_report()
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Diagnostic completed successfully!")
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        traceback.print_exc()
        sys.exit(1)
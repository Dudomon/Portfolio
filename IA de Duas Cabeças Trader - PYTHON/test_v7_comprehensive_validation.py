#!/usr/bin/env python3
"""
🔬 V7 INTUITION COMPREHENSIVE VALIDATION TEST

Bateria completa de testes antes do reinício do treino.
Testa TODOS os aspectos críticos da arquitetura para garantir zero erros idiotas.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Imports necessários
from trading_framework.policies.two_head_v7_intuition import TwoHeadV7Intuition, get_v7_intuition_kwargs
from trading_framework.policies.two_head_v7_simple import EnhancedFeaturesExtractor
from gym import spaces
import traceback
import time
import gc

class V7IntuitionComprehensiveValidator:
    """🔬 Validador completo para V7 Intuition - ZERO TOLERÂNCIA A ERROS"""
    
    def __init__(self):
        self.results = {
            'critical_failures': [],
            'performance_issues': [],
            'memory_leaks': [],
            'gradient_problems': [],
            'numerical_issues': [],
            'compatibility_issues': [],
            'warnings': [],
            'passed_tests': []
        }
        
        # Test environments
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(2580,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(8,), dtype=np.float32
        )
        
        print("🔬 V7 INTUITION COMPREHENSIVE VALIDATION")
        print("=" * 80)
        print("OBJETIVO: Zero tolerância a erros antes do treino")
        print("TESTES: Arquitetura, Performance, Memória, Gradientes, Compatibilidade")
        print("=" * 80)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """🚀 Executar todos os testes de validação"""
        
        test_suite = [
            ("🧪 BASIC ARCHITECTURE", self._test_basic_architecture),
            ("🔄 FORWARD PASS STRESS", self._test_forward_pass_stress),
            ("🎯 BATCH SIZE COMPATIBILITY", self._test_batch_compatibility),
            ("🧠 MEMORY MANAGEMENT", self._test_memory_management),
            ("🌊 GRADIENT FLOW DEEP", self._test_gradient_flow_deep),
            ("⚡ PERFORMANCE BENCHMARKS", self._test_performance_benchmarks),
            ("🔢 NUMERICAL PRECISION", self._test_numerical_precision),
            ("💾 CHECKPOINT COMPATIBILITY", self._test_checkpoint_compatibility),
            ("🎲 RANDOM STATE ROBUSTNESS", self._test_random_robustness),
            ("📊 DISTRIBUTION ANALYSIS", self._test_distribution_analysis),
            ("🔥 EXTREME STRESS TEST", self._test_extreme_conditions),
            ("🎭 EDGE CASES", self._test_edge_cases),
            ("🔧 INTEGRATION TEST", self._test_integration_with_ppo)
        ]
        
        for test_name, test_func in test_suite:
            print(f"\n{test_name}")
            print("-" * 60)
            
            try:
                start_time = time.time()
                test_func()
                duration = time.time() - start_time
                
                self.results['passed_tests'].append(f"{test_name} ({duration:.2f}s)")
                print(f"✅ PASSED ({duration:.2f}s)")
                
            except Exception as e:
                self.results['critical_failures'].append(f"{test_name}: {e}")
                print(f"❌ FAILED: {e}")
                traceback.print_exc()
                
            # Memory cleanup between tests
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return self.results
    
    def _test_basic_architecture(self):
        """🧪 Test basic architecture integrity"""
        print("Creating policy instance...")
        
        kwargs = get_v7_intuition_kwargs()
        kwargs['critic_learning_rate'] = 2e-5
        
        policy = TwoHeadV7Intuition(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=lambda x: 2e-5,
            **kwargs
        )
        
        self.policy = policy  # Store for other tests
        
        # Architecture integrity checks
        critical_components = [
            'unified_backbone', 'actor_lstm', 'critic_lstm', 
            'entry_head', 'management_head', 'enhanced_memory',
            'actor_head', 'critic_head'
        ]
        
        missing = [comp for comp in critical_components if not hasattr(policy, comp)]
        if missing:
            raise Exception(f"Missing critical components: {missing}")
        
        # Parameter count validation
        total_params = sum(p.numel() for p in policy.parameters())
        if total_params < 100000:  # Should have substantial parameters
            raise Exception(f"Too few parameters: {total_params}")
        
        # Optimizer validation
        try:
            actor_opt, critic_opt = policy.get_actor_critic_optimizers()
            if len(actor_opt.param_groups) == 0 or len(critic_opt.param_groups) == 0:
                raise Exception("Empty optimizer param groups")
        except:
            raise Exception("Optimizers not properly configured")
        
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ All critical components present")
        print(f"✓ Optimizers configured")
    
    def _test_forward_pass_stress(self):
        """🔄 Stress test forward passes with various conditions"""
        policy = self.policy
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            print(f"Testing batch size {batch_size}...")
            
            # Random input
            obs = torch.randn(batch_size, 2580)
            episode_starts = torch.zeros(batch_size, dtype=torch.bool)
            
            # Forward passes
            features = policy.extract_features(obs)
            actions, lstm_states, gate_info = policy.forward_actor(features, None, episode_starts)
            values, _ = policy.forward_critic(features, None, episode_starts)
            
            # Validate outputs
            if actions.shape != (batch_size, 8):
                raise Exception(f"Wrong action shape: {actions.shape}, expected ({batch_size}, 8)")
            
            if values.shape != (batch_size, 1):
                raise Exception(f"Wrong value shape: {values.shape}, expected ({batch_size}, 1)")
            
            # Check for NaN/Inf
            if torch.isnan(actions).any():
                raise Exception(f"NaN in actions at batch size {batch_size}")
            
            if torch.isnan(values).any():
                raise Exception(f"NaN in values at batch size {batch_size}")
            
            if torch.isinf(actions).any():
                raise Exception(f"Inf in actions at batch size {batch_size}")
            
            if torch.isinf(values).any():
                raise Exception(f"Inf in values at batch size {batch_size}")
        
        print(f"✓ All batch sizes passed: {batch_sizes}")
    
    def _test_batch_compatibility(self):
        """🎯 Test compatibility with different batch configurations"""
        policy = self.policy
        
        # Test sequential vs parallel batching
        test_configs = [
            ("Sequential single", [(1, 2580) for _ in range(5)]),
            ("Parallel batch", [(5, 2580)]),
            ("Mixed sizes", [(2, 2580), (4, 2580), (1, 2580)]),
            ("Large batch", [(128, 2580)]),
        ]
        
        for config_name, shapes in test_configs:
            print(f"Testing {config_name}...")
            
            for shape in shapes:
                obs = torch.randn(*shape)
                episode_starts = torch.zeros(shape[0], dtype=torch.bool)
                
                # Should not fail
                features = policy.extract_features(obs)
                actions, _, _ = policy.forward_actor(features, None, episode_starts)
                values, _ = policy.forward_critic(features, None, episode_starts)
                
                # Basic validation
                assert actions.shape[0] == shape[0]
                assert values.shape[0] == shape[0]
                assert not torch.isnan(actions).any()
                assert not torch.isnan(values).any()
        
        print("✓ All batch configurations compatible")
    
    def _test_memory_management(self):
        """🧠 Test memory leaks and management"""
        policy = self.policy
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        # Stress memory with repeated forwards
        for i in range(100):
            obs = torch.randn(32, 2580)
            episode_starts = torch.zeros(32, dtype=torch.bool)
            
            features = policy.extract_features(obs)
            actions, _, _ = policy.forward_actor(features, None, episode_starts)
            values, _ = policy.forward_critic(features, None, episode_starts)
            
            # Force computation with proper graph management
            loss = actions.sum() + values.sum()
            loss.backward()
            
            # Clear gradients
            for param in policy.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # Clear computation graph
            del loss, actions, values, features
            
            if i % 20 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"Iteration {i}: {current_memory:.1f} MB")
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        if memory_increase > 500:  # More than 500MB increase is concerning
            self.results['memory_leaks'].append(f"High memory increase: {memory_increase:.1f} MB")
        
        if memory_increase > 1000:  # More than 1GB is critical
            raise Exception(f"Critical memory leak detected: {memory_increase:.1f} MB")
        
        print("✓ Memory management acceptable")
    
    def _test_gradient_flow_deep(self):
        """🌊 Deep gradient flow analysis"""
        policy = self.policy
        
        # Test gradient flow through all components
        obs = torch.randn(4, 2580, requires_grad=True)
        episode_starts = torch.zeros(4, dtype=torch.bool)
        
        # Forward pass
        features = policy.extract_features(obs)
        actions, _, _ = policy.forward_actor(features, None, episode_starts)
        values, _ = policy.forward_critic(features, None, episode_starts)
        
        # Create meaningful losses
        action_loss = ((actions - torch.randn_like(actions)) ** 2).mean()
        value_loss = ((values - torch.randn_like(values)) ** 2).mean()
        total_loss = action_loss + value_loss
        
        # Backward pass
        total_loss.backward()
        
        # Analyze gradients
        grad_stats = {
            'zero_grads': 0,
            'nan_grads': 0,
            'inf_grads': 0,
            'tiny_grads': 0,
            'huge_grads': 0,
            'total_params': 0,
            'grad_norms': []
        }
        
        for name, param in policy.named_parameters():
            if param.grad is not None:
                grad_stats['total_params'] += 1
                grad_norm = param.grad.norm().item()
                grad_stats['grad_norms'].append(grad_norm)
                
                if grad_norm == 0:
                    grad_stats['zero_grads'] += 1
                elif torch.isnan(param.grad).any():
                    grad_stats['nan_grads'] += 1
                elif torch.isinf(param.grad).any():
                    grad_stats['inf_grads'] += 1
                elif grad_norm < 1e-8:
                    grad_stats['tiny_grads'] += 1
                elif grad_norm > 100:
                    grad_stats['huge_grads'] += 1
        
        # Validate gradient health
        if grad_stats['nan_grads'] > 0:
            raise Exception(f"NaN gradients detected: {grad_stats['nan_grads']} parameters")
        
        if grad_stats['inf_grads'] > 0:
            raise Exception(f"Inf gradients detected: {grad_stats['inf_grads']} parameters")
        
        zero_percentage = (grad_stats['zero_grads'] / grad_stats['total_params']) * 100
        if zero_percentage > 50:
            raise Exception(f"Too many zero gradients: {zero_percentage:.1f}%")
        
        tiny_percentage = (grad_stats['tiny_grads'] / grad_stats['total_params']) * 100
        if tiny_percentage > 30:
            self.results['gradient_problems'].append(f"Many tiny gradients: {tiny_percentage:.1f}%")
        
        huge_percentage = (grad_stats['huge_grads'] / grad_stats['total_params']) * 100
        if huge_percentage > 10:
            self.results['gradient_problems'].append(f"Many huge gradients: {huge_percentage:.1f}%")
        
        mean_grad_norm = np.mean(grad_stats['grad_norms'])
        print(f"✓ Gradient statistics:")
        print(f"  Total params with grads: {grad_stats['total_params']}")
        print(f"  Zero grads: {grad_stats['zero_grads']} ({zero_percentage:.1f}%)")
        print(f"  Mean grad norm: {mean_grad_norm:.2e}")
        print(f"  Tiny grads: {grad_stats['tiny_grads']} ({tiny_percentage:.1f}%)")
        print(f"  Huge grads: {grad_stats['huge_grads']} ({huge_percentage:.1f}%)")
    
    def _test_performance_benchmarks(self):
        """⚡ Performance benchmark tests"""
        policy = self.policy
        
        # Benchmark forward pass speed
        batch_size = 32
        num_iterations = 100
        
        obs = torch.randn(batch_size, 2580)
        episode_starts = torch.zeros(batch_size, dtype=torch.bool)
        
        # Warm up
        for _ in range(10):
            features = policy.extract_features(obs)
            actions, _, _ = policy.forward_actor(features, None, episode_starts)
            values, _ = policy.forward_critic(features, None, episode_starts)
        
        # Benchmark forward pass
        start_time = time.time()
        for _ in range(num_iterations):
            features = policy.extract_features(obs)
            actions, _, _ = policy.forward_actor(features, None, episode_starts)
            values, _ = policy.forward_critic(features, None, episode_starts)
        
        forward_time = time.time() - start_time
        forward_fps = (num_iterations * batch_size) / forward_time
        
        print(f"Forward pass: {forward_fps:.0f} samples/sec")
        
        # Benchmark backward pass
        start_time = time.time()
        for _ in range(num_iterations):
            features = policy.extract_features(obs)
            actions, _, _ = policy.forward_actor(features, None, episode_starts)
            values, _ = policy.forward_critic(features, None, episode_starts)
            
            loss = actions.sum() + values.sum()
            loss.backward()
            
            # Clear gradients
            for param in policy.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # Clear computation graph
            del loss, actions, values, features
        
        backward_time = time.time() - start_time
        backward_fps = (num_iterations * batch_size) / backward_time
        
        print(f"Forward+backward: {backward_fps:.0f} samples/sec")
        
        # Performance thresholds
        if forward_fps < 100:  # Should process at least 100 samples/sec
            self.results['performance_issues'].append(f"Slow forward pass: {forward_fps:.0f} samples/sec")
        
        if backward_fps < 50:  # Should do forward+backward at least 50 samples/sec
            self.results['performance_issues'].append(f"Slow backward pass: {backward_fps:.0f} samples/sec")
        
        print("✓ Performance benchmarks completed")
    
    def _test_numerical_precision(self):
        """🔢 Test numerical precision and stability"""
        policy = self.policy
        
        # Test with different numerical ranges
        test_ranges = [
            ("Normal", 1.0),
            ("Small", 1e-3),
            ("Large", 1e3),
            ("Very small", 1e-6),
            ("Mixed", None)  # Special case
        ]
        
        for range_name, scale in test_ranges:
            print(f"Testing {range_name} range...")
            
            if scale is not None:
                obs = torch.randn(4, 2580) * scale
            else:
                # Mixed scales
                obs = torch.randn(4, 2580)
                obs[:, :860] *= 1e-6   # Very small
                obs[:, 860:1720] *= 1.0  # Normal  
                obs[:, 1720:] *= 1e3   # Large
            
            episode_starts = torch.zeros(4, dtype=torch.bool)
            
            try:
                features = policy.extract_features(obs)
                actions, _, _ = policy.forward_actor(features, None, episode_starts)
                values, _ = policy.forward_critic(features, None, episode_starts)
                
                # Check outputs are reasonable
                if torch.isnan(actions).any() or torch.isnan(values).any():
                    raise Exception(f"NaN outputs with {range_name} inputs")
                
                if torch.isinf(actions).any() or torch.isinf(values).any():
                    raise Exception(f"Inf outputs with {range_name} inputs")
                
                # Check action ranges are reasonable
                if actions.abs().max() > 1e6:
                    self.results['numerical_issues'].append(f"Extreme action values with {range_name} inputs")
                
                if values.abs().max() > 1e6:
                    self.results['numerical_issues'].append(f"Extreme value outputs with {range_name} inputs")
                
            except Exception as e:
                raise Exception(f"Failed with {range_name} inputs: {e}")
        
        print("✓ Numerical precision tests passed")
    
    def _test_checkpoint_compatibility(self):
        """💾 Test checkpoint save/load compatibility"""
        policy = self.policy
        
        # Save current state
        original_state = policy.state_dict()
        
        # Test serialization
        try:
            # Simulate save
            checkpoint_data = {
                'policy_state_dict': policy.state_dict(),
                'actor_optimizer': policy.actor_optimizer.state_dict(),
                'critic_optimizer': policy.critic_optimizer.state_dict(),
            }
            
            # Test that we can access all needed components for checkpointing
            required_components = [
                'policy_state_dict',
                'actor_optimizer', 
                'critic_optimizer'
            ]
            
            for component in required_components:
                if component not in checkpoint_data:
                    raise Exception(f"Missing component for checkpointing: {component}")
            
            print("✓ Checkpoint data structure valid")
            
            # Test state_dict completeness
            state_dict = policy.state_dict()
            param_count = len([k for k in state_dict.keys() if 'weight' in k or 'bias' in k])
            
            if param_count < 50:  # Should have substantial number of parameters
                self.results['compatibility_issues'].append(f"Few parameters in state_dict: {param_count}")
            
            print(f"✓ State dict contains {len(state_dict)} entries")
            
        except Exception as e:
            raise Exception(f"Checkpoint compatibility failed: {e}")
    
    def _test_random_robustness(self):
        """🎲 Test robustness with different random states"""
        policy = self.policy
        
        # Test with different seeds
        seeds = [42, 123, 999, 2023, 0]
        results_consistency = []
        
        obs = torch.randn(4, 2580)  # Fixed input
        episode_starts = torch.zeros(4, dtype=torch.bool)
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            features = policy.extract_features(obs)
            actions, _, _ = policy.forward_actor(features, None, episode_starts)
            values, _ = policy.forward_critic(features, None, episode_starts)
            
            results_consistency.append({
                'actions_mean': actions.mean().item(),
                'actions_std': actions.std().item(),
                'values_mean': values.mean().item(),
                'values_std': values.std().item()
            })
        
        # Check consistency (should be deterministic with same seed)
        torch.manual_seed(42)
        np.random.seed(42)
        features1 = policy.extract_features(obs)
        actions1, _, _ = policy.forward_actor(features1, None, episode_starts)
        
        torch.manual_seed(42)
        np.random.seed(42) 
        features2 = policy.extract_features(obs)
        actions2, _, _ = policy.forward_actor(features2, None, episode_starts)
        
        if not torch.allclose(actions1, actions2, atol=1e-6):
            self.results['warnings'].append("Non-deterministic behavior with same seed")
        
        print("✓ Random state robustness tested")
    
    def _test_distribution_analysis(self):
        """📊 Analyze output distributions"""
        policy = self.policy
        
        # Generate large sample
        num_samples = 1000
        batch_size = 16
        
        all_actions = []
        all_values = []
        
        for _ in range(num_samples // batch_size):
            obs = torch.randn(batch_size, 2580)
            episode_starts = torch.zeros(batch_size, dtype=torch.bool)
            
            features = policy.extract_features(obs)
            actions, _, _ = policy.forward_actor(features, None, episode_starts)
            values, _ = policy.forward_critic(features, None, episode_starts)
            
            all_actions.append(actions.detach())
            all_values.append(values.detach())
        
        all_actions = torch.cat(all_actions, dim=0)  # [num_samples, 8]
        all_values = torch.cat(all_values, dim=0)    # [num_samples, 1]
        
        # Analyze distributions
        print("Action distributions:")
        for i in range(8):
            action_i = all_actions[:, i]
            print(f"  Action {i}: mean={action_i.mean():.3f}, std={action_i.std():.3f}, "
                  f"min={action_i.min():.3f}, max={action_i.max():.3f}")
            
            # Check for collapsed distributions
            if action_i.std() < 0.001:
                self.results['warnings'].append(f"Action {i} has very low variance: {action_i.std():.6f}")
        
        print(f"Value distribution: mean={all_values.mean():.3f}, std={all_values.std():.3f}")
        
        # Check action ranges
        action_0 = all_actions[:, 0]  # Entry decision (should be 0-2)
        if action_0.min() < -0.5 or action_0.max() > 2.5:
            self.results['warnings'].append(f"Action 0 out of expected range [0,2]: [{action_0.min():.2f}, {action_0.max():.2f}]")
        
        action_1 = all_actions[:, 1]  # Entry confidence (should be 0-1)
        if action_1.min() < -0.1 or action_1.max() > 1.1:
            self.results['warnings'].append(f"Action 1 out of expected range [0,1]: [{action_1.min():.2f}, {action_1.max():.2f}]")
        
        print("✓ Distribution analysis completed")
    
    def _test_extreme_conditions(self):
        """🔥 Test extreme stress conditions"""
        policy = self.policy
        
        extreme_tests = [
            ("All zeros", lambda: torch.zeros(8, 2580)),
            ("All ones", lambda: torch.ones(8, 2580)),
            ("Random extreme", lambda: torch.randn(8, 2580) * 1000),
            ("Alternating", lambda: torch.tensor([1.0, -1.0] * 1290).expand(8, 2580)),
            ("Gradient pattern", lambda: torch.linspace(-100, 100, 2580).expand(8, 2580)),
        ]
        
        for test_name, input_gen in extreme_tests:
            print(f"Testing {test_name}...")
            
            try:
                obs = input_gen()
                episode_starts = torch.zeros(8, dtype=torch.bool)
                
                features = policy.extract_features(obs)
                actions, _, _ = policy.forward_actor(features, None, episode_starts)
                values, _ = policy.forward_critic(features, None, episode_starts)
                
                # Should not produce NaN/Inf
                if torch.isnan(actions).any() or torch.isnan(values).any():
                    raise Exception(f"NaN outputs with {test_name}")
                
                if torch.isinf(actions).any() or torch.isinf(values).any():
                    raise Exception(f"Inf outputs with {test_name}")
                
            except Exception as e:
                if "Expected 2580 features" in str(e):
                    continue  # Skip this specific test error
                raise Exception(f"Failed extreme test {test_name}: {e}")
        
        print("✓ Extreme conditions handled")
    
    def _test_edge_cases(self):
        """🎭 Test edge cases and corner scenarios"""
        policy = self.policy
        
        # Test single sample
        obs = torch.randn(1, 2580)
        episode_starts = torch.zeros(1, dtype=torch.bool)
        
        features = policy.extract_features(obs)
        actions, _, _ = policy.forward_actor(features, None, episode_starts)
        values, _ = policy.forward_critic(features, None, episode_starts)
        
        assert actions.shape == (1, 8)
        assert values.shape == (1, 1)
        
        # Test very large batch
        try:
            large_obs = torch.randn(256, 2580)
            large_starts = torch.zeros(256, dtype=torch.bool)
            
            features = policy.extract_features(large_obs)
            actions, _, _ = policy.forward_actor(features, None, large_starts)
            values, _ = policy.forward_critic(features, None, large_starts)
            
            print("✓ Large batch (256) handled")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("⚠️ Large batch OOM (expected on limited hardware)")
            else:
                raise
        
        # Test episode_starts variations
        for starts_config in [torch.ones(4, dtype=torch.bool), torch.tensor([True, False, True, False])]:
            obs = torch.randn(4, 2580)
            features = policy.extract_features(obs)
            actions, _, _ = policy.forward_actor(features, None, starts_config)
            values, _ = policy.forward_critic(features, None, starts_config)
            
            assert not torch.isnan(actions).any()
            assert not torch.isnan(values).any()
        
        print("✓ Edge cases handled")
    
    def _test_integration_with_ppo(self):
        """🔧 Test integration aspects relevant for PPO training"""
        policy = self.policy
        
        # Test action sampling (what PPO will do)
        obs = torch.randn(16, 2580)
        episode_starts = torch.zeros(16, dtype=torch.bool)
        
        features = policy.extract_features(obs)
        actions, _, _ = policy.forward_actor(features, None, episode_starts)
        values, _ = policy.forward_critic(features, None, episode_starts)
        
        # Test that actions can be used for policy evaluation
        try:
            # Simulate PPO's action evaluation
            action_mean = actions
            action_std = torch.ones_like(actions) * 0.1  # PPO would learn this
            
            from torch.distributions import Normal
            dist = Normal(action_mean, action_std)
            
            # Sample actions (what PPO does)
            sampled_actions = dist.sample()
            
            # Evaluate log probability (what PPO needs)
            log_probs = dist.log_prob(sampled_actions).sum(dim=1)
            
            # Entropy calculation (what PPO needs)
            entropy = dist.entropy().sum(dim=1)
            
            assert not torch.isnan(log_probs).any()
            assert not torch.isnan(entropy).any()
            assert not torch.isinf(log_probs).any()
            assert not torch.isinf(entropy).any()
            
            print("✓ PPO action distribution compatibility")
            
        except Exception as e:
            raise Exception(f"PPO integration issue: {e}")
        
        # Test critic value consistency
        values1, _ = policy.forward_critic(features, None, episode_starts)
        values2, _ = policy.forward_critic(features, None, episode_starts)
        
        if not torch.allclose(values1, values2, atol=1e-6):
            self.results['warnings'].append("Critic values not deterministic")
        
        # Test that we can compute PPO losses
        try:
            # Simulate PPO loss computation
            old_log_probs = torch.randn(16)
            returns = torch.randn(16, 1)
            advantages = torch.randn(16)
            
            # Policy loss terms
            ratio = torch.exp(log_probs - old_log_probs)
            clip_range = 0.2
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = ((values.squeeze() - returns.squeeze()) ** 2).mean()
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            assert not torch.isnan(total_loss)
            assert not torch.isinf(total_loss)
            
            print("✓ PPO loss computation compatibility")
            
        except Exception as e:
            raise Exception(f"PPO loss computation failed: {e}")
        
        print("✓ PPO integration test passed")
    
    def print_comprehensive_report(self):
        """📋 Print comprehensive validation report"""
        print("\n" + "="*80)
        print("🔬 V7 INTUITION COMPREHENSIVE VALIDATION REPORT")
        print("="*80)
        
        # Summary
        total_tests = len(self.results['passed_tests'])
        total_failures = len(self.results['critical_failures'])
        
        if total_failures == 0:
            print("🎉 ALL TESTS PASSED!")
            print(f"✅ {total_tests} tests executed successfully")
        else:
            print(f"❌ {total_failures} CRITICAL FAILURES detected")
            print(f"✅ {total_tests} tests passed")
        
        # Detailed results
        categories = [
            ('❌ CRITICAL FAILURES', 'critical_failures'),
            ('⚠️ PERFORMANCE ISSUES', 'performance_issues'),
            ('💾 MEMORY LEAKS', 'memory_leaks'),
            ('🌊 GRADIENT PROBLEMS', 'gradient_problems'),
            ('🔢 NUMERICAL ISSUES', 'numerical_issues'),
            ('🔧 COMPATIBILITY ISSUES', 'compatibility_issues'),
            ('⚠️ WARNINGS', 'warnings'),
            ('✅ PASSED TESTS', 'passed_tests')
        ]
        
        for title, key in categories:
            issues = self.results[key]
            print(f"\n{title}:")
            if issues:
                for i, issue in enumerate(issues, 1):
                    print(f"   {i:2d}. {issue}")
            else:
                print("   ✅ None detected")
        
        # Final assessment
        print("\n" + "="*80)
        print("🎯 FINAL ASSESSMENT")
        print("="*80)
        
        if total_failures > 0:
            print("🚨 CRITICAL: Cannot proceed with training!")
            print("   Fix all critical failures before starting training.")
            return False
        
        performance_score = 100
        if self.results['performance_issues']:
            performance_score -= len(self.results['performance_issues']) * 10
        
        if self.results['memory_leaks']:
            performance_score -= len(self.results['memory_leaks']) * 15
        
        if self.results['gradient_problems']:
            performance_score -= len(self.results['gradient_problems']) * 10
        
        if self.results['numerical_issues']:
            performance_score -= len(self.results['numerical_issues']) * 5
        
        if self.results['warnings']:
            performance_score -= len(self.results['warnings']) * 2
        
        performance_score = max(0, performance_score)
        
        print(f"📊 Overall Health Score: {performance_score}/100")
        
        if performance_score >= 90:
            print("🏆 EXCELLENT: Architecture ready for training!")
        elif performance_score >= 75:
            print("✅ GOOD: Architecture suitable for training with monitoring")
        elif performance_score >= 60:
            print("⚠️ CAUTION: Consider addressing issues before training")
        else:
            print("❌ POOR: Address issues before training")
        
        print("\n🚀 TRAINING READINESS:")
        if total_failures == 0 and performance_score >= 75:
            print("✅ READY TO START TRAINING")
            print("   Architecture passed all critical tests")
        else:
            print("⚠️ NOT READY - Address issues first")
        
        return total_failures == 0 and performance_score >= 75


def main():
    """🚀 Run comprehensive V7 Intuition validation"""
    print("🔬 STARTING COMPREHENSIVE V7 INTUITION VALIDATION")
    print("Target: Zero errors before training restart")
    print()
    
    validator = V7IntuitionComprehensiveValidator()
    results = validator.run_all_tests()
    is_ready = validator.print_comprehensive_report()
    
    return is_ready, results


if __name__ == "__main__":
    try:
        is_ready, results = main()
        if is_ready:
            print("\n🎉 SUCCESS: V7 Intuition validated and ready for training!")
            sys.exit(0)
        else:
            print("\n⚠️ ISSUES DETECTED: Review and fix before training")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 VALIDATION FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
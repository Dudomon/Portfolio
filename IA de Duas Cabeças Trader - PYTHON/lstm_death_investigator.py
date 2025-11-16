#!/usr/bin/env python3
"""
🔍 LSTM DEATH INVESTIGATOR
Descobrir sistematicamente o que está matando os LSTMs da V8Heritage
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class LSTMDeathInvestigator(BaseCallback):
    """
    🔍 Investigador forense para descobrir o que mata os LSTMs
    """
    
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.investigation_data = []
        self.lstm_health_history = {}
        self.step_count = 0
        self.death_detected = False
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Investigar a cada 50 steps
        if self.step_count % 50 == 0:
            self._investigate_lstm_health()
            
        return True
    
    def _investigate_lstm_health(self):
        """🔍 Investigação forense completa dos LSTMs"""
        
        if not hasattr(self.model, 'policy'):
            return
            
        policy = self.model.policy
        
        # Verificar se é V8Heritage
        if not hasattr(policy, 'neural_architecture'):
            return
            
        print(f"\n🔍 INVESTIGAÇÃO FORENSE - Step {self.step_count}")
        print("=" * 60)
        
        neural_arch = policy.neural_architecture
        
        for lstm_name in ['actor_lstm', 'critic_lstm']:
            if hasattr(neural_arch, lstm_name):
                lstm = getattr(neural_arch, lstm_name)
                self._investigate_single_lstm(lstm_name, lstm)
    
    def _investigate_single_lstm(self, lstm_name, lstm):
        """🔍 Investigação detalhada de um LSTM específico"""
        
        print(f"\n🎯 INVESTIGANDO {lstm_name.upper()}")
        print("-" * 40)
        
        investigation = {
            'step': self.step_count,
            'lstm_name': lstm_name,
            'params': {},
            'gradients': {},
            'optimizer_state': {},
            'suspicious_findings': []
        }
        
        # 1. ANÁLISE DOS PARÂMETROS
        for param_name, param in lstm.named_parameters():
            if param is not None:
                zeros_count = (param.data.abs() < 1e-8).sum().item()
                total_params = param.data.numel()
                zero_ratio = zeros_count / total_params
                
                param_stats = {
                    'zero_ratio': zero_ratio,
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'has_nan': torch.isnan(param.data).any().item(),
                    'has_inf': torch.isinf(param.data).any().item()
                }
                
                investigation['params'][param_name] = param_stats
                
                print(f"📊 {param_name}:")
                print(f"   Zeros: {zero_ratio*100:.1f}% | Mean: {param_stats['mean']:.6f} | Std: {param_stats['std']:.6f}")
                print(f"   Range: [{param_stats['min']:.6f}, {param_stats['max']:.6f}] | NaN: {param_stats['has_nan']} | Inf: {param_stats['has_inf']}")
                
                # Detectar morte súbita
                if zero_ratio > 0.8:  # 80% zeros = morte
                    investigation['suspicious_findings'].append(f"{param_name}: MORTE DETECTADA ({zero_ratio*100:.1f}% zeros)")
                    print(f"   🚨 MORTE DETECTADA: {zero_ratio*100:.1f}% zeros!")
                    self.death_detected = True
                
                # 2. ANÁLISE DOS GRADIENTES
                if param.grad is not None:
                    grad_zeros = (param.grad.abs() < 1e-8).sum().item()
                    grad_total = param.grad.numel()
                    grad_zero_ratio = grad_zeros / grad_total
                    
                    grad_stats = {
                        'zero_ratio': grad_zero_ratio,
                        'mean': param.grad.mean().item(),
                        'std': param.grad.std().item(),
                        'norm': param.grad.norm().item(),
                        'has_nan': torch.isnan(param.grad).any().item(),
                        'has_inf': torch.isinf(param.grad).any().item()
                    }
                    
                    investigation['gradients'][param_name] = grad_stats
                    
                    print(f"   🎯 Gradientes: Zeros: {grad_zero_ratio*100:.1f}% | Norm: {grad_stats['norm']:.6f}")
                    print(f"   🎯 Grad Range: Mean: {grad_stats['mean']:.6f} | Std: {grad_stats['std']:.6f}")
                    
                    # Detectar problemas de gradientes
                    if grad_zero_ratio > 0.9:
                        investigation['suspicious_findings'].append(f"{param_name}: GRADIENTES MORTOS ({grad_zero_ratio*100:.1f}% zeros)")
                        print(f"   🚨 GRADIENTES MORTOS: {grad_zero_ratio*100:.1f}% zeros!")
                    
                    if grad_stats['norm'] < 1e-8:
                        investigation['suspicious_findings'].append(f"{param_name}: GRADIENTES MICROSCÓPICOS (norm={grad_stats['norm']:.2e})")
                        print(f"   🚨 GRADIENTES MICROSCÓPICOS: norm={grad_stats['norm']:.2e}")
                    
                    if grad_stats['norm'] > 100:
                        investigation['suspicious_findings'].append(f"{param_name}: GRADIENTES EXPLOSIVOS (norm={grad_stats['norm']:.2e})")
                        print(f"   🚨 GRADIENTES EXPLOSIVOS: norm={grad_stats['norm']:.2e}")
                        
                else:
                    investigation['gradients'][param_name] = None
                    print(f"   ⚠️ SEM GRADIENTES!")
                    investigation['suspicious_findings'].append(f"{param_name}: SEM GRADIENTES")
        
        # 3. ANÁLISE DO OPTIMIZER STATE
        self._investigate_optimizer_state(lstm_name, lstm, investigation)
        
        # 4. SALVAR DADOS DA INVESTIGAÇÃO
        self.investigation_data.append(investigation)
        
        # 5. RELATÓRIO DE SUSPEITAS
        if investigation['suspicious_findings']:
            print(f"\n🚨 SUSPEITAS DETECTADAS EM {lstm_name.upper()}:")
            for finding in investigation['suspicious_findings']:
                print(f"   🔍 {finding}")
    
    def _investigate_optimizer_state(self, lstm_name, lstm, investigation):
        """🔍 Investigar estado do optimizer"""
        
        try:
            optimizer = self.model.policy.optimizer
            if optimizer is None:
                print("   ⚠️ Optimizer não encontrado!")
                return
                
            print(f"\n🔧 OPTIMIZER INFO:")
            print(f"   Tipo: {type(optimizer).__name__}")
            
            # Verificar learning rate
            for param_group in optimizer.param_groups:
                lr = param_group.get('lr', 'N/A')
                weight_decay = param_group.get('weight_decay', 'N/A')
                print(f"   LR: {lr} | Weight Decay: {weight_decay}")
                
                investigation['optimizer_state']['lr'] = lr
                investigation['optimizer_state']['weight_decay'] = weight_decay
                
                # Detectar problemas
                if isinstance(lr, float) and lr > 1e-2:
                    investigation['suspicious_findings'].append(f"LR MUITO ALTO: {lr}")
                    print(f"   🚨 LR MUITO ALTO: {lr}")
                
                if isinstance(weight_decay, float) and weight_decay > 1e-2:
                    investigation['suspicious_findings'].append(f"WEIGHT DECAY MUITO ALTO: {weight_decay}")
                    print(f"   🚨 WEIGHT DECAY MUITO ALTO: {weight_decay}")
            
            # Verificar estado dos parâmetros LSTM no optimizer
            for param_name, param in lstm.named_parameters():
                if param in optimizer.state:
                    state = optimizer.state[param]
                    print(f"   📊 {param_name} optimizer state: {list(state.keys())}")
                    
                    # Verificar momentum/adam states
                    if 'exp_avg' in state:
                        exp_avg_norm = state['exp_avg'].norm().item()
                        print(f"      exp_avg norm: {exp_avg_norm:.6f}")
                        if exp_avg_norm > 10:
                            investigation['suspicious_findings'].append(f"{param_name}: MOMENTUM EXPLOSIVO ({exp_avg_norm:.2e})")
                    
                    if 'exp_avg_sq' in state:
                        exp_avg_sq_norm = state['exp_avg_sq'].norm().item()
                        print(f"      exp_avg_sq norm: {exp_avg_sq_norm:.6f}")
                        if exp_avg_sq_norm > 100:
                            investigation['suspicious_findings'].append(f"{param_name}: SECOND MOMENT EXPLOSIVO ({exp_avg_sq_norm:.2e})")
                
        except Exception as e:
            print(f"   ❌ Erro ao investigar optimizer: {e}")
            investigation['optimizer_state']['error'] = str(e)
    
    def get_investigation_report(self):
        """📋 Gerar relatório completo da investigação"""
        
        if not self.investigation_data:
            return "Nenhum dado de investigação coletado."
        
        report = []
        report.append("🔍 RELATÓRIO FORENSE: MORTE DOS LSTMs V8HERITAGE")
        report.append("=" * 60)
        
        # Análise cronológica das mortes
        deaths_detected = []
        for data in self.investigation_data:
            if data['suspicious_findings']:
                deaths_detected.append(data)
        
        if deaths_detected:
            report.append(f"\n🚨 {len(deaths_detected)} EVENTOS SUSPEITOS DETECTADOS:")
            
            for death in deaths_detected:
                report.append(f"\nStep {death['step']} - {death['lstm_name']}:")
                for finding in death['suspicious_findings']:
                    report.append(f"  🔍 {finding}")
        
        # Padrões identificados
        report.append(f"\n📊 ANÁLISE DE PADRÕES:")
        report.append(f"Total de steps investigados: {len(self.investigation_data)}")
        report.append(f"Eventos suspeitos: {len(deaths_detected)}")
        
        return "\n".join(report)

# Função para adicionar o investigador aos callbacks
def create_lstm_death_investigator():
    """🔍 Criar investigador forense para LSTMs"""
    return LSTMDeathInvestigator(verbose=1)
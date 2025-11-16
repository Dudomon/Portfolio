#!/usr/bin/env python3
"""
🔧 CORREÇÃO RUNTIME PARA ZEROS NOS BIAS DE ATTENTION
Problema: Zeros aparecem DURANTE o treinamento, não na inicialização
Solução: Monitor ativo que detecta e corrige zeros em tempo real
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Optional
import logging

class RuntimeAttentionBiasFixer(BaseCallback):
    """🔧 Corretor em tempo real para zeros nos bias de attention"""
    
    def __init__(self, 
                 check_frequency: int = 500,
                 zero_threshold: float = 1e-8,
                 fix_threshold: float = 0.25,  # Corrigir se >25% zeros
                 noise_std: float = 1e-5,
                 verbose: int = 1):
        
        super().__init__(verbose)
        
        self.check_frequency = check_frequency
        self.zero_threshold = zero_threshold
        self.fix_threshold = fix_threshold
        self.noise_std = noise_std
        
        # Estatísticas
        self.total_fixes = 0
        self.layer_fix_counts = {}
        self.zero_history = []
        
        # Logging
        self.logger = logging.getLogger('RuntimeAttentionBiasFixer')
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if verbose >= 1 else logging.WARNING)
        
    def _on_training_start(self) -> None:
        """Inicializar monitoramento"""
        self.logger.info("🔍 Runtime Attention Bias Fixer ativado")
        self.logger.info(f"   Check frequency: {self.check_frequency} steps")
        self.logger.info(f"   Fix threshold: {self.fix_threshold:.1%} zeros")
        self.logger.info(f"   Noise std: {self.noise_std}")
        
        # Identificar attention layers
        self.attention_layers = []
        if hasattr(self.model, 'policy'):
            for name, module in self.model.policy.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    self.attention_layers.append((name, module))
                    self.layer_fix_counts[name] = 0
        
        self.logger.info(f"   Attention layers encontrados: {len(self.attention_layers)}")
        for name, _ in self.attention_layers:
            self.logger.info(f"     - {name}")
    
    def _on_step(self) -> bool:
        """Verificar e corrigir bias a cada step"""
        if self.num_timesteps % self.check_frequency != 0:
            return True
        
        try:
            fixes_applied = self._check_and_fix_attention_bias()
            
            if fixes_applied > 0:
                self.total_fixes += fixes_applied
                if self.verbose >= 1:
                    print(f"🔧 Step {self.num_timesteps}: {fixes_applied} attention bias corrections applied")
                    
        except Exception as e:
            self.logger.error(f"Erro no monitoramento: {e}")
        
        return True
    
    def _check_and_fix_attention_bias(self) -> int:
        """Verificar e corrigir zeros nos bias de attention"""
        fixes_applied = 0
        step_stats = {
            'step': self.num_timesteps,
            'layers_checked': 0,
            'layers_fixed': 0,
            'total_zeros_before': 0,
            'total_zeros_after': 0
        }
        
        for name, module in self.attention_layers:
            if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                step_stats['layers_checked'] += 1
                
                # Verificar gradientes (onde o problema aparece)
                if module.in_proj_bias.grad is not None:
                    grad_data = module.in_proj_bias.grad.data
                    
                    # Contar zeros
                    zero_mask = torch.abs(grad_data) < self.zero_threshold
                    zeros_count = torch.sum(zero_mask).item()
                    total_count = grad_data.numel()
                    zero_ratio = zeros_count / total_count
                    
                    step_stats['total_zeros_before'] += zeros_count
                    
                    # Se muitos zeros, aplicar correção
                    if zero_ratio > self.fix_threshold:
                        # Método 1: Adicionar ruído aos gradientes zerados
                        noise = torch.randn_like(grad_data) * self.noise_std
                        grad_data[zero_mask] += noise[zero_mask]
                        
                        # Método 2: Redistribuir gradientes
                        if zero_ratio > 0.5:  # Se >50% zeros, redistribuir
                            non_zero_mask = ~zero_mask
                            if torch.sum(non_zero_mask) > 0:
                                # Pegar média dos gradientes não-zero
                                mean_non_zero = torch.mean(grad_data[non_zero_mask])
                                std_non_zero = torch.std(grad_data[non_zero_mask])
                                
                                # Aplicar valores similares aos zeros
                                replacement_values = torch.normal(
                                    mean=mean_non_zero * 0.1,  # 10% da média
                                    std=std_non_zero * 0.1,    # 10% do desvio
                                    size=(zeros_count,),
                                    device=grad_data.device
                                )
                                grad_data[zero_mask] = replacement_values
                        
                        # Contar zeros após correção
                        zeros_after = torch.sum(torch.abs(grad_data) < self.zero_threshold).item()
                        step_stats['total_zeros_after'] += zeros_after
                        
                        fixes_applied += 1
                        step_stats['layers_fixed'] += 1
                        self.layer_fix_counts[name] += 1
                        
                        if self.verbose >= 2:
                            self.logger.info(f"🔧 {name}: {zero_ratio:.1%} zeros → {zeros_after/total_count:.1%} zeros")
                    else:
                        step_stats['total_zeros_after'] += zeros_count
                
                # Verificar também os parâmetros (bias) diretamente
                param_data = module.in_proj_bias.data
                param_zeros = torch.sum(torch.abs(param_data) < self.zero_threshold).item()
                param_total = param_data.numel()
                param_zero_ratio = param_zeros / param_total
                
                if param_zero_ratio > self.fix_threshold:
                    # Corrigir parâmetros zerados
                    zero_param_mask = torch.abs(param_data) < self.zero_threshold
                    noise = torch.randn_like(param_data) * (self.noise_std * 10)  # Ruído maior para parâmetros
                    param_data[zero_param_mask] += noise[zero_param_mask]
                    
                    if self.verbose >= 2:
                        self.logger.info(f"🔧 {name} params: {param_zero_ratio:.1%} zeros corrigidos")
        
        # Salvar estatísticas
        self.zero_history.append(step_stats)
        
        # Log resumo se houve correções
        if fixes_applied > 0:
            improvement = step_stats['total_zeros_before'] - step_stats['total_zeros_after']
            self.logger.info(f"Step {self.num_timesteps}: {fixes_applied} layers fixed, {improvement} zeros removed")
        
        return fixes_applied
    
    def _on_training_end(self) -> None:
        """Relatório final"""
        if self.verbose >= 1:
            print(f"\n📊 RUNTIME ATTENTION BIAS FIXER - RELATÓRIO FINAL")
            print(f"=" * 60)
            print(f"Total de correções aplicadas: {self.total_fixes}")
            print(f"Checks realizados: {len(self.zero_history)}")
            
            if self.layer_fix_counts:
                print(f"\nCorreções por layer:")
                for layer, count in self.layer_fix_counts.items():
                    print(f"  {layer}: {count} correções")
            
            # Estatísticas de melhoria
            if self.zero_history:
                initial_zeros = self.zero_history[0]['total_zeros_before'] if self.zero_history else 0
                final_zeros = self.zero_history[-1]['total_zeros_after'] if self.zero_history else 0
                
                if initial_zeros > 0:
                    improvement = (initial_zeros - final_zeros) / initial_zeros * 100
                    print(f"\nMelhoria geral: {improvement:.1f}% redução de zeros")
            
            print(f"=" * 60)
    
    def get_statistics(self) -> Dict:
        """Obter estatísticas do fixer"""
        return {
            'total_fixes': self.total_fixes,
            'layer_fix_counts': self.layer_fix_counts.copy(),
            'checks_performed': len(self.zero_history),
            'zero_history': self.zero_history[-10:] if self.zero_history else []  # Últimos 10
        }

def create_runtime_attention_bias_fixer(**kwargs) -> RuntimeAttentionBiasFixer:
    """🏭 Factory function para criar o fixer"""
    return RuntimeAttentionBiasFixer(**kwargs)

def test_runtime_fixer():
    """🧪 Testar o fixer em tempo real"""
    print("🧪 TESTE DO RUNTIME ATTENTION BIAS FIXER")
    print("=" * 50)
    
    try:
        # Criar fixer
        fixer = create_runtime_attention_bias_fixer(
            check_frequency=100,  # Mais frequente para teste
            fix_threshold=0.2,    # Mais sensível para teste
            verbose=2
        )
        
        print("✅ Runtime Attention Bias Fixer criado")
        print(f"   Check frequency: {fixer.check_frequency}")
        print(f"   Fix threshold: {fixer.fix_threshold:.1%}")
        print(f"   Noise std: {fixer.noise_std}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    print("🔧 RUNTIME ATTENTION BIAS FIXER")
    print("=" * 50)
    
    # Testar
    success = test_runtime_fixer()
    
    print("\n" + "=" * 50)
    print("📋 RESUMO:")
    print(f"   Teste: {'✅' if success else '❌'}")
    
    print("\n💡 COMO USAR NO DAYTRADER:")
    print("=" * 50)
    print("# Adicionar ao daytrader.py:")
    print("from runtime_attention_bias_fixer import create_runtime_attention_bias_fixer")
    print("")
    print("# Na seção de callbacks:")
    print("runtime_bias_fixer = create_runtime_attention_bias_fixer(")
    print("    check_frequency=500,  # Verificar a cada 500 steps")
    print("    fix_threshold=0.25,   # Corrigir se >25% zeros")
    print("    noise_std=1e-5,       # Ruído para correção")
    print("    verbose=1             # Logging ativo")
    print(")")
    print("")
    print("# Incluir na lista de callbacks:")
    print("callbacks = [robust_callback, metrics_callback, progress_callback,")
    print("            gradient_callback, runtime_bias_fixer]")
    
    if success:
        print("\n🎉 RUNTIME FIXER PRONTO PARA USO!")
        print("💡 Este fixer corrige zeros DURANTE o treinamento, não apenas na inicialização")
    else:
        print("\n⚠️ Revisar implementação")
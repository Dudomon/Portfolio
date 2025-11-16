#!/usr/bin/env python3
"""
🚀 LIGHTWEIGHT GRADIENT MONITOR - ULTRA-OTIMIZADO
Versão minimalista para manter velocidade máxima (150it/s)

OTIMIZAÇÕES:
- Apenas métricas essenciais
- Sem loops desnecessários  
- Sem logging pesado
- Sem análise de tendências
- Cálculos em batch quando possível
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

class LightweightGradientMonitor:
    """🚀 Monitor ultra-leve de gradientes"""
    
    def __init__(self, 
                 model: nn.Module,
                 check_frequency: int = 1000,  # Menos frequente
                 enable_auto_fix: bool = True,
                 alert_threshold: float = 0.3):
        
        self.model = model
        self.check_frequency = check_frequency
        self.enable_auto_fix = enable_auto_fix
        self.alert_threshold = alert_threshold
        
        # Apenas contadores essenciais
        self.step_count = 0
        self.corrections_applied = 0
        self.last_health_score = 1.0
        
        print(f"✅ Gradient Health Monitor ativado (check a cada {check_frequency} steps)")
        print(f"   Check frequency: {check_frequency} steps")
        print(f"   Auto-fix: {enable_auto_fix}")
        print(f"   Alert threshold: {alert_threshold}")
    
    def quick_health_check(self, step: int) -> Optional[Dict]:
        """🚀 ULTRA-RÁPIDO: Check minimalista"""
        self.step_count = step
        
        # Só verificar na frequência definida
        if step % self.check_frequency != 0:
            return None
        
        # OTIMIZAÇÃO: Contar apenas o essencial
        zero_params = 0
        total_params = 0
        total_norm_sq = 0.0
        
        # Loop único e otimizado
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                total_params += 1
                
                # Verificar zeros (mais rápido que torch.norm)  
                grad_data = param.grad.data
                if torch.sum(torch.abs(grad_data) < 1e-8).item() == grad_data.numel():
                    zero_params += 1
                
                # Norma acumulada (evita cálculos individuais)
                total_norm_sq += torch.sum(grad_data ** 2).item()
        
        # Cálculos finais simples
        total_norm = np.sqrt(total_norm_sq) if total_norm_sq > 0 else 0.0
        zero_ratio = zero_params / max(total_params, 1)
        
        # Score simplificado (0-1)
        health_score = 1.0 - min(zero_ratio * 2, 1.0)  # Penalizar zeros
        if total_norm > 100.0:  # Gradientes explodindo
            health_score *= 0.5
        elif total_norm < 0.001:  # Gradientes desaparecendo
            health_score *= 0.8
            
        self.last_health_score = health_score
        
        # Print minimalista apenas se necessário
        if health_score < self.alert_threshold or step % (self.check_frequency * 5) == 0:
            print(f"🔍 HEALTH CHECK - Step {step}")
            print(f"   Health Score: {health_score:.3f}")
            print(f"   Zero params: {zero_params}/{total_params} ({zero_ratio*100:.1f}%)")
            print(f"   Parameter norm: {total_norm:.4f}")
        
        return {
            'step': step,
            'health_score': health_score,
            'zero_params': zero_params,
            'total_params': total_params,
            'parameter_norm': total_norm
        }
    
    def apply_quick_fixes(self) -> int:
        """🔧 ULTRA-RÁPIDO: Correções essenciais"""
        if not self.enable_auto_fix:
            return 0
            
        corrections = 0
        
        # Loop único para correções críticas
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_data = param.grad.data
                
                # Só corrigir NaN/Inf (crítico)
                if torch.isnan(grad_data).any() or torch.isinf(grad_data).any():
                    param.grad.data = torch.zeros_like(grad_data)
                    corrections += 1
        
        if corrections > 0:
            self.corrections_applied += corrections
            print(f"🔧 Corrigidos {corrections} gradientes NaN/Inf")
        
        return corrections
    
    def get_status(self) -> str:
        """📊 Status atual simplificado"""
        if self.last_health_score > 0.8:
            return "HEALTHY"
        elif self.last_health_score > 0.5:
            return "WARNING"
        else:
            return "CRITICAL"


def setup_lightweight_monitoring(model: nn.Module, check_frequency: int = 1000) -> LightweightGradientMonitor:
    """🏭 Setup rápido do monitor leve"""
    return LightweightGradientMonitor(
        model=model,
        check_frequency=check_frequency,
        enable_auto_fix=True,
        alert_threshold=0.3
    )


# INTEGRAÇÃO DIRETA NO TRAINING LOOP
class FastGradientCallback:
    """⚡ Callback ultra-otimizado para training loop"""
    
    def __init__(self, model, check_frequency: int = 1000):
        self.monitor = LightweightGradientMonitor(model, check_frequency)
        
    def __call__(self, step: int):
        """Chamada direta no training loop"""
        report = self.monitor.quick_health_check(step)
        if report and report['health_score'] < 0.3:
            self.monitor.apply_quick_fixes()
        return report


if __name__ == "__main__":
    # Teste de performance
    import time
    
    print("🚀 Testando Lightweight Gradient Monitor...")
    
    model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(), 
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    
    monitor = setup_lightweight_monitoring(model, check_frequency=10)
    
    start_time = time.time()
    
    # Simular 1000 steps
    for step in range(1000):
        # Forward/backward
        x = torch.randn(64, 1000)
        y = model(x)
        loss = torch.mean(y ** 2)
        loss.backward()
        
        # Monitor (apenas quando necessário)
        monitor.quick_health_check(step)
        
        model.zero_grad()
    
    elapsed = time.time() - start_time
    print(f"✅ 1000 steps em {elapsed:.2f}s = {1000/elapsed:.1f} it/s")
    print(f"📊 Status final: {monitor.get_status()}")
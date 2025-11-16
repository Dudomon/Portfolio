#!/usr/bin/env python3
"""
🔍 GRADIENT HEALTH MONITOR
Sistema para monitorar e garantir qualidade dos gradientes durante treinamento

FUNCIONALIDADES:
1. Monitoramento em tempo real dos gradientes
2. Detecção de gradientes zerados/explodindo
3. Correção automática de problemas
4. Logging detalhado para análise
5. Alertas e intervenções automáticas
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
import time
from datetime import datetime
import json
import os

class GradientHealthMonitor:
    """🔍 Monitor de saúde dos gradientes"""
    
    def __init__(self, 
                 model: nn.Module,
                 log_dir: str = "gradient_logs",
                 check_frequency: int = 100,
                 gradient_clip_value: float = 1.0,
                 min_gradient_norm: float = 1e-8,
                 max_gradient_norm: float = 10.0):
        
        self.model = model
        self.log_dir = log_dir
        self.check_frequency = check_frequency
        self.gradient_clip_value = gradient_clip_value
        self.min_gradient_norm = min_gradient_norm
        self.max_gradient_norm = max_gradient_norm
        
        # Criar diretório de logs
        os.makedirs(log_dir, exist_ok=True)
        
        # Configurar logging
        self.logger = logging.getLogger('GradientHealthMonitor')
        handler = logging.FileHandler(f'{log_dir}/gradient_health_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Histórico de gradientes
        self.gradient_history = defaultdict(lambda: deque(maxlen=1000))
        self.health_metrics = deque(maxlen=1000)
        
        # Contadores
        self.step_count = 0
        self.zero_gradient_warnings = 0
        self.exploding_gradient_warnings = 0
        self.corrections_applied = 0
        
        # Status
        self.last_check_time = time.time()
        self.problematic_layers = set()
        
        self.logger.info("GradientHealthMonitor inicializado")
        self.logger.info(f"Configurações: clip={gradient_clip_value}, min_norm={min_gradient_norm}, max_norm={max_gradient_norm}")
    
    def check_gradient_health(self, step: int) -> Dict[str, any]:
        """🔍 Verificar saúde dos gradientes"""
        self.step_count = step
        
        if step % self.check_frequency != 0:
            return {}
        
        health_report = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'total_params': 0,
            'params_with_grad': 0,
            'zero_gradients': 0,
            'small_gradients': 0,
            'large_gradients': 0,
            'nan_gradients': 0,
            'inf_gradients': 0,
            'gradient_norms': {},
            'problematic_layers': [],
            'health_score': 0.0,
            'recommendations': []
        }
        
        total_grad_norm = 0.0
        layer_problems = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                health_report['total_params'] += 1
                
                if param.grad is not None:
                    health_report['params_with_grad'] += 1
                    grad_data = param.grad.data
                    
                    # Calcular norma do gradiente
                    grad_norm = torch.norm(grad_data).item()
                    health_report['gradient_norms'][name] = grad_norm
                    total_grad_norm += grad_norm ** 2
                    
                    # Verificar problemas
                    if torch.isnan(grad_data).any():
                        health_report['nan_gradients'] += 1
                        layer_problems.append(f"{name}: NaN gradients")
                        self.problematic_layers.add(name)
                    
                    elif torch.isinf(grad_data).any():
                        health_report['inf_gradients'] += 1
                        layer_problems.append(f"{name}: Inf gradients")
                        self.problematic_layers.add(name)
                    
                    elif grad_norm < self.min_gradient_norm:
                        health_report['zero_gradients'] += 1
                        if grad_norm == 0.0:
                            layer_problems.append(f"{name}: Zero gradients")
                            self.problematic_layers.add(name)
                    
                    elif grad_norm < 1e-6:
                        health_report['small_gradients'] += 1
                        layer_problems.append(f"{name}: Very small gradients ({grad_norm:.2e})")
                    
                    elif grad_norm > self.max_gradient_norm:
                        health_report['large_gradients'] += 1
                        layer_problems.append(f"{name}: Large gradients ({grad_norm:.2f})")
                        self.problematic_layers.add(name)
                    
                    # Armazenar histórico
                    self.gradient_history[name].append({
                        'step': step,
                        'norm': grad_norm,
                        'mean': torch.mean(torch.abs(grad_data)).item(),
                        'std': torch.std(grad_data).item()
                    })
        
        # Calcular norma total
        total_grad_norm = np.sqrt(total_grad_norm)
        health_report['total_gradient_norm'] = total_grad_norm
        health_report['problematic_layers'] = layer_problems
        
        # Calcular score de saúde (0-1)
        health_score = self._calculate_health_score(health_report)
        health_report['health_score'] = health_score
        
        # Gerar recomendações
        recommendations = self._generate_recommendations(health_report)
        health_report['recommendations'] = recommendations
        
        # Armazenar métricas
        self.health_metrics.append(health_report)
        
        # Log se houver problemas
        if health_score < 0.7 or layer_problems:
            self.logger.warning(f"Step {step}: Health score {health_score:.3f}")
            for problem in layer_problems[:5]:  # Mostrar apenas os primeiros 5
                self.logger.warning(f"  {problem}")
            for rec in recommendations:
                self.logger.info(f"  Recomendação: {rec}")
        
        return health_report
    
    def _calculate_health_score(self, report: Dict) -> float:
        """Calcular score de saúde dos gradientes (0-1)"""
        if report['total_params'] == 0:
            return 0.0
        
        score = 1.0
        
        # Penalizar gradientes problemáticos
        if report['nan_gradients'] > 0:
            score -= 0.5  # NaN é muito grave
        
        if report['inf_gradients'] > 0:
            score -= 0.4  # Inf é grave
        
        # Penalizar muitos gradientes zerados
        zero_ratio = report['zero_gradients'] / report['total_params']
        if zero_ratio > 0.5:
            score -= 0.3
        elif zero_ratio > 0.2:
            score -= 0.1
        
        # Penalizar gradientes muito grandes
        large_ratio = report['large_gradients'] / report['total_params']
        if large_ratio > 0.1:
            score -= 0.2
        
        # Penalizar norma total muito alta ou muito baixa
        total_norm = report.get('total_gradient_norm', 0)
        if total_norm > 50.0:
            score -= 0.2
        elif total_norm < 0.001:
            score -= 0.1
        
        return max(0.0, score)
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Gerar recomendações baseadas no relatório"""
        recommendations = []
        
        if report['nan_gradients'] > 0:
            recommendations.append("CRÍTICO: Gradientes NaN detectados - verificar loss function e dados")
        
        if report['inf_gradients'] > 0:
            recommendations.append("CRÍTICO: Gradientes Inf detectados - reduzir learning rate")
        
        if report['zero_gradients'] > report['total_params'] * 0.3:
            recommendations.append("Muitos gradientes zerados - verificar arquitetura e forward pass")
        
        if report['large_gradients'] > report['total_params'] * 0.1:
            recommendations.append("Gradientes explodindo - aplicar gradient clipping mais agressivo")
        
        total_norm = report.get('total_gradient_norm', 0)
        if total_norm > 20.0:
            recommendations.append(f"Norma total alta ({total_norm:.2f}) - reduzir learning rate")
        elif total_norm < 0.01:
            recommendations.append(f"Norma total baixa ({total_norm:.4f}) - aumentar learning rate")
        
        if not recommendations:
            recommendations.append("Gradientes saudáveis - continuar treinamento")
        
        return recommendations
    
    def apply_gradient_corrections(self) -> int:
        """🔧 Aplicar correções automáticas nos gradientes"""
        corrections = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_data = param.grad.data
                
                # Correção 1: Substituir NaN e Inf
                if torch.isnan(grad_data).any() or torch.isinf(grad_data).any():
                    param.grad.data = torch.zeros_like(grad_data)
                    corrections += 1
                    self.logger.warning(f"Corrigido NaN/Inf em {name}")
                
                # Correção 2: Clipping de gradientes extremos
                grad_norm = torch.norm(grad_data).item()
                if grad_norm > self.max_gradient_norm:
                    param.grad.data = param.grad.data * (self.max_gradient_norm / grad_norm)
                    corrections += 1
                    self.logger.info(f"Aplicado clipping em {name}: {grad_norm:.2f} -> {self.max_gradient_norm}")
        
        # Aplicar gradient clipping global
        if corrections > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
            self.corrections_applied += corrections
        
        return corrections
    
    def get_health_summary(self) -> Dict:
        """📊 Obter resumo da saúde dos gradientes"""
        if not self.health_metrics:
            return {"status": "no_data"}
        
        recent_metrics = list(self.health_metrics)[-10:]  # Últimas 10 verificações
        
        avg_health_score = np.mean([m['health_score'] for m in recent_metrics])
        avg_zero_gradients = np.mean([m['zero_gradients'] for m in recent_metrics])
        avg_total_norm = np.mean([m.get('total_gradient_norm', 0) for m in recent_metrics])
        
        return {
            'status': 'healthy' if avg_health_score > 0.7 else 'problematic',
            'avg_health_score': avg_health_score,
            'avg_zero_gradients': avg_zero_gradients,
            'avg_total_norm': avg_total_norm,
            'total_corrections': self.corrections_applied,
            'problematic_layers_count': len(self.problematic_layers),
            'most_problematic_layers': list(self.problematic_layers)[:5]
        }
    
    def save_detailed_report(self, filename: Optional[str] = None):
        """💾 Salvar relatório detalhado"""
        if filename is None:
            filename = f"{self.log_dir}/gradient_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'summary': self.get_health_summary(),
            'configuration': {
                'check_frequency': self.check_frequency,
                'gradient_clip_value': self.gradient_clip_value,
                'min_gradient_norm': self.min_gradient_norm,
                'max_gradient_norm': self.max_gradient_norm
            },
            'recent_metrics': list(self.health_metrics)[-50:],  # Últimas 50 verificações
            'gradient_trends': self._analyze_gradient_trends()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Relatório detalhado salvo: {filename}")
        return filename
    
    def _analyze_gradient_trends(self) -> Dict:
        """📈 Analisar tendências dos gradientes"""
        trends = {}
        
        for layer_name, history in self.gradient_history.items():
            if len(history) < 5:
                continue
            
            recent_norms = [h['norm'] for h in list(history)[-20:]]
            
            if recent_norms:
                trends[layer_name] = {
                    'mean_norm': np.mean(recent_norms),
                    'std_norm': np.std(recent_norms),
                    'trend': 'increasing' if recent_norms[-1] > recent_norms[0] else 'decreasing',
                    'stability': 'stable' if np.std(recent_norms) < np.mean(recent_norms) * 0.5 else 'unstable'
                }
        
        return trends

def create_gradient_monitor(model: nn.Module, **kwargs) -> GradientHealthMonitor:
    """🏭 Factory function para criar monitor de gradientes"""
    return GradientHealthMonitor(model, **kwargs)

# Exemplo de uso como callback para Stable-Baselines3
class GradientHealthCallback:
    """📊 Callback para integração com Stable-Baselines3"""
    
    def __init__(self, model, check_frequency: int = 1000, **monitor_kwargs):
        self.monitor = GradientHealthMonitor(model.policy, check_frequency=check_frequency, **monitor_kwargs)
        self.last_report = None
    
    def on_training_step(self, step: int):
        """Chamado a cada step de treinamento"""
        report = self.monitor.check_gradient_health(step)
        
        if report:
            self.last_report = report
            
            # Aplicar correções se necessário
            if report['health_score'] < 0.5:
                corrections = self.monitor.apply_gradient_corrections()
                if corrections > 0:
                    print(f"Step {step}: Aplicadas {corrections} correções de gradiente")
            
            # Alertar se saúde muito baixa
            if report['health_score'] < 0.3:
                print(f"⚠️ Step {step}: Saúde dos gradientes baixa ({report['health_score']:.3f})")
                for rec in report['recommendations'][:2]:
                    print(f"   💡 {rec}")
    
    def get_summary(self):
        """Obter resumo da saúde"""
        return self.monitor.get_health_summary()
    
    def save_report(self):
        """Salvar relatório"""
        return self.monitor.save_detailed_report()

if __name__ == "__main__":
    # Teste básico
    print("🔍 Testando Gradient Health Monitor...")
    
    # Criar modelo de teste
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Criar monitor
    monitor = create_gradient_monitor(model, check_frequency=1)
    
    # Simular treinamento
    for step in range(5):
        # Forward pass
        x = torch.randn(32, 100)
        y = model(x)
        loss = torch.mean(y ** 2)
        
        # Backward pass
        loss.backward()
        
        # Verificar gradientes
        report = monitor.check_gradient_health(step)
        if report:
            print(f"Step {step}: Health score = {report['health_score']:.3f}")
        
        # Aplicar correções
        corrections = monitor.apply_gradient_corrections()
        if corrections > 0:
            print(f"  Aplicadas {corrections} correções")
        
        # Limpar gradientes
        model.zero_grad()
    
    # Salvar relatório
    report_file = monitor.save_detailed_report()
    print(f"✅ Relatório salvo: {report_file}")
    
    # Resumo final
    summary = monitor.get_health_summary()
    print(f"📊 Resumo: {summary}")
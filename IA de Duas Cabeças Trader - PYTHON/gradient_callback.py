#!/usr/bin/env python3
"""
🔧 GRADIENT CALLBACK
Callback para integração automática do monitoramento de gradientes
com Stable-Baselines3 e RecurrentPPO
"""

import os
import sys
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Optional

class GradientHealthCallback(BaseCallback):
    """
    🔍 Callback para monitoramento automático de gradientes
    
    Funcionalidades:
    - Monitora saúde dos gradientes em tempo real
    - Aplica correções automáticas quando necessário
    - Gera alertas para problemas críticos
    - Salva relatórios detalhados
    """
    
    def __init__(self, 
                 check_frequency: int = 500,
                 auto_fix: bool = True,
                 alert_threshold: float = 0.3,
                 log_dir: str = "gradient_logs",
                 verbose: int = 1):
        
        super().__init__(verbose)
        
        self.check_frequency = check_frequency
        self.auto_fix = auto_fix
        self.alert_threshold = alert_threshold
        self.log_dir = log_dir
        
        # Status
        self.monitoring_active = False
        self.total_corrections = 0
        self.last_health_score = 1.0
        self.critical_alerts = 0
        
        # Criar diretório de logs
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_training_start(self) -> None:
        """Inicializar monitoramento no início do treinamento"""
        try:
            # Verificar se o modelo suporta monitoramento de gradientes
            if hasattr(self.model.policy, 'setup_gradient_monitoring'):
                success = self.model.policy.setup_gradient_monitoring(
                    check_frequency=self.check_frequency,
                    log_dir=self.log_dir
                )
                
                if success:
                    self.monitoring_active = True
                    if self.verbose >= 1:
                        print(f"✅ Gradient Health Monitoring ativado")
                        print(f"   Check frequency: {self.check_frequency} steps")
                        print(f"   Auto-fix: {self.auto_fix}")
                        print(f"   Alert threshold: {self.alert_threshold}")
                else:
                    if self.verbose >= 1:
                        print("⚠️ Gradient Health Monitoring não pôde ser ativado")
            else:
                if self.verbose >= 1:
                    print("⚠️ Modelo não suporta Gradient Health Monitoring")
                    
        except Exception as e:
            if self.verbose >= 1:
                print(f"❌ Erro ao inicializar Gradient Health Monitoring: {e}")
    
    def _on_step(self) -> bool:
        """Verificar gradientes a cada step"""
        if not self.monitoring_active:
            return True
        
        try:
            # Verificar e corrigir gradientes
            if hasattr(self.model.policy, 'check_and_fix_gradients'):
                health_report = self.model.policy.check_and_fix_gradients(self.num_timesteps)
                
                if health_report:
                    health_score = health_report.get('health_score', 1.0)
                    self.last_health_score = health_score
                    
                    # Contar correções
                    if 'corrections_applied' in health_report:
                        self.total_corrections += health_report['corrections_applied']
                    
                    # Alertas críticos
                    if health_score < self.alert_threshold:
                        self.critical_alerts += 1
                        
                        if self.verbose >= 1:
                            print(f"\n⚠️ ALERTA CRÍTICO - Step {self.num_timesteps}")
                            print(f"   Saúde dos gradientes: {health_score:.3f}")
                            print(f"   Problemas detectados: {len(health_report.get('problematic_layers', []))}")
                            
                            # Mostrar recomendações principais
                            for rec in health_report.get('recommendations', [])[:2]:
                                print(f"   💡 {rec}")
                    
                    # Log periódico de status
                    elif self.num_timesteps % (self.check_frequency * 10) == 0 and self.verbose >= 2:
                        print(f"🔍 Step {self.num_timesteps}: Gradient health = {health_score:.3f}")
            
        except Exception as e:
            if self.verbose >= 1:
                print(f"❌ Erro no monitoramento de gradientes (step {self.num_timesteps}): {e}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Finalizar monitoramento e gerar relatório final"""
        if not self.monitoring_active:
            return
        
        try:
            if self.verbose >= 1:
                print(f"\n📊 RESUMO DO GRADIENT HEALTH MONITORING")
                print(f"=" * 50)
            
            # Obter resumo final
            if hasattr(self.model.policy, 'get_gradient_health_summary'):
                summary = self.model.policy.get_gradient_health_summary()
                
                if self.verbose >= 1:
                    print(f"Status final: {summary.get('status', 'unknown')}")
                    print(f"Saúde média: {summary.get('avg_health_score', 0):.3f}")
                    print(f"Total de correções: {summary.get('total_corrections', 0)}")
                    print(f"Alertas críticos: {self.critical_alerts}")
                    
                    if summary.get('most_problematic_layers'):
                        print(f"Layers mais problemáticos:")
                        for layer in summary['most_problematic_layers'][:3]:
                            print(f"  - {layer}")
            
            # Salvar relatório detalhado
            if hasattr(self.model.policy, 'save_gradient_report'):
                report_file = self.model.policy.save_gradient_report()
                if report_file and self.verbose >= 1:
                    print(f"📄 Relatório detalhado salvo: {report_file}")
            
            if self.verbose >= 1:
                print(f"=" * 50)
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"❌ Erro ao finalizar monitoramento: {e}")
    
    def get_monitoring_stats(self) -> Dict:
        """📊 Obter estatísticas do monitoramento"""
        return {
            'monitoring_active': self.monitoring_active,
            'total_corrections': self.total_corrections,
            'last_health_score': self.last_health_score,
            'critical_alerts': self.critical_alerts,
            'check_frequency': self.check_frequency
        }

def create_gradient_callback(**kwargs) -> GradientHealthCallback:
    """🏭 Factory function para criar callback de gradientes"""
    return GradientHealthCallback(**kwargs)

# Exemplo de uso
if __name__ == "__main__":
    print("🔧 Gradient Health Callback - Exemplo de uso:")
    print()
    print("# Integração com treinamento:")
    print("from gradient_callback import create_gradient_callback")
    print()
    print("# Criar callback")
    print("gradient_callback = create_gradient_callback(")
    print("    check_frequency=500,  # Verificar a cada 500 steps")
    print("    auto_fix=True,        # Aplicar correções automáticas")
    print("    alert_threshold=0.3,  # Alertar se saúde < 0.3")
    print("    verbose=1             # Nível de logging")
    print(")")
    print()
    print("# Usar no treinamento:")
    print("model.learn(")
    print("    total_timesteps=1000000,")
    print("    callback=[gradient_callback, other_callbacks...]")
    print(")")
    print()
    print("✅ Callback pronto para uso!")
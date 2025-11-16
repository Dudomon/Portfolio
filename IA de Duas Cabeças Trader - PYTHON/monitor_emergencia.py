#!/usr/bin/env python3
"""
🚨 MONITOR DE EMERGÊNCIA - CONVERGÊNCIA
======================================
Monitora em tempo real se as correções emergenciais estão funcionando:
- Explained Variance deve subir de -0.00667 para > 0.1
- Clip Fraction deve cair de 0.774 para < 0.2  
- Approx KL deve cair de 0.167 para < 0.05
- Drawdown deve melhorar de 95.9% para < 50%
"""

import time
import re
import os
from datetime import datetime
import numpy as np

class EmergencyMonitor:
    """🚨 Monitor de emergência para convergência"""
    
    def __init__(self):
        self.metrics_history = {
            'explained_variance': [],
            'clip_fraction': [],
            'approx_kl': [],
            'drawdown': [],
            'portfolio': [],
            'timestamps': []
        }
        
        # Valores críticos detectados
        self.critical_values = {
            'explained_variance': -0.00667,
            'clip_fraction': 0.774,
            'approx_kl': 0.167,
            'drawdown': 95.9
        }
        
        # Targets após correção
        self.targets = {
            'explained_variance': 0.1,
            'clip_fraction': 0.2,
            'approx_kl': 0.05,
            'drawdown': 50.0
        }
        
        self.start_time = datetime.now()
        self.last_check = None
        
    def extract_metrics_from_log(self, log_content):
        """📊 Extrair métricas dos logs de treinamento"""
        metrics = {}
        
        # Explained Variance
        ev_match = re.search(r'explained_variance\s*\|\s*([-\d\.e\-\+]+)', log_content)
        if ev_match:
            metrics['explained_variance'] = float(ev_match.group(1))
        
        # Clip Fraction
        cf_match = re.search(r'clip_fraction\s*\|\s*([\d\.]+)', log_content)
        if cf_match:
            metrics['clip_fraction'] = float(cf_match.group(1))
        
        # Approx KL
        kl_match = re.search(r'approx_kl\s*\|\s*([\d\.e\-\+]+)', log_content)
        if kl_match:
            metrics['approx_kl'] = float(kl_match.group(1))
        
        # Drawdown
        dd_match = re.search(r'DD=([\d\.]+)%', log_content)
        if dd_match:
            metrics['drawdown'] = float(dd_match.group(1))
        
        # Portfolio
        port_match = re.search(r'Portfolio=\$?([\d\.]+)', log_content)
        if port_match:
            metrics['portfolio'] = float(port_match.group(1))
        
        return metrics
    
    def update_metrics(self, metrics):
        """📈 Atualizar histórico de métricas"""
        timestamp = datetime.now()
        self.metrics_history['timestamps'].append(timestamp)
        
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Manter apenas últimos 50 pontos
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > 50:
                self.metrics_history[key] = self.metrics_history[key][-50:]
    
    def check_improvement(self):
        """✅ Verificar se as correções estão funcionando"""
        if not self.metrics_history['timestamps']:
            return None
        
        improvements = {}
        current_status = {}
        
        for metric in ['explained_variance', 'clip_fraction', 'approx_kl', 'drawdown']:
            if self.metrics_history[metric]:
                current_value = self.metrics_history[metric][-1]
                critical_value = self.critical_values[metric]
                target_value = self.targets[metric]
                
                current_status[metric] = current_value
                
                if metric == 'explained_variance':
                    # Para EV, maior é melhor
                    improvement = (current_value - critical_value) / abs(target_value - critical_value) * 100
                    improvements[metric] = min(100, max(-100, improvement))
                    
                elif metric in ['clip_fraction', 'approx_kl', 'drawdown']:
                    # Para estes, menor é melhor
                    improvement = (critical_value - current_value) / abs(critical_value - target_value) * 100
                    improvements[metric] = min(100, max(-100, improvement))
        
        return improvements, current_status
    
    def generate_status_report(self):
        """📋 Gerar relatório de status"""
        improvements, current_status = self.check_improvement()
        
        if not improvements:
            return "❌ Nenhuma métrica disponível ainda"
        
        elapsed = datetime.now() - self.start_time
        
        report = []
        report.append(f"🚨 === MONITOR DE EMERGÊNCIA - CONVERGÊNCIA ===")
        report.append(f"⏱️ Tempo desde correções: {elapsed}")
        report.append(f"📊 Última atualização: {self.metrics_history['timestamps'][-1].strftime('%H:%M:%S')}")
        report.append("")
        
        # Status de cada métrica
        for metric, improvement in improvements.items():
            current = current_status[metric]
            critical = self.critical_values[metric]
            target = self.targets[metric]
            
            if improvement >= 80:
                status = "✅ EXCELENTE"
                icon = "🟢"
            elif improvement >= 50:
                status = "🔶 BOM"
                icon = "🟡"
            elif improvement >= 20:
                status = "⚠️ MELHORANDO"
                icon = "🟠"
            elif improvement > 0:
                status = "🔄 LEVE MELHORA"
                icon = "🔵"
            else:
                status = "❌ SEM MELHORA"
                icon = "🔴"
            
            report.append(f"{icon} {metric.upper().replace('_', ' ')}:")
            report.append(f"   Crítico: {critical:.6f} → Atual: {current:.6f} → Target: {target:.6f}")
            report.append(f"   Progresso: {improvement:.1f}% {status}")
            report.append("")
        
        # Score geral
        overall_score = np.mean(list(improvements.values()))
        if overall_score >= 80:
            overall_status = "🎯 CONVERGÊNCIA EXCELENTE"
        elif overall_score >= 50:
            overall_status = "🔶 CONVERGÊNCIA BOA"
        elif overall_score >= 20:
            overall_status = "⚠️ CONVERGÊNCIA MELHORANDO"
        else:
            overall_status = "❌ CONVERGÊNCIA PROBLEMÁTICA"
        
        report.append(f"📊 SCORE GERAL: {overall_score:.1f}% - {overall_status}")
        
        return "\n".join(report)
    
    def monitor_training_logs(self, log_file_pattern="logs/*.log"):
        """🔍 Monitorar logs de treinamento"""
        print("🚨 Iniciando monitoramento de emergência...")
        print("Pressione Ctrl+C para parar")
        
        try:
            while True:
                # Procurar por arquivos de log
                import glob
                log_files = glob.glob(log_file_pattern)
                
                if log_files:
                    # Pegar o log mais recente
                    latest_log = max(log_files, key=os.path.getmtime)
                    
                    try:
                        with open(latest_log, 'r', encoding='utf-8') as f:
                            # Ler últimas 100 linhas
                            lines = f.readlines()
                            recent_content = ''.join(lines[-100:])
                            
                        metrics = self.extract_metrics_from_log(recent_content)
                        
                        if metrics:
                            self.update_metrics(metrics)
                            
                            # Gerar relatório a cada 30 segundos
                            now = datetime.now()
                            if not self.last_check or (now - self.last_check).seconds >= 30:
                                print("\n" + "="*70)
                                print(self.generate_status_report())
                                print("="*70)
                                self.last_check = now
                    
                    except Exception as e:
                        print(f"⚠️ Erro ao ler log {latest_log}: {e}")
                
                time.sleep(5)  # Verificar a cada 5 segundos
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoramento interrompido pelo usuário")
            if self.metrics_history['timestamps']:
                print("\n📋 RELATÓRIO FINAL:")
                print(self.generate_status_report())

def monitor_console_output():
    """🖥️ Monitorar saída do console em tempo real"""
    monitor = EmergencyMonitor()
    
    print("🚨 === MONITOR DE EMERGÊNCIA ATIVO ===")
    print("📊 Monitorando métricas críticas de convergência:")
    print("   - Explained Variance: -0.00667 → > 0.1")
    print("   - Clip Fraction: 0.774 → < 0.2")
    print("   - Approx KL: 0.167 → < 0.05")
    print("   - Drawdown: 95.9% → < 50%")
    print("\n🔍 Procurando por logs de treinamento...")
    
    # Monitorar logs
    monitor.monitor_training_logs()

if __name__ == "__main__":
    monitor_console_output() 
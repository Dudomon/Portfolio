#!/usr/bin/env python3
"""
📊 MONITOR DE CONVERGÊNCIA EM TEMPO REAL
Acompanha métricas críticas durante o treinamento
"""

import time
import os
import re
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

class ConvergenceMonitor:
    """📊 Monitor de convergência em tempo real"""
    
    def __init__(self, log_file_pattern="logs/*.txt"):
        self.log_files = []
        self.metrics_history = {
            'explained_variance': deque(maxlen=100),
            'clip_fraction': deque(maxlen=100),
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy_loss': deque(maxlen=100),
            'learning_rate': deque(maxlen=100),
            'portfolio_value': deque(maxlen=100),
            'episode_reward': deque(maxlen=100),
            'timestamps': deque(maxlen=100)
        }
        
        self.convergence_status = {
            'explained_variance_ok': False,
            'clip_fraction_ok': False,
            'learning_stable': False,
            'portfolio_growing': False
        }
        
        self.thresholds = {
            'explained_variance_min': 0.1,
            'clip_fraction_max': 0.3,
            'policy_loss_max': 1.0,
            'value_loss_max': 1.0
        }
        
        print("📊 Monitor de Convergência Inicializado")
        print(f"🎯 Thresholds: explained_variance > {self.thresholds['explained_variance_min']}")
        print(f"🎯 Thresholds: clip_fraction < {self.thresholds['clip_fraction_max']}")
    
    def find_latest_log(self):
        """🔍 Encontrar arquivo de log mais recente"""
        log_dirs = ['logs', 'treino_principal/logs', '.']
        
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                files = [f for f in os.listdir(log_dir) if f.endswith('.txt') or f.endswith('.log')]
                if files:
                    latest_file = max([os.path.join(log_dir, f) for f in files], key=os.path.getmtime)
                    return latest_file
        
        return None
    
    def parse_log_line(self, line):
        """📝 Extrair métricas de uma linha de log"""
        metrics = {}
        
        # Padrões de regex para diferentes métricas
        patterns = {
            'explained_variance': r'explained_variance[:\s]+(-?[\d.e-]+)',
            'clip_fraction': r'clip_fraction[:\s]+([\d.e-]+)',
            'policy_loss': r'policy_loss[:\s]+(-?[\d.e-]+)',
            'value_loss': r'value_loss[:\s]+(-?[\d.e-]+)',
            'entropy_loss': r'entropy_loss[:\s]+(-?[\d.e-]+)',
            'learning_rate': r'learning_rate[:\s]+([\d.e-]+)',
            'portfolio': r'portfolio[:\s$]+([\d.]+)',
            'reward': r'reward[:\s]+(-?[\d.e-]+)',
            'episode_reward': r'episode_reward[:\s]+(-?[\d.e-]+)'
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, line.lower())
            if match:
                try:
                    value = float(match.group(1))
                    metrics[metric] = value
                except:
                    pass
        
        return metrics
    
    def update_metrics(self, metrics):
        """📊 Atualizar histórico de métricas"""
        timestamp = datetime.now()
        self.metrics_history['timestamps'].append(timestamp)
        
        for metric, value in metrics.items():
            if metric in self.metrics_history:
                self.metrics_history[metric].append(value)
            elif metric == 'portfolio':
                self.metrics_history['portfolio_value'].append(value)
            elif metric in ['reward', 'episode_reward']:
                self.metrics_history['episode_reward'].append(value)
    
    def check_convergence_status(self):
        """✅ Verificar status de convergência"""
        if not self.metrics_history['explained_variance']:
            return
        
        # Explained Variance
        recent_ev = list(self.metrics_history['explained_variance'])[-5:]  # Últimos 5 valores
        if recent_ev:
            avg_ev = np.mean(recent_ev)
            self.convergence_status['explained_variance_ok'] = avg_ev > self.thresholds['explained_variance_min']
        
        # Clip Fraction
        recent_cf = list(self.metrics_history['clip_fraction'])[-5:]
        if recent_cf:
            avg_cf = np.mean(recent_cf)
            self.convergence_status['clip_fraction_ok'] = avg_cf < self.thresholds['clip_fraction_max']
        
        # Learning Stability (policy loss não explodindo)
        recent_pl = list(self.metrics_history['policy_loss'])[-5:]
        if recent_pl:
            avg_pl = np.mean(recent_pl)
            self.convergence_status['learning_stable'] = avg_pl < self.thresholds['policy_loss_max']
        
        # Portfolio Growth
        if len(self.metrics_history['portfolio_value']) >= 10:
            recent_portfolio = list(self.metrics_history['portfolio_value'])[-10:]
            if len(recent_portfolio) >= 2:
                trend = recent_portfolio[-1] - recent_portfolio[0]
                self.convergence_status['portfolio_growing'] = trend > 0
    
    def print_status(self):
        """📊 Imprimir status atual"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("📊 MONITOR DE CONVERGÊNCIA EM TEMPO REAL")
        print("=" * 70)
        print(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
        
        # Métricas atuais
        print(f"\n📈 MÉTRICAS ATUAIS:")
        print("-" * 40)
        
        if self.metrics_history['explained_variance']:
            ev = self.metrics_history['explained_variance'][-1]
            status = "✅" if ev > self.thresholds['explained_variance_min'] else "❌"
            print(f"{status} Explained Variance: {ev:.6f} (target: >{self.thresholds['explained_variance_min']})")
        
        if self.metrics_history['clip_fraction']:
            cf = self.metrics_history['clip_fraction'][-1]
            status = "✅" if cf < self.thresholds['clip_fraction_max'] else "❌"
            print(f"{status} Clip Fraction: {cf:.4f} (target: <{self.thresholds['clip_fraction_max']})")
        
        if self.metrics_history['policy_loss']:
            pl = self.metrics_history['policy_loss'][-1]
            status = "✅" if pl < self.thresholds['policy_loss_max'] else "❌"
            print(f"{status} Policy Loss: {pl:.6f} (target: <{self.thresholds['policy_loss_max']})")
        
        if self.metrics_history['value_loss']:
            vl = self.metrics_history['value_loss'][-1]
            status = "✅" if vl < self.thresholds['value_loss_max'] else "❌"
            print(f"{status} Value Loss: {vl:.6f} (target: <{self.thresholds['value_loss_max']})")
        
        if self.metrics_history['learning_rate']:
            lr = self.metrics_history['learning_rate'][-1]
            print(f"📚 Learning Rate: {lr:.2e}")
        
        if self.metrics_history['portfolio_value']:
            pv = self.metrics_history['portfolio_value'][-1]
            print(f"💰 Portfolio Value: ${pv:.2f}")
        
        # Status de convergência
        print(f"\n🎯 STATUS DE CONVERGÊNCIA:")
        print("-" * 40)
        
        convergence_checks = [
            ("Value Network Learning", self.convergence_status['explained_variance_ok']),
            ("Policy Stability", self.convergence_status['clip_fraction_ok']),
            ("Training Stability", self.convergence_status['learning_stable']),
            ("Portfolio Growth", self.convergence_status['portfolio_growing'])
        ]
        
        total_ok = sum(1 for _, status in convergence_checks if status)
        
        for check_name, status in convergence_checks:
            icon = "✅" if status else "❌"
            print(f"{icon} {check_name}")
        
        print(f"\n🏆 SCORE DE CONVERGÊNCIA: {total_ok}/4")
        
        if total_ok == 4:
            print("🎉 CONVERGÊNCIA ALCANÇADA! Modelo treinando corretamente!")
        elif total_ok >= 2:
            print("⚠️ Convergência parcial - monitorar evolução")
        else:
            print("🚨 Problemas de convergência detectados")
        
        # Tendências
        if len(self.metrics_history['explained_variance']) >= 5:
            recent_ev = list(self.metrics_history['explained_variance'])[-5:]
            trend = "📈" if recent_ev[-1] > recent_ev[0] else "📉"
            print(f"\n{trend} Tendência Explained Variance: {recent_ev[0]:.4f} → {recent_ev[-1]:.4f}")
        
        if len(self.metrics_history['clip_fraction']) >= 5:
            recent_cf = list(self.metrics_history['clip_fraction'])[-5:]
            trend = "📉" if recent_cf[-1] < recent_cf[0] else "📈"  # Queremos que diminua
            print(f"{trend} Tendência Clip Fraction: {recent_cf[0]:.4f} → {recent_cf[-1]:.4f}")
        
        print(f"\n💡 Pressione Ctrl+C para parar o monitoramento")
    
    def monitor_realtime(self, update_interval=10):
        """🔄 Monitoramento em tempo real"""
        print("🚀 Iniciando monitoramento em tempo real...")
        print(f"🔄 Atualizando a cada {update_interval} segundos")
        
        last_position = 0
        
        try:
            while True:
                log_file = self.find_latest_log()
                
                if log_file and os.path.exists(log_file):
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            f.seek(last_position)
                            new_lines = f.readlines()
                            last_position = f.tell()
                        
                        # Processar novas linhas
                        for line in new_lines:
                            metrics = self.parse_log_line(line)
                            if metrics:
                                self.update_metrics(metrics)
                        
                        if new_lines:
                            self.check_convergence_status()
                            self.print_status()
                    
                    except Exception as e:
                        print(f"Erro ao ler log: {e}")
                
                else:
                    print(f"⏳ Aguardando arquivo de log... ({datetime.now().strftime('%H:%M:%S')})")
                
                time.sleep(update_interval)
        
        except KeyboardInterrupt:
            print(f"\n\n🛑 Monitoramento interrompido pelo usuário")
            return self.generate_summary()
    
    def generate_summary(self):
        """📋 Gerar resumo final"""
        print(f"\n📋 RESUMO FINAL DO MONITORAMENTO")
        print("=" * 50)
        
        if not any(self.metrics_history[key] for key in self.metrics_history if key != 'timestamps'):
            print("❌ Nenhuma métrica coletada")
            return
        
        summary = {}
        
        for metric, values in self.metrics_history.items():
            if values and metric != 'timestamps':
                summary[metric] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': np.mean(values),
                    'final': values[-1] if values else 0,
                    'count': len(values)
                }
        
        print(f"📊 Métricas coletadas: {len(summary)}")
        print(f"📈 Pontos de dados: {len(self.metrics_history['timestamps'])}")
        
        if 'explained_variance' in summary:
            ev = summary['explained_variance']
            print(f"\n🎯 Explained Variance:")
            print(f"   Final: {ev['final']:.6f}")
            print(f"   Média: {ev['avg']:.6f}")
            print(f"   Range: {ev['min']:.6f} - {ev['max']:.6f}")
        
        if 'clip_fraction' in summary:
            cf = summary['clip_fraction']
            print(f"\n🎯 Clip Fraction:")
            print(f"   Final: {cf['final']:.4f}")
            print(f"   Média: {cf['avg']:.4f}")
            print(f"   Range: {cf['min']:.4f} - {cf['max']:.4f}")
        
        return summary

def main():
    """🚀 Função principal"""
    monitor = ConvergenceMonitor()
    
    print("📊 MONITOR DE CONVERGÊNCIA")
    print("Este monitor acompanha as métricas críticas em tempo real")
    print("para verificar se as correções estão funcionando.\n")
    
    try:
        summary = monitor.monitor_realtime(update_interval=5)
        return summary
    except Exception as e:
        print(f"❌ Erro no monitoramento: {e}")
        return None

if __name__ == "__main__":
    main() 
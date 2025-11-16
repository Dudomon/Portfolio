"""
🕵️ MONITOR DE REWARD HACKING EM TEMPO REAL
Monitora JSONL logs durante treinamento para detectar reward hacking
"""

import json
import time
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque, defaultdict
import threading
import sys

class RewardHackingMonitor:
    """Monitor em tempo real para detectar reward hacking nos logs JSONL"""
    
    def __init__(self, jsonl_dir="avaliacoes", check_interval=10):
        self.jsonl_dir = jsonl_dir
        self.check_interval = check_interval
        self.is_monitoring = False
        
        # Buffers para análise temporal
        self.recent_rewards = deque(maxlen=100)  # Últimos 100 rewards
        self.recent_sharpe = deque(maxlen=50)    # Últimos 50 sharpe calculations
        self.recent_clip_fraction = deque(maxlen=50)  # Últimas 50 clip fractions
        self.recent_portfolio_values = deque(maxlen=100)  # Portfolio values
        
        # Métricas de alerta
        self.alert_thresholds = {
            'clip_fraction_high': 0.35,      # Clip fraction muito alto
            'reward_sharpe_divergence': 0.5,  # Divergência entre reward e Sharpe
            'reward_inflation': 2.0,          # Rewards crescendo muito rápido
            'performance_stagnation': 0.1     # Performance real estagnada
        }
        
        # Estado atual
        self.current_stats = {
            'total_steps': 0,
            'avg_reward': 0.0,
            'avg_clip_fraction': 0.0,
            'estimated_sharpe': 0.0,
            'portfolio_growth': 0.0,
            'last_update': None
        }
        
        # Alertas ativos
        self.active_alerts = set()
        
    def get_latest_jsonl_files(self):
        """Encontra os arquivos JSONL mais recentes"""
        patterns = [
            f"{self.jsonl_dir}/training_*.jsonl",
            f"{self.jsonl_dir}/rewards_*.jsonl", 
            f"{self.jsonl_dir}/performance_*.jsonl"
        ]
        
        latest_files = {}
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                # Pega o mais recente baseado no timestamp do nome
                latest = max(files, key=os.path.getmtime)
                file_type = os.path.basename(latest).split('_')[0]
                latest_files[file_type] = latest
                
        return latest_files
    
    def parse_jsonl_line(self, line):
        """Parse uma linha do JSONL com tratamento de erro"""
        try:
            data = json.loads(line.strip())
            return data
        except (json.JSONDecodeError, ValueError):
            return None
    
    def read_recent_jsonl_data(self, file_path, max_lines=200):
        """Lê as últimas linhas de um arquivo JSONL"""
        try:
            if not os.path.exists(file_path):
                return []
                
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = deque(f, maxlen=max_lines)
                
            data = []
            for line in lines:
                parsed = self.parse_jsonl_line(line)
                if parsed:
                    data.append(parsed)
                    
            return data
        except Exception as e:
            print(f"⚠️ Erro lendo {file_path}: {e}")
            return []
    
    def analyze_training_data(self, training_data):
        """Analisa dados de training para sinais de hacking"""
        if not training_data:
            return
            
        for entry in training_data[-50:]:  # Últimas 50 entradas
            if 'clip_fraction' in entry:
                self.recent_clip_fraction.append(entry['clip_fraction'])
            
            if 'step' in entry:
                self.current_stats['total_steps'] = max(self.current_stats['total_steps'], entry['step'])
    
    def analyze_rewards_data(self, rewards_data):
        """Analisa dados de rewards para detectar inflação"""
        if not rewards_data:
            return
            
        for entry in rewards_data[-100:]:  # Últimas 100 entradas
            if 'reward' in entry:
                self.recent_rewards.append(entry['reward'])
            elif 'episode_reward_mean' in entry:
                self.recent_rewards.append(entry['episode_reward_mean'])
    
    def analyze_performance_data(self, performance_data):
        """Analisa dados de performance para calcular métricas reais"""
        if not performance_data:
            return
            
        portfolio_values = []
        for entry in performance_data[-100:]:
            if 'portfolio_value' in entry:
                portfolio_values.append(entry['portfolio_value'])
                self.recent_portfolio_values.append(entry['portfolio_value'])
        
        # Estimar Sharpe a partir do portfolio
        if len(portfolio_values) >= 20:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            if np.std(returns) > 0:
                sharpe_estimate = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Anualizado
                self.recent_sharpe.append(sharpe_estimate)
    
    def detect_reward_hacking(self):
        """Detecta sinais de reward hacking"""
        alerts = set()
        
        # 1. CLIP FRACTION MUITO ALTO
        if self.recent_clip_fraction:
            avg_clip = np.mean(list(self.recent_clip_fraction)[-20:])  # Últimas 20
            self.current_stats['avg_clip_fraction'] = avg_clip
            
            if avg_clip > self.alert_thresholds['clip_fraction_high']:
                alerts.add('HIGH_CLIP_FRACTION')
        
        # 2. REWARDS INFLANDO SEM PERFORMANCE REAL
        if len(self.recent_rewards) >= 20 and len(self.recent_sharpe) >= 10:
            recent_rewards_trend = np.polyfit(range(len(self.recent_rewards)), list(self.recent_rewards), 1)[0]
            recent_sharpe_trend = np.polyfit(range(len(self.recent_sharpe)), list(self.recent_sharpe), 1)[0]
            
            self.current_stats['avg_reward'] = np.mean(list(self.recent_rewards)[-20:])
            self.current_stats['estimated_sharpe'] = np.mean(list(self.recent_sharpe)[-10:])
            
            # Rewards subindo mas Sharpe não
            if (recent_rewards_trend > 0.01 and 
                abs(recent_sharpe_trend) < self.alert_thresholds['performance_stagnation']):
                alerts.add('REWARD_INFLATION')
        
        # 3. DIVERGÊNCIA REWARDS vs PERFORMANCE REAL  
        if (len(self.recent_rewards) >= 30 and 
            len(self.recent_portfolio_values) >= 30):
            
            # Correlação entre rewards e performance real
            rewards_array = np.array(list(self.recent_rewards)[-30:])
            portfolio_array = np.array(list(self.recent_portfolio_values)[-30:])
            
            if len(rewards_array) == len(portfolio_array) and np.std(rewards_array) > 0 and np.std(portfolio_array) > 0:
                correlation = np.corrcoef(rewards_array, portfolio_array)[0, 1]
                
                if abs(correlation) < self.alert_thresholds['reward_sharpe_divergence']:
                    alerts.add('REWARD_PERFORMANCE_DIVERGENCE')
        
        # 4. PORTFOLIO GROWTH STAGNATION
        if len(self.recent_portfolio_values) >= 50:
            portfolio_growth = (list(self.recent_portfolio_values)[-1] / list(self.recent_portfolio_values)[0] - 1) * 100
            self.current_stats['portfolio_growth'] = portfolio_growth
            
            # Se rewards altos mas portfolio não cresce
            if (self.current_stats['avg_reward'] > 0.5 and 
                abs(portfolio_growth) < 1.0):  # Menos de 1% crescimento
                alerts.add('PORTFOLIO_STAGNATION')
        
        return alerts
    
    def print_status(self):
        """Imprime status atual do monitoramento"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n🕵️ MONITOR REWARD HACKING - {timestamp}")
        print("="*60)
        print(f"📊 Steps: {self.current_stats['total_steps']:,}")
        print(f"🎯 Avg Reward: {self.current_stats['avg_reward']:.4f}")
        print(f"✂️ Avg Clip Fraction: {self.current_stats['avg_clip_fraction']:.3f}")
        print(f"📈 Estimated Sharpe: {self.current_stats['estimated_sharpe']:.3f}")
        print(f"💰 Portfolio Growth: {self.current_stats['portfolio_growth']:.2f}%")
        
        # Status dos buffers
        print(f"\n📊 Buffer Status:")
        print(f"   Rewards: {len(self.recent_rewards)}/100")
        print(f"   Sharpe: {len(self.recent_sharpe)}/50") 
        print(f"   Clip Fraction: {len(self.recent_clip_fraction)}/50")
        print(f"   Portfolio: {len(self.recent_portfolio_values)}/100")
    
    def print_alerts(self, alerts):
        """Imprime alertas ativos"""
        if not alerts:
            print("✅ NENHUM SINAL DE REWARD HACKING DETECTADO")
            return
            
        print(f"\n🚨 ALERTAS ATIVOS ({len(alerts)}):")
        print("-"*40)
        
        alert_descriptions = {
            'HIGH_CLIP_FRACTION': f'Clip Fraction muito alto ({self.current_stats["avg_clip_fraction"]:.3f} > {self.alert_thresholds["clip_fraction_high"]})',
            'REWARD_INFLATION': 'Rewards crescendo sem melhoria real de performance',
            'REWARD_PERFORMANCE_DIVERGENCE': 'Rewards não correlacionados com performance real',
            'PORTFOLIO_STAGNATION': 'Portfolio estagnado apesar de rewards altos'
        }
        
        for alert in alerts:
            if alert in alert_descriptions:
                print(f"⚠️ {alert_descriptions[alert]}")
            else:
                print(f"⚠️ {alert}")
        
        # Recomendações
        print(f"\n💡 RECOMENDAÇÕES:")
        if 'HIGH_CLIP_FRACTION' in alerts:
            print("   • Considere reduzir learning rate")
            print("   • Monitore se performance real melhora")
        if 'REWARD_INFLATION' in alerts:
            print("   • POSSÍVEL REWARD HACKING - investigar reward function")
            print("   • Verificar se agent encontrou exploit")
        if 'REWARD_PERFORMANCE_DIVERGENCE' in alerts:
            print("   • Rewards podem não refletir performance real")
            print("   • Revisar design do reward system")
    
    def monitor_loop(self):
        """Loop principal de monitoramento"""
        print("🚀 Iniciando monitor de reward hacking...")
        print(f"📂 Diretório: {self.jsonl_dir}")
        print(f"⏱️ Intervalo: {self.check_interval}s")
        
        while self.is_monitoring:
            try:
                # Buscar arquivos JSONL mais recentes
                latest_files = self.get_latest_jsonl_files()
                
                if not latest_files:
                    print("⏳ Aguardando arquivos JSONL...")
                    time.sleep(self.check_interval)
                    continue
                
                # Analisar dados de cada tipo
                if 'training' in latest_files:
                    training_data = self.read_recent_jsonl_data(latest_files['training'])
                    self.analyze_training_data(training_data)
                
                if 'rewards' in latest_files:
                    rewards_data = self.read_recent_jsonl_data(latest_files['rewards'])
                    self.analyze_rewards_data(rewards_data)
                
                if 'performance' in latest_files:
                    performance_data = self.read_recent_jsonl_data(latest_files['performance'])
                    self.analyze_performance_data(performance_data)
                
                # Detectar reward hacking
                alerts = self.detect_reward_hacking()
                
                # Atualizar timestamp
                self.current_stats['last_update'] = datetime.now()
                
                # Imprimir status e alertas
                os.system('cls' if os.name == 'nt' else 'clear')  # Limpar tela
                self.print_status()
                self.print_alerts(alerts)
                
                # Atualizar alertas ativos
                self.active_alerts = alerts
                
            except Exception as e:
                print(f"❌ Erro no monitor: {e}")
            
            time.sleep(self.check_interval)
    
    def start_monitoring(self):
        """Inicia monitoramento em thread separada"""
        self.is_monitoring = True
        monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def stop_monitoring(self):
        """Para o monitoramento"""
        self.is_monitoring = False

def main():
    """Função principal"""
    print("🕵️ MONITOR DE REWARD HACKING V5")
    print("="*50)
    
    # Configurações via argumentos ou padrão
    jsonl_dir = "avaliacoes"
    check_interval = 15  # Segundos
    
    if len(sys.argv) > 1:
        check_interval = int(sys.argv[1])
    if len(sys.argv) > 2:
        jsonl_dir = sys.argv[2]
    
    # Criar e iniciar monitor
    monitor = RewardHackingMonitor(jsonl_dir=jsonl_dir, check_interval=check_interval)
    
    try:
        monitor_thread = monitor.start_monitoring()
        
        # Manter script rodando
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Parando monitor...")
        monitor.stop_monitoring()
        print("✅ Monitor finalizado!")

if __name__ == "__main__":
    main()
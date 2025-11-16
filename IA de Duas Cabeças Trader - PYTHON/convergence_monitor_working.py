#!/usr/bin/env python3
"""
CONVERGENCE MONITOR: Monitor visual de convergencia em tempo real
"""

import time
import os
import json
from datetime import datetime
import sys

class ConvergenceMonitor:
    """Monitor de convergencia simples que funciona no terminal"""
    
    def __init__(self, update_interval=30):
        self.update_interval = update_interval
        self.last_update = 0
        self.data_file = "convergence_data.json"
        
    def collect_data(self):
        """Coleta dados de convergencia dos arquivos de debug"""
        current_data = {
            'timestamp': time.time(),
            'step': 0,
            'gradient_zeros': 0.0,
            'alert_count': 0
        }
        
        # Ler ultimo debug zeros report
        try:
            debug_files = [f for f in os.listdir('.') if f.startswith('debug_zeros_report_step_') and f.endswith('.txt')]
            if debug_files:
                # Pegar o arquivo mais recente
                latest_debug = sorted(debug_files, key=lambda x: int(x.split('_')[4].split('.')[0]))[-1]
                
                with open(latest_debug, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extrair dados do report
                for line in content.split('\n'):
                    if 'Recent avg zeros:' in line:
                        try:
                            zeros_pct = float(line.split('Recent avg zeros: ')[1].split('%')[0])
                            current_data['gradient_zeros'] = zeros_pct
                        except:
                            pass
                    if 'Alert count:' in line:
                        try:
                            alert_count = int(line.split('Alert count: ')[1])
                            current_data['alert_count'] = alert_count
                        except:
                            pass
                            
                # Extrair step do filename
                step = int(latest_debug.split('_')[4].split('.')[0])
                current_data['step'] = step
                
        except Exception as e:
            print(f"Error reading debug files: {e}")
            
        return current_data
        
    def save_data(self, data):
        """Salvar dados para historico"""
        history = []
        
        # Ler historico existente
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    history = json.load(f)
            except:
                history = []
                
        # Adicionar novo ponto
        history.append(data)
        
        # Manter apenas ultimos 100 pontos
        if len(history) > 100:
            history = history[-100:]
            
        # Salvar
        with open(self.data_file, 'w') as f:
            json.dump(history, f, indent=2)
            
        return history
        
    def print_convergence_status(self, data, history):
        """Print status detalhado de convergencia"""
        
        # Clear screen (funciona no Windows)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 70)
        print("         CONVERGENCE MONITOR - Real Time Status")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current Step: {data['step']:,}")
        print("=" * 70)
        
        # Status atual
        gradient_zeros = data['gradient_zeros']
        alert_count = data['alert_count']
        
        print("CURRENT STATUS:")
        print("-" * 30)
        
        # Gradient Health
        if gradient_zeros < 2.0:
            status = "EXCELLENT"
            icon = "[+++]"
        elif gradient_zeros < 5.0:
            status = "HEALTHY"
            icon = "[++]"
        elif gradient_zeros < 10.0:
            status = "WARNING"  
            icon = "[+]"
        else:
            status = "CRITICAL"
            icon = "[!!!]"
            
        print(f"Gradient Zeros: {gradient_zeros:.2f}% {icon} {status}")
        
        # Alert status
        if alert_count == 0:
            alert_status = "NO ALERTS"
            alert_icon = "[OK]"
        else:
            alert_status = f"{alert_count} ACTIVE ALERTS"
            alert_icon = "[!]"
            
        print(f"Alert Count: {alert_count} {alert_icon} {alert_status}")
        
        print("=" * 70)
        
        # Historico recente (ultimos 10 pontos)
        if len(history) > 1:
            print("RECENT HISTORY (Last 10 measurements):")
            print("-" * 50)
            print("Step      | Gradient Zeros | Alerts | Trend")
            print("-" * 50)
            
            recent = history[-10:] if len(history) >= 10 else history
            
            for i, point in enumerate(recent):
                step = point['step']
                zeros = point['gradient_zeros']
                alerts = point['alert_count']
                
                # Calcular trend
                if i > 0:
                    prev_zeros = recent[i-1]['gradient_zeros']
                    if zeros < prev_zeros:
                        trend = "IMPROVING"
                    elif zeros > prev_zeros:
                        trend = "DEGRADING"
                    else:
                        trend = "STABLE"
                else:
                    trend = "BASELINE"
                    
                print(f"{step:8,} | {zeros:12.2f}% | {alerts:6} | {trend}")
                
        print("=" * 70)
        
        # Estatisticas
        if len(history) > 5:
            recent_zeros = [h['gradient_zeros'] for h in history[-10:]]
            avg_zeros = sum(recent_zeros) / len(recent_zeros)
            min_zeros = min(recent_zeros)
            max_zeros = max(recent_zeros)
            
            print("STATISTICS (Last 10 measurements):")
            print("-" * 40)
            print(f"Average Gradient Zeros: {avg_zeros:.2f}%")
            print(f"Best (Minimum): {min_zeros:.2f}%")
            print(f"Worst (Maximum): {max_zeros:.2f}%")
            
            # Trend analysis
            if len(recent_zeros) >= 5:
                first_half = sum(recent_zeros[:len(recent_zeros)//2]) / (len(recent_zeros)//2)
                second_half = sum(recent_zeros[len(recent_zeros)//2:]) / (len(recent_zeros) - len(recent_zeros)//2)
                
                if second_half < first_half:
                    trend = "IMPROVING TREND"
                elif second_half > first_half:
                    trend = "DEGRADING TREND" 
                else:
                    trend = "STABLE TREND"
                    
                print(f"Overall Trend: {trend}")
            
        print("=" * 70)
        print(f"Next update in: {self.update_interval} seconds")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 70)
        
    def run(self):
        """Executar monitoramento continuo"""
        print("CONVERGENCE MONITOR STARTED")
        print("Monitoring gradient convergence every 30 seconds...")
        print("Press Ctrl+C to stop")
        print("")
        
        try:
            while True:
                current_time = time.time()
                
                if current_time - self.last_update >= self.update_interval:
                    # Coletar dados
                    data = self.collect_data()
                    
                    # Salvar historico
                    history = self.save_data(data)
                    
                    # Mostrar status
                    self.print_convergence_status(data, history)
                    
                    self.last_update = current_time
                    
                time.sleep(1)  # Check every second
                
        except KeyboardInterrupt:
            print("\n")
            print("CONVERGENCE MONITOR STOPPED")
            print("Final data saved to:", self.data_file)

if __name__ == "__main__":
    monitor = ConvergenceMonitor(update_interval=30)
    monitor.run()
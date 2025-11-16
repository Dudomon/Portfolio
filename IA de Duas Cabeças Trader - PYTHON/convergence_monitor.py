#!/usr/bin/env python3
"""
🎯 CONVERGENCE MONITOR: Monitor visual de convergência em tempo real
"""

import time
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
import numpy as np

class ConvergenceMonitor:
    """Monitor de convergência em tempo real para transformer + daytrader"""
    
    def __init__(self, update_interval=30):
        self.update_interval = update_interval  # segundos
        self.last_update = 0
        self.data_file = "convergence_data.json"
        
        # Configurar matplotlib para não bloquear
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('🎯 CONVERGENCE MONITORING - Real Time', fontsize=16)
        
        # Configurar subplots
        self.setup_plots()
        
    def setup_plots(self):
        """Configurar os 6 gráficos de monitoramento"""
        
        # Plot 1: Gradient Zeros (mais importante)
        self.axes[0,0].set_title('🔥 Gradient Zeros %')
        self.axes[0,0].set_xlabel('Steps')
        self.axes[0,0].set_ylabel('Zero %')
        self.axes[0,0].axhline(y=5.0, color='r', linestyle='--', alpha=0.7, label='Alert Threshold')
        self.axes[0,0].legend()
        
        # Plot 2: Learnable Pooling Evolution  
        self.axes[0,1].set_title('🧠 Learnable Pooling Weights')
        self.axes[0,1].set_xlabel('Steps')
        self.axes[0,1].set_ylabel('Recent Bias %')
        
        # Plot 3: Projection Saturation
        self.axes[0,2].set_title('⚡ Projection Health')
        self.axes[0,2].set_xlabel('Steps') 
        self.axes[0,2].set_ylabel('Saturation %')
        self.axes[0,2].axhline(y=10.0, color='r', linestyle='--', alpha=0.7, label='Problem Threshold')
        self.axes[0,2].legend()
        
        # Plot 4: Action Distribution (Short bias)
        self.axes[1,0].set_title('📊 Action Distribution')
        self.axes[1,0].set_xlabel('Steps')
        self.axes[1,0].set_ylabel('Short %')
        
        # Plot 5: Performance Metrics
        self.axes[1,1].set_title('🚀 Performance')
        self.axes[1,1].set_xlabel('Steps')
        self.axes[1,1].set_ylabel('Processing Time (ms)')
        
        # Plot 6: Gradient Balance
        self.axes[1,2].set_title('⚖️ Gradient Balance')
        self.axes[1,2].set_xlabel('Steps')
        self.axes[1,2].set_ylabel('Norm Ratio')
        
        plt.tight_layout()
        
    def collect_data(self):
        """Coleta dados de convergência dos arquivos de debug"""
        current_data = {
            'timestamp': time.time(),
            'step': 0,
            'gradient_zeros': 0.0,
            'pooling_recent_bias': 0.0,
            'projection_saturation': 0.0,
            'short_percentage': 0.0,
            'action_time_ms': 0.0,
            'gradient_balance': 1.0
        }
        
        # 1. Ler último debug zeros report
        try:
            debug_files = [f for f in os.listdir('.') if f.startswith('debug_zeros_report_step_') and f.endswith('.txt')]
            if debug_files:
                latest_debug = sorted(debug_files, key=lambda x: int(x.split('_')[4].split('.')[0]))[-1]
                
                with open(latest_debug, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extrair dados do report
                if 'Recent avg zeros:' in content:
                    line = [l for l in content.split('\n') if 'Recent avg zeros:' in l][0]
                    zeros_pct = float(line.split('Recent avg zeros: ')[1].split('%')[0])
                    current_data['gradient_zeros'] = zeros_pct
                    
                # Extrair step do filename
                step = int(latest_debug.split('_')[4].split('.')[0])
                current_data['step'] = step
                
        except Exception as e:
            print(f"❌ Error reading debug files: {e}")
            
        # 2. Tentar ler dados de convergência dos objetos (se disponível)
        # Isso seria mais complexo, por agora usar dados simulados para demonstração
        
        return current_data
        
    def save_data(self, data):
        """Salvar dados para histórico"""
        history = []
        
        # Ler histórico existente
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    history = json.load(f)
            except:
                history = []
                
        # Adicionar novo ponto
        history.append(data)
        
        # Manter apenas últimos 1000 pontos
        if len(history) > 1000:
            history = history[-1000:]
            
        # Salvar
        with open(self.data_file, 'w') as f:
            json.dump(history, f)
            
        return history
        
    def update_plots(self, history):
        """Atualizar gráficos com dados históricos"""
        if len(history) < 2:
            return
            
        # Extrair dados para plotting
        steps = [d['step'] for d in history]
        gradient_zeros = [d['gradient_zeros'] for d in history]
        pooling_bias = [d['pooling_recent_bias'] for d in history]
        projection_sat = [d['projection_saturation'] for d in history]
        short_pct = [d['short_percentage'] for d in history]
        action_times = [d['action_time_ms'] for d in history]
        grad_balance = [d['gradient_balance'] for d in history]
        
        # Limpar plots
        for ax in self.axes.flat:
            ax.clear()
            
        # Reconfigurar
        self.setup_plots()
        
        # Plot 1: Gradient Zeros (CRÍTICO)
        self.axes[0,0].plot(steps, gradient_zeros, 'b-', linewidth=2, label='Gradient Zeros')
        self.axes[0,0].axhline(y=5.0, color='r', linestyle='--', alpha=0.7, label='Alert (5%)')
        if gradient_zeros:
            latest = gradient_zeros[-1]
            color = 'green' if latest < 5.0 else 'red'
            self.axes[0,0].text(0.02, 0.98, f'Latest: {latest:.1f}%', 
                               transform=self.axes[0,0].transAxes, 
                               verticalalignment='top', 
                               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        self.axes[0,0].legend()
        
        # Plot 2: Learnable Pooling
        self.axes[0,1].plot(steps, pooling_bias, 'g-', linewidth=2, label='Recent Bias')
        self.axes[0,1].axhline(y=33.3, color='orange', linestyle='--', alpha=0.7, label='Target (33%)')
        if pooling_bias:
            latest = pooling_bias[-1]
            self.axes[0,1].text(0.02, 0.98, f'Latest: {latest:.1f}%', 
                               transform=self.axes[0,1].transAxes, 
                               verticalalignment='top')
        self.axes[0,1].legend()
        
        # Plot 3: Projection Saturation
        self.axes[0,2].plot(steps, projection_sat, 'orange', linewidth=2, label='Saturation')
        self.axes[0,2].axhline(y=10.0, color='r', linestyle='--', alpha=0.7, label='Problem (10%)')
        if projection_sat:
            latest = projection_sat[-1]
            color = 'green' if latest < 10.0 else 'red'
            self.axes[0,2].text(0.02, 0.98, f'Latest: {latest:.1f}%', 
                               transform=self.axes[0,2].transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        self.axes[0,2].legend()
        
        # Plot 4: Short Percentage  
        self.axes[1,0].plot(steps, short_pct, 'purple', linewidth=2, label='Short %')
        self.axes[1,0].axhline(y=15.0, color='g', linestyle='--', alpha=0.7, label='Target (15%)')
        if short_pct:
            latest = short_pct[-1]
            self.axes[1,0].text(0.02, 0.98, f'Latest: {latest:.1f}%', 
                               transform=self.axes[1,0].transAxes, 
                               verticalalignment='top')
        self.axes[1,0].legend()
        
        # Plot 5: Performance
        if action_times and any(t > 0 for t in action_times):
            self.axes[1,1].plot(steps, action_times, 'red', linewidth=2, label='Action Time')
            if action_times:
                latest = action_times[-1]
                self.axes[1,1].text(0.02, 0.98, f'Latest: {latest:.1f}ms', 
                                   transform=self.axes[1,1].transAxes, 
                                   verticalalignment='top')
        self.axes[1,1].legend()
        
        # Plot 6: Gradient Balance
        self.axes[1,2].plot(steps, grad_balance, 'cyan', linewidth=2, label='Market/Position')
        self.axes[1,2].axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Balanced')
        if grad_balance:
            latest = grad_balance[-1]
            self.axes[1,2].text(0.02, 0.98, f'Latest: {latest:.2f}', 
                               transform=self.axes[1,2].transAxes, 
                               verticalalignment='top')
        self.axes[1,2].legend()
        
        # Refresh display
        plt.draw()
        plt.pause(0.01)
        
    def print_status(self, data):
        """Print status summary"""
        print(f"\n🎯 CONVERGENCE STATUS - Step {data['step']} - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 70)
        
        # Gradient Health
        gradient_status = "✅ HEALTHY" if data['gradient_zeros'] < 5.0 else "🚨 CRITICAL"
        print(f"🔥 Gradient Zeros: {data['gradient_zeros']:.1f}% - {gradient_status}")
        
        # Pooling Evolution  
        pooling_status = "🧠 LEARNING" if data['pooling_recent_bias'] > 20 else "⏳ EVOLVING"
        print(f"🧠 Pooling Recent Bias: {data['pooling_recent_bias']:.1f}% - {pooling_status}")
        
        # Projection Health
        proj_status = "✅ HEALTHY" if data['projection_saturation'] < 10.0 else "⚠️ SATURATED"
        print(f"⚡ Projection Saturation: {data['projection_saturation']:.1f}% - {proj_status}")
        
        # Action Distribution
        short_status = "✅ BALANCED" if data['short_percentage'] > 10 else "⚠️ SHORT BIAS LOW"
        print(f"📊 Short Actions: {data['short_percentage']:.1f}% - {short_status}")
        
        print("=" * 70)
        
    def run(self):
        """Executar monitoramento contínuo"""
        print("🎯 CONVERGENCE MONITOR STARTED")
        print("📊 Updating plots every 30 seconds...")
        print("❌ Press Ctrl+C to stop")
        
        try:
            while True:
                current_time = time.time()
                
                if current_time - self.last_update >= self.update_interval:
                    # Coletar dados
                    data = self.collect_data()
                    
                    # Salvar histórico
                    history = self.save_data(data)
                    
                    # Atualizar plots
                    self.update_plots(history)
                    
                    # Print status
                    self.print_status(data)
                    
                    self.last_update = current_time
                    
                time.sleep(1)  # Check every second
                
        except KeyboardInterrupt:
            print("\n🛑 CONVERGENCE MONITOR STOPPED")
            plt.ioff()
            plt.show()  # Keep final plot open

if __name__ == "__main__":
    monitor = ConvergenceMonitor(update_interval=30)
    monitor.run()
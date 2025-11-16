#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 MONITOR DE CONVERGÊNCIA E GRADIENTES EM TEMPO REAL

Dashboard interativo para monitoramento em tempo real usando dados JSON streaming.
Substitui necessidade de ler CSVs grandes, permitindo análise instantânea.

Features:
- Dashboard web interativo
- Gráficos em tempo real de gradientes
- Análise de convergência automática
- Alertas visuais para problemas
- Estatísticas detalhadas por componente
"""

import json
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly não disponível - usando matplotlib")

from real_time_logger import LogReader


class RealTimeMonitor:
    """
    📊 MONITOR EM TEMPO REAL PARA TREINAMENTO DE RL
    
    Monitora convergência e gradientes usando dados JSON streaming,
    permitindo análise instantânea sem overhead de CSV parsing.
    """
    
    def __init__(self, 
                 log_path: str = "D:/Projeto/avaliacoes",
                 refresh_interval: float = 2.0,
                 history_window: int = 1000):
        
        self.log_path = Path(log_path)
        self.refresh_interval = refresh_interval
        self.history_window = history_window
        
        # Reader para logs JSON
        self.log_reader = LogReader(str(self.log_path))
        
        # Dados em memória para plots
        self.data_cache = {
            'training': [],
            'gradients': [],
            'convergence': [],
            'rewards': [],
            'performance': []
        }
        
        # Estado do monitor
        self.running = False
        self.monitor_thread = None
        self.session_id = None
        
        # Alertas ativos
        self.active_alerts = []
        
        # Configurações de plot
        self.plot_config = {
            'gradient_threshold_high': 5.0,
            'gradient_threshold_low': 1e-6,
            'convergence_window': 50,
            'alert_retention_minutes': 30
        }
        
        print(f"📊 RealTimeMonitor inicializado")
        print(f"📁 Monitorando: {self.log_path}")
    
    def start_monitoring(self, session_id: Optional[str] = None):
        """Inicia monitoramento em tempo real"""
        if self.running:
            print("⚠️ Monitor já está rodando")
            return
        
        if session_id is None:
            session_id = self.log_reader.get_latest_session()
            if session_id is None:
                print("❌ Nenhuma sessão ativa encontrada")
                return
        
        self.session_id = session_id
        self.running = True
        
        print(f"🚀 Iniciando monitoramento - Sessão: {session_id}")
        
        # Thread para atualizações contínuas
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Iniciar dashboard se plotly disponível
        if PLOTLY_AVAILABLE:
            self._create_dashboard()
        else:
            self._create_matplotlib_dashboard()
    
    def stop_monitoring(self):
        """Para monitoramento"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        print("🛑 Monitoramento parado")
    
    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.running:
            try:
                self._update_data_cache()
                self._analyze_current_state()
                time.sleep(self.refresh_interval)
            except Exception as e:
                print(f"⚠️ Erro no monitoramento: {e}")
                time.sleep(self.refresh_interval * 2)
    
    def _update_data_cache(self):
        """Atualiza cache com dados mais recentes"""
        categories = ['training', 'gradients', 'convergence', 'rewards', 'performance']
        
        for category in categories:
            try:
                # Buscar dados recentes
                recent_data = self.log_reader.tail_stream(
                    category, 
                    self.history_window, 
                    self.session_id
                )
                
                # Filtrar apenas dados relevantes (não headers/alerts)
                filtered_data = [
                    entry for entry in recent_data 
                    if entry.get('type', '').startswith(category.rstrip('s'))
                ]
                
                self.data_cache[category] = filtered_data
                
            except Exception as e:
                print(f"⚠️ Erro ao atualizar {category}: {e}")
    
    def _analyze_current_state(self):
        """Análise do estado atual do treinamento"""
        current_time = datetime.now()
        
        # Limpar alertas antigos
        self.active_alerts = [
            alert for alert in self.active_alerts
            if (current_time - datetime.fromisoformat(alert['timestamp'])).total_seconds() < 
               self.plot_config['alert_retention_minutes'] * 60
        ]
        
        # Análise de gradientes
        self._analyze_gradient_health()
        
        # Análise de convergência
        self._analyze_convergence_trends()
        
        # Análise de rewards
        self._analyze_reward_patterns()
        
        # Imprimir status
        if len(self.data_cache['training']) > 0:
            latest = self.data_cache['training'][-1]
            step = latest.get('step', 0)
            loss = latest.get('loss', 'N/A')
            print(f"📈 Step {step:6d} | Loss: {loss} | Alertas: {len(self.active_alerts)}")
    
    def _analyze_gradient_health(self):
        """Análise de saúde dos gradientes"""
        gradient_data = self.data_cache.get('gradients', [])
        if len(gradient_data) < 5:
            return
        
        recent_gradients = gradient_data[-10:]  # Últimos 10 pontos
        
        for entry in recent_gradients:
            grad_norm = entry.get('grad_norm', 0)
            zeros_ratio = entry.get('grad_zeros_ratio', 0)
            step = entry.get('step', 0)
            
            # Detectar problemas
            if grad_norm > self.plot_config['gradient_threshold_high']:
                self._add_alert(f"Gradient explosion (step {step}): {grad_norm:.4f}", "error")
            elif grad_norm < self.plot_config['gradient_threshold_low']:
                self._add_alert(f"Vanishing gradients (step {step}): {grad_norm:.8f}", "warning")
            
            if zeros_ratio > 0.7:
                self._add_alert(f"Muitos zeros nos gradientes (step {step}): {zeros_ratio*100:.1f}%", "warning")
    
    def _analyze_convergence_trends(self):
        """Análise de tendências de convergência"""
        training_data = self.data_cache.get('training', [])
        if len(training_data) < self.plot_config['convergence_window']:
            return
        
        recent_data = training_data[-self.plot_config['convergence_window']:]
        losses = [entry.get('loss', 0) for entry in recent_data if 'loss' in entry]
        
        if len(losses) < 10:
            return
        
        # Análise de tendência
        x = np.arange(len(losses))
        trend = np.polyfit(x, losses, 1)[0]
        
        if trend > 0.01:  # Loss aumentando significativamente
            latest_step = recent_data[-1].get('step', 0)
            self._add_alert(f"Loss divergindo (step {latest_step}): tendência +{trend:.6f}", "error")
        elif abs(trend) < 1e-8:  # Estagnação
            latest_step = recent_data[-1].get('step', 0)
            self._add_alert(f"Possível estagnação (step {latest_step}): tendência {trend:.8f}", "info")
    
    def _analyze_reward_patterns(self):
        """Análise de padrões de reward"""
        reward_data = self.data_cache.get('rewards', [])
        if len(reward_data) < 20:
            return
        
        recent_rewards = reward_data[-50:]  # Últimos 50 episódios
        rewards = [entry.get('episode_reward', 0) for entry in recent_rewards if 'episode_reward' in entry]
        
        if len(rewards) < 10:
            return
        
        # Detectar padrões problemáticos
        reward_std = np.std(rewards)
        reward_mean = np.mean(rewards)
        
        if reward_std < 1.0 and len(rewards) > 30:  # Rewards muito uniformes
            latest_step = recent_rewards[-1].get('step', 0)
            self._add_alert(f"Rewards muito uniformes (step {latest_step}): std={reward_std:.3f}", "info")
        
        if reward_mean < -500 and len(rewards) > 20:  # Performance muito ruim
            latest_step = recent_rewards[-1].get('step', 0)
            self._add_alert(f"Performance baixa (step {latest_step}): mean={reward_mean:.1f}", "warning")
    
    def _add_alert(self, message: str, level: str = "info"):
        """Adiciona novo alerta"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'level': level
        }
        
        # Evitar duplicatas recentes
        recent_messages = [a['message'] for a in self.active_alerts[-5:]]
        if message not in recent_messages:
            self.active_alerts.append(alert)
            
            # Print com cor baseado no nível
            colors = {'error': '🔴', 'warning': '🟡', 'info': '🔵'}
            icon = colors.get(level, '⚪')
            print(f"{icon} {message}")
    
    def _create_dashboard(self):
        """Cria dashboard interativo usando Plotly"""
        if not PLOTLY_AVAILABLE:
            return
        
        print("📊 Criando dashboard Plotly...")
        
        # Thread para atualização do dashboard
        dashboard_thread = threading.Thread(target=self._update_plotly_dashboard, daemon=True)
        dashboard_thread.start()
    
    def _update_plotly_dashboard(self):
        """Atualiza dashboard Plotly em tempo real"""
        dashboard_file = self.log_path / f"dashboard_{self.session_id}.html"
        
        while self.running:
            try:
                # Criar subplots
                fig = sp.make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Loss & Learning Rate', 'Gradient Health', 
                                   'Reward Trends', 'Training Metrics'),
                    specs=[[{"secondary_y": True}, {"secondary_y": True}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Plot 1: Loss e Learning Rate
                self._add_loss_plot(fig, row=1, col=1)
                
                # Plot 2: Gradient Health
                self._add_gradient_plot(fig, row=1, col=2)
                
                # Plot 3: Reward Trends
                self._add_reward_plot(fig, row=2, col=1)
                
                # Plot 4: Training Metrics
                self._add_training_metrics_plot(fig, row=2, col=2)
                
                # Layout
                fig.update_layout(
                    title=f'🚀 Training Monitor - Session {self.session_id}',
                    height=800,
                    showlegend=True,
                    template='plotly_dark'
                )
                
                # Salvar dashboard
                plot(fig, filename=str(dashboard_file), auto_open=False)
                
                time.sleep(self.refresh_interval * 2)  # Dashboard atualiza mais devagar
                
            except Exception as e:
                print(f"⚠️ Erro no dashboard Plotly: {e}")
                time.sleep(10)
    
    def _add_loss_plot(self, fig, row: int, col: int):
        """Adiciona plot de loss"""
        training_data = self.data_cache.get('training', [])
        if not training_data:
            return
        
        steps = [entry.get('step', 0) for entry in training_data]
        losses = [entry.get('loss', 0) for entry in training_data if 'loss' in entry]
        lrs = [entry.get('learning_rate', 0) for entry in training_data if 'learning_rate' in entry]
        
        if losses:
            fig.add_trace(
                go.Scatter(x=steps[:len(losses)], y=losses, name='Loss', line=dict(color='red')),
                row=row, col=col
            )
        
        if lrs:
            fig.add_trace(
                go.Scatter(x=steps[:len(lrs)], y=lrs, name='Learning Rate', 
                          line=dict(color='blue'), yaxis='y2'),
                row=row, col=col, secondary_y=True
            )
    
    def _add_gradient_plot(self, fig, row: int, col: int):
        """Adiciona plot de gradientes"""
        gradient_data = self.data_cache.get('gradients', [])
        if not gradient_data:
            return
        
        steps = [entry.get('step', 0) for entry in gradient_data]
        grad_norms = [entry.get('grad_norm', 0) for entry in gradient_data]
        zeros_ratios = [entry.get('grad_zeros_ratio', 0) * 100 for entry in gradient_data]
        
        if grad_norms:
            fig.add_trace(
                go.Scatter(x=steps, y=grad_norms, name='Gradient Norm', line=dict(color='green')),
                row=row, col=col
            )
        
        if zeros_ratios:
            fig.add_trace(
                go.Scatter(x=steps, y=zeros_ratios, name='Zeros %', 
                          line=dict(color='orange'), yaxis='y2'),
                row=row, col=col, secondary_y=True
            )
    
    def _add_reward_plot(self, fig, row: int, col: int):
        """Adiciona plot de rewards"""
        reward_data = self.data_cache.get('rewards', [])
        if not reward_data:
            return
        
        steps = [entry.get('step', 0) for entry in reward_data]
        rewards = [entry.get('episode_reward', 0) for entry in reward_data if 'episode_reward' in entry]
        
        if rewards:
            # Moving average
            if len(rewards) > 10:
                ma_window = min(20, len(rewards) // 4)
                ma_rewards = pd.Series(rewards).rolling(ma_window).mean().tolist()
                fig.add_trace(
                    go.Scatter(x=steps[:len(ma_rewards)], y=ma_rewards, 
                              name=f'Reward MA({ma_window})', line=dict(color='purple')),
                    row=row, col=col
                )
            
            fig.add_trace(
                go.Scatter(x=steps[:len(rewards)], y=rewards, name='Episode Reward', 
                          mode='markers', marker=dict(color='cyan', size=3)),
                row=row, col=col
            )
    
    def _add_training_metrics_plot(self, fig, row: int, col: int):
        """Adiciona plot de métricas de treinamento"""
        training_data = self.data_cache.get('training', [])
        if not training_data:
            return
        
        steps = [entry.get('step', 0) for entry in training_data]
        
        # Diferentes métricas que podem estar disponíveis
        metrics_to_plot = ['entropy_loss', 'value_loss', 'policy_loss', 'explained_variance']
        colors = ['yellow', 'pink', 'lightblue', 'lightgreen']
        
        for i, metric in enumerate(metrics_to_plot):
            values = [entry.get(metric, 0) for entry in training_data if metric in entry]
            if values:
                fig.add_trace(
                    go.Scatter(x=steps[:len(values)], y=values, name=metric.title(), 
                              line=dict(color=colors[i % len(colors)])),
                    row=row, col=col
                )
    
    def _create_matplotlib_dashboard(self):
        """Cria dashboard usando matplotlib (fallback)"""
        print("📊 Criando dashboard matplotlib...")
        
        plt.ion()  # Interactive mode
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Monitor - Session {self.session_id}')
        
        # Thread para atualização
        def update_matplotlib():
            while self.running:
                try:
                    self._update_matplotlib_plots(axes)
                    plt.draw()
                    plt.pause(self.refresh_interval)
                except Exception as e:
                    print(f"⚠️ Erro no matplotlib: {e}")
                    time.sleep(5)
        
        matplotlib_thread = threading.Thread(target=update_matplotlib, daemon=True)
        matplotlib_thread.start()
        
        plt.show()
    
    def _update_matplotlib_plots(self, axes):
        """Atualiza plots matplotlib"""
        # Clear previous plots
        for ax_row in axes:
            for ax in ax_row:
                ax.clear()
        
        # Plot 1: Loss
        training_data = self.data_cache.get('training', [])
        if training_data:
            steps = [entry.get('step', 0) for entry in training_data]
            losses = [entry.get('loss', 0) for entry in training_data if 'loss' in entry]
            if losses:
                axes[0, 0].plot(steps[:len(losses)], losses, 'r-', label='Loss')
                axes[0, 0].set_title('Training Loss')
                axes[0, 0].legend()
        
        # Plot 2: Gradients
        gradient_data = self.data_cache.get('gradients', [])
        if gradient_data:
            steps = [entry.get('step', 0) for entry in gradient_data]
            grad_norms = [entry.get('grad_norm', 0) for entry in gradient_data]
            if grad_norms:
                axes[0, 1].semilogy(steps, grad_norms, 'g-', label='Grad Norm')
                axes[0, 1].set_title('Gradient Norms')
                axes[0, 1].legend()
        
        # Plot 3: Rewards
        reward_data = self.data_cache.get('rewards', [])
        if reward_data:
            steps = [entry.get('step', 0) for entry in reward_data]
            rewards = [entry.get('episode_reward', 0) for entry in reward_data if 'episode_reward' in entry]
            if rewards:
                axes[1, 0].plot(steps[:len(rewards)], rewards, 'b.', alpha=0.5, label='Rewards')
                # Moving average
                if len(rewards) > 10:
                    ma_rewards = pd.Series(rewards).rolling(10).mean()
                    axes[1, 0].plot(steps[:len(ma_rewards)], ma_rewards, 'b-', label='MA(10)')
                axes[1, 0].set_title('Episode Rewards')
                axes[1, 0].legend()
        
        # Plot 4: Alerts
        axes[1, 1].text(0.1, 0.9, f'Active Alerts: {len(self.active_alerts)}', 
                        transform=axes[1, 1].transAxes, fontsize=12)
        
        # Mostrar últimos alertas
        for i, alert in enumerate(self.active_alerts[-5:]):
            level_colors = {'error': 'red', 'warning': 'orange', 'info': 'blue'}
            color = level_colors.get(alert['level'], 'black')
            axes[1, 1].text(0.1, 0.8 - i*0.1, alert['message'][:50] + '...', 
                           transform=axes[1, 1].transAxes, fontsize=8, color=color)
        
        axes[1, 1].set_title('Recent Alerts')
        axes[1, 1].axis('off')
    
    def get_current_status(self) -> Dict[str, Any]:
        """Retorna status atual do monitoramento"""
        return {
            'session_id': self.session_id,
            'running': self.running,
            'data_points': {k: len(v) for k, v in self.data_cache.items()},
            'active_alerts': len(self.active_alerts),
            'latest_step': self.data_cache['training'][-1].get('step', 0) if self.data_cache['training'] else 0,
            'last_update': datetime.now().isoformat()
        }
    
    def export_analysis_report(self) -> str:
        """Exporta relatório de análise"""
        report_file = self.log_path / f"analysis_report_{self.session_id}.json"
        
        report = {
            'session_id': self.session_id,
            'generated_at': datetime.now().isoformat(),
            'status': self.get_current_status(),
            'alerts_summary': {
                'total_alerts': len(self.active_alerts),
                'by_level': {}
            },
            'training_summary': {},
            'gradient_summary': {},
            'convergence_analysis': {}
        }
        
        # Análise de alertas
        for alert in self.active_alerts:
            level = alert['level']
            if level not in report['alerts_summary']['by_level']:
                report['alerts_summary']['by_level'][level] = 0
            report['alerts_summary']['by_level'][level] += 1
        
        # Análise de treinamento
        training_data = self.data_cache.get('training', [])
        if training_data:
            losses = [e.get('loss', 0) for e in training_data if 'loss' in e]
            if losses:
                report['training_summary'] = {
                    'total_steps': len(training_data),
                    'loss_trend': float(np.polyfit(range(len(losses)), losses, 1)[0]) if len(losses) > 1 else 0,
                    'current_loss': losses[-1] if losses else 0,
                    'min_loss': min(losses) if losses else 0,
                    'max_loss': max(losses) if losses else 0
                }
        
        # Salvar relatório
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Relatório exportado: {report_file}")
        return str(report_file)


def create_monitor(**kwargs) -> RealTimeMonitor:
    """Factory function para criar monitor"""
    return RealTimeMonitor(**kwargs)


# Exemplo de uso
if __name__ == "__main__":
    print("🚀 Iniciando RealTimeMonitor...")
    
    monitor = create_monitor(refresh_interval=1.0)
    
    try:
        monitor.start_monitoring()
        
        # Manter rodando
        while True:
            status = monitor.get_current_status()
            print(f"📊 Status: {status['data_points']} | Alertas: {status['active_alerts']}")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n🛑 Parando monitor...")
        monitor.stop_monitoring()
        
        # Exportar relatório final
        report_file = monitor.export_analysis_report()
        print(f"📋 Relatório final: {report_file}")
    
    print("✅ Monitor finalizado!")
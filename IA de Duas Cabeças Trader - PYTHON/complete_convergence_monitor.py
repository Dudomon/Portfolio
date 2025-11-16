#!/usr/bin/env python3
"""
🔥 MONITOR ULTRA RIGOROSO DE CONVERGÊNCIA V7 TWOHEAD
Combina gradient health + métricas reais de convergência dos logs JSONL

V7 TWOHEAD ULTRA RIGOROSO (Anti-premature convergence):
- Loss trend: STABLE/IMPROVING (entropy loss alto -24,-100+ é NORMAL para TwoHead)
- Performance: 25%+ return, 60%+ win rate, 200+ trades (excepcional)  
- Trading Profitability: 10%+ lucro mínimo consistente
- Confirmações: 20 consecutivas (vs 3 original)
- Calibrado para arquiteturas TwoHead com entropy loss naturalmente alto
"""

import time
import json
import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional

class CompleteConvergenceMonitor:
    """
    Monitor completo que combina:
    - Gradient health (debug zeros reports)  
    - Convergência real (JSONL training data)
    - Performance de trading (JSONL rewards/performance)
    """
    
    def __init__(self, 
                 avaliacoes_path: str = "D:/Projeto/avaliacoes",
                 update_interval: float = 30.0,
                 history_size: int = 100):
        
        self.avaliacoes_path = Path(avaliacoes_path)
        self.update_interval = update_interval
        self.history_size = history_size
        
        # Buffers para dados históricos
        self.gradient_history = deque(maxlen=history_size)
        self.training_history = deque(maxlen=history_size) 
        self.rewards_history = deque(maxlen=history_size)
        self.convergence_analysis = {}
        
        # Sessão ativa
        self.current_session = None
        self.last_update = 0
        
        # Convergência tracking
        self.convergence_criteria_met = 0
        self.convergence_start_time = None
        self.converged = False
        self.convergence_step = None  # Step onde convergência foi detectada
        self.convergence_threshold = 20  # V7 ULTRA RIGOROSO: 20 confirmações consecutivas
        
        print(f"[FIRE] Complete Convergence Monitor iniciado")
        print(f"[PATH] Path: {self.avaliacoes_path}")
    
    def find_latest_session(self) -> Optional[str]:
        """Encontra a sessão mais recente"""
        sessions = set()
        
        if not self.avaliacoes_path.exists():
            return None
            
        for file in self.avaliacoes_path.glob("*.jsonl"):
            # Extract session_id: category_YYYYMMDD_HHMMSS.jsonl
            parts = file.stem.split('_')
            if len(parts) >= 3:
                session_id = '_'.join(parts[-2:])  # YYYYMMDD_HHMMSS
                sessions.add(session_id)
        
        return max(sessions) if sessions else None
    
    def read_jsonl_tail(self, category: str, session_id: str, n: int = 50) -> List[Dict]:
        """Lê últimas N entradas de um arquivo JSONL"""
        filename = f"{category}_{session_id}.jsonl"
        filepath = self.avaliacoes_path / filename
        
        if not filepath.exists():
            return []
        
        entries = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            if entry.get('type') != 'header':  # Skip header
                                entries.append(entry)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"⚠️ Erro lendo {filepath}: {e}")
            return []
        
        return entries[-n:] if entries else []
    
    def collect_gradient_data(self) -> Dict[str, Any]:
        """Coleta dados de gradient health dos debug reports"""
        gradient_data = {
            'timestamp': time.time(),
            'step': 0,
            'gradient_zeros': 0.0,
            'alert_count': 0,
            'status': 'UNKNOWN'
        }
        
        try:
            # Buscar último debug report
            debug_files = [f for f in os.listdir('.') if f.startswith('debug_zeros_report_step_') and f.endswith('.txt')]
            if debug_files:
                latest_debug = sorted(debug_files, key=lambda x: int(x.split('_')[4].split('.')[0]))[-1]
                
                with open(latest_debug, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for line in content.split('\n'):
                    if 'Recent avg zeros:' in line:
                        try:
                            zeros_pct = float(line.split('Recent avg zeros: ')[1].split('%')[0])
                            gradient_data['gradient_zeros'] = zeros_pct
                        except:
                            pass
                    if 'Alert count:' in line:
                        try:
                            alert_count = int(line.split('Alert count: ')[1])
                            gradient_data['alert_count'] = alert_count
                        except:
                            pass
                
                # Extract step from filename
                step = int(latest_debug.split('_')[4].split('.')[0])
                gradient_data['step'] = step
                
                # Determine status (ajustado para LSTM - thresholds maiores)
                zeros = gradient_data['gradient_zeros']
                if zeros < 5.0:
                    gradient_data['status'] = 'EXCELLENT'
                elif zeros < 15.0:
                    gradient_data['status'] = 'HEALTHY'
                elif zeros < 25.0:
                    gradient_data['status'] = 'WARNING'
                else:
                    gradient_data['status'] = 'CRITICAL'
                    
        except Exception as e:
            print(f"⚠️ Erro coletando gradient data: {e}")
            
        return gradient_data
    
    def collect_training_data(self, session_id: str) -> Dict[str, Any]:
        """Coleta dados de treinamento do JSONL"""
        training_entries = self.read_jsonl_tail('training', session_id, 50)
        
        if not training_entries:
            return {'status': 'NO_DATA'}
        
        # Extrair métricas dos últimos entries
        recent_losses = []
        recent_rewards = []
        recent_lrs = []
        
        for entry in training_entries:
            if 'loss' in entry:
                recent_losses.append(entry['loss'])
            if 'episode_reward' in entry:
                recent_rewards.append(entry['episode_reward'])
            if 'learning_rate' in entry:
                recent_lrs.append(entry['learning_rate'])
        
        # Análise de tendências
        loss_trend = 'STABLE'
        if len(recent_losses) >= 10:
            # Linear regression nos últimos 10 pontos
            x = np.arange(len(recent_losses[-10:]))
            y = np.array(recent_losses[-10:])
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                if slope < -0.01:
                    loss_trend = 'IMPROVING'
                elif slope > 0.01:
                    loss_trend = 'DEGRADING'
        
        reward_trend = 'STABLE'  
        if len(recent_rewards) >= 10:
            x = np.arange(len(recent_rewards[-10:]))
            y = np.array(recent_rewards[-10:])
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                if slope > 1.0:
                    reward_trend = 'IMPROVING'
                elif slope < -1.0:
                    reward_trend = 'DEGRADING'
        
        return {
            'status': 'ACTIVE',
            'latest_entry': training_entries[-1],
            'current_loss': recent_losses[-1] if recent_losses else 0.0,
            'loss_mean': np.mean(recent_losses) if recent_losses else 0.0,
            'loss_std': np.std(recent_losses) if recent_losses else 0.0,
            'loss_trend': loss_trend,
            'current_reward': recent_rewards[-1] if recent_rewards else 0.0,
            'reward_mean': np.mean(recent_rewards) if recent_rewards else 0.0,
            'reward_std': np.std(recent_rewards) if recent_rewards else 0.0,
            'reward_trend': reward_trend,
            'current_lr': recent_lrs[-1] if recent_lrs else 0.0,
            'total_entries': len(training_entries)
        }
    
    def collect_performance_data(self, session_id: str) -> Dict[str, Any]:
        """Coleta dados de performance/rewards do JSONL - ANÁLISE MULTI-EPISÓDIOS"""
        performance_entries = self.read_jsonl_tail('performance', session_id, 200)  # Mais dados para análise
        
        if not performance_entries:
            return {'status': 'NO_DATA'}
        
        data = {'status': 'ACTIVE'}
        
        # ANÁLISE MULTI-EPISÓDIOS: Agrupar por episódios (2k steps cada)
        episodes_data = defaultdict(list)
        
        for entry in performance_entries:
            step = entry.get('step', 0)
            episode_id = step // 2000  # Episódios de 2000 steps
            episodes_data[episode_id].append(entry)
        
        # Analisar últimos 10-20 episódios
        recent_episodes = list(episodes_data.keys())[-20:] if len(episodes_data) > 20 else list(episodes_data.keys())
        
        episode_returns = []
        episode_win_rates = []
        episode_trade_counts = []
        episode_drawdowns = []
        episode_sharpe_ratios = []
        
        for episode_id in recent_episodes:
            episode_entries = episodes_data[episode_id]
            
            # Métricas do episódio (usar última entrada do episódio)
            if episode_entries:
                final_entry = episode_entries[-1]
                
                if 'portfolio_value' in final_entry:
                    initial_value = episode_entries[0].get('portfolio_value', 500)
                    final_value = final_entry['portfolio_value']
                    episode_return = (final_value - initial_value) / initial_value
                    episode_returns.append(episode_return * 100)
                
                if 'win_rate' in final_entry:
                    episode_win_rates.append(final_entry['win_rate'])
                
                if 'trades_count' in final_entry:
                    episode_trade_counts.append(final_entry['trades_count'])
                
                if 'drawdown' in final_entry:
                    episode_drawdowns.append(final_entry['drawdown'])
                
                if 'sharpe_ratio' in final_entry:
                    episode_sharpe_ratios.append(final_entry['sharpe_ratio'])
        
        # ESTATÍSTICAS MULTI-EPISÓDIOS
        if episode_returns:
            data.update({
                'episodes_analyzed': len(episode_returns),
                'avg_episode_return': np.mean(episode_returns),
                'best_episode_return': np.max(episode_returns),
                'worst_episode_return': np.min(episode_returns),
                'return_std': np.std(episode_returns),
                'positive_episodes_pct': (np.array(episode_returns) > 0).mean() * 100,
                'recent_episodes_trend': 'IMPROVING' if len(episode_returns) >= 5 and np.mean(episode_returns[-5:]) > np.mean(episode_returns[:5]) else 'STABLE'
            })
        
        if episode_win_rates:
            data.update({
                'avg_win_rate': np.mean(episode_win_rates),
                'best_win_rate': np.max(episode_win_rates),
                'worst_win_rate': np.min(episode_win_rates),
                'current_win_rate': episode_win_rates[-1] if episode_win_rates else 0
            })
        
        if episode_trade_counts:
            data.update({
                'avg_trades_per_episode': np.mean(episode_trade_counts),
                'total_trades_analyzed': np.sum(episode_trade_counts)
            })
        
        if episode_drawdowns:
            data.update({
                'avg_drawdown': np.mean(episode_drawdowns),
                'max_drawdown': np.max(episode_drawdowns),
                'current_drawdown': episode_drawdowns[-1] if episode_drawdowns else 0
            })
        
        if episode_sharpe_ratios:
            data.update({
                'avg_sharpe_ratio': np.mean(episode_sharpe_ratios),
                'best_sharpe_ratio': np.max(episode_sharpe_ratios)
            })
        
        # Adicionar métricas de retorno total para compatibilidade
        if episode_returns:
            data.update({
                'total_return_pct': data.get('avg_episode_return', 0),  # Usar média dos episódios como retorno total
                'total_trades': data.get('total_trades_analyzed', 0)     # Compatibilidade
            })
        else:
            data.update({
                'total_return_pct': 0,
                'total_trades': 0
            })
        
        data['performance_entries'] = len(performance_entries)
        
        return data
    
    def check_convergence_criteria(self, gradient_data: Dict, training_data: Dict, performance_data: Dict) -> bool:
        """Verifica se os critérios de convergência foram atendidos"""
        criteria_met = []
        
        # 1. Gradient Health: Deve estar healthy ou excellent
        grad_status = gradient_data.get('status', 'UNKNOWN')
        gradient_healthy = grad_status in ['EXCELLENT', 'HEALTHY']
        criteria_met.append(('Gradient Health OK', gradient_healthy))
        
        # 2. Loss Stability: Loss deve estar stable ou improving E em range válido para V7
        if training_data.get('status') == 'ACTIVE':
            loss_trend = training_data.get('loss_trend', 'UNKNOWN')
            current_loss = training_data.get('current_loss', float('inf'))
            # V7 TWOHEAD: Loss pode ser muito negativo (normal), foco na tendência
            # Para TwoHead, entropy loss -24, -100+ é NORMAL e não indica problema
            loss_stable = (loss_trend in ['STABLE', 'IMPROVING'])  # Volta ao foco na tendência
            criteria_met.append(('Loss Converged', loss_stable))
        else:
            criteria_met.append(('Loss Converged', False))
        
        # 3. Performance Stability: Trading deve estar MUITO positivo para V7 complexo
        if performance_data.get('status') == 'ACTIVE':
            total_return = performance_data.get('total_return_pct', -100)
            win_rate = performance_data.get('current_win_rate', 0)
            trades = performance_data.get('total_trades_analyzed', 0)
            
            # V7 ULTRA RIGOROSO: Performance deve ser EXCEPCIONAL antes de convergir
            performance_good = (total_return > 25.0 and  # 25%+ retorno MÍNIMO
                              win_rate > 0.60 and       # 60%+ win rate (bem superior)
                              trades > 200)             # 200+ trades (muita experiência)
            criteria_met.append(('Performance Stable', performance_good))
        else:
            criteria_met.append(('Performance Stable', False))
        
        # 4. Volatility Check: Últimos 5 updates devem ser consistentes
        volatility_ok = True
        if len(self.training_history) >= 5:
            recent_losses = []
            recent_returns = []
            
            for hist in list(self.training_history)[-5:]:
                if hist.get('status') == 'ACTIVE':
                    recent_losses.append(hist.get('current_loss', 0))
            
            for hist in list(self.rewards_history)[-5:]:
                if hist.get('status') == 'ACTIVE':
                    recent_returns.append(hist.get('total_return_pct', 0))
            
            if recent_losses and len(recent_losses) >= 3:
                loss_volatility = np.std(recent_losses) / (np.mean(recent_losses) + 1e-8)
                volatility_ok = loss_volatility < 0.1  # Variação < 10%
            
        criteria_met.append(('Low Volatility', volatility_ok))
        
        # 5. TRADING PERFORMANCE: O QUE REALMENTE IMPORTA para TwoHead
        # Entropy loss alto é NORMAL para TwoHead, foco na performance real
        trading_profitable = True
        if performance_data.get('status') == 'ACTIVE':
            total_return = performance_data.get('total_return_pct', -100)
            # Modelo DEVE estar ganhando dinheiro de forma consistente
            if total_return < 10.0:  # Pelo menos 10% lucro mínimo
                trading_profitable = False
        criteria_met.append(('Trading Profitable', trading_profitable))
        
        # Imprimir status dos critérios
        print(f"\n[CONVERGENCE CHECK]:")
        all_met = True
        for criterion, met in criteria_met:
            status = "[OK]" if met else "[FAIL]"
            print(f"  {status} {criterion}")
            if not met:
                all_met = False
        
        return all_met
    
    def analyze_convergence_status(self, gradient_data: Dict, training_data: Dict, performance_data: Dict) -> Dict[str, Any]:
        """Análise completa do status de convergência"""
        
        # Status geral baseado em múltiplas fontes
        overall_status = 'UNKNOWN'
        issues = []
        recommendations = []
        
        # 1. Análise de Gradient Health
        grad_status = gradient_data.get('status', 'UNKNOWN')
        grad_zeros = gradient_data.get('gradient_zeros', 0.0)
        
        if grad_status == 'CRITICAL':
            issues.append(f"Gradient death detectado ({grad_zeros:.1f}% zeros)")
            recommendations.append("Verificar layer normalization e learning rate")
        elif grad_status == 'WARNING':
            issues.append(f"Gradientes degradando ({grad_zeros:.1f}% zeros)")
            recommendations.append("Monitorar próximos steps")
        
        # 2. Análise de Training Loss
        if training_data.get('status') == 'ACTIVE':
            loss_trend = training_data.get('loss_trend', 'STABLE')
            current_loss = training_data.get('current_loss', 0.0)
            
            if loss_trend == 'DEGRADING':
                issues.append(f"Loss divergindo (atual: {current_loss:.3f})")
                recommendations.append("Considerar reduzir learning rate")
            elif loss_trend == 'STABLE' and current_loss > 5.0:
                issues.append("Loss alta e estagnada")
                recommendations.append("Verificar arquitetura do modelo")
        
        # 3. Análise de Performance Trading
        if performance_data.get('status') == 'ACTIVE':
            win_rate = performance_data.get('current_win_rate', 0.0)
            total_return = performance_data.get('total_return_pct', 0.0)
            drawdown = performance_data.get('current_drawdown', 0.0)
            trades = performance_data.get('total_trades_analyzed', 0)
            
            if win_rate < 0.4:
                issues.append(f"Win rate baixa ({win_rate*100:.1f}%)")
                recommendations.append("Ajustar estratégia de entrada")
            
            if total_return < -5.0:
                issues.append(f"Retorno negativo ({total_return:.1f}%)")
                recommendations.append("Revisar estratégia de trading")
                
            if drawdown > 20.0:
                issues.append(f"Drawdown alto ({drawdown:.1f}%)")
                recommendations.append("Implementar melhor gestão de risco")
                
            if trades < 5:
                issues.append(f"Poucos trades ({trades})")
                recommendations.append("Verificar critérios de entrada")
        
        # Determinar status geral
        if not issues:
            overall_status = 'EXCELLENT'
        elif len(issues) == 1 and grad_status in ['EXCELLENT', 'HEALTHY']:
            overall_status = 'GOOD'  
        elif grad_status == 'CRITICAL' or len(issues) >= 3:
            overall_status = 'CRITICAL'
        else:
            overall_status = 'WARNING'
        
        return {
            'overall_status': overall_status,
            'issues': issues,
            'recommendations': recommendations,
            'gradient_health': grad_status,
            'training_trend': training_data.get('loss_trend', 'UNKNOWN'),
            'performance_trend': training_data.get('reward_trend', 'UNKNOWN')
        }
    
    def print_complete_status(self, gradient_data: Dict, training_data: Dict, performance_data: Dict, analysis: Dict):
        """Exibe status completo formatado"""
        
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # CONVERGENCE STATUS (moved to top)
        print("=" * 80)
        print("CONVERGENCE STATUS:")
        print("-" * 30)
        if self.converged:
            elapsed = time.time() - self.convergence_start_time if self.convergence_start_time else 0
            convergence_info = f" at step {self.convergence_step:,}" if self.convergence_step else ""
            print(f"[CONVERGED] Model converged{convergence_info}! ({elapsed/60:.1f} min ago)")
            print(f"[SUCCESS] Training can be stopped safely")
        elif self.convergence_criteria_met > 0:
            print(f"[PROGRESS] Convergence criteria met {self.convergence_criteria_met}/{self.convergence_threshold} times")
            print(f"[WAITING] Need {self.convergence_threshold - self.convergence_criteria_met} more consecutive confirmations")
        else:
            print(f"[TRAINING] Model still training - convergence not detected")
        
        print("=" * 80)
        print("         [FIRE] COMPLETE CONVERGENCE MONITOR")  
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Session: {self.current_session}")
        print("=" * 80)
        
        # Status Geral
        overall = analysis['overall_status']
        status_icons = {
            'EXCELLENT': '[+++]',
            'GOOD': '[++]', 
            'WARNING': '[+]',
            'CRITICAL': '[!!!]',
            'UNKNOWN': '[?]'
        }
        
        print("OVERALL STATUS:")
        print("-" * 40)
        print(f"{status_icons.get(overall, '[?]')} {overall}")
        
        if analysis['issues']:
            print(f"\n[!] ISSUES ({len(analysis['issues'])}):")
            for issue in analysis['issues']:
                print(f"  - {issue}")
        
        if analysis['recommendations']:
            print(f"\n[REC] RECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                print(f"  > {rec}")
        
        print("\n" + "=" * 80)
        
        # Gradient Health
        print("GRADIENT HEALTH:")
        print("-" * 30)
        grad_status = gradient_data['status']
        grad_zeros = gradient_data['gradient_zeros']
        grad_step = gradient_data['step']
        alerts = gradient_data['alert_count']
        
        print(f"Status: {status_icons.get(grad_status, '[?]')} {grad_status}")
        print(f"Gradient Zeros: {grad_zeros:.2f}%")
        print(f"Latest Step: {grad_step:,}")
        print(f"Alert Count: {alerts}")
        
        print("\n" + "=" * 80)
        
        # Training Metrics
        print("TRAINING METRICS:")
        print("-" * 30)
        if training_data.get('status') == 'ACTIVE':
            current_loss = training_data['current_loss']
            loss_mean = training_data['loss_mean']
            loss_trend = training_data['loss_trend']
            current_lr = training_data['current_lr']
            
            trend_icons = {
                'IMPROVING': '[UP]',
                'DEGRADING': '[DOWN]', 
                'STABLE': '[FLAT]'
            }
            
            print(f"Current Loss: {current_loss:.4f}")
            print(f"Average Loss: {loss_mean:.4f}")
            print(f"Loss Trend: {trend_icons.get(loss_trend, '[FLAT]')} {loss_trend}")
            print(f"Learning Rate: {current_lr:.2e}")
            print(f"Total Entries: {training_data['total_entries']}")
        else:
            print("[!] No training data available")
        
        print("\n" + "=" * 80)
        
        # Performance Metrics - MULTI-EPISÓDIOS
        print("PERFORMANCE METRICS (MULTI-EPISODES):")
        print("-" * 40)
        if performance_data.get('status') == 'ACTIVE':
            episodes_count = performance_data.get('episodes_analyzed', 0)
            
            if episodes_count > 0:
                print(f"Episodes Analyzed: {episodes_count}")
                print(f"Avg Episode Return: {performance_data.get('avg_episode_return', 0):.2f}%")
                print(f"Best Episode: {performance_data.get('best_episode_return', 0):.2f}%")
                print(f"Worst Episode: {performance_data.get('worst_episode_return', 0):.2f}%")
                print(f"Return Volatility: {performance_data.get('return_std', 0):.2f}%")
                print(f"Positive Episodes: {performance_data.get('positive_episodes_pct', 0):.1f}%")
                print(f"Trend: {performance_data.get('recent_episodes_trend', 'STABLE')}")
                
                print(f"\nTRADING STATS:")
                print(f"Avg Win Rate: {performance_data.get('avg_win_rate', 0)*100:.1f}%")
                print(f"Avg Trades/Episode: {performance_data.get('avg_trades_per_episode', 0):.1f}")
                print(f"Avg Drawdown: {performance_data.get('avg_drawdown', 0):.2f}%")
                print(f"Max Drawdown: {performance_data.get('max_drawdown', 0):.2f}%")
                print(f"Avg Sharpe Ratio: {performance_data.get('avg_sharpe_ratio', 0):.2f}")
            else:
                print("[!] No episodes data available")
        else:
            print("[!] No performance data available")
        
        print("\n" + "=" * 80)
        print(f"Next update in: {self.update_interval} seconds")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)
    
    def run_monitoring(self):
        """Executa monitoramento contínuo"""
        print("[FIRE] Starting complete convergence monitoring...")
        print("Finding latest session...")
        
        self.current_session = self.find_latest_session()
        if not self.current_session:
            print("[ERROR] No JSONL sessions found in avaliacoes/")
            print("Make sure training is running with RealTimeLogger")
            return
        
        print(f"[MONITOR] Monitoring session: {self.current_session}")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                current_time = time.time()
                
                if current_time - self.last_update >= self.update_interval:
                    # Coletar dados de todas as fontes
                    gradient_data = self.collect_gradient_data()
                    training_data = self.collect_training_data(self.current_session)
                    performance_data = self.collect_performance_data(self.current_session)
                    
                    # Análise completa
                    analysis = self.analyze_convergence_status(gradient_data, training_data, performance_data)
                    
                    # Check if convergence criteria met
                    criteria_met = self.check_convergence_criteria(gradient_data, training_data, performance_data)
                    
                    if criteria_met:
                        self.convergence_criteria_met += 1
                        if self.convergence_start_time is None:
                            self.convergence_start_time = current_time
                        
                        # Se critérios foram atendidos N vezes consecutivas, convergiu
                        if self.convergence_criteria_met >= self.convergence_threshold and not self.converged:
                            self.converged = True
                            # Capturar step atual para tracking
                            current_step = max(
                                gradient_data.get('step', 0),
                                training_data.get('current_step', 0)
                            )
                            self.convergence_step = current_step
                            print(f"\n[CONVERGENCE DETECTED AT STEP {current_step:,}] Model has converged!")
                            print(f"[CRITERIA MET] {self.convergence_criteria_met} consecutive confirmations")
                            print(f"[RECOMMENDATION] Training can be stopped")
                            
                            # Enviar beep de notificação
                            for _ in range(3):
                                print("\a", end="", flush=True)
                                time.sleep(0.5)
                    else:
                        # Reset counter se critérios não foram atendidos
                        self.convergence_criteria_met = 0
                        self.convergence_start_time = None
                    
                    # Salvar no histórico
                    self.gradient_history.append(gradient_data)
                    self.training_history.append(training_data)
                    self.rewards_history.append(performance_data)
                    
                    # Exibir status
                    self.print_complete_status(gradient_data, training_data, performance_data, analysis)
                    
                    self.last_update = current_time
                
                time.sleep(1)  # Check every second
                
        except KeyboardInterrupt:
            print("\n\n[STOP] Complete Convergence Monitor stopped")
            print("[SUMMARY] Final summary:")
            print(f"   - Gradient history: {len(self.gradient_history)} points")
            print(f"   - Training history: {len(self.training_history)} points") 
            print(f"   - Performance history: {len(self.rewards_history)} points")
            print("[CHECK] Monitoring session ended")

if __name__ == "__main__":
    monitor = CompleteConvergenceMonitor(
        avaliacoes_path="D:/Projeto/avaliacoes",
        update_interval=30.0,  # Update every 30 seconds
        history_size=200       # Keep 200 points in memory
    )
    
    monitor.run_monitoring()
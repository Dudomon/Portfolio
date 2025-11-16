#!/usr/bin/env python3
"""
🔥 STATS V11 - ANÁLISE EM TEMPO REAL DOS LOGS JSONL DE TREINAMENTO
=================================================================

Monitora e analiza em tempo real:
✅ Convergence: KL divergence, loss trends, stability
✅ Performance: Portfolio, trades, win rate, drawdown
✅ Training: Learning curves, gradients, clip fractions
✅ Rewards: Episode rewards, patterns, distribution

Uso:
    python statsv11.py [--live] [--session SESSION_ID] [--tail N]
"""

import json
import glob
import os
import sys
import time
import argparse
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt

# Configurações
AVALIACOES_PATH = "D:/Projeto/avaliacoes/"
REFRESH_INTERVAL = 2.0  # segundos
LIVE_TAIL_SIZE = 200    # últimas N entradas - aumentado para trading reliability
PLOT_WINDOW = 1000      # janela para plots

class TrainingStatsAnalyzer:
    def __init__(self):
        self.sessions = {}
        self.current_session = None
        
    def find_recent_session(self):
        """🔍 Encontrar sessão mais recente com FILTRO DE STEPS MÍNIMO"""
        files = glob.glob(os.path.join(AVALIACOES_PATH, "*.jsonl"))
        if not files:
            return None

        # Extrair session_id dos arquivos mais recentes
        valid_sessions = {}
        for file in files:
            if os.path.getsize(file) < 200:  # Skip arquivos apenas com header
                continue

            basename = os.path.basename(file)
            parts = basename.split('_')
            if len(parts) >= 4:
                session_id = '_'.join(parts[1:])  # Remove categoria
                session_id = session_id.replace('.jsonl', '')

                # 🎯 FILTRO DE STEPS: Verificar se sessão tem steps suficientes
                max_steps = self._get_session_max_steps(session_id)

                # 🎯 FILTRO AJUSTADO: Aceitar sessões com > 10 steps (incluir sessões novas)
                if max_steps >= 50000:  # Incluir sessões recém-iniciadas
                    mod_time = os.path.getmtime(file)
                    valid_sessions[session_id] = {
                        'mod_time': mod_time,
                        'max_steps': max_steps
                    }

        if not valid_sessions:
            print("⚠️  Nenhuma sessão de treino encontrada")
            return None

        # 🎯 PRIORIZAR SESSÃO MAIS RECENTE (não a com mais steps)
        selected_session = max(valid_sessions.keys(), key=lambda k: valid_sessions[k]['mod_time'])

        # 📊 Mostrar informações da sessão selecionada
        session_info = valid_sessions[selected_session]
        mod_time_str = datetime.fromtimestamp(session_info['mod_time']).strftime("%Y-%m-%d %H:%M:%S")
        print(f"🎯 SESSÃO SELECIONADA: {selected_session}")
        print(f"   📅 Timestamp: {mod_time_str}")
        print(f"   📊 Max Steps: {session_info['max_steps']:,}")
        print(f"   🔄 Total sessões encontradas: {len(valid_sessions)}")

        return selected_session

    def _get_session_max_steps(self, session_id):
        """🔍 Obter máximo de steps de uma sessão"""
        categories = ['convergence', 'performance', 'training', 'rewards']
        max_steps = 0

        for category in categories:
            pattern = os.path.join(AVALIACOES_PATH, f"{category}_{session_id}.jsonl")
            files = glob.glob(pattern)

            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue

                            try:
                                data = json.loads(line)
                                if data.get('type') == 'header':
                                    continue

                                step = data.get('step', 0)
                                if step > max_steps:
                                    max_steps = step
                            except:
                                continue
                except:
                    continue

        return max_steps
    
    def load_session_data(self, session_id):
        """📊 Carregar dados de uma sessão"""
        categories = ['convergence', 'performance', 'training', 'gradients', 'rewards']
        session_data = {
            'session_id': session_id,
            'data': defaultdict(list),
            'latest_step': 0,
            'start_time': None
        }
        
        for category in categories:
            pattern = os.path.join(AVALIACOES_PATH, f"{category}_{session_id}.jsonl")
            files = glob.glob(pattern)
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                                
                            data = json.loads(line)
                            if data.get('type') == 'header':
                                if session_data['start_time'] is None:
                                    session_data['start_time'] = data.get('start_time')
                                continue
                                
                            session_data['data'][category].append(data)
                            
                            # Atualizar latest_step
                            step = data.get('step', 0)
                            if step > session_data['latest_step']:
                                session_data['latest_step'] = step
                                
                except Exception as e:
                    print(f"⚠️ Erro lendo {file_path}: {e}")
        
        return session_data
    
    def analyze_convergence(self, data):
        """🎯 Analisar métricas de convergência"""
        if not data:
            return {}
            
        convergence_scores = [d.get('convergence_score', 0) for d in data[-LIVE_TAIL_SIZE:]]
        loss_trends = [d.get('loss_trend', 'unknown') for d in data[-LIVE_TAIL_SIZE:]]
        
        analysis = {
            'avg_convergence': np.mean(convergence_scores) if convergence_scores else 0,
            'convergence_trend': 'improving' if len(convergence_scores) > 1 and convergence_scores[-1] > convergence_scores[0] else 'stable',
            'stable_count': loss_trends.count('stable'),
            'improving_count': loss_trends.count('improving'),
            'degrading_count': loss_trends.count('degrading'),
            'latest_score': convergence_scores[-1] if convergence_scores else 0
        }
        
        return analysis
    
    def ultra_reliable_sharpe(self, portfolio_values, risk_free_rate=0.02):
        """🔥 ULTRA RELIABLE Sharpe calculation with mathematical stability"""
        if len(portfolio_values) < 100:  # Need minimum data for reliable Sharpe
            return 0
            
        try:
            # Calculate returns
            portfolio_array = np.array(portfolio_values)
            returns = np.diff(portfolio_array) / portfolio_array[:-1]
            
            # Remove outliers (ultra reliable)
            q75, q25 = np.percentile(returns, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            clean_returns = returns[(returns >= lower_bound) & (returns <= upper_bound)]
            
            if len(clean_returns) < 10:
                clean_returns = returns
            
            # Calculate Sharpe with multiple windows for validation
            daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
            excess_returns = clean_returns - daily_risk_free
            
            returns_std = np.std(excess_returns)
            
            # 🎯 MATHEMATICAL STABILITY: Prevent division by near-zero
            if returns_std == 0 or returns_std < 1e-5:  # Too small = unstable
                return 0  # Portfolio too stable for meaningful Sharpe
                
            sharpe = np.mean(excess_returns) / returns_std * np.sqrt(252)
            
            # 🎯 REASONABLE BOUNDS: Cap realistic Sharpe range
            sharpe = np.clip(sharpe, -50, 50)  # Professional trading range
                
            return sharpe
            
        except Exception as e:
            print(f"🚨 Sharpe calculation error: {e}")
            return 0

    def ultra_reliable_calmar(self, portfolio_values):
        """🔥 ULTRA RELIABLE Calmar ratio calculation"""
        if len(portfolio_values) < 150:  # Need more data for reliable Calmar
            return 0
            
        try:
            portfolio_array = np.array(portfolio_values)
            
            # Calculate cumulative return
            total_return = (portfolio_array[-1] / portfolio_array[0]) - 1
            
            # Annualize the return (assume data is daily)
            days = len(portfolio_array)
            annual_return = (1 + total_return) ** (252 / days) - 1
            
            # Calculate maximum drawdown
            peak = np.maximum.accumulate(portfolio_array)
            drawdown = (portfolio_array - peak) / peak
            max_drawdown = abs(np.min(drawdown))
            
            if max_drawdown < 0.001:  # Avoid division by near-zero
                return annual_return * 1000  # High Calmar for minimal drawdown
                
            calmar = annual_return / max_drawdown
            
            # Ultra reliable validation
            if abs(calmar) > 50:  # Extreme values
                calmar = np.sign(calmar) * 50
                
            return calmar
            
        except Exception as e:
            print(f"🚨 Calmar calculation error: {e}")
            return 0

    def ultra_reliable_peak_detection(self, data):
        """🔥 ULTRA RELIABLE peak detection with automatic checkpoint saving"""
        if len(data) < 100:
            return []
            
        try:
            # Extract Sharpe values with steps
            sharpe_data = []
            for entry in data:
                portfolio = entry.get('portfolio_value', 500)
                step = entry.get('step', 0)
                if portfolio != 500:  # Skip default values
                    sharpe_data.append({'portfolio': portfolio, 'step': step})
            
            if len(sharpe_data) < 100:
                return []
            
            # Calculate rolling Sharpe for peak detection
            window_size = min(50, len(sharpe_data) // 4)
            peaks = []

            for i in range(window_size, len(sharpe_data) - window_size):
                current_step = sharpe_data[i]['step']

                # 🔥 FILTRO: Só considerar peaks APÓS 200k steps
                if current_step < 200000:
                    continue

                window_portfolios = [d['portfolio'] for d in sharpe_data[i-window_size:i+window_size]]
                sharpe = self.ultra_reliable_sharpe(window_portfolios)

                if sharpe > 0.5:  # Only consider good Sharpe ratios
                    peaks.append({
                        'step': current_step,
                        'sharpe': sharpe,
                        'portfolio': sharpe_data[i]['portfolio'],
                        'confidence': min(1.0, len(window_portfolios) / 50)  # Confidence based on data
                    })
            
            # Sort by Sharpe ratio and return top peaks
            peaks.sort(key=lambda x: x['sharpe'], reverse=True)
            top_peaks = peaks[:10]  # Top 10 peaks
            
            # 🔥 ULTRA RELIABLE: Save peak information for checkpoints
            self.save_ultra_reliable_peaks(top_peaks)
            
            return top_peaks
            
        except Exception as e:
            print(f"🚨 Peak detection error: {e}")
            return []

    def save_ultra_reliable_peaks(self, peaks):
        """🔥 ULTRA RELIABLE: Save peak information for checkpoint decisions"""
        try:
            if not peaks:
                return
                
            peak_file = "ultra_reliable_peaks.txt"
            with open(peak_file, "w", encoding='utf-8') as f:
                f.write("🔥 ULTRA RELIABLE PEAKS - CHECKPOINT CANDIDATES\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for i, peak in enumerate(peaks, 1):
                    marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
                    f.write(f"{marker} PEAK #{i}:\n")
                    f.write(f"   Step: {peak['step']:,}\n")
                    f.write(f"   Sharpe: {peak['sharpe']:.4f}\n")
                    f.write(f"   Portfolio: ${peak['portfolio']:.2f}\n")
                    f.write(f"   Confidence: {peak['confidence']:.2f}\n")
                    f.write(f"   📁 CHECKPOINT RECOMMENDATION: SAVE AT STEP {peak['step']}\n\n")
                    
            print(f"🔥 PEAKS SAVED: {peak_file} ({len(peaks)} peaks)")
            
        except Exception as e:
            print(f"🚨 Error saving peaks: {e}")

    def calculate_sharpe_ratio(self, portfolio_values, risk_free_rate=0.02):
        """📊 Calcular Sharpe ratio"""
        if len(portfolio_values) < 2:
            return 0.0
            
        # Calcular returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[~np.isnan(returns)]  # Remove NaNs
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Sharpe ratio anualizado (assumindo dados diários)
        excess_returns = np.mean(returns) - risk_free_rate/252
        return (excess_returns / np.std(returns)) * np.sqrt(252)
    
    def calculate_calmar_ratio(self, portfolio_values):
        """📊 Calcular Calmar ratio"""
        if len(portfolio_values) < 2:
            return 0.0
            
        # Calcular retorno total anualizado
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        # Calcular máximo drawdown
        peak = portfolio_values[0]
        max_dd = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        if max_dd == 0:
            return float('inf') if total_return > 0 else 0.0
            
        return total_return / max_dd
    
    def find_sharpe_peaks(self, data, window_size=50):
        """🎯 Encontrar picos de Sharpe ratio durante o treinamento"""
        if len(data) < window_size * 2:
            return []
            
        sharpe_history = []
        steps = []
        
        # Calcular Sharpe para janelas deslizantes
        for i in range(window_size, len(data)):
            window_data = data[i-window_size:i]
            portfolio_values = [d.get('portfolio_value', 500) for d in window_data]
            
            sharpe = self.calculate_sharpe_ratio(portfolio_values)
            sharpe_history.append(sharpe)
            steps.append(data[i].get('step', 0))
        
        # Encontrar picos locais
        peaks = []
        for i in range(1, len(sharpe_history) - 1):
            if (sharpe_history[i] > sharpe_history[i-1] and 
                sharpe_history[i] > sharpe_history[i+1] and
                sharpe_history[i] > 0.5):  # Apenas picos significativos
                peaks.append({
                    'step': steps[i],
                    'sharpe': sharpe_history[i],
                    'index': i + window_size
                })
        
        # Retornar apenas os N melhores picos
        peaks.sort(key=lambda x: x['sharpe'], reverse=True)
        return peaks[:3]  # Top 3 picos

    def analyze_performance(self, data):
        """💹 Analisar métricas de performance"""
        if not data:
            return {}
            
        recent_data = data[-LIVE_TAIL_SIZE:]
        all_portfolio = [d.get('portfolio_value', 500) for d in data]
        recent_portfolio = [d.get('portfolio_value', 500) for d in recent_data]
        drawdowns = [d.get('drawdown', 0) for d in recent_data]
        trades_counts = [d.get('trades_count', 0) for d in recent_data]
        win_rates = [d.get('win_rate', 0) for d in recent_data if d.get('trades_count', 0) > 0]
        
        # 🔥 ULTRA RELIABLE METRICS CALCULATION
        try:
            # ULTRA RELIABLE Sharpe calculation with multiple windows
            sharpe_recent = self.ultra_reliable_sharpe(recent_portfolio)
            sharpe_all = self.ultra_reliable_sharpe(all_portfolio)
            
            # ULTRA RELIABLE Calmar calculation
            calmar_recent = self.ultra_reliable_calmar(recent_portfolio)
            calmar_all = self.ultra_reliable_calmar(all_portfolio)
            
            # ULTRA RELIABLE peak detection with automatic saving
            sharpe_peaks = self.ultra_reliable_peak_detection(data)
            
        except Exception as e:
            print(f"🚨 ERROR in ultra reliable calculations: {e}")
            sharpe_recent = sharpe_all = calmar_recent = calmar_all = 0
            sharpe_peaks = []
        
        analysis = {
            'current_portfolio': recent_portfolio[-1] if recent_portfolio else 500,
            'portfolio_change': ((recent_portfolio[-1] - 500) / 500 * 100) if recent_portfolio else 0,
            'max_drawdown': max(drawdowns) if drawdowns else 0,
            'current_drawdown': drawdowns[-1] if drawdowns else 0,
            'total_trades': trades_counts[-1] if trades_counts else 0,
            'avg_win_rate': np.mean(win_rates) if win_rates else 0,
            'portfolio_std': np.std(recent_portfolio) if len(recent_portfolio) > 1 else 0,
            'sharpe_recent': sharpe_recent,
            'sharpe_all': sharpe_all,
            'calmar_recent': calmar_recent,
            'calmar_all': calmar_all,
            'sharpe_peaks': sharpe_peaks,
            'best_sharpe': max(sharpe_peaks, key=lambda x: x['sharpe']) if sharpe_peaks else None
        }
        
        return analysis
    
    def analyze_training(self, data):
        """🚀 Analiizar métricas de treinamento"""
        if not data:
            return {}
            
        recent_data = data[-LIVE_TAIL_SIZE:]
        policy_losses = [d.get('policy_loss', 0) for d in recent_data if d.get('policy_loss') is not None]
        value_losses = [d.get('value_loss', 0) for d in recent_data if d.get('value_loss') is not None]
        entropy_losses = [d.get('entropy_loss', 0) for d in recent_data if d.get('entropy_loss') is not None]
        clip_fractions = [d.get('clip_fraction', 0) for d in recent_data if d.get('clip_fraction') is not None]
        losses = [d.get('loss', 0) for d in recent_data if d.get('loss') is not None]
        learning_rates = [d.get('learning_rate', 0) for d in recent_data if d.get('learning_rate') is not None]
        explained_variances = [d.get('explained_variance', 0) for d in recent_data if d.get('explained_variance') is not None]
        
        analysis = {
            'avg_policy_loss': np.mean(policy_losses) if policy_losses else 0,
            'avg_value_loss': np.mean(value_losses) if value_losses else 0,
            'avg_entropy_loss': np.mean(entropy_losses) if entropy_losses else 0,
            'avg_clip_fraction': np.mean(clip_fractions) if clip_fractions else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'avg_learning_rate': np.mean(learning_rates) if learning_rates else 0,
            'avg_explained_variance': np.mean(explained_variances) if explained_variances else 0,
            'clip_trend': 'low' if clip_fractions and np.mean(clip_fractions) < 0.1 else 'high',
            'latest_policy_loss': policy_losses[-1] if policy_losses else 0,
            'latest_value_loss': value_losses[-1] if value_losses else 0,
            'latest_entropy_loss': entropy_losses[-1] if entropy_losses else 0,
            'latest_clip': clip_fractions[-1] if clip_fractions else 0,
            'latest_lr': learning_rates[-1] if learning_rates else 0,
            'latest_explained_var': explained_variances[-1] if explained_variances else 0
        }
        
        return analysis
    
    def print_live_stats(self, session_data):
        """📊 Imprimir estatísticas em tempo real"""
        # Só limpa uma vez no início, depois atualiza os stats importantes
        if not hasattr(self, '_first_print_done'):
            os.system('cls' if os.name == 'nt' else 'clear')
            self._first_print_done = True
        else:
            # Limpa apenas as linhas que mudam (método inteligente)
            os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🔥 STATS V11 ULTRA RELIABLE - MÁXIMA CONFIABILIDADE")
        print("=" * 70)
        print(f"📅 Sessão: {session_data['session_id']}")
        print(f"⏰ Iniciado: {session_data.get('start_time', 'N/A')}")
        print(f"📈 Step atual: {session_data['latest_step']:,}")
        print(f"🔄 Atualizado: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        # Análise de convergência
        conv_analysis = self.analyze_convergence(session_data['data']['convergence'])
        if conv_analysis:
            print("🎯 CONVERGÊNCIA")
            print(f"   Score médio: {conv_analysis['avg_convergence']:.3f}")
            print(f"   Score atual: {conv_analysis['latest_score']:.3f}")
            print(f"   Tendência: {conv_analysis['convergence_trend']}")
            print(f"   Estabilidade: {conv_analysis['stable_count']}/{LIVE_TAIL_SIZE} stable")
            print()
        
        # Análise de performance
        perf_analysis = self.analyze_performance(session_data['data']['performance'])
        if perf_analysis:
            print("💹 ULTRA RELIABLE PERFORMANCE")
            print(f"   💰 Portfolio: ${perf_analysis['current_portfolio']:.2f}")
            print(f"   📈 Mudança: {perf_analysis['portfolio_change']:+.2f}%")
            print(f"   📉 Max DD: {perf_analysis['max_drawdown']:.2f}%")
            print(f"   📉 DD atual: {perf_analysis['current_drawdown']:.2f}%")
            print(f"   🔄 Trades: {perf_analysis['total_trades']}")
            print(f"   ✅ Win Rate: {perf_analysis['avg_win_rate']:.1%}")
            print(f"   📊 Portfolio σ: {perf_analysis['portfolio_std']:.2f}")
            print()
            
            # 🔥 ULTRA RELIABLE SHARPE ANALYSIS
            print("📊 ULTRA RELIABLE SHARPE ANALYSIS")
            sharpe_status_recent = "🟢 EXCELENTE" if perf_analysis['sharpe_recent'] > 1.5 else "🟡 BOM" if perf_analysis['sharpe_recent'] > 1.0 else "🟠 MÉDIO" if perf_analysis['sharpe_recent'] > 0.5 else "🔴 RUIM"
            sharpe_status_all = "🟢 EXCELENTE" if perf_analysis['sharpe_all'] > 1.5 else "🟡 BOM" if perf_analysis['sharpe_all'] > 1.0 else "🟠 MÉDIO" if perf_analysis['sharpe_all'] > 0.5 else "🔴 RUIM"
            
            print(f"   📊 Sharpe (Recente): {perf_analysis['sharpe_recent']:.4f} {sharpe_status_recent}")
            print(f"   📊 Sharpe (Total):   {perf_analysis['sharpe_all']:.4f} {sharpe_status_all}")
            print()
            
            # 🔥 ULTRA RELIABLE CALMAR ANALYSIS  
            print("📈 ULTRA RELIABLE CALMAR ANALYSIS")
            calmar_status_recent = "🟢 EXCELENTE" if perf_analysis['calmar_recent'] > 2.0 else "🟡 BOM" if perf_analysis['calmar_recent'] > 1.0 else "🟠 MÉDIO" if perf_analysis['calmar_recent'] > 0.5 else "🔴 RUIM"
            calmar_status_all = "🟢 EXCELENTE" if perf_analysis['calmar_all'] > 2.0 else "🟡 BOM" if perf_analysis['calmar_all'] > 1.0 else "🟠 MÉDIO" if perf_analysis['calmar_all'] > 0.5 else "🔴 RUIM"
            
            print(f"   📈 Calmar (Recente): {perf_analysis['calmar_recent']:.4f} {calmar_status_recent}")
            print(f"   📈 Calmar (Total):   {perf_analysis['calmar_all']:.4f} {calmar_status_all}")
            print()
            
            # 🔥 ULTRA RELIABLE PEAK DETECTION
            if perf_analysis['sharpe_peaks']:
                print("🏔️ ULTRA RELIABLE PEAKS - CHECKPOINT CANDIDATES")
                print("   🔥 AUTOMATIC PEAK SAVING ACTIVE!")
                for i, peak in enumerate(perf_analysis['sharpe_peaks'][:5], 1):
                    marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅" if i == 4 else "⭐"
                    confidence_status = "🔥 ULTRA ALTA" if peak['confidence'] > 0.8 else "✅ ALTA" if peak['confidence'] > 0.6 else "🟡 MÉDIA"
                    print(f"   {marker} PEAK #{i}: Sharpe {peak['sharpe']:.4f} | Step {peak['step']:,} | {confidence_status}")
                    
                if perf_analysis['best_sharpe']:
                    best = perf_analysis['best_sharpe'] 
                    print(f"   🎯 MELHOR CHECKPOINT: Step {best['step']:,} (Sharpe {best['sharpe']:.4f})")
                print()
            else:
                print("🔍 PEAK DETECTION: Aguardando dados suficientes...")
                print()
        
        # Análise de treinamento
        train_analysis = self.analyze_training(session_data['data']['training'])
        if train_analysis:
            print("🚀 TREINAMENTO")
            print(f"   Policy Loss: {train_analysis['avg_policy_loss']:.6f}")
            print(f"   Value Loss: {train_analysis['avg_value_loss']:.6f}")
            print(f"   Entropy Loss: {train_analysis['avg_entropy_loss']:.6f}")
            print(f"   Clip fraction: {train_analysis['avg_clip_fraction']:.3f} ({train_analysis['clip_trend']})")
            print(f"   Learning Rate: {train_analysis['avg_learning_rate']:.2e}")
            print(f"   Explained Var: {train_analysis['avg_explained_variance']:.3f}")
            print(f"   🔄 Latest:")
            print(f"     Policy: {train_analysis['latest_policy_loss']:.6f}")
            print(f"     Value: {train_analysis['latest_value_loss']:.6f}")
            print(f"     Entropy: {train_analysis['latest_entropy_loss']:.6f}")
            print(f"     Clip: {train_analysis['latest_clip']:.3f}")
            print(f"     LR: {train_analysis['latest_lr']:.2e}")
            print()
        
        # Totais de dados
        total_entries = sum(len(session_data['data'][cat]) for cat in session_data['data'])
        print("📊 DADOS CARREGADOS")
        for category, data_list in session_data['data'].items():
            if data_list:
                print(f"   {category.title()}: {len(data_list):,} entradas")
        print(f"   Total: {total_entries:,} entradas")
        print()
        
        print("🔄 Pressione Ctrl+C para sair")
    
    def run_live_analysis(self, session_id=None, refresh_interval=REFRESH_INTERVAL):
        """🔴 Executar análise em tempo real"""
        if session_id is None:
            session_id = self.find_recent_session()
            if not session_id:
                print("❌ Nenhuma sessão de treinamento encontrada!")
                return
        
        print(f"🔥 Iniciando monitoramento da sessão: {session_id}")
        print(f"📁 Pasta: {AVALIACOES_PATH}")
        print(f"⏱️ Intervalo: {refresh_interval}s")
        print("=" * 60)
        
        try:
            while True:
                session_data = self.load_session_data(session_id)
                self.print_live_stats(session_data)
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitoramento interrompido pelo usuário")
        except Exception as e:
            print(f"\n❌ Erro durante monitoramento: {e}")
    
    def analyze_session(self, session_id):
        """📈 Análise completa de uma sessão"""
        session_data = self.load_session_data(session_id)
        
        print(f"📊 ANÁLISE COMPLETA - {session_id}")
        print("=" * 60)
        
        conv_analysis = self.analyze_convergence(session_data['data']['convergence'])
        perf_analysis = self.analyze_performance(session_data['data']['performance'])
        train_analysis = self.analyze_training(session_data['data']['training'])
        
        # Imprimir análises detalhadas
        if conv_analysis:
            print("\n🎯 CONVERGÊNCIA:")
            for key, value in conv_analysis.items():
                print(f"   {key}: {value}")
        
        if perf_analysis:
            print("\n💹 PERFORMANCE:")
            print(f"   Portfolio Atual: ${perf_analysis['current_portfolio']:.2f}")
            print(f"   Mudança Total: {perf_analysis['portfolio_change']:+.2f}%")
            print(f"   Max Drawdown: {perf_analysis['max_drawdown']:.2f}%")
            print(f"   Total Trades: {perf_analysis['total_trades']}")
            print(f"   Win Rate: {perf_analysis['avg_win_rate']:.1%}")
            print(f"   Volatilidade: ${perf_analysis['portfolio_std']:.2f}")
            print(f"\n   📊 SHARPE RATIO:")
            print(f"      Recente (últimos {LIVE_TAIL_SIZE}): {perf_analysis['sharpe_recent']:.3f}")
            print(f"      Total (sessão completa): {perf_analysis['sharpe_all']:.3f}")
            print(f"   📈 CALMAR RATIO:")
            print(f"      Recente (últimos {LIVE_TAIL_SIZE}): {perf_analysis['calmar_recent']:.3f}")
            print(f"      Total (sessão completa): {perf_analysis['calmar_all']:.3f}")
            
            if perf_analysis['best_sharpe']:
                best = perf_analysis['best_sharpe']
                print(f"\n   🎯 MELHOR PICO DE SHARPE:")
                print(f"      Valor: {best['sharpe']:.3f}")
                print(f"      Step: {best['step']:,}")
                print(f"      📍 PONTO DE INTERESSE para salvar checkpoint!")
            
            if perf_analysis['sharpe_peaks']:
                print(f"\n   🏆 TOP 3 PICOS DE SHARPE:")
                for i, peak in enumerate(perf_analysis['sharpe_peaks'], 1):
                    marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                    print(f"      {marker} #{i}: {peak['sharpe']:.3f} (Step {peak['step']:,})")
            
            # Análise de qualidade
            sharpe_quality = "🔴 Ruim" if perf_analysis['sharpe_all'] < 0.5 else "🟡 Mediano" if perf_analysis['sharpe_all'] < 1.0 else "🟢 Bom" if perf_analysis['sharpe_all'] < 1.5 else "🏆 Excelente"
            calmar_quality = "🔴 Ruim" if perf_analysis['calmar_all'] < 0.5 else "🟡 Mediano" if perf_analysis['calmar_all'] < 1.0 else "🟢 Bom" if perf_analysis['calmar_all'] < 2.0 else "🏆 Excelente"
            
            print(f"\n   🎖️ AVALIAÇÃO DE QUALIDADE:")
            print(f"      Sharpe: {sharpe_quality} ({perf_analysis['sharpe_all']:.3f})")
            print(f"      Calmar: {calmar_quality} ({perf_analysis['calmar_all']:.3f})")
        
        if train_analysis:
            print("\n🚀 TREINAMENTO:")
            print(f"   Policy Loss (avg): {train_analysis['avg_policy_loss']:.6f}")
            print(f"   Value Loss (avg): {train_analysis['avg_value_loss']:.6f}")
            print(f"   Entropy Loss (avg): {train_analysis['avg_entropy_loss']:.6f}")
            print(f"   Clip Fraction (avg): {train_analysis['avg_clip_fraction']:.3f}")
            print(f"   Learning Rate (avg): {train_analysis['avg_learning_rate']:.2e}")
            print(f"   Explained Variance (avg): {train_analysis['avg_explained_variance']:.3f}")
            print(f"   Latest Policy Loss: {train_analysis['latest_policy_loss']:.6f}")
            print(f"   Latest Value Loss: {train_analysis['latest_value_loss']:.6f}")
            print(f"   Latest Clip Fraction: {train_analysis['latest_clip']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Stats V11 - Análise JSONL Training')
    parser.add_argument('--live', action='store_true', help='Modo análise em tempo real')
    parser.add_argument('--session', type=str, help='ID da sessão específica')
    parser.add_argument('--interval', type=float, default=REFRESH_INTERVAL, help='Intervalo de atualização (segundos)')
    
    args = parser.parse_args()
    
    analyzer = TrainingStatsAnalyzer()
    
    if args.live:
        analyzer.run_live_analysis(args.session, args.interval)
    else:
        session_id = args.session or analyzer.find_recent_session()
        if not session_id:
            print("❌ Nenhuma sessão encontrada!")
            return
        analyzer.analyze_session(session_id)

if __name__ == "__main__":
    main()
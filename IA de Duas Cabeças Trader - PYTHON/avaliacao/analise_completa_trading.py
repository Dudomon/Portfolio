#!/usr/bin/env python3
"""
🎯 ANÁLISE COMPLETA DE TRADING - GUIA PARA SHARPE 1.5+
=====================================================

Analisa TUDO que impacta performance e sugere mudanças específicas
para otimizar o modelo SILUS rumo ao Sharpe ratio de 1.5

Uso:
    python analise_completa_trading.py [--session SESSION_ID]
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict, deque
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

AVALIACOES_PATH = "D:/Projeto/avaliacoes/"

class TradingPerformanceAnalyzer:
    def __init__(self):
        self.data = defaultdict(list)
        self.session_id = None
        
    def load_session_data(self, session_id=None):
        """📊 Carregar dados da sessão mais recente ou específica"""
        if session_id is None:
            # Encontrar sessão mais recente
            files = glob.glob(os.path.join(AVALIACOES_PATH, "*.jsonl"))
            if not files:
                print("❌ Nenhum arquivo JSONL encontrado!")
                return False
                
            # Extrair session_ids
            sessions = {}
            for file in files:
                if os.path.getsize(file) < 200:
                    continue
                basename = os.path.basename(file)
                parts = basename.split('_')
                if len(parts) >= 4:
                    sid = '_'.join(parts[1:]).replace('.jsonl', '')
                    sessions[sid] = max(sessions.get(sid, 0), os.path.getmtime(file))
            
            if not sessions:
                print("❌ Nenhuma sessão válida encontrada!")
                return False
                
            session_id = max(sessions.keys(), key=lambda k: sessions[k])
        
        self.session_id = session_id
        print(f"📊 Carregando sessão: {session_id}")
        
        # Carregar todos os tipos de dados
        categories = ['convergence', 'performance', 'training', 'gradients', 'rewards']
        total_loaded = 0
        
        for category in categories:
            pattern = os.path.join(AVALIACOES_PATH, f"{category}_{session_id}.jsonl")
            files = glob.glob(pattern)
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        count = 0
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                                
                            data = json.loads(line)
                            if data.get('type') == 'header':
                                continue
                                
                            self.data[category].append(data)
                            count += 1
                            
                    print(f"✅ {category}: {count:,} entradas")
                    total_loaded += count
                    
                except Exception as e:
                    print(f"⚠️ Erro lendo {file_path}: {e}")
        
        print(f"📊 Total carregado: {total_loaded:,} entradas")
        return total_loaded > 0
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """📈 Calcular Sharpe Ratio"""
        if len(returns) < 2:
            return 0
        
        # Converter para retornos anualizados (assumindo dados diários)
        excess_returns = np.array(returns) - (risk_free_rate / 252)
        
        if np.std(excess_returns) == 0:
            return 0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=0.02):
        """📉 Calcular Sortino Ratio (apenas downside risk)"""
        if len(returns) < 2:
            return 0
        
        excess_returns = np.array(returns) - (risk_free_rate / 252)
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0 or np.std(negative_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0
        
        return np.mean(excess_returns) / np.std(negative_returns) * np.sqrt(252)
    
    def calculate_max_drawdown(self, portfolio_values):
        """📉 Calcular Maximum Drawdown e duração"""
        if len(portfolio_values) < 2:
            return 0, 0
        
        cumulative = np.array(portfolio_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = abs(np.min(drawdown))
        
        # Duração do drawdown
        in_drawdown = drawdown < -0.01  # 1% threshold
        dd_duration = 0
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                dd_duration = max(dd_duration, current_duration)
            else:
                current_duration = 0
        
        return max_dd, dd_duration
    
    def analyze_performance_metrics(self):
        """💰 Análise completa de métricas de performance"""
        perf_data = self.data['performance']
        if not perf_data:
            print("❌ Nenhum dado de performance encontrado!")
            return {}
        
        print(f"\n💰 ANÁLISE DE PERFORMANCE ({len(perf_data):,} amostras)")
        print("=" * 60)
        
        # Extrair dados
        portfolio_values = [d.get('portfolio_value', 500) for d in perf_data]
        win_rates = [d.get('win_rate', 0) for d in perf_data if d.get('total_trades', 0) > 0]
        drawdowns = [d.get('drawdown', 0) for d in perf_data]
        trade_counts = [d.get('total_trades', 0) for d in perf_data]
        steps = [d.get('step', 0) for d in perf_data]
        
        # Calcular retornos
        returns = []
        if len(portfolio_values) > 1:
            for i in range(1, len(portfolio_values)):
                if portfolio_values[i-1] > 0:
                    ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                    returns.append(ret)
        
        # Métricas principais
        current_value = portfolio_values[-1] if portfolio_values else 500
        initial_value = 500
        total_return = (current_value - initial_value) / initial_value * 100
        
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        max_dd, dd_duration = self.calculate_max_drawdown(portfolio_values)
        
        # Profit Factor
        positive_returns = [r for r in returns if r > 0]
        negative_returns = [r for r in returns if r < 0]
        
        gross_profit = sum(positive_returns) if positive_returns else 0
        gross_loss = abs(sum(negative_returns)) if negative_returns else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Win/Loss Statistics
        win_rate_avg = np.mean(win_rates) if win_rates else 0
        avg_win = np.mean(positive_returns) if positive_returns else 0
        avg_loss = np.mean([abs(r) for r in negative_returns]) if negative_returns else 0
        avg_trade_value = avg_win - avg_loss
        
        # Volatilidade
        portfolio_volatility = np.std(portfolio_values) if len(portfolio_values) > 1 else 0
        returns_volatility = np.std(returns) if len(returns) > 1 else 0
        
        metrics = {
            'portfolio_current': current_value,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd * 100,
            'drawdown_duration': dd_duration,
            'profit_factor': profit_factor,
            'win_rate_avg': win_rate_avg * 100,
            'avg_win_pct': avg_win * 100,
            'avg_loss_pct': avg_loss * 100,
            'avg_trade_value': avg_trade_value * 100,
            'portfolio_volatility': portfolio_volatility,
            'returns_volatility': returns_volatility * 100,
            'total_trades': trade_counts[-1] if trade_counts else 0,
            'trading_frequency': len([t for t in trade_counts if t > 0]) / len(trade_counts) if trade_counts else 0
        }
        
        # Output formatado
        print(f"🎯 Portfolio Atual: ${metrics['portfolio_current']:.2f}")
        print(f"📈 Retorno Total: {metrics['total_return_pct']:+.2f}%")
        print(f"🏆 SHARPE RATIO: {metrics['sharpe_ratio']:.3f}")
        print(f"📊 Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"📉 Max Drawdown: {metrics['max_drawdown']:.2f}% ({metrics['drawdown_duration']} períodos)")
        print(f"💰 Profit Factor: {metrics['profit_factor']:.3f}")
        print(f"🎲 Win Rate: {metrics['win_rate_avg']:.1f}%")
        print(f"💚 Ganho Médio: {metrics['avg_win_pct']:+.3f}%")
        print(f"❌ Perda Média: {metrics['avg_loss_pct']:+.3f}%")
        print(f"⚖️ Valor Médio/Trade: {metrics['avg_trade_value']:+.3f}%")
        print(f"📊 Volatilidade: Portfolio ${metrics['portfolio_volatility']:.2f}, Returns {metrics['returns_volatility']:.2f}%")
        print(f"🔄 Total Trades: {metrics['total_trades']}")
        
        return metrics
    
    def analyze_training_correlation(self):
        """🧠 Análise da correlação entre parâmetros de treino e performance"""
        train_data = self.data['training']
        perf_data = self.data['performance']
        
        if not train_data or not perf_data:
            print("❌ Dados insuficientes para análise de correlação")
            return {}
        
        print(f"\n🧠 ANÁLISE DE CORRELAÇÃO TREINO-PERFORMANCE")
        print("=" * 60)
        
        # Sincronizar dados por step
        train_by_step = {d.get('step', 0): d for d in train_data if d.get('step')}
        perf_by_step = {d.get('step', 0): d for d in perf_data if d.get('step')}
        
        common_steps = set(train_by_step.keys()) & set(perf_by_step.keys())
        if len(common_steps) < 10:
            print("❌ Poucos dados sincronizados para correlação")
            return {}
        
        # Preparar dados para correlação
        data_for_corr = []
        for step in sorted(common_steps):
            train_entry = train_by_step[step]
            perf_entry = perf_by_step[step]
            
            portfolio_value = perf_entry.get('portfolio_value', 500)
            
            data_for_corr.append({
                'step': step,
                'portfolio_value': portfolio_value,
                'portfolio_return': (portfolio_value - 500) / 500 * 100,
                'policy_loss': train_entry.get('policy_loss', 0),
                'value_loss': train_entry.get('value_loss', 0),
                'entropy_loss': train_entry.get('entropy_loss', 0),
                'clip_fraction': train_entry.get('clip_fraction', 0),
                'learning_rate': train_entry.get('learning_rate', 0),
                'explained_variance': train_entry.get('explained_variance', 0),
                'total_loss': train_entry.get('loss', 0),
                'win_rate': perf_entry.get('win_rate', 0),
                'drawdown': perf_entry.get('drawdown', 0)
            })
        
        df = pd.DataFrame(data_for_corr)
        
        # Calcular correlações
        performance_cols = ['portfolio_return', 'win_rate', 'drawdown']
        training_cols = ['policy_loss', 'value_loss', 'entropy_loss', 'clip_fraction', 'learning_rate', 'explained_variance', 'total_loss']
        
        correlations = {}
        for perf_col in performance_cols:
            correlations[perf_col] = {}
            for train_col in training_cols:
                if df[perf_col].std() > 0 and df[train_col].std() > 0:
                    corr = df[perf_col].corr(df[train_col])
                    correlations[perf_col][train_col] = corr
                else:
                    correlations[perf_col][train_col] = 0
        
        # Output das correlações mais importantes
        print("📊 CORRELAÇÕES CRÍTICAS:")
        for perf_metric in correlations:
            print(f"\n{perf_metric.upper()}:")
            sorted_corrs = sorted(correlations[perf_metric].items(), 
                                key=lambda x: abs(x[1]), reverse=True)
            for train_param, corr in sorted_corrs:
                if abs(corr) > 0.1:  # Apenas correlações significativas
                    direction = "📈" if corr > 0 else "📉"
                    strength = "🔥" if abs(corr) > 0.5 else "⚡" if abs(corr) > 0.3 else "💡"
                    print(f"  {strength} {direction} {train_param}: {corr:+.3f}")
        
        return correlations, df
    
    def detect_convergence_patterns(self):
        """🎯 Detectar padrões de convergência e pontos críticos"""
        conv_data = self.data['convergence']
        if not conv_data:
            print("❌ Nenhum dado de convergência encontrado!")
            return {}
        
        print(f"\n🎯 ANÁLISE DE CONVERGÊNCIA ({len(conv_data):,} amostras)")
        print("=" * 60)
        
        # Extrair métricas ao longo do tempo
        steps = [d.get('step', 0) for d in conv_data]
        convergence_scores = [d.get('convergence_score', 0) for d in conv_data if d.get('convergence_score') is not None]
        
        if not convergence_scores:
            print("❌ Nenhum convergence_score encontrado!")
            return {}
        
        # Análise de tendências
        recent_window = min(500, len(convergence_scores) // 4)  # Últimos 25%
        recent_scores = convergence_scores[-recent_window:]
        older_scores = convergence_scores[:recent_window] if len(convergence_scores) > recent_window else convergence_scores
        
        # Estatísticas
        current_avg = np.mean(recent_scores)
        historical_avg = np.mean(older_scores)
        trend_change = current_avg - historical_avg
        
        # Detectar picos e vales
        scores_array = np.array(convergence_scores)
        peaks = []
        valleys = []
        
        window = min(50, len(scores_array) // 10)
        for i in range(window, len(scores_array) - window):
            if scores_array[i] == max(scores_array[i-window:i+window+1]):
                peaks.append((i, scores_array[i]))
            elif scores_array[i] == min(scores_array[i-window:i+window+1]):
                valleys.append((i, scores_array[i]))
        
        # Estabilidade
        stability_window = min(100, len(convergence_scores) // 5)
        recent_stability = np.std(convergence_scores[-stability_window:]) if len(convergence_scores) >= stability_window else np.std(convergence_scores)
        
        results = {
            'current_avg_convergence': current_avg,
            'historical_avg_convergence': historical_avg,
            'convergence_trend': trend_change,
            'convergence_stability': recent_stability,
            'convergence_peaks': len(peaks),
            'convergence_valleys': len(valleys),
            'best_convergence': max(convergence_scores),
            'worst_convergence': min(convergence_scores)
        }
        
        # Output
        trend_emoji = "📈" if trend_change > 0.01 else "📉" if trend_change < -0.01 else "➡️"
        stability_emoji = "🟢" if recent_stability < 0.05 else "🟡" if recent_stability < 0.1 else "🔴"
        
        print(f"📊 Convergência Atual: {current_avg:.4f}")
        print(f"📈 Convergência Histórica: {historical_avg:.4f}")
        print(f"{trend_emoji} Tendência: {trend_change:+.4f}")
        print(f"{stability_emoji} Estabilidade: {recent_stability:.4f}")
        print(f"🎯 Melhor Score: {results['best_convergence']:.4f}")
        print(f"💥 Pior Score: {results['worst_convergence']:.4f}")
        print(f"🏔️ Picos detectados: {results['convergence_peaks']}")
        print(f"🕳️ Vales detectados: {results['convergence_valleys']}")
        
        return results
    
    def generate_optimization_recommendations(self, performance_metrics, correlations=None):
        """🎯 Gerar recomendações específicas para otimização"""
        print(f"\n🎯 RECOMENDAÇÕES PARA SHARPE 1.5+")
        print("=" * 60)
        
        current_sharpe = performance_metrics.get('sharpe_ratio', 0)
        target_sharpe = 1.5
        sharpe_gap = target_sharpe - current_sharpe
        
        print(f"🏆 Sharpe Atual: {current_sharpe:.3f}")
        print(f"🎯 Meta Sharpe: {target_sharpe:.3f}")
        print(f"📏 Gap: {sharpe_gap:+.3f}")
        print()
        
        recommendations = []
        
        # 1. Análise baseada em Sharpe Ratio
        if current_sharpe < 0.5:
            recommendations.append({
                'priority': '🔥 CRÍTICO',
                'area': 'Reward System',
                'issue': 'Sharpe extremamente baixo - modelo não está aprendendo',
                'actions': [
                    'Revisar reward function - pode estar muito penalizando risk',
                    'Verificar se early termination está funcionando corretamente',
                    'Aumentar reward por trades vencedores',
                    'Reduzir penalty por volatilidade'
                ]
            })
        elif current_sharpe < 1.0:
            recommendations.append({
                'priority': '⚡ ALTO',
                'area': 'Risk Management',
                'issue': f'Sharpe {current_sharpe:.3f} - precisa melhor risk/return balance',
                'actions': [
                    'Otimizar position sizing',
                    'Implementar stop-loss mais inteligente',
                    'Ajustar reward por holding positions',
                    'Considerar volatility targeting'
                ]
            })
        
        # 2. Análise baseada em correlações
        if correlations:
            portfolio_corrs = correlations.get('portfolio_return', {})
            
            # Learning Rate
            lr_corr = portfolio_corrs.get('learning_rate', 0)
            if abs(lr_corr) > 0.3:
                if lr_corr < -0.3:
                    recommendations.append({
                        'priority': '⚡ ALTO',
                        'area': 'Learning Rate',
                        'issue': 'LR negativamente correlacionado com performance',
                        'actions': [
                            'Reduzir learning rate inicial',
                            'Implementar learning rate scheduling mais agressivo',
                            'Considerar adaptive learning rate'
                        ]
                    })
                elif lr_corr > 0.3:
                    recommendations.append({
                        'priority': '💡 MÉDIO',
                        'area': 'Learning Rate',
                        'issue': 'LR positivamente correlacionado - pode aumentar',
                        'actions': [
                            'Tentar learning rate ligeiramente maior',
                            'Verificar se não está underfitting'
                        ]
                    })
            
            # Clip Fraction
            clip_corr = portfolio_corrs.get('clip_fraction', 0)
            if abs(clip_corr) > 0.2:
                if clip_corr < -0.2:
                    recommendations.append({
                        'priority': '⚡ ALTO',
                        'area': 'PPO Clipping',
                        'issue': 'Alto clip fraction prejudicando performance',
                        'actions': [
                            'Reduzir clip_range (ex: 0.2 → 0.1)',
                            'Implementar adaptive clipping',
                            'Verificar se updates são muito agressivos'
                        ]
                    })
            
            # Entropy Loss
            entropy_corr = portfolio_corrs.get('entropy_loss', 0)
            if entropy_corr > 0.2:
                recommendations.append({
                    'priority': '💡 MÉDIO',
                    'area': 'Exploration',
                    'issue': 'Mais entropy correlaciona com melhor performance',
                    'actions': [
                        'Aumentar entropy coefficient',
                        'Manter exploration por mais tempo',
                        'Considerar curiosity-driven exploration'
                    ]
                })
            elif entropy_corr < -0.2:
                recommendations.append({
                    'priority': '💡 MÉDIO',
                    'area': 'Exploitation',
                    'issue': 'Menos entropy correlaciona com melhor performance',
                    'actions': [
                        'Reduzir entropy coefficient',
                        'Focar mais em exploitation',
                        'Implementar entropy decay schedule'
                    ]
                })
        
        # 3. Análise baseada em métricas específicas
        max_dd = performance_metrics.get('max_drawdown', 0)
        if max_dd > 20:
            recommendations.append({
                'priority': '🔥 CRÍTICO',
                'area': 'Risk Control',
                'issue': f'Drawdown muito alto ({max_dd:.1f}%)',
                'actions': [
                    'Implementar position sizing baseado em volatility',
                    'Adicionar circuit breakers',
                    'Revisar stop-loss levels',
                    'Considerar Kelly criterion para sizing'
                ]
            })
        
        profit_factor = performance_metrics.get('profit_factor', 0)
        if profit_factor < 1.2:
            recommendations.append({
                'priority': '⚡ ALTO',
                'area': 'Trade Quality',
                'issue': f'Profit factor baixo ({profit_factor:.2f})',
                'actions': [
                    'Melhorar trade selection criteria',
                    'Otimizar exit strategies',
                    'Reduzir overtrading',
                    'Focar em trades de maior qualidade'
                ]
            })
        
        win_rate = performance_metrics.get('win_rate_avg', 0)
        if win_rate < 45:
            recommendations.append({
                'priority': '💡 MÉDIO',
                'area': 'Win Rate',
                'issue': f'Win rate baixo ({win_rate:.1f}%)',
                'actions': [
                    'Revisar entry signals',
                    'Implementar confirmation filters',
                    'Otimizar timing de entries',
                    'Considerar ensemble methods'
                ]
            })
        
        # 4. Recomendações de arquitetura
        if current_sharpe < 0.8:
            recommendations.append({
                'priority': '💡 MÉDIO',
                'area': 'Model Architecture',
                'issue': 'Performance geral baixa - considerar mudanças estruturais',
                'actions': [
                    'Testar diferentes network architectures',
                    'Implementar attention mechanisms',
                    'Considerar ensemble de políticas',
                    'Adicionar market regime detection'
                ]
            })
        
        # Output das recomendações
        for i, rec in enumerate(recommendations, 1):
            print(f"{rec['priority']} {i}. {rec['area'].upper()}")
            print(f"   📋 {rec['issue']}")
            print(f"   🛠️ Ações:")
            for action in rec['actions']:
                print(f"      • {action}")
            print()
        
        # Resumo final
        print("🎯 PRÓXIMOS PASSOS PRIORITÁRIOS:")
        critical_items = [r for r in recommendations if r['priority'] == '🔥 CRÍTICO']
        high_items = [r for r in recommendations if r['priority'] == '⚡ ALTO']
        
        if critical_items:
            print("🔥 CRÍTICO - Resolver IMEDIATAMENTE:")
            for item in critical_items:
                print(f"   • {item['area']}: {item['actions'][0]}")
        
        if high_items:
            print("⚡ ALTA PRIORIDADE - Próximas modificações:")
            for item in high_items:
                print(f"   • {item['area']}: {item['actions'][0]}")
        
        return recommendations
    
    def run_complete_analysis(self, session_id=None):
        """🚀 Executar análise completa"""
        print("🎯 ANÁLISE COMPLETA DE TRADING - GUIA PARA SHARPE 1.5+")
        print("=" * 80)
        
        if not self.load_session_data(session_id):
            return
        
        print(f"📅 Sessão analisada: {self.session_id}")
        print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Análise de performance
        performance_metrics = self.analyze_performance_metrics()
        
        # 2. Análise de convergência
        convergence_results = self.detect_convergence_patterns()
        
        # 3. Análise de correlações
        try:
            correlations, df = self.analyze_training_correlation()
        except:
            correlations = None
            print("⚠️ Análise de correlação falhou - dados insuficientes")
        
        # 4. Recomendações
        recommendations = self.generate_optimization_recommendations(
            performance_metrics, correlations
        )
        
        # 5. Salvar relatório
        self.save_analysis_report(performance_metrics, convergence_results, 
                                recommendations, correlations)
        
        return {
            'performance': performance_metrics,
            'convergence': convergence_results,
            'correlations': correlations,
            'recommendations': recommendations
        }
    
    def save_analysis_report(self, performance, convergence, recommendations, correlations):
        """💾 Salvar relatório completo"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"D:/Projeto/avaliacao/relatorio_completo_{self.session_id}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("🎯 RELATÓRIO COMPLETO DE ANÁLISE DE TRADING\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Sessão: {self.session_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Performance
            f.write("💰 MÉTRICAS DE PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            for key, value in performance.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Convergence
            f.write("🎯 ANÁLISE DE CONVERGÊNCIA\n")
            f.write("-" * 30 + "\n")
            for key, value in convergence.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Recommendations
            f.write("🛠️ RECOMENDAÇÕES DE OTIMIZAÇÃO\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. [{rec['priority']}] {rec['area']}\n")
                f.write(f"   Issue: {rec['issue']}\n")
                f.write(f"   Actions:\n")
                for action in rec['actions']:
                    f.write(f"   - {action}\n")
                f.write("\n")
        
        print(f"💾 Relatório salvo: {filename}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Análise Completa de Trading')
    parser.add_argument('--session', type=str, help='ID da sessão específica')
    
    args = parser.parse_args()
    
    analyzer = TradingPerformanceAnalyzer()
    results = analyzer.run_complete_analysis(args.session)
    
    if results:
        print("\n✅ ANÁLISE COMPLETA FINALIZADA!")
        sharpe = results['performance'].get('sharpe_ratio', 0)
        if sharpe >= 1.5:
            print("🎉 PARABÉNS! Meta de Sharpe 1.5+ ATINGIDA!")
        elif sharpe >= 1.0:
            print("🎯 Sharpe > 1.0 - Caminho certo para 1.5!")
        else:
            print("🔧 Sharpe < 1.0 - Implementar recomendações ASAP!")

if __name__ == "__main__":
    main()
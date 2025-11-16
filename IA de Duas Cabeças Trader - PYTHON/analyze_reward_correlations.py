#!/usr/bin/env python3
"""
ANÁLISE DE CORRELAÇÃO - SISTEMA DE REWARDS V2
Análise profunda das correlações entre componentes de reward
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import json
import glob
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class RewardCorrelationAnalyzer:
    """Analisador de correlações do sistema de rewards"""
    
    def __init__(self):
        self.data = []
        self.analysis_results = {}
        
    def load_historical_data(self):
        """Carregar dados históricos de diferentes fontes"""
        
        # 1. Simular dados baseados no sistema conhecido
        print("📊 Simulando dados baseados no sistema V2...")
        self.simulate_reward_data()
        
        # 2. Analisar estrutura atual do sistema
        self.analyze_reward_system_structure()
        
    def simulate_reward_data(self, n_samples=1000):
        """Simular dados de reward baseados no sistema atual"""
        np.random.seed(42)
        
        # Simular variáveis de entrada
        pnl_values = np.random.normal(0, 15, n_samples)  # PnL média 0, std 15
        position_sizes = np.random.uniform(0.005, 0.02, n_samples)
        durations = np.random.randint(10, 60, n_samples)
        
        # Calcular componentes baseados na lógica V2
        data = []
        
        for i in range(n_samples):
            pnl = pnl_values[i]
            pos_size = position_sizes[i]
            duration = durations[i]
            
            # 1. PnL Component (40% weight)
            pnl_percent = pnl / 1000.0
            base_reward = pnl_percent * 100 * 1.0
            
            if pnl > 0:
                win_bonus = min(0.5, pnl_percent * 50)
                pnl_component = np.clip(base_reward + win_bonus, -2.0, 2.0)
            else:
                loss_penalty = max(-0.5, abs(pnl_percent) * -50)
                pnl_component = np.clip(base_reward + loss_penalty, -2.0, 2.0)
            
            # 2. Risk Management Component (30% weight)
            risk_component = 0.0
            
            # Position sizing bonus
            if 0.005 <= pos_size <= 0.02:
                if pnl > 0:
                    risk_component += 1.5  # full bonus for winning trades
                else:
                    risk_component += 1.5 * 0.2  # reduced for losing trades
            
            # Risk-reward ratio (simulated)
            sl_points = np.random.uniform(3, 8)
            tp_points = np.random.uniform(6, 15)
            rr_ratio = tp_points / sl_points
            
            if 1.5 <= rr_ratio <= 3.0:
                if pnl > 0:
                    risk_component += 2.0
                else:
                    risk_component += 2.0 * 0.1
            elif rr_ratio > 3.0:
                if pnl > 0:
                    risk_component += 1.0
                else:
                    risk_component += 1.0 * 0.05
            
            risk_component = np.clip(risk_component, -2.0, 2.0)
            
            # 3. Consistency Component (20% weight) - simplified
            consistency_component = 0.0
            
            # Simulated win rate bonus
            if np.random.random() > 0.45:  # 55% win rate target
                consistency_component += 1.0
            
            # Simulated sharpe bonus
            if np.random.random() > 0.7:  # 30% chance
                consistency_component += 1.5
            
            consistency_component = np.clip(consistency_component, 0.0, 4.0)
            
            # 4. Timing Component (10% weight)
            timing_component = 0.0
            
            if 10 <= duration <= 60 and pnl > 0:
                timing_component += 0.5
            
            if 20 <= duration <= 40:
                timing_component += 0.3
            
            if pnl > 0:
                timing_component += 0.2
            
            timing_component = np.clip(timing_component, 0.0, 1.0)
            
            # 5. Gaming Penalty
            gaming_penalty = 0.0
            
            # Simulate micro-trade penalty
            if abs(pnl) < 1.0 and np.random.random() > 0.8:
                gaming_penalty -= 2.0
            
            gaming_penalty = np.clip(gaming_penalty, -10.0, 0.0)
            
            # Total reward (with clipping)
            total_reward = pnl_component + risk_component + consistency_component + timing_component + gaming_penalty
            total_reward = np.clip(total_reward, -3.0, 3.0)
            
            # Simulated performance metrics
            policy_loss = np.random.uniform(-0.1, 0.1)
            value_loss = np.random.uniform(0, 5.0)
            explained_variance = np.random.uniform(-0.5, 0.95)
            clip_fraction = np.random.uniform(0, 0.3)
            
            # Phase calculation
            step = i * 10  # Simulate step progression
            if step < 100_000:
                phase = "exploration"
                phase_multiplier = 0.8
            elif step < 500_000:
                phase = "refinement"
                phase_multiplier = 1.0
            else:
                phase = "mastery"
                phase_multiplier = 1.2
            
            data.append({
                'step': step,
                'pnl': pnl,
                'position_size': pos_size,
                'duration': duration,
                'pnl_component': pnl_component,
                'risk_component': risk_component,
                'consistency_component': consistency_component,
                'timing_component': timing_component,
                'gaming_penalty': gaming_penalty,
                'total_reward': total_reward,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'explained_variance': explained_variance,
                'clip_fraction': clip_fraction,
                'training_phase': phase,
                'phase_multiplier': phase_multiplier,
                'sl_points': sl_points,
                'tp_points': tp_points,
                'rr_ratio': rr_ratio
            })
        
        self.data = pd.DataFrame(data)
        print(f"✅ Dados simulados: {len(self.data)} samples")
        
    def analyze_reward_system_structure(self):
        """Analisar estrutura atual do sistema de rewards"""
        print("\n🔍 ANÁLISE DA ESTRUTURA DO SISTEMA V2")
        print("="*60)
        
        # Pesos teóricos do sistema
        theoretical_weights = {
            'PnL': 0.40,        # 40% - reduzido de 70%
            'Risk': 0.30,       # 30% - aumentado de 10%
            'Consistency': 0.20, # 20% - aumentado de 5%
            'Timing': 0.10      # 10% - reduzido de 15%
        }
        
        print("Pesos Teóricos (V2.0):")
        for component, weight in theoretical_weights.items():
            print(f"  {component}: {weight:.0%}")
        
        # Análise dos ranges de clipping
        clipping_ranges = {
            'pnl_component': (-2.0, 2.0),
            'risk_component': (-2.0, 2.0),
            'consistency_component': (0.0, 4.0),
            'timing_component': (0.0, 1.0),
            'gaming_penalty': (-10.0, 0.0),
            'total_reward': (-3.0, 3.0)
        }
        
        print("\nRanges de Clipping:")
        for component, (min_val, max_val) in clipping_ranges.items():
            print(f"  {component}: [{min_val:.1f}, {max_val:.1f}]")
    
    def calculate_correlation_matrix(self):
        """Calcular matriz de correlação entre componentes"""
        print("\n📊 ANÁLISE DE CORRELAÇÕES INTER-COMPONENTES")
        print("="*60)
        
        # Selecionar colunas para correlação
        correlation_cols = [
            'pnl_component', 'risk_component', 'consistency_component', 
            'timing_component', 'gaming_penalty', 'total_reward',
            'policy_loss', 'value_loss', 'explained_variance', 'clip_fraction',
            'pnl', 'position_size', 'duration', 'rr_ratio'
        ]
        
        corr_data = self.data[correlation_cols]
        
        # Correlação de Pearson
        pearson_corr = corr_data.corr()
        
        print("CORRELAÇÕES INTER-COMPONENTES (Pearson):")
        
        # Correlações entre componentes de reward
        reward_components = ['pnl_component', 'risk_component', 'consistency_component', 'timing_component']
        
        print("\n1. CORRELAÇÕES ENTRE COMPONENTES DE REWARD:")
        for i, comp1 in enumerate(reward_components):
            for comp2 in reward_components[i+1:]:
                corr_val = pearson_corr.loc[comp1, comp2]
                strength = self.interpret_correlation_strength(abs(corr_val))
                print(f"  {comp1} vs {comp2}: {corr_val:.3f} ({strength})")
        
        # Correlações com performance
        performance_metrics = ['policy_loss', 'value_loss', 'explained_variance', 'clip_fraction']
        
        print("\n2. CORRELAÇÕES REWARD vs PERFORMANCE:")
        for perf_metric in performance_metrics:
            corr_val = pearson_corr.loc['total_reward', perf_metric]
            strength = self.interpret_correlation_strength(abs(corr_val))
            print(f"  Total Reward vs {perf_metric}: {corr_val:.3f} ({strength})")
        
        # Correlações com inputs originais
        input_metrics = ['pnl', 'position_size', 'duration']
        
        print("\n3. CORRELAÇÕES REWARD vs INPUTS:")
        for input_metric in input_metrics:
            corr_val = pearson_corr.loc['total_reward', input_metric]
            strength = self.interpret_correlation_strength(abs(corr_val))
            print(f"  Total Reward vs {input_metric}: {corr_val:.3f} ({strength})")
        
        self.correlation_matrix = pearson_corr
        return pearson_corr
    
    def analyze_phase_correlations(self):
        """Analisar correlações por fase de treinamento"""
        print("\n🎯 ANÁLISE POR FASE DE TREINAMENTO")
        print("="*60)
        
        phases = self.data['training_phase'].unique()
        
        for phase in phases:
            phase_data = self.data[self.data['training_phase'] == phase]
            print(f"\n--- FASE: {phase.upper()} ---")
            print(f"Samples: {len(phase_data)}")
            
            # Correlação total_reward vs componentes
            components = ['pnl_component', 'risk_component', 'consistency_component', 'timing_component']
            
            for component in components:
                if len(phase_data) > 10:  # Mínimo de samples para correlação válida
                    corr_val = phase_data['total_reward'].corr(phase_data[component])
                    print(f"  Total vs {component}: {corr_val:.3f}")
            
            # Média dos componentes por fase
            print("  Médias dos componentes:")
            for component in components:
                mean_val = phase_data[component].mean()
                print(f"    {component}: {mean_val:.3f}")
    
    def detect_gaming_patterns(self):
        """Detectar padrões de gaming nas correlações"""
        print("\n🚫 DETECÇÃO DE PADRÕES DE GAMING")
        print("="*60)
        
        # 1. Correlação entre gaming penalty e outros componentes
        gaming_corr = self.correlation_matrix['gaming_penalty']
        
        print("Correlações Gaming Penalty:")
        relevant_components = ['pnl_component', 'risk_component', 'total_reward', 'pnl']
        
        for component in relevant_components:
            corr_val = gaming_corr[component]
            if abs(corr_val) > 0.1:  # Threshold para relevância
                strength = self.interpret_correlation_strength(abs(corr_val))
                print(f"  Gaming vs {component}: {corr_val:.3f} ({strength})")
        
        # 2. Identificar trades com suspeita de gaming
        gaming_trades = self.data[self.data['gaming_penalty'] < -0.5]
        
        print(f"\nTrades com Gaming Penalty: {len(gaming_trades)}/{len(self.data)} ({len(gaming_trades)/len(self.data)*100:.1f}%)")
        
        if len(gaming_trades) > 0:
            print("Características dos trades com gaming:")
            print(f"  PnL médio: {gaming_trades['pnl'].mean():.2f}")
            print(f"  PnL std: {gaming_trades['pnl'].std():.2f}")
            print(f"  Position size médio: {gaming_trades['position_size'].mean():.4f}")
            print(f"  Duration médio: {gaming_trades['duration'].mean():.1f}")
        
        # 3. Correlação entre variabilidade baixa e gaming
        pnl_cv = self.data['pnl'].std() / abs(self.data['pnl'].mean()) if self.data['pnl'].mean() != 0 else float('inf')
        print(f"\nCoeficiente de Variação PnL: {pnl_cv:.2f}")
        
        if pnl_cv < 1.0:
            print("⚠️  ALERTA: Baixa variabilidade em PnL pode indicar gaming")
    
    def analyze_clipping_effects(self):
        """Analisar efeitos do clipping nos componentes"""
        print("\n✂️ ANÁLISE DE EFEITOS DO CLIPPING")
        print("="*60)
        
        clipping_analysis = {}
        
        components_ranges = {
            'pnl_component': (-2.0, 2.0),
            'risk_component': (-2.0, 2.0),
            'consistency_component': (0.0, 4.0),
            'timing_component': (0.0, 1.0),
            'total_reward': (-3.0, 3.0)
        }
        
        for component, (min_clip, max_clip) in components_ranges.items():
            if component in self.data.columns:
                values = self.data[component]
                
                # Contar valores clipped
                clipped_min = (values <= min_clip).sum()
                clipped_max = (values >= max_clip).sum()
                total_clipped = clipped_min + clipped_max
                
                clipping_rate = total_clipped / len(values) * 100
                
                print(f"{component}:")
                print(f"  Range: [{min_clip:.1f}, {max_clip:.1f}]")
                print(f"  Clipped Min: {clipped_min} ({clipped_min/len(values)*100:.1f}%)")
                print(f"  Clipped Max: {clipped_max} ({clipped_max/len(values)*100:.1f}%)")
                print(f"  Total Clipped: {total_clipped} ({clipping_rate:.1f}%)")
                print(f"  Mean: {values.mean():.3f}, Std: {values.std():.3f}")
                
                clipping_analysis[component] = {
                    'clipping_rate': clipping_rate,
                    'mean': values.mean(),
                    'std': values.std(),
                    'clipped_min': clipped_min,
                    'clipped_max': clipped_max
                }
                print()
        
        # Identificar componentes com clipping excessivo
        high_clipping = [comp for comp, data in clipping_analysis.items() 
                        if data['clipping_rate'] > 10.0]
        
        if high_clipping:
            print("⚠️  COMPONENTES COM CLIPPING EXCESSIVO (>10%):")
            for comp in high_clipping:
                rate = clipping_analysis[comp]['clipping_rate']
                print(f"  {comp}: {rate:.1f}%")
    
    def calculate_reward_efficiency(self):
        """Calcular eficiência do sistema de rewards"""
        print("\n⚡ ANÁLISE DE EFICIÊNCIA DO SISTEMA")
        print("="*60)
        
        # 1. Variância explicada por componentes
        total_var = self.data['total_reward'].var()
        
        components = ['pnl_component', 'risk_component', 'consistency_component', 'timing_component']
        component_vars = {}
        
        print("Contribuição de Variância por Componente:")
        for component in components:
            if component in self.data.columns:
                comp_var = self.data[component].var()
                var_contribution = comp_var / total_var * 100
                component_vars[component] = var_contribution
                print(f"  {component}: {var_contribution:.1f}%")
        
        # 2. Signal-to-noise ratio
        signal = abs(self.data['total_reward'].mean())
        noise = self.data['total_reward'].std()
        snr = signal / noise if noise > 0 else 0
        
        print(f"\nSignal-to-Noise Ratio: {snr:.3f}")
        
        # 3. Correlação entre reward e performance real
        explained_var_corr = self.data['total_reward'].corr(self.data['explained_variance'])
        
        print(f"Correlação Reward vs Explained Variance: {explained_var_corr:.3f}")
        
        # 4. Detectar reward inversion (rewards negativos para trades positivos)
        positive_trades = self.data[self.data['pnl'] > 0]
        negative_rewards_on_positive_trades = (positive_trades['total_reward'] < 0).sum()
        
        negative_trades = self.data[self.data['pnl'] < 0]
        positive_rewards_on_negative_trades = (negative_trades['total_reward'] > 0).sum()
        
        print(f"\nReward Inversion Analysis:")
        print(f"  Trades positivos com reward negativo: {negative_rewards_on_positive_trades}/{len(positive_trades)} ({negative_rewards_on_positive_trades/len(positive_trades)*100:.1f}%)")
        print(f"  Trades negativos com reward positivo: {positive_rewards_on_negative_trades}/{len(negative_trades)} ({positive_rewards_on_negative_trades/len(negative_trades)*100:.1f}%)")
    
    def interpret_correlation_strength(self, abs_corr):
        """Interpretar força da correlação"""
        if abs_corr >= 0.7:
            return "FORTE"
        elif abs_corr >= 0.4:
            return "MODERADA"
        elif abs_corr >= 0.2:
            return "FRACA"
        else:
            return "NEGLIGÍVEL"
    
    def generate_recommendations(self):
        """Gerar recomendações baseadas na análise"""
        print("\n💡 RECOMENDAÇÕES BASEADAS NA ANÁLISE")
        print("="*60)
        
        recommendations = []
        
        # Análise do clipping
        for component in ['pnl_component', 'risk_component', 'total_reward']:
            if component in self.data.columns:
                values = self.data[component]
                if component == 'pnl_component':
                    clipped = ((values <= -2.0) | (values >= 2.0)).sum()
                elif component == 'risk_component':
                    clipped = ((values <= -2.0) | (values >= 2.0)).sum()
                else:  # total_reward
                    clipped = ((values <= -3.0) | (values >= 3.0)).sum()
                
                clipping_rate = clipped / len(values) * 100
                
                if clipping_rate > 15:
                    recommendations.append(f"CRÍTICO: {component} tem {clipping_rate:.1f}% de clipping - ajustar ranges")
        
        # Análise de correlação
        if hasattr(self, 'correlation_matrix'):
            # Correlação PnL vs Risk muito alta pode indicar problema
            pnl_risk_corr = abs(self.correlation_matrix.loc['pnl_component', 'risk_component'])
            if pnl_risk_corr > 0.8:
                recommendations.append(f"ATENÇÃO: Correlação PnL-Risk muito alta ({pnl_risk_corr:.3f}) - componentes podem estar redundantes")
            
            # Correlação total_reward vs explained_variance baixa é problemática
            reward_performance_corr = abs(self.correlation_matrix.loc['total_reward', 'explained_variance'])
            if reward_performance_corr < 0.3:
                recommendations.append(f"PROBLEMA: Correlação Reward-Performance baixa ({reward_performance_corr:.3f}) - reward não está guiando aprendizado")
        
        # Análise de gaming
        gaming_trades = (self.data['gaming_penalty'] < -0.5).sum()
        gaming_rate = gaming_trades / len(self.data) * 100
        
        if gaming_rate > 20:
            recommendations.append(f"ALERTA: Taxa de gaming alta ({gaming_rate:.1f}%) - fortalecer anti-gaming")
        
        # Output das recomendações
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("✅ Sistema aparenta estar bem balanceado!")
    
    def run_full_analysis(self):
        """Executar análise completa"""
        print("🎯 ANÁLISE COMPLETA DE CORRELAÇÃO - SISTEMA DE REWARDS V2")
        print("="*80)
        
        self.load_historical_data()
        self.calculate_correlation_matrix()
        self.analyze_phase_correlations()
        self.detect_gaming_patterns()
        self.analyze_clipping_effects()
        self.calculate_reward_efficiency()
        self.generate_recommendations()
        
        print("\n" + "="*80)
        print("✅ ANÁLISE COMPLETA FINALIZADA")

if __name__ == '__main__':
    analyzer = RewardCorrelationAnalyzer()
    analyzer.run_full_analysis()
#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO COMPLETO DE BLOQUEIO DE TRADES
Analisa cada ponto do pipeline onde trades podem ser bloqueados
"""

import numpy as np
import pandas as pd
import torch
import sys
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Adicionar paths do projeto
sys.path.append("Modelo PPO Trader")
sys.path.append(".")

# Imports do sistema
from daytrader import TradingEnv, TRADING_CONFIG, TRIAL_2_TRADING_PARAMS
from trading_framework.policies.two_head_v7_intuition import TwoHeadV7Intuition, get_v7_intuition_kwargs
from trading_framework.policies.two_head_v7_simple import SpecializedEntryHead
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

class TradeDiagnosticAnalyzer:
    """🔍 Analisador completo de diagnóstico de trades"""
    
    def __init__(self, df_path="dados/ES_5min_processado.csv"):
        self.df_path = df_path
        self.diagnostics = {
            'action_analysis': {},
            'threshold_analysis': {},
            'gate_analysis': {},
            'env_constraints': {},
            'execution_flow': {},
            'statistics': {}
        }
        
        # Contadores para análise
        self.action_counter = Counter()
        self.threshold_crossings = []
        self.gate_values = []
        self.execution_attempts = []
        
        print("🔍 [DIAGNOSTIC] Inicializando analisador de diagnóstico...")
        
    def load_data_and_env(self):
        """Carregar dados e criar ambiente"""
        print("📊 [DATA] Carregando dados...")
        
        if os.path.exists(self.df_path):
            df = pd.read_csv(self.df_path, index_col=0, parse_dates=True)
        else:
            print(f"❌ Arquivo não encontrado: {self.df_path}")
            # Criar dados mock para teste
            dates = pd.date_range('2023-01-01', periods=1000, freq='5min')
            df = pd.DataFrame({
                'close_5m': 4000 + np.cumsum(np.random.randn(1000) * 0.1),
                'high_5m': 4000 + np.cumsum(np.random.randn(1000) * 0.1) + 2,
                'low_5m': 4000 + np.cumsum(np.random.randn(1000) * 0.1) - 2,
                'volume_5m': np.random.randint(1000, 10000, 1000),
            }, index=dates)
            print("🔧 [MOCK] Usando dados mock para diagnóstico")
        
        print(f"✅ [DATA] Dados carregados: {len(df)} barras")
        
        # Criar ambiente
        self.env = TradingEnv(
            df=df,
            window_size=20,
            is_training=True,
            initial_balance=TRADING_CONFIG["portfolio_inicial"],
            trading_params=TRIAL_2_TRADING_PARAMS
        )
        
        print(f"✅ [ENV] Ambiente criado com {len(df)} barras")
        return df
    
    def create_mock_model(self):
        """Criar modelo mock para análise"""
        print("🤖 [MODEL] Criando modelo mock V7 Intuition...")
        
        # Criar ambiente vectorizado
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Configurar V7 Intuition
        v7_kwargs = get_v7_intuition_kwargs()
        v7_kwargs['features_extractor_kwargs']['window_size'] = 20
        
        # Criar modelo RecurrentPPO com V7 Intuition
        model = RecurrentPPO(
            TwoHeadV7Intuition,
            vec_env,
            policy_kwargs=v7_kwargs,
            verbose=0,
            tensorboard_log=None
        )
        
        print("✅ [MODEL] Modelo V7 Intuition criado")
        return model
    
    def analyze_action_space_distribution(self, model, num_samples=1000):
        """Analisar distribuição do action space"""
        print(f"🎯 [ACTIONS] Analisando {num_samples} ações...")
        
        obs = self.env.reset()
        actions_raw = []
        actions_processed = []
        
        for i in range(num_samples):
            # Gerar ação usando o modelo
            action, _states = model.predict(obs, deterministic=False)
            actions_raw.append(action.copy())
            
            # Processar ação conforme daytrader.py
            raw_decision = float(action[0])
            if raw_decision < 0.5:
                entry_decision = 0  # HOLD
            elif raw_decision < 1.5:
                entry_decision = 1  # LONG
            else:
                entry_decision = 2  # SHORT
                
            actions_processed.append({
                'raw_decision': raw_decision,
                'entry_decision': entry_decision,
                'entry_confidence': float(action[1]),
                'temporal_signal': float(action[2]),
                'risk_appetite': float(action[3]),
                'market_regime_bias': float(action[4]),
                'full_action': action.copy()
            })
            
            self.action_counter[entry_decision] += 1
            
            # Step no ambiente para próxima observação
            try:
                obs, reward, done, info = self.env.step(action)
                if done:
                    obs = self.env.reset()
            except Exception as e:
                print(f"⚠️ Erro no step {i}: {e}")
                obs = self.env.reset()
        
        # Análise estatística
        actions_df = pd.DataFrame(actions_processed)
        
        self.diagnostics['action_analysis'] = {
            'total_samples': num_samples,
            'hold_count': self.action_counter[0],
            'long_count': self.action_counter[1], 
            'short_count': self.action_counter[2],
            'hold_percentage': (self.action_counter[0] / num_samples) * 100,
            'long_percentage': (self.action_counter[1] / num_samples) * 100,
            'short_percentage': (self.action_counter[2] / num_samples) * 100,
            'raw_decision_stats': {
                'mean': actions_df['raw_decision'].mean(),
                'std': actions_df['raw_decision'].std(),
                'min': actions_df['raw_decision'].min(),
                'max': actions_df['raw_decision'].max(),
                'percentiles': {
                    '25%': actions_df['raw_decision'].quantile(0.25),
                    '50%': actions_df['raw_decision'].quantile(0.50),
                    '75%': actions_df['raw_decision'].quantile(0.75)
                }
            },
            'confidence_stats': {
                'mean': actions_df['entry_confidence'].mean(),
                'std': actions_df['entry_confidence'].std(),
                'min': actions_df['entry_confidence'].min(),
                'max': actions_df['entry_confidence'].max()
            }
        }
        
        return actions_df
    
    def analyze_v7_gates(self, model, num_samples=500):
        """Analisar gates internos da V7 Intuition"""
        print(f"🧠 [V7 GATES] Analisando gates internos da V7...")
        
        # Hook para capturar gates
        gate_data = []
        
        def capture_gates_hook(module, input, output):
            if hasattr(output, '__len__') and len(output) >= 3:
                # output = (decision, confidence, gate_info)
                if len(output) >= 3 and isinstance(output[2], dict):
                    gate_info = output[2]
                    gate_data.append({
                        'composite_score': gate_info.get('composite_score', 0),
                        'final_gate': gate_info.get('final_gate', 0),
                        'temporal_gate': gate_info.get('temporal_gate', 0),
                        'validation_gate': gate_info.get('validation_gate', 0),
                        'risk_gate': gate_info.get('risk_gate', 0),
                        'market_gate': gate_info.get('market_gate', 0),
                        'quality_gate': gate_info.get('quality_gate', 0),
                        'confidence_gate': gate_info.get('confidence_gate', 0),
                        'scores': gate_info.get('scores', {})
                    })
        
        # Registrar hook na entry_head se disponível
        hook_handle = None
        if hasattr(model.policy, 'entry_head'):
            hook_handle = model.policy.entry_head.register_forward_hook(capture_gates_hook)
            print("✅ [HOOK] Hook registrado na entry_head")
        
        # Gerar samples
        obs = self.env.reset()
        for i in range(num_samples):
            try:
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, done, info = self.env.step(action)
                if done:
                    obs = self.env.reset()
            except Exception as e:
                print(f"⚠️ Erro na análise V7 step {i}: {e}")
                obs = self.env.reset()
        
        # Remover hook
        if hook_handle:
            hook_handle.remove()
        
        # Análise dos gates
        if gate_data:
            gates_df = pd.DataFrame(gate_data)
            
            # Extrair tensors para valores numéricos
            for col in gates_df.columns:
                if col != 'scores':
                    gates_df[col] = gates_df[col].apply(
                        lambda x: float(x.item()) if hasattr(x, 'item') else float(x)
                    )
            
            self.diagnostics['gate_analysis'] = {
                'samples_captured': len(gate_data),
                'composite_score_stats': {
                    'mean': gates_df['composite_score'].mean(),
                    'std': gates_df['composite_score'].std(),
                    'min': gates_df['composite_score'].min(),
                    'max': gates_df['composite_score'].max(),
                    'above_threshold_50': (gates_df['composite_score'] > 0.5).sum(),
                    'above_threshold_50_pct': (gates_df['composite_score'] > 0.5).mean() * 100,
                    'above_threshold_60': (gates_df['composite_score'] > 0.6).sum(),
                    'above_threshold_60_pct': (gates_df['composite_score'] > 0.6).mean() * 100,
                },
                'final_gate_stats': {
                    'mean': gates_df['final_gate'].mean(),
                    'passed_count': (gates_df['final_gate'] > 0.5).sum(),
                    'passed_percentage': (gates_df['final_gate'] > 0.5).mean() * 100,
                    'blocked_percentage': (gates_df['final_gate'] <= 0.5).mean() * 100
                },
                'individual_gates': {
                    gate: {
                        'mean': gates_df[gate].mean(),
                        'std': gates_df[gate].std(),
                        'pass_rate': (gates_df[gate] > 0.5).mean() * 100
                    }
                    for gate in ['temporal_gate', 'validation_gate', 'risk_gate', 
                                'market_gate', 'quality_gate', 'confidence_gate']
                    if gate in gates_df.columns
                }
            }
            
            return gates_df
        else:
            print("❌ [V7 GATES] Nenhum gate capturado")
            return pd.DataFrame()
    
    def analyze_environment_constraints(self, num_samples=200):
        """Analisar constraints do ambiente"""
        print(f"🏭 [ENV] Analisando constraints do ambiente...")
        
        obs = self.env.reset()
        constraint_data = []
        
        for i in range(num_samples):
            # Estado atual
            current_state = {
                'step': i,
                'current_positions': len(self.env.positions),
                'max_positions': self.env.max_positions,
                'portfolio_value': self.env.portfolio_value,
                'positions_full': len(self.env.positions) >= self.env.max_positions,
                'positions_list': [pos['type'] for pos in self.env.positions]
            }
            
            # Simular diferentes tipos de entrada
            for entry_type in [1, 2]:  # Long, Short
                can_enter = len(self.env.positions) < self.env.max_positions
                constraint_data.append({
                    'step': i,
                    'entry_type': 'long' if entry_type == 1 else 'short',
                    'can_enter': can_enter,
                    'positions_count': len(self.env.positions),
                    'max_positions': self.env.max_positions,
                    'portfolio_value': self.env.portfolio_value,
                    'constraint_reason': 'max_positions' if not can_enter else 'none'
                })
            
            # Step com ação neutra
            try:
                neutral_action = np.array([0.3, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                obs, reward, done, info = self.env.step(neutral_action)
                if done:
                    obs = self.env.reset()
            except Exception as e:
                print(f"⚠️ Erro constraint analysis step {i}: {e}")
                obs = self.env.reset()
        
        # Análise
        constraints_df = pd.DataFrame(constraint_data)
        
        self.diagnostics['env_constraints'] = {
            'max_positions_limit': self.env.max_positions,
            'positions_blocking_rate': (constraints_df['can_enter'] == False).mean() * 100,
            'average_positions': constraints_df['positions_count'].mean(),
            'max_positions_reached': (constraints_df['positions_count'] >= self.env.max_positions).sum(),
            'constraint_breakdown': constraints_df['constraint_reason'].value_counts().to_dict()
        }
        
        return constraints_df
    
    def analyze_full_execution_flow(self, model, num_samples=300):
        """Analisar fluxo completo de execução"""
        print(f"🔄 [FLOW] Analisando fluxo completo de execução...")
        
        obs = self.env.reset()
        flow_data = []
        
        for i in range(num_samples):
            try:
                # 1. Gerar ação
                action, _states = model.predict(obs, deterministic=False)
                
                # 2. Processar decisão
                raw_decision = float(action[0])
                if raw_decision < 0.5:
                    entry_decision = 0  # HOLD
                elif raw_decision < 1.5:
                    entry_decision = 1  # LONG
                else:
                    entry_decision = 2  # SHORT
                
                # 3. Verificar constraints
                positions_before = len(self.env.positions)
                can_enter_positions = positions_before < self.env.max_positions
                
                # 4. Executar step
                obs, reward, done, info = self.env.step(action)
                positions_after = len(self.env.positions)
                
                # 5. Verificar se trade foi executado
                trade_executed = positions_after > positions_before
                
                flow_data.append({
                    'step': i,
                    'raw_decision': raw_decision,
                    'entry_decision': entry_decision,
                    'entry_confidence': float(action[1]),
                    'wanted_to_trade': entry_decision > 0,
                    'positions_before': positions_before,
                    'positions_after': positions_after,
                    'can_enter_positions': can_enter_positions,
                    'trade_executed': trade_executed,
                    'blocked_by_positions': (entry_decision > 0) and not can_enter_positions,
                    'blocked_by_other': (entry_decision > 0) and can_enter_positions and not trade_executed,
                    'portfolio_value': self.env.portfolio_value,
                    'reward': float(reward)
                })
                
                if done:
                    obs = self.env.reset()
                    
            except Exception as e:
                print(f"⚠️ Erro flow analysis step {i}: {e}")
                obs = self.env.reset()
        
        # Análise do fluxo
        flow_df = pd.DataFrame(flow_data)
        
        if len(flow_df) > 0:
            total_wanted_trades = flow_df['wanted_to_trade'].sum()
            total_executed_trades = flow_df['trade_executed'].sum()
            blocked_by_positions = flow_df['blocked_by_positions'].sum()
            blocked_by_other = flow_df['blocked_by_other'].sum()
            
            self.diagnostics['execution_flow'] = {
                'total_samples': len(flow_df),
                'wanted_trades': int(total_wanted_trades),
                'executed_trades': int(total_executed_trades),
                'blocked_by_positions': int(blocked_by_positions),
                'blocked_by_other': int(blocked_by_other),
                'execution_rate': (total_executed_trades / max(total_wanted_trades, 1)) * 100,
                'position_blocking_rate': (blocked_by_positions / max(total_wanted_trades, 1)) * 100,
                'other_blocking_rate': (blocked_by_other / max(total_wanted_trades, 1)) * 100,
                'hold_rate': (flow_df['entry_decision'] == 0).mean() * 100,
                'long_attempt_rate': (flow_df['entry_decision'] == 1).mean() * 100,
                'short_attempt_rate': (flow_df['entry_decision'] == 2).mean() * 100
            }
        
        return flow_df
    
    def generate_comprehensive_report(self):
        """Gerar relatório completo"""
        print("\n" + "="*80)
        print("📋 RELATÓRIO COMPLETO DE DIAGNÓSTICO DE TRADES")
        print("="*80)
        
        # 1. Análise de Ações
        if 'action_analysis' in self.diagnostics:
            action_data = self.diagnostics['action_analysis']
            print(f"\n🎯 ANÁLISE DO ACTION SPACE:")
            print(f"  Total de amostras: {action_data['total_samples']}")
            print(f"  Hold: {action_data['hold_count']} ({action_data['hold_percentage']:.1f}%)")
            print(f"  Long: {action_data['long_count']} ({action_data['long_percentage']:.1f}%)")
            print(f"  Short: {action_data['short_count']} ({action_data['short_percentage']:.1f}%)")
            print(f"  Raw Decision Stats: μ={action_data['raw_decision_stats']['mean']:.3f}, σ={action_data['raw_decision_stats']['std']:.3f}")
            print(f"  Confidence Stats: μ={action_data['confidence_stats']['mean']:.3f}, σ={action_data['confidence_stats']['std']:.3f}")
        
        # 2. Análise de Gates V7
        if 'gate_analysis' in self.diagnostics:
            gate_data = self.diagnostics['gate_analysis']
            print(f"\n🧠 ANÁLISE DOS GATES V7 INTUITION:")
            print(f"  Samples capturados: {gate_data['samples_captured']}")
            if gate_data['samples_captured'] > 0:
                comp_stats = gate_data['composite_score_stats']
                print(f"  Composite Score: μ={comp_stats['mean']:.3f}, σ={comp_stats['std']:.3f}")
                print(f"  Acima threshold 0.5: {comp_stats['above_threshold_50']} ({comp_stats['above_threshold_50_pct']:.1f}%)")
                print(f"  Acima threshold 0.6: {comp_stats['above_threshold_60']} ({comp_stats['above_threshold_60_pct']:.1f}%)")
                
                gate_stats = gate_data['final_gate_stats']
                print(f"  Final Gate Pass Rate: {gate_stats['passed_percentage']:.1f}%")
                print(f"  Final Gate Block Rate: {gate_stats['blocked_percentage']:.1f}%")
                
                if 'individual_gates' in gate_data:
                    print(f"  Gates Individuais:")
                    for gate_name, stats in gate_data['individual_gates'].items():
                        print(f"    {gate_name}: Pass Rate {stats['pass_rate']:.1f}%")
        
        # 3. Análise de Constraints do Ambiente
        if 'env_constraints' in self.diagnostics:
            env_data = self.diagnostics['env_constraints']
            print(f"\n🏭 ANÁLISE DE CONSTRAINTS DO AMBIENTE:")
            print(f"  Max Positions: {env_data['max_positions_limit']}")
            print(f"  Positions Blocking Rate: {env_data['positions_blocking_rate']:.1f}%")
            print(f"  Average Positions: {env_data['average_positions']:.1f}")
            print(f"  Max Positions Reached: {env_data['max_positions_reached']} vezes")
        
        # 4. Análise do Fluxo de Execução
        if 'execution_flow' in self.diagnostics:
            flow_data = self.diagnostics['execution_flow']
            print(f"\n🔄 ANÁLISE DO FLUXO DE EXECUÇÃO:")
            print(f"  Total de samples: {flow_data['total_samples']}")
            print(f"  Queria fazer trades: {flow_data['wanted_trades']}")
            print(f"  Trades executados: {flow_data['executed_trades']}")
            print(f"  Taxa de execução: {flow_data['execution_rate']:.1f}%")
            print(f"  Bloqueado por posições: {flow_data['blocked_by_positions']} ({flow_data['position_blocking_rate']:.1f}%)")
            print(f"  Bloqueado por outros: {flow_data['blocked_by_other']} ({flow_data['other_blocking_rate']:.1f}%)")
            print(f"  Hold Rate: {flow_data['hold_rate']:.1f}%")
            print(f"  Long Attempt Rate: {flow_data['long_attempt_rate']:.1f}%")
            print(f"  Short Attempt Rate: {flow_data['short_attempt_rate']:.1f}%")
        
        # 5. Análise Consolidada
        print(f"\n🎯 DIAGNÓSTICO CONSOLIDADO:")
        
        if 'execution_flow' in self.diagnostics and 'action_analysis' in self.diagnostics:
            flow = self.diagnostics['execution_flow']
            actions = self.diagnostics['action_analysis']
            
            trade_intention_rate = actions['long_percentage'] + actions['short_percentage']
            actual_execution_rate = flow['execution_rate'] if flow['wanted_trades'] > 0 else 0
            
            print(f"  🎲 Intenção de Trade: {trade_intention_rate:.1f}%")
            print(f"  ✅ Taxa de Execução Real: {actual_execution_rate:.1f}%")
            print(f"  🚫 Taxa de Bloqueio Total: {100 - actual_execution_rate:.1f}%")
            
            # Identificar principais bloqueadores
            blockers = []
            if 'gate_analysis' in self.diagnostics and self.diagnostics['gate_analysis']['samples_captured'] > 0:
                gate_block_rate = self.diagnostics['gate_analysis']['final_gate_stats']['blocked_percentage']
                blockers.append(f"V7 Gates: {gate_block_rate:.1f}%")
            
            if flow['position_blocking_rate'] > 0:
                blockers.append(f"Max Positions: {flow['position_blocking_rate']:.1f}%")
            
            if flow['other_blocking_rate'] > 0:
                blockers.append(f"Outros: {flow['other_blocking_rate']:.1f}%")
            
            if blockers:
                print(f"  🔍 Principais Bloqueadores:")
                for blocker in blockers:
                    print(f"    - {blocker}")
        
        print("\n" + "="*80)
        
        return self.diagnostics
    
    def save_diagnostics_to_file(self, filename="diagnostico_trades_report.json"):
        """Salvar diagnósticos em arquivo"""
        try:
            # Converter tensors para valores serializáveis
            def convert_tensors(obj):
                if hasattr(obj, 'item'):
                    return float(obj.item())
                elif isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().numpy().tolist()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_tensors(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_tensors(item) for item in obj]
                else:
                    return obj
            
            clean_diagnostics = convert_tensors(self.diagnostics)
            clean_diagnostics['timestamp'] = datetime.now().isoformat()
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(clean_diagnostics, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Diagnósticos salvos em: {filename}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar diagnósticos: {e}")

def run_complete_diagnosis():
    """Executar diagnóstico completo"""
    print("🚀 INICIANDO DIAGNÓSTICO COMPLETO DE TRADES")
    print("="*80)
    
    analyzer = TradeDiagnosticAnalyzer()
    
    try:
        # 1. Carregar dados e ambiente
        df = analyzer.load_data_and_env()
        
        # 2. Criar modelo
        model = analyzer.create_mock_model()
        
        # 3. Análise do action space
        actions_df = analyzer.analyze_action_space_distribution(model, num_samples=1000)
        
        # 4. Análise dos gates V7
        gates_df = analyzer.analyze_v7_gates(model, num_samples=500)
        
        # 5. Análise de constraints do ambiente
        constraints_df = analyzer.analyze_environment_constraints(num_samples=200)
        
        # 6. Análise do fluxo completo
        flow_df = analyzer.analyze_full_execution_flow(model, num_samples=500)
        
        # 7. Gerar relatório
        diagnostics = analyzer.generate_comprehensive_report()
        
        # 8. Salvar diagnósticos
        analyzer.save_diagnostics_to_file()
        
        return analyzer, diagnostics
        
    except Exception as e:
        print(f"❌ ERRO NO DIAGNÓSTICO: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    analyzer, diagnostics = run_complete_diagnosis()
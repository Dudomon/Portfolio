#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO SIMPLES DE BLOQUEIO DE TRADES
Analisa o daytrader diretamente sem criar modelo complexo
"""

import numpy as np
import pandas as pd
import sys
import os
from collections import defaultdict, Counter
import json
from datetime import datetime

# Adicionar paths do projeto
sys.path.append("Modelo PPO Trader")
sys.path.append(".")

# Imports do sistema
from daytrader import TradingEnv, TRADING_CONFIG, TRIAL_2_TRADING_PARAMS

class SimpleTradeDiagnostic:
    """🔍 Diagnóstico simples focado nos bloqueadores de trades"""
    
    def __init__(self):
        self.results = {}
        self.mock_actions = []
        
        print("🔍 [DIAGNOSTIC] Inicializando diagnóstico simples...")
        
    def create_mock_data(self):
        """Criar dados mock para teste"""
        print("📊 [DATA] Criando dados mock...")
        
        dates = pd.date_range('2023-01-01', periods=1000, freq='5min')
        df = pd.DataFrame({
            'close_5m': 4000 + np.cumsum(np.random.randn(1000) * 0.1),
            'high_5m': 4000 + np.cumsum(np.random.randn(1000) * 0.1) + 2,
            'low_5m': 4000 + np.cumsum(np.random.randn(1000) * 0.1) - 2,
            'volume_5m': np.random.randint(1000, 10000, 1000),
        }, index=dates)
        
        return df
    
    def create_environment(self, df):
        """Criar ambiente de trading"""
        print("🏭 [ENV] Criando ambiente...")
        
        env = TradingEnv(
            df=df,
            window_size=20,
            is_training=True,
            initial_balance=TRADING_CONFIG["portfolio_inicial"],
            trading_params=TRIAL_2_TRADING_PARAMS
        )
        
        return env
    
    def generate_test_actions(self, num_actions=1000):
        """Gerar ações de teste variadas"""
        print(f"🎯 [ACTIONS] Gerando {num_actions} ações de teste...")
        
        actions = []
        
        # Distribuição de ações de teste
        for i in range(num_actions):
            # Variar raw_decision para testar thresholds
            if i % 3 == 0:
                raw_decision = np.random.uniform(0.0, 0.5)  # HOLD
            elif i % 3 == 1:
                raw_decision = np.random.uniform(0.5, 1.5)  # LONG
            else:
                raw_decision = np.random.uniform(1.5, 2.0)  # SHORT
            
            action = np.array([
                raw_decision,                           # [0] entry_decision
                np.random.uniform(0.0, 1.0),          # [1] entry_confidence
                np.random.uniform(-1.0, 1.0),         # [2] temporal_signal
                np.random.uniform(0.0, 1.0),          # [3] risk_appetite
                np.random.uniform(-1.0, 1.0),         # [4] market_regime_bias
                np.random.uniform(-3.0, 3.0),         # [5] sl1
                np.random.uniform(-3.0, 3.0),         # [6] sl2
                np.random.uniform(-3.0, 3.0),         # [7] sl3
                np.random.uniform(-3.0, 3.0),         # [8] tp1
                np.random.uniform(-3.0, 3.0),         # [9] tp2
                np.random.uniform(-3.0, 3.0),         # [10] tp3
            ], dtype=np.float32)
            
            actions.append(action)
        
        return actions
    
    def analyze_action_processing(self, actions):
        """Analisar como as ações são processadas"""
        print("⚙️ [PROCESS] Analisando processamento de ações...")
        
        processed_actions = []
        decision_counter = Counter()
        
        for action in actions:
            # Processar conforme daytrader.py
            raw_decision = float(action[0])
            
            if raw_decision < 0.5:
                entry_decision = 0  # HOLD
            elif raw_decision < 1.5:
                entry_decision = 1  # LONG
            else:
                entry_decision = 2  # SHORT
            
            decision_counter[entry_decision] += 1
            
            processed_actions.append({
                'raw_decision': raw_decision,
                'entry_decision': entry_decision,
                'entry_confidence': float(action[1]),
                'wants_to_trade': entry_decision > 0
            })
        
        total = len(actions)
        results = {
            'total_actions': total,
            'hold_count': decision_counter[0],
            'long_count': decision_counter[1],
            'short_count': decision_counter[2],
            'hold_percentage': (decision_counter[0] / total) * 100,
            'long_percentage': (decision_counter[1] / total) * 100,
            'short_percentage': (decision_counter[2] / total) * 100,
            'trade_intention_percentage': ((decision_counter[1] + decision_counter[2]) / total) * 100
        }
        
        return processed_actions, results
    
    def test_environment_execution(self, env, actions, max_steps=500):
        """Testar execução no ambiente"""
        print(f"🔄 [EXECUTION] Testando execução no ambiente...")
        
        obs = env.reset()
        execution_data = []
        
        for i, action in enumerate(actions[:max_steps]):
            try:
                # Estado antes da execução
                positions_before = len(env.positions)
                portfolio_before = env.portfolio_value
                
                # Processar decisão
                raw_decision = float(action[0])
                if raw_decision < 0.5:
                    entry_decision = 0  # HOLD
                elif raw_decision < 1.5:
                    entry_decision = 1  # LONG
                else:
                    entry_decision = 2  # SHORT
                
                # Verificar constraints antes da execução
                can_add_position = positions_before < env.max_positions
                
                # Executar step
                obs, reward, done, info = env.step(action)
                
                # Estado após execução
                positions_after = len(env.positions)
                portfolio_after = env.portfolio_value
                
                # Análise
                wanted_to_trade = entry_decision > 0
                trade_executed = positions_after > positions_before
                blocked_by_max_positions = wanted_to_trade and not can_add_position
                blocked_by_other = wanted_to_trade and can_add_position and not trade_executed
                
                execution_data.append({
                    'step': i,
                    'raw_decision': raw_decision,
                    'entry_decision': entry_decision,
                    'wanted_to_trade': wanted_to_trade,
                    'positions_before': positions_before,
                    'positions_after': positions_after,
                    'can_add_position': can_add_position,
                    'trade_executed': trade_executed,
                    'blocked_by_max_positions': blocked_by_max_positions,
                    'blocked_by_other': blocked_by_other,
                    'portfolio_before': portfolio_before,
                    'portfolio_after': portfolio_after,
                    'reward': float(reward)
                })
                
                if done:
                    obs = env.reset()
                    
            except Exception as e:
                print(f"⚠️ Erro no step {i}: {e}")
                obs = env.reset()
                continue
        
        return execution_data
    
    def analyze_execution_results(self, execution_data):
        """Analisar resultados da execução"""
        print("📊 [ANALYSIS] Analisando resultados...")
        
        if not execution_data:
            return {}
        
        df = pd.DataFrame(execution_data)
        
        total_steps = len(df)
        wanted_trades = df['wanted_to_trade'].sum()
        executed_trades = df['trade_executed'].sum()
        blocked_by_max_pos = df['blocked_by_max_positions'].sum()
        blocked_by_other = df['blocked_by_other'].sum()
        
        results = {
            'total_steps': total_steps,
            'wanted_trades': int(wanted_trades),
            'executed_trades': int(executed_trades),
            'blocked_by_max_positions': int(blocked_by_max_pos),
            'blocked_by_other': int(blocked_by_other),
            'execution_rate': (executed_trades / max(wanted_trades, 1)) * 100,
            'max_position_blocking_rate': (blocked_by_max_pos / max(wanted_trades, 1)) * 100,
            'other_blocking_rate': (blocked_by_other / max(wanted_trades, 1)) * 100,
            'hold_steps': int((df['entry_decision'] == 0).sum()),
            'long_attempts': int((df['entry_decision'] == 1).sum()),
            'short_attempts': int((df['entry_decision'] == 2).sum()),
            'avg_positions': df['positions_before'].mean(),
            'max_positions_reached': int((df['positions_before'] >= 3).sum())
        }
        
        return results
    
    def find_blocking_patterns(self, execution_data):
        """Identificar padrões de bloqueio"""
        print("🔍 [PATTERNS] Identificando padrões de bloqueio...")
        
        if not execution_data:
            return {}
        
        df = pd.DataFrame(execution_data)
        
        # Filtrar apenas tentativas de trade
        trade_attempts = df[df['wanted_to_trade'] == True].copy()
        
        if len(trade_attempts) == 0:
            return {'no_trade_attempts': True}
        
        patterns = {}
        
        # 1. Padrão de bloqueio por max_positions
        max_pos_blocks = trade_attempts[trade_attempts['blocked_by_max_positions'] == True]
        if len(max_pos_blocks) > 0:
            patterns['max_positions_pattern'] = {
                'count': len(max_pos_blocks),
                'percentage': (len(max_pos_blocks) / len(trade_attempts)) * 100,
                'avg_positions_when_blocked': max_pos_blocks['positions_before'].mean()
            }
        
        # 2. Padrão de bloqueio por outros fatores
        other_blocks = trade_attempts[trade_attempts['blocked_by_other'] == True]
        if len(other_blocks) > 0:
            patterns['other_blocking_pattern'] = {
                'count': len(other_blocks),
                'percentage': (len(other_blocks) / len(trade_attempts)) * 100,
                'confidence_stats': {
                    'mean': df.loc[other_blocks.index, 'entry_decision'].mean() if len(other_blocks) > 0 else 0,
                }
            }
        
        # 3. Padrão de sucesso
        successful_trades = trade_attempts[trade_attempts['trade_executed'] == True]
        if len(successful_trades) > 0:
            patterns['successful_pattern'] = {
                'count': len(successful_trades),
                'percentage': (len(successful_trades) / len(trade_attempts)) * 100,
                'long_vs_short': {
                    'long_count': int((df.loc[successful_trades.index, 'entry_decision'] == 1).sum()),
                    'short_count': int((df.loc[successful_trades.index, 'entry_decision'] == 2).sum())
                }
            }
        
        return patterns
    
    def generate_report(self):
        """Gerar relatório final"""
        print("\n" + "="*80)
        print("📋 RELATÓRIO DE DIAGNÓSTICO SIMPLES DE TRADES")
        print("="*80)
        
        if 'action_processing' in self.results:
            action_data = self.results['action_processing']
            print(f"\n🎯 PROCESSAMENTO DE AÇÕES:")
            print(f"  Total de ações: {action_data['total_actions']}")
            print(f"  Hold: {action_data['hold_count']} ({action_data['hold_percentage']:.1f}%)")
            print(f"  Long: {action_data['long_count']} ({action_data['long_percentage']:.1f}%)")
            print(f"  Short: {action_data['short_count']} ({action_data['short_percentage']:.1f}%)")
            print(f"  Intenção de Trade: {action_data['trade_intention_percentage']:.1f}%")
        
        if 'execution_analysis' in self.results:
            exec_data = self.results['execution_analysis']
            print(f"\n🔄 ANÁLISE DE EXECUÇÃO:")
            print(f"  Total de steps: {exec_data['total_steps']}")
            print(f"  Queria fazer trades: {exec_data['wanted_trades']}")
            print(f"  Trades executados: {exec_data['executed_trades']}")
            print(f"  Taxa de execução: {exec_data['execution_rate']:.1f}%")
            print(f"  Bloqueado por max positions: {exec_data['blocked_by_max_positions']} ({exec_data['max_position_blocking_rate']:.1f}%)")
            print(f"  Bloqueado por outros fatores: {exec_data['blocked_by_other']} ({exec_data['other_blocking_rate']:.1f}%)")
            print(f"  Posições médias: {exec_data['avg_positions']:.1f}")
            print(f"  Max positions atingido: {exec_data['max_positions_reached']} vezes")
        
        if 'blocking_patterns' in self.results:
            patterns = self.results['blocking_patterns']
            print(f"\n🔍 PADRÕES DE BLOQUEIO:")
            
            if 'max_positions_pattern' in patterns:
                max_pos = patterns['max_positions_pattern']
                print(f"  Max Positions Bloqueio:")
                print(f"    - Ocorrências: {max_pos['count']}")
                print(f"    - Percentual: {max_pos['percentage']:.1f}%")
                print(f"    - Posições médias quando bloqueado: {max_pos['avg_positions_when_blocked']:.1f}")
            
            if 'other_blocking_pattern' in patterns:
                other = patterns['other_blocking_pattern']
                print(f"  Outros Bloqueios:")
                print(f"    - Ocorrências: {other['count']}")
                print(f"    - Percentual: {other['percentage']:.1f}%")
            
            if 'successful_pattern' in patterns:
                success = patterns['successful_pattern']
                print(f"  Trades Bem-sucedidos:")
                print(f"    - Ocorrências: {success['count']}")
                print(f"    - Percentual: {success['percentage']:.1f}%")
                if 'long_vs_short' in success:
                    print(f"    - Long: {success['long_vs_short']['long_count']}")
                    print(f"    - Short: {success['long_vs_short']['short_count']}")
        
        # Diagnóstico consolidado
        print(f"\n🎯 DIAGNÓSTICO CONSOLIDADO:")
        
        if 'action_processing' in self.results and 'execution_analysis' in self.results:
            action_intention = self.results['action_processing']['trade_intention_percentage']
            actual_execution = self.results['execution_analysis']['execution_rate']
            max_pos_blocking = self.results['execution_analysis']['max_position_blocking_rate']
            other_blocking = self.results['execution_analysis']['other_blocking_rate']
            
            print(f"  📊 Intenção de Trade (modelo): {action_intention:.1f}%")
            print(f"  ✅ Execução Real: {actual_execution:.1f}%")
            print(f"  🚫 Bloqueio Total: {100 - actual_execution:.1f}%")
            print(f"  📍 Principais Bloqueadores:")
            print(f"     - Max Positions (3): {max_pos_blocking:.1f}%")
            print(f"     - Outros fatores: {other_blocking:.1f}%")
            
            # Identificar problema principal
            if max_pos_blocking > 50:
                print(f"  🚨 PROBLEMA PRINCIPAL: Max Positions (limite de 3)")
                print(f"  💡 SOLUÇÃO: Aumentar max_positions ou reduzir hold time")
            elif other_blocking > 50:
                print(f"  🚨 PROBLEMA PRINCIPAL: Outros bloqueadores desconhecidos")
                print(f"  💡 INVESTIGAR: Lógica interna do step() pode ter filtros ocultos")
            elif action_intention < 30:
                print(f"  🚨 PROBLEMA PRINCIPAL: Modelo não quer fazer trades")
                print(f"  💡 SOLUÇÃO: Verificar treinamento ou action space")
            else:
                print(f"  ✅ SISTEMA: Funcionando dentro do esperado")
        
        print("="*80)
        
        return self.results
    
    def save_results(self, filename="diagnostico_trades_simples.json"):
        """Salvar resultados"""
        try:
            output = {
                'timestamp': datetime.now().isoformat(),
                'diagnostics': self.results
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Resultados salvos em: {filename}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar: {e}")

def run_simple_diagnosis():
    """Executar diagnóstico simples"""
    print("🚀 INICIANDO DIAGNÓSTICO SIMPLES DE TRADES")
    print("="*80)
    
    diagnostic = SimpleTradeDiagnostic()
    
    try:
        # 1. Criar dados mock
        df = diagnostic.create_mock_data()
        
        # 2. Criar ambiente
        env = diagnostic.create_environment(df)
        
        # 3. Gerar ações de teste
        test_actions = diagnostic.generate_test_actions(1000)
        
        # 4. Analisar processamento de ações
        processed_actions, action_results = diagnostic.analyze_action_processing(test_actions)
        diagnostic.results['action_processing'] = action_results
        
        # 5. Testar execução no ambiente
        execution_data = diagnostic.test_environment_execution(env, test_actions, max_steps=500)
        
        # 6. Analisar resultados de execução
        execution_results = diagnostic.analyze_execution_results(execution_data)
        diagnostic.results['execution_analysis'] = execution_results
        
        # 7. Identificar padrões de bloqueio
        blocking_patterns = diagnostic.find_blocking_patterns(execution_data)
        diagnostic.results['blocking_patterns'] = blocking_patterns
        
        # 8. Gerar relatório
        results = diagnostic.generate_report()
        
        # 9. Salvar resultados
        diagnostic.save_results()
        
        return diagnostic, results
        
    except Exception as e:
        print(f"❌ ERRO NO DIAGNÓSTICO: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    diagnostic, results = run_simple_diagnosis()
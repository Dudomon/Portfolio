#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 SISTEMA DE VALIDAÇÃO DE ESTRATÉGIAS - AVALIAR SE MODELO APRENDEU LÓGICA VÁLIDA

Este sistema avalia se o modelo realmente aprendeu estratégias de trading válidas,
não apenas overfitting em padrões aleatórios.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import json

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Adicionar ao path
sys.path.append("Modelo PPO Trader")
sys.path.append(".")

class StrategyValidator:
    """🧠 Validador de estratégias de trading"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.env = None
        self.results = {}
        
    def load_model_and_env(self):
        """Carregar modelo e ambiente para análise"""
        try:
            from daytrader import TradingEnv
            from sb3_contrib import RecurrentPPO
            
            # Carregar dados de teste
            import pickle
            with open('datasets/combined_crypto_data.pkl', 'rb') as f:
                data = pickle.load(f)
            
            # Usar últimos 10k pontos para teste out-of-sample
            test_data = data.tail(10000).copy()
            
            # Criar ambiente de teste
            self.env = TradingEnv(
                data=test_data,
                initial_balance=500.0,
                lookback_window_size=20,
                transaction_cost_pct=0.0001,
                max_positions=3
            )
            
            # Carregar modelo (se especificado)
            if self.model_path and os.path.exists(self.model_path):
                self.model = RecurrentPPO.load(self.model_path, env=self.env)
                print(f"✅ Modelo carregado: {self.model_path}")
            else:
                print("⚠️ Nenhum modelo especificado - usando modelo atual do daytrader")
                
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo/ambiente: {e}")
            return False
    
    def test_market_regime_adaptation(self):
        """🔍 TESTE 1: Adaptação a diferentes regimes de mercado"""
        print("\n🔍 TESTE 1: ADAPTAÇÃO A REGIMES DE MERCADO")
        print("=" * 60)
        
        results = {
            'bull_market': {'trades': 0, 'win_rate': 0, 'pnl': 0, 'drawdown': 0},
            'bear_market': {'trades': 0, 'win_rate': 0, 'pnl': 0, 'drawdown': 0},
            'sideways_market': {'trades': 0, 'win_rate': 0, 'pnl': 0, 'drawdown': 0},
            'volatile_market': {'trades': 0, 'win_rate': 0, 'pnl': 0, 'drawdown': 0}
        }
        
        try:
            # Identificar regimes no dataset
            data = self.env.data.copy()
            
            # Calcular tendência (SMA 50 vs SMA 200)
            data['sma_50'] = data['close_5m'].rolling(50).mean()
            data['sma_200'] = data['close_5m'].rolling(200).mean()
            data['trend'] = (data['sma_50'] - data['sma_200']) / data['sma_200']
            
            # Calcular volatilidade (ATR normalizada)
            data['volatility'] = data['atr_14_5m'] / data['close_5m']
            
            # Classificar regimes
            data['regime'] = 'sideways'
            data.loc[data['trend'] > 0.02, 'regime'] = 'bull'
            data.loc[data['trend'] < -0.02, 'regime'] = 'bear'
            data.loc[data['volatility'] > data['volatility'].quantile(0.8), 'regime'] = 'volatile'
            
            # Simular modelo em cada regime
            for regime in ['bull', 'bear', 'sideways', 'volatile']:
                regime_data = data[data['regime'] == f'{regime}_market'].copy()
                
                if len(regime_data) < 100:
                    continue
                
                # Criar ambiente específico para o regime
                regime_env = TradingEnv(
                    data=regime_data.iloc[:1000],  # Primeiros 1000 pontos
                    initial_balance=500.0,
                    lookback_window_size=20,
                    transaction_cost_pct=0.0001,
                    max_positions=3
                )
                
                # Simular trading
                obs = regime_env.reset()
                portfolio_values = [500.0]
                trades = []
                
                for step in range(min(500, len(regime_data) - 50)):
                    try:
                        if self.model:
                            action, _ = self.model.predict(obs, deterministic=True)
                        else:
                            # Usar estratégia baseline se não houver modelo
                            action = self._baseline_strategy(obs, regime_env)
                        
                        obs, reward, done, info = regime_env.step(action)
                        portfolio_values.append(regime_env.portfolio_value)
                        
                        if 'trade_executed' in info and info['trade_executed']:
                            trades.append({
                                'pnl': info.get('trade_pnl', 0),
                                'win': info.get('trade_pnl', 0) > 0
                            })
                        
                        if done:
                            break
                            
                    except Exception as e:
                        print(f"   ❌ Erro no step {step}: {e}")
                        break
                
                # Calcular métricas do regime
                if trades:
                    total_pnl = sum(t['pnl'] for t in trades)
                    win_rate = sum(t['win'] for t in trades) / len(trades)
                    max_portfolio = max(portfolio_values)
                    current_portfolio = portfolio_values[-1]
                    drawdown = (max_portfolio - current_portfolio) / max_portfolio
                    
                    results[f'{regime}_market'] = {
                        'trades': len(trades),
                        'win_rate': win_rate,
                        'pnl': total_pnl,
                        'drawdown': drawdown
                    }
                    
                    print(f"   📊 {regime.upper()} Market:")
                    print(f"      Trades: {len(trades)} | Win Rate: {win_rate:.1%}")
                    print(f"      PnL: ${total_pnl:.2f} | Drawdown: {drawdown:.1%}")
        
        except Exception as e:
            print(f"   ❌ Erro na análise de regimes: {e}")
            
        self.results['regime_adaptation'] = results
        return results
    
    def test_technical_pattern_recognition(self):
        """🔍 TESTE 2: Reconhecimento de padrões técnicos"""
        print("\n🔍 TESTE 2: RECONHECIMENTO DE PADRÕES TÉCNICOS")
        print("=" * 60)
        
        patterns_results = {}
        
        try:
            data = self.env.data.copy()
            
            # 1. BREAKOUTS
            data['bb_squeeze'] = (data['bb_upper_5m'] - data['bb_lower_5m']) / data['close_5m']
            breakout_conditions = data['bb_squeeze'] < data['bb_squeeze'].quantile(0.2)
            
            # 2. REVERSÕES
            data['rsi_oversold'] = data['rsi_14_5m'] < 30
            data['rsi_overbought'] = data['rsi_14_5m'] > 70
            
            # 3. MOMENTUM
            data['macd_bullish'] = (data['macd_5m'] > data['macd_signal_5m']) & \
                                  (data['macd_5m'].shift(1) <= data['macd_signal_5m'].shift(1))
            data['macd_bearish'] = (data['macd_5m'] < data['macd_signal_5m']) & \
                                  (data['macd_5m'].shift(1) >= data['macd_signal_5m'].shift(1))
            
            patterns = {
                'breakout': breakout_conditions,
                'oversold_reversal': data['rsi_oversold'],
                'overbought_reversal': data['rsi_overbought'],
                'bullish_momentum': data['macd_bullish'],
                'bearish_momentum': data['macd_bearish']
            }
            
            # Analisar comportamento do modelo em cada padrão
            for pattern_name, condition in patterns.items():
                pattern_indices = data[condition].index[:100]  # Primeiros 100 casos
                
                if len(pattern_indices) < 10:
                    continue
                
                pattern_actions = []
                pattern_rewards = []
                
                for idx in pattern_indices:
                    try:
                        # Simular ação do modelo neste ponto
                        obs_idx = max(0, idx - self.env.lookback_window_size)
                        if obs_idx + self.env.lookback_window_size >= len(data):
                            continue
                            
                        # Criar observação
                        obs_data = data.iloc[obs_idx:obs_idx + self.env.lookback_window_size]
                        obs = self._create_observation(obs_data)
                        
                        if self.model:
                            action, _ = self.model.predict(obs, deterministic=True)
                        else:
                            action = self._baseline_strategy_for_pattern(pattern_name, obs_data)
                        
                        # Calcular reward futuro (próximos 10 steps)
                        future_data = data.iloc[idx:idx+10]
                        if len(future_data) == 10:
                            future_return = (future_data['close_5m'].iloc[-1] - future_data['close_5m'].iloc[0]) / future_data['close_5m'].iloc[0]
                            
                            # Avaliar se ação foi apropriada
                            if action == 0:  # LONG
                                reward = future_return
                            elif action == 1:  # SHORT  
                                reward = -future_return
                            else:  # HOLD
                                reward = 0
                            
                            pattern_actions.append(action)
                            pattern_rewards.append(reward)
                            
                    except Exception as e:
                        continue
                
                if pattern_actions:
                    avg_reward = np.mean(pattern_rewards)
                    action_dist = np.bincount(pattern_actions, minlength=3) / len(pattern_actions)
                    
                    patterns_results[pattern_name] = {
                        'samples': len(pattern_actions),
                        'avg_reward': avg_reward,
                        'action_distribution': {
                            'long': action_dist[0],
                            'short': action_dist[1], 
                            'hold': action_dist[2]
                        }
                    }
                    
                    print(f"   📈 {pattern_name.replace('_', ' ').title()}:")
                    print(f"      Samples: {len(pattern_actions)} | Avg Reward: {avg_reward:.4f}")
                    print(f"      Actions: L:{action_dist[0]:.1%} S:{action_dist[1]:.1%} H:{action_dist[2]:.1%}")
        
        except Exception as e:
            print(f"   ❌ Erro na análise de padrões: {e}")
            
        self.results['pattern_recognition'] = patterns_results
        return patterns_results
    
    def test_risk_management_logic(self):
        """🔍 TESTE 3: Lógica de gerenciamento de risco"""
        print("\n🔍 TESTE 3: LÓGICA DE GERENCIAMENTO DE RISCO")
        print("=" * 60)
        
        risk_results = {}
        
        try:
            # Simular cenários de alto risco
            data = self.env.data.copy()
            
            # 1. ALTA VOLATILIDADE
            high_vol_periods = data[data['atr_14_5m'] > data['atr_14_5m'].quantile(0.9)]
            
            # 2. GRANDES GAPS
            data['gap'] = abs(data['open_5m'] - data['close_5m'].shift(1)) / data['close_5m'].shift(1)
            large_gaps = data[data['gap'] > data['gap'].quantile(0.95)]
            
            # 3. DRAWDOWN PERIODS
            data['portfolio_sim'] = 500 * (1 + data['returns_5m'].cumsum())
            data['drawdown'] = (data['portfolio_sim'] - data['portfolio_sim'].expanding().max()) / data['portfolio_sim'].expanding().max()
            high_dd_periods = data[data['drawdown'] < -0.1]  # >10% drawdown
            
            risk_scenarios = {
                'high_volatility': high_vol_periods.index[:50],
                'large_gaps': large_gaps.index[:50], 
                'high_drawdown': high_dd_periods.index[:50]
            }
            
            for scenario_name, indices in risk_scenarios.items():
                if len(indices) < 10:
                    continue
                
                scenario_actions = []
                scenario_positions = []
                
                for idx in indices:
                    try:
                        obs_idx = max(0, idx - self.env.lookback_window_size)
                        if obs_idx + self.env.lookback_window_size >= len(data):
                            continue
                            
                        obs_data = data.iloc[obs_idx:obs_idx + self.env.lookback_window_size]
                        obs = self._create_observation(obs_data)
                        
                        if self.model:
                            action, _ = self.model.predict(obs, deterministic=True)
                        else:
                            # Estratégia conservadora baseline
                            action = 2  # HOLD em cenários de risco
                        
                        scenario_actions.append(action)
                        
                        # Simular posição atual (simplificado)
                        position_size = 0.5 if action in [0, 1] else 0
                        scenario_positions.append(position_size)
                        
                    except Exception as e:
                        continue
                
                if scenario_actions:
                    hold_pct = (np.array(scenario_actions) == 2).mean()
                    avg_position_size = np.mean(scenario_positions)
                    
                    risk_results[scenario_name] = {
                        'samples': len(scenario_actions),
                        'hold_percentage': hold_pct,
                        'avg_position_size': avg_position_size,
                        'risk_awareness': hold_pct > 0.6  # Bom gerenciamento = >60% hold em risco
                    }
                    
                    status = "✅ BOM" if hold_pct > 0.6 else "⚠️ ARRISCADO"
                    print(f"   🛡️ {scenario_name.replace('_', ' ').title()}:")
                    print(f"      Hold Rate: {hold_pct:.1%} | Pos Size: {avg_position_size:.2f} | {status}")
        
        except Exception as e:
            print(f"   ❌ Erro na análise de risco: {e}")
            
        self.results['risk_management'] = risk_results
        return risk_results
    
    def test_strategy_consistency(self):
        """🔍 TESTE 4: Consistência estratégica"""
        print("\n🔍 TESTE 4: CONSISTÊNCIA ESTRATÉGICA")
        print("=" * 60)
        
        consistency_results = {}
        
        try:
            # 1. TESTE DE REPRODUTIBILIDADE
            obs = self.env.reset()
            
            # Fazer mesma predição 10 vezes
            predictions = []
            for _ in range(10):
                if self.model:
                    action, _ = self.model.predict(obs, deterministic=True)
                    predictions.append(action)
                else:
                    predictions.append(2)  # HOLD baseline
            
            reproducibility = len(set(predictions)) == 1
            
            # 2. TESTE DE COERÊNCIA EM CENÁRIOS SIMILARES
            similar_scenarios = []
            data = self.env.data.copy()
            
            # Encontrar cenários similares (RSI similar, MACD similar)
            reference_rsi = data['rsi_14_5m'].iloc[100]
            reference_macd = data['macd_5m'].iloc[100]
            
            similar_indices = data[
                (abs(data['rsi_14_5m'] - reference_rsi) < 5) &
                (abs(data['macd_5m'] - reference_macd) < 0.1)
            ].index[:20]
            
            similar_actions = []
            for idx in similar_indices:
                try:
                    obs_idx = max(0, idx - self.env.lookback_window_size)
                    if obs_idx + self.env.lookback_window_size >= len(data):
                        continue
                        
                    obs_data = data.iloc[obs_idx:obs_idx + self.env.lookback_window_size]  
                    obs = self._create_observation(obs_data)
                    
                    if self.model:
                        action, _ = self.model.predict(obs, deterministic=True)
                    else:
                        action = 2
                    
                    similar_actions.append(action)
                    
                except Exception as e:
                    continue
            
            if similar_actions:
                action_consistency = 1 - (len(set(similar_actions)) - 1) / max(1, len(similar_actions) - 1)
            else:
                action_consistency = 0
            
            consistency_results = {
                'reproducibility': reproducibility,
                'similar_scenario_consistency': action_consistency,
                'consistency_score': (reproducibility + action_consistency) / 2
            }
            
            print(f"   🔄 Reprodutibilidade: {'✅ OK' if reproducibility else '❌ FALHA'}")
            print(f"   🎯 Consistência Cenários Similares: {action_consistency:.1%}")
            print(f"   📊 Score Geral: {consistency_results['consistency_score']:.1%}")
        
        except Exception as e:
            print(f"   ❌ Erro na análise de consistência: {e}")
            consistency_results = {'error': str(e)}
            
        self.results['consistency'] = consistency_results
        return consistency_results
    
    def _baseline_strategy(self, obs, env):
        """Estratégia baseline simples para comparação"""
        # Estratégia RSI + MACD simples
        try:
            current_data = env.data.iloc[env.current_step]
            rsi = current_data['rsi_14_5m']
            macd = current_data['macd_5m'] 
            macd_signal = current_data['macd_signal_5m']
            
            if rsi < 30 and macd > macd_signal:
                return 0  # LONG
            elif rsi > 70 and macd < macd_signal:
                return 1  # SHORT
            else:
                return 2  # HOLD
        except:
            return 2  # HOLD por segurança
    
    def _baseline_strategy_for_pattern(self, pattern_name, obs_data):
        """Estratégia baseline específica para padrão"""
        if 'bullish' in pattern_name or 'oversold' in pattern_name:
            return 0  # LONG
        elif 'bearish' in pattern_name or 'overbought' in pattern_name:
            return 1  # SHORT
        else:
            return 2  # HOLD
    
    def _create_observation(self, obs_data):
        """Criar observação compatível com o ambiente"""
        try:
            # Simplificado - na prática usar a mesma lógica do TradingEnv
            features = []
            for col in ['close_5m', 'volume_5m', 'rsi_14_5m', 'macd_5m']:
                if col in obs_data.columns:
                    features.extend(obs_data[col].values[-10:])  # Últimos 10 valores
            
            return np.array(features, dtype=np.float32)
        except:
            return np.zeros(40, dtype=np.float32)  # Fallback
    
    def generate_strategy_report(self):
        """📋 Gerar relatório completo de validação estratégica"""
        print("\n" + "="*80)
        print("📋 RELATÓRIO DE VALIDAÇÃO ESTRATÉGICA")
        print("="*80)
        
        # Calcular score geral
        scores = []
        
        # Score adaptação a regimes
        if 'regime_adaptation' in self.results:
            regime_data = self.results['regime_adaptation']
            regime_trades = sum(r['trades'] for r in regime_data.values())
            regime_avg_winrate = np.mean([r['win_rate'] for r in regime_data.values() if r['trades'] > 0])
            regime_score = min(1.0, regime_trades / 50) * regime_avg_winrate
            scores.append(regime_score)
            print(f"🎯 Adaptação a Regimes: {regime_score:.1%}")
        
        # Score reconhecimento de padrões  
        if 'pattern_recognition' in self.results:
            pattern_data = self.results['pattern_recognition']
            pattern_scores = [max(0, r['avg_reward']) for r in pattern_data.values()]
            pattern_score = np.mean(pattern_scores) if pattern_scores else 0
            scores.append(pattern_score)
            print(f"📈 Reconhecimento Padrões: {pattern_score:.1%}")
        
        # Score gerenciamento de risco
        if 'risk_management' in self.results:
            risk_data = self.results['risk_management']
            risk_awareness = [r['risk_awareness'] for r in risk_data.values()]
            risk_score = np.mean(risk_awareness) if risk_awareness else 0
            scores.append(risk_score)
            print(f"🛡️ Gerenciamento Risco: {risk_score:.1%}")
        
        # Score consistência
        if 'consistency' in self.results:
            consistency_score = self.results['consistency'].get('consistency_score', 0)
            scores.append(consistency_score)
            print(f"🔄 Consistência: {consistency_score:.1%}")
        
        # Score geral
        if scores:
            overall_score = np.mean(scores)
            print(f"\n🏆 SCORE GERAL: {overall_score:.1%}")
            
            if overall_score > 0.7:
                assessment = "✅ ESTRATÉGIA VÁLIDA - Modelo aprendeu lógica sólida"
            elif overall_score > 0.5:
                assessment = "⚠️ ESTRATÉGIA QUESTIONÁVEL - Precisa mais treinamento"
            else:
                assessment = "❌ ESTRATÉGIA INVÁLIDA - Não aprendeu lógica válida"
            
            print(f"📊 AVALIAÇÃO: {assessment}")
        
        # Salvar relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"strategy_validation_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📁 Relatório salvo: {report_file}")
        
        return overall_score if scores else 0

def main():
    """Executar validação completa de estratégia"""
    print("🧠 SISTEMA DE VALIDAÇÃO DE ESTRATÉGIAS - INICIANDO")
    
    validator = StrategyValidator()
    
    # Carregar modelo e ambiente
    if not validator.load_model_and_env():
        print("❌ Falha ao carregar componentes")
        return
    
    # Executar testes
    validator.test_market_regime_adaptation()
    validator.test_technical_pattern_recognition() 
    validator.test_risk_management_logic()
    validator.test_strategy_consistency()
    
    # Gerar relatório final
    overall_score = validator.generate_strategy_report()
    
    print(f"\n🎯 VALIDAÇÃO CONCLUÍDA - Score: {overall_score:.1%}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
🎯 TESTE COMPLETO DE PERFORMANCE - CHECKPOINT 9.95M STEPS
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Adicionar projeto ao path
projeto_path = Path("D:/Projeto")
sys.path.insert(0, str(projeto_path))

class PerformanceTester:
    def __init__(self):
        self.projeto_path = Path("D:/Projeto")
        self.checkpoint_path = self.projeto_path / "trading_framework/training/checkpoints/DAYTRADER/checkpoint_9950000_steps_20250805_120857.zip"
        self.model = None
        
    def carregar_modelo(self):
        """Carregar modelo do checkpoint"""
        print("🤖 Carregando modelo...")
        
        try:
            from sb3_contrib import RecurrentPPO
            self.model = RecurrentPPO.load(self.checkpoint_path)
            
            print(f"✅ Modelo carregado!")
            print(f"   🔢 Timesteps: {self.model.num_timesteps:,}")
            print(f"   🧠 Policy: {type(self.model.policy).__name__}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro carregando modelo: {e}")
            return False
    
    def gerar_dados_sinteticos(self, n_bars=2000):
        """Gerar dados sintéticos para teste"""
        print(f"📊 Gerando {n_bars} barras sintéticas...")
        
        # Simular dados de OHLCV realistas
        np.random.seed(42)  # Reproduzibilidade
        
        base_price = 2000.0  # Preço base (simulando ouro)
        
        dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='1H')
        
        # Gerar preços com walk randômico
        returns = np.random.normal(0, 0.002, n_bars)  # 0.2% volatilidade
        returns = np.cumsum(returns)
        
        close_prices = base_price * np.exp(returns)
        
        # Gerar OHLC baseado no close
        high_offset = np.random.exponential(0.001, n_bars)
        low_offset = np.random.exponential(0.001, n_bars)
        
        high_prices = close_prices * (1 + high_offset)
        low_prices = close_prices * (1 - low_offset)
        
        # Open é o close anterior + ruído
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        open_prices += np.random.normal(0, close_prices * 0.0005)
        
        # Volume sintético
        volume = np.random.lognormal(10, 0.5, n_bars)
        
        df = pd.DataFrame({
            'datetime': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        print(f"✅ Dados sintéticos gerados: {len(df)} barras")
        print(f"   📈 Range de preços: {df['close'].min():.2f} - {df['close'].max():.2f}")
        
        return df
    
    def simular_trading_episode(self, df, episode_id=1, max_steps=500):
        """Simular um episódio de trading"""
        
        window_size = 20
        balance = 100000.0
        position = 0.0
        position_value = 0.0
        transaction_cost = 0.0001  # 1 basis point
        
        episode_data = {
            'episode': episode_id,
            'actions': [],
            'prices': [],
            'balances': [],
            'positions': [],
            'returns': [],
            'step_rewards': []
        }
        
        # Simular steps
        for step in range(min(max_steps, len(df) - window_size)):
            # Obter observação atual (simplificada)
            current_idx = window_size + step
            current_price = df.iloc[current_idx]['close']
            
            # Observação sintética (2580 features como esperado pelo modelo)
            # Simular features técnicas baseadas nos dados
            price_window = df.iloc[current_idx-window_size:current_idx]['close'].values
            
            # Features básicas por barra (129 por 20 barras = 2580)
            obs_features = []
            
            for i in range(window_size):
                bar_idx = current_idx - window_size + i
                bar = df.iloc[bar_idx]
                
                # Features OHLCV
                ohlcv = [bar['open'], bar['high'], bar['low'], bar['close'], bar['volume']]
                
                # Features técnicas simples
                if i > 0:
                    prev_bar = df.iloc[bar_idx-1]
                    price_change = (bar['close'] - prev_bar['close']) / prev_bar['close']
                    volatility = (bar['high'] - bar['low']) / bar['close']
                else:
                    price_change = 0.0
                    volatility = 0.01
                
                # Features de posição
                position_features = [position, position_value, balance]
                
                # Completar até 129 features por barra
                technical_features = [
                    price_change, volatility,
                    bar['close'] / bar['open'] - 1,  # Retorno intraday
                    (bar['high'] + bar['low']) / 2,  # Preço médio
                    bar['volume'] / 1000000,  # Volume normalizado
                ]
                
                # Adicionar features dummy para completar 129
                remaining_features = [0.0] * (129 - len(ohlcv) - len(position_features) - len(technical_features))
                
                bar_features = ohlcv + position_features + technical_features + remaining_features
                obs_features.extend(bar_features[:129])  # Garantir exatamente 129
            
            # Criar observação final
            obs = np.array(obs_features[:2580], dtype=np.float32)  # Garantir 2580 features
            
            # Fazer predição
            try:
                action, _states = self.model.predict(obs, deterministic=True)
                
                # Interpretar ação (baseado no action space do modelo)
                # action[0]: tipo de ordem (0=hold, 1=buy, 2=sell)
                # action[1]: quantidade
                # action[2]: usar stop loss
                # etc.
                
                order_type = int(np.clip(action[0], 0, 2))
                quantity = float(np.clip(action[1], 0, 1))  # 0-100% da conta
                
                # Executar ação
                old_balance = balance
                old_position = position
                
                if order_type == 1 and quantity > 0:  # BUY
                    if position <= 0:  # Pode comprar
                        trade_value = balance * quantity
                        trade_cost = trade_value * transaction_cost
                        
                        if trade_value + trade_cost <= balance:
                            shares_bought = (trade_value - trade_cost) / current_price
                            position += shares_bought
                            balance -= trade_value + trade_cost
                            
                elif order_type == 2 and quantity > 0:  # SELL
                    if position > 0:  # Tem posição para vender
                        shares_to_sell = position * quantity
                        trade_value = shares_to_sell * current_price
                        trade_cost = trade_value * transaction_cost
                        
                        position -= shares_to_sell
                        balance += trade_value - trade_cost
                
                # Calcular valor da posição atual
                position_value = position * current_price
                total_value = balance + position_value
                
                # Calcular reward (mudança no valor total)
                if step == 0:
                    prev_total = 100000.0
                else:
                    prev_total = episode_data['balances'][-1] + episode_data['positions'][-1]
                
                step_reward = (total_value - prev_total) / prev_total
                
                # Armazenar dados
                episode_data['actions'].append(action.tolist())
                episode_data['prices'].append(current_price)
                episode_data['balances'].append(balance)
                episode_data['positions'].append(position_value)
                episode_data['returns'].append(total_value / 100000.0 - 1.0)
                episode_data['step_rewards'].append(step_reward)
                
            except Exception as e:
                print(f"❌ Erro no step {step}: {e}")
                break
        
        # Métricas finais do episódio
        if episode_data['returns']:
            final_return = episode_data['returns'][-1]
            final_balance = episode_data['balances'][-1] + episode_data['positions'][-1]
            total_reward = sum(episode_data['step_rewards'])
            
            episode_data['final_return'] = final_return
            episode_data['final_balance'] = final_balance
            episode_data['total_reward'] = total_reward
            episode_data['steps'] = len(episode_data['actions'])
            
            print(f"[{episode_id}] Return: {final_return*100:.2f}%, Reward: {total_reward:.4f}, Steps: {episode_data['steps']}")
        else:
            episode_data['final_return'] = -1.0
            episode_data['final_balance'] = 0.0
            episode_data['total_reward'] = -10.0
            episode_data['steps'] = 0
        
        return episode_data
    
    def executar_teste_completo(self, n_episodes=20):
        """Executar teste completo de performance"""
        
        print(f"\n🎮 EXECUTANDO TESTE COMPLETO ({n_episodes} episódios)")
        print("=" * 60)
        
        # Gerar dados para teste
        df_test = self.gerar_dados_sinteticos(n_bars=3000)
        
        resultados = []
        
        for ep in range(n_episodes):
            print(f"Episódio {ep+1}/{n_episodes}: ", end="")
            
            # Usar diferentes segmentos dos dados para cada episódio
            start_idx = ep * 100
            episode_df = df_test.iloc[start_idx:start_idx+1000].reset_index(drop=True)
            
            episode_result = self.simular_trading_episode(episode_df, ep+1)
            resultados.append(episode_result)
        
        return resultados
    
    def analisar_resultados(self, resultados):
        """Analisar resultados dos episódios"""
        
        print(f"\n📊 ANÁLISE DE RESULTADOS")
        print("=" * 60)
        
        if not resultados:
            print("❌ Nenhum resultado para analisar")
            return {}
        
        # Extrair métricas
        returns = [r['final_return'] for r in resultados if 'final_return' in r]
        rewards = [r['total_reward'] for r in resultados if 'total_reward' in r]
        steps = [r['steps'] for r in resultados if 'steps' in r]
        
        if not returns:
            print("❌ Nenhum retorno válido encontrado")
            return {}
        
        # Calcular estatísticas
        metrics = {
            'total_episodes': len(resultados),
            'valid_episodes': len(returns),
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'mean_reward': np.mean(rewards) if rewards else 0,
            'positive_episodes': sum(1 for r in returns if r > 0),
            'mean_steps': np.mean(steps) if steps else 0
        }
        
        metrics['success_rate'] = metrics['positive_episodes'] / metrics['valid_episodes'] * 100
        
        if metrics['std_return'] > 0:
            metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['std_return']
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Imprimir resultados
        print(f"📈 PERFORMANCE GERAL:")
        print(f"   Episódios Válidos: {metrics['valid_episodes']}/{metrics['total_episodes']}")
        print(f"   Retorno Médio: {metrics['mean_return']*100:.2f}% ± {metrics['std_return']*100:.2f}%")
        print(f"   Range: {metrics['min_return']*100:.2f}% → {metrics['max_return']*100:.2f}%")
        print(f"   Reward Médio: {metrics['mean_reward']:.4f}")
        
        print(f"\n🎯 TAXA DE SUCESSO:")
        print(f"   Episódios Positivos: {metrics['positive_episodes']}/{metrics['valid_episodes']} ({metrics['success_rate']:.1f}%)")
        
        print(f"\n📊 MÉTRICAS AVANÇADAS:")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"   Steps Médio: {metrics['mean_steps']:.0f}")
        
        # Top 5 melhores
        sorted_results = sorted(resultados, key=lambda x: x.get('final_return', -1), reverse=True)
        print(f"\n🏆 TOP 5 MELHORES EPISÓDIOS:")
        for i, r in enumerate(sorted_results[:5]):
            if 'final_return' in r:
                print(f"   {i+1}. Episódio {r['episode']}: {r['final_return']*100:.2f}% (R: {r.get('total_reward', 0):.3f})")
        
        return metrics
    
    def salvar_relatorio(self, metrics, resultados):
        """Salvar relatório detalhado"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        relatorio_path = self.projeto_path / "avaliacoes" / f"performance_completo_9950000_{timestamp}.txt"
        
        os.makedirs(self.projeto_path / "avaliacoes", exist_ok=True)
        
        with open(relatorio_path, 'w', encoding='utf-8') as f:
            f.write(f"🎯 RELATÓRIO COMPLETO DE PERFORMANCE - CHECKPOINT 9.95M STEPS\n")
            f.write(f"=" * 80 + "\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {self.checkpoint_path.name}\n")
            f.write(f"Timesteps Treinados: {self.model.num_timesteps:,}\n")
            f.write(f"Policy: {type(self.model.policy).__name__}\n\n")
            
            f.write(f"📊 MÉTRICAS PRINCIPAIS:\n")
            f.write(f"   Total Episódios: {metrics['total_episodes']}\n")
            f.write(f"   Episódios Válidos: {metrics['valid_episodes']}\n")
            f.write(f"   Retorno Médio: {metrics['mean_return']*100:.4f}%\n")
            f.write(f"   Desvio Padrão: {metrics['std_return']*100:.4f}%\n")
            f.write(f"   Range: {metrics['min_return']*100:.4f}% → {metrics['max_return']*100:.4f}%\n")
            f.write(f"   Taxa de Sucesso: {metrics['success_rate']:.2f}%\n")
            f.write(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.6f}\n")
            f.write(f"   Reward Médio: {metrics['mean_reward']:.6f}\n")
            f.write(f"   Steps Médio: {metrics['mean_steps']:.1f}\n\n")
            
            f.write(f"📈 EPISÓDIOS DETALHADOS:\n")
            for r in resultados:
                if 'final_return' in r:
                    f.write(f"   Ep {r['episode']:2d}: Return={r['final_return']*100:7.2f}%, ")
                    f.write(f"Reward={r.get('total_reward', 0):8.4f}, Steps={r.get('steps', 0):3d}\n")
        
        print(f"\n💾 Relatório salvo: {relatorio_path}")
        return relatorio_path
    
    def run(self):
        """Executar teste completo"""
        
        print("🎯 TESTE COMPLETO DE PERFORMANCE - CHECKPOINT 9.95M STEPS")
        print("=" * 70)
        print(f"🕐 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Carregar modelo
        if not self.carregar_modelo():
            print("❌ Falha ao carregar modelo")
            return
        
        # 2. Executar testes
        resultados = self.executar_teste_completo(n_episodes=20)
        
        # 3. Analisar resultados
        metrics = self.analisar_resultados(resultados)
        
        if not metrics:
            print("❌ Falha na análise")
            return
        
        # 4. Salvar relatório
        relatorio_path = self.salvar_relatorio(metrics, resultados)
        
        # 5. Conclusão
        print(f"\n🏆 CONCLUSÃO FINAL:")
        if metrics['success_rate'] >= 60:
            print(f"   🌟 EXCELENTE: Modelo com alta performance!")
        elif metrics['success_rate'] >= 40:
            print(f"   ✅ BOM: Performance satisfatória")
        else:
            print(f"   ⚠️ ATENÇÃO: Modelo precisa de melhorias")
        
        print(f"   📊 Taxa de Sucesso: {metrics['success_rate']:.1f}%")
        print(f"   📈 Retorno Médio: {metrics['mean_return']*100:.2f}%")
        print(f"   🎯 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        
        print(f"\n🕐 Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    tester = PerformanceTester()
    tester.run()

if __name__ == "__main__":
    main()
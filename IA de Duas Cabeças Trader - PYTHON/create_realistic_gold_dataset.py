#!/usr/bin/env python3
"""
🚀 GERADOR DE DATASET SINTÉTICO BASEADO EM OURO REAL
Cria 2M barras com distribuição realista baseada em análise do ouro
"""

import numpy as np
import pandas as pd
import os
from scipy import stats
import pickle
from datetime import datetime, timedelta

class RealisticGoldGenerator:
    def __init__(self):
        """Inicializar com parâmetros baseados em dados reais do ouro"""
        
        # PARÂMETROS EXTRAÍDOS DOS DADOS REAIS
        self.base_price = 2000.0  # Preço base realista
        self.daily_drift = 0.00000116  # Drift diário observado
        self.base_volatility = 0.000616  # Volatilidade base observada
        
        # REGIMES DE VOLATILIDADE (baseado em análise real)
        self.volatility_regimes = {
            'consolidation': {
                'prob': 0.45,  # 45% do tempo (ajustado do real 25% muito baixo)
                'vol_range': (0.0001, 0.0008),  # 0.01% - 0.08%
                'mean_duration': 120  # ~10 horas
            },
            'trending': {
                'prob': 0.35,  # 35% do tempo  
                'vol_range': (0.0008, 0.0025),  # 0.08% - 0.25%
                'mean_duration': 200  # ~16 horas
            },
            'breakout': {
                'prob': 0.15,  # 15% do tempo
                'vol_range': (0.0025, 0.008),   # 0.25% - 0.8%
                'mean_duration': 80   # ~6 horas
            },
            'extreme': {
                'prob': 0.05,  # 5% do tempo
                'vol_range': (0.008, 0.020),    # 0.8% - 2.0%
                'mean_duration': 40   # ~3 horas
            }
        }
        
        # PADRÕES INTRADAY (horários de maior/menor atividade)
        self.intraday_multipliers = {
            0: 0.3, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5,  # Madrugada
            6: 0.7, 7: 0.9, 8: 1.3, 9: 1.5, 10: 1.4, 11: 1.2, # Manhã
            12: 1.0, 13: 1.1, 14: 1.3, 15: 1.4, 16: 1.2, 17: 1.0, # Tarde
            18: 0.8, 19: 0.6, 20: 0.5, 21: 0.4, 22: 0.3, 23: 0.3  # Noite
        }
        
        # SUPORTE E RESISTÊNCIA (níveis psicológicos)
        self.support_resistance_levels = [1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200]
        
        # STATE TRACKING
        self.current_regime = 'consolidation'
        self.regime_duration = 0
        self.regime_max_duration = 120
        
    def _select_new_regime(self):
        """Selecionar novo regime baseado em probabilidades"""
        regime_names = list(self.volatility_regimes.keys())
        probabilities = [self.volatility_regimes[r]['prob'] for r in regime_names]
        
        # Adicionar bias para não repetir o mesmo regime
        current_idx = regime_names.index(self.current_regime)
        probabilities[current_idx] *= 0.5  # Reduzir chance de repetir
        
        # Normalizar probabilidades
        total_prob = sum(probabilities)
        probabilities = [p/total_prob for p in probabilities]
        
        new_regime = np.random.choice(regime_names, p=probabilities)
        return new_regime
    
    def _get_regime_volatility(self):
        """Obter volatilidade do regime atual"""
        regime_info = self.volatility_regimes[self.current_regime]
        vol_min, vol_max = regime_info['vol_range']
        
        # Volatilidade varia suavemente dentro do range
        volatility = np.random.uniform(vol_min, vol_max)
        return volatility
    
    def _apply_intraday_seasonality(self, base_vol, hour):
        """Aplicar sazonalidade intraday"""
        multiplier = self.intraday_multipliers.get(hour, 1.0)
        return base_vol * multiplier
    
    def _apply_support_resistance(self, price, returns):
        """Aplicar efeito de suporte/resistência"""
        # Encontrar nível mais próximo
        closest_level = min(self.support_resistance_levels, key=lambda x: abs(x - price))
        distance = abs(price - closest_level)
        
        # Se muito próximo de S/R (±$5), adicionar resistência
        if distance <= 5:
            resistance_strength = (5 - distance) / 5  # 0-1
            # Adicionar força contrária ao movimento
            if price > closest_level:  # Resistência
                returns -= resistance_strength * 0.0002
            else:  # Suporte
                returns += resistance_strength * 0.0002
                
        return returns
    
    def _add_realistic_noise(self, returns):
        """Adicionar ruído realista (bid-ask, HFT, etc)"""
        # Micro reversões (simula HFT)
        if np.random.random() < 0.3:  # 30% chance
            micro_reversal = np.random.normal(0, 0.00005)  # Muito pequeno
            returns += micro_reversal
            
        # Spread bias
        spread_bias = np.random.normal(0, 0.00002)
        returns += spread_bias
        
        return returns
    
    def _add_weekend_gaps(self, returns, is_monday_open):
        """Adicionar gaps de fim de semana"""
        if is_monday_open and np.random.random() < 0.7:  # 70% chance gap segunda
            gap_size = np.random.normal(0, 0.002)  # Gap médio 0.2%
            returns += gap_size
            
        return returns
    
    def generate_realistic_gold_sequence(self, num_bars=2000000):
        """Gerar sequência realista de ouro com 2M barras"""
        
        print(f"GERANDO {num_bars:,} BARRAS DE OURO SINTETICO REALISTA")
        print("="*60)
        
        # Inicializar arrays
        prices = np.zeros(num_bars)
        volumes = np.zeros(num_bars)
        timestamps = []
        regimes = []
        
        # Preço inicial
        current_price = self.base_price
        prices[0] = current_price
        
        # Timestamp inicial
        start_date = datetime(2023, 1, 1, 9, 0)  # Começar 9h segunda
        current_time = start_date
        
        print("Gerando dados...")
        progress_checkpoints = [num_bars//10 * i for i in range(1, 11)]
        
        for i in range(1, num_bars):
            
            # Progress tracking
            if i in progress_checkpoints:
                progress = (i / num_bars) * 100
                print(f"  Progresso: {progress:.0f}% - {i:,} barras geradas")
            
            # Atualizar timestamp (5min bars)
            current_time += timedelta(minutes=5)
            timestamps.append(current_time)
            
            # Verificar se precisa trocar regime
            if self.regime_duration >= self.regime_max_duration:
                self.current_regime = self._select_new_regime()
                regime_info = self.volatility_regimes[self.current_regime]
                self.regime_max_duration = int(np.random.exponential(regime_info['mean_duration']))
                self.regime_duration = 0
            
            regimes.append(self.current_regime)
            self.regime_duration += 1
            
            # Gerar retorno base
            base_volatility = self._get_regime_volatility()
            
            # Aplicar sazonalidade intraday
            hour = current_time.hour
            adjusted_vol = self._apply_intraday_seasonality(base_volatility, hour)
            
            # Gerar retorno com drift + volatilidade
            returns = np.random.normal(self.daily_drift, adjusted_vol)
            
            # Aplicar efeitos especiais
            returns = self._apply_support_resistance(current_price, returns)
            returns = self._add_realistic_noise(returns)
            
            # Weekend gaps (segunda-feira)
            is_monday_open = current_time.weekday() == 0 and current_time.hour == 9
            if is_monday_open:
                returns = self._add_weekend_gaps(returns, True)
            
            # Atualizar preço
            current_price = current_price * (1 + returns)
            prices[i] = current_price
            
            # Gerar volume (correlacionado com volatilidade)
            base_volume = 5000
            vol_multiplier = adjusted_vol / self.base_volatility
            volume = int(base_volume * vol_multiplier * np.random.uniform(0.5, 2.0))
            volumes[i] = volume
        
        print("Geracao concluida!")
        
        # Criar DataFrame
        df = pd.DataFrame({
            'timestamp': [start_date] + timestamps,
            'open': prices,  # Simplificado: open = close anterior
            'high': prices * (1 + np.random.uniform(0, 0.002, num_bars)),
            'low': prices * (1 - np.random.uniform(0, 0.002, num_bars)),
            'close': prices,
            'volume': volumes.astype(int),
            'regime': ['consolidation'] + regimes
        })
        
        # Ajustar OHLC para ser consistente
        for i in range(len(df)):
            o, h, l, c = df.iloc[i][['open', 'high', 'low', 'close']]
            # Garantir High >= max(O,C) e Low <= min(O,C)
            df.iloc[i, df.columns.get_loc('high')] = max(h, o, c)
            df.iloc[i, df.columns.get_loc('low')] = min(l, o, c)
        
        return df
    
    def validate_dataset(self, df):
        """Validar qualidade do dataset gerado"""
        print(f"\nVALIDACAO DO DATASET GERADO")
        print("="*40)
        
        returns = df['close'].pct_change().dropna()
        
        print(f"Total de barras: {len(df):,}")
        print(f"Periodo simulado: {df['timestamp'].iloc[0]} ate {df['timestamp'].iloc[-1]}")
        print(f"Variacao de preco: ${df['close'].iloc[0]:.2f} -> ${df['close'].iloc[-1]:.2f}")
        
        print(f"\nEstatisticas dos retornos:")
        print(f"  Retorno medio: {returns.mean()*100:.6f}%")
        print(f"  Volatilidade: {returns.std()*100:.4f}%")
        print(f"  Skewness: {stats.skew(returns):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(returns):.4f}")
        
        # Distribuição de regimes
        regime_dist = df['regime'].value_counts(normalize=True)
        print(f"\nDistribuicao de regimes:")
        for regime, pct in regime_dist.items():
            print(f"  {regime}: {pct*100:.1f}%")
        
        return True

def main():
    """Função principal"""
    
    print("CRIANDO DATASET SINTETICO DE OURO - 2M BARRAS")
    print("="*60)
    
    # Criar gerador
    generator = RealisticGoldGenerator()
    
    # Gerar dataset
    df = generator.generate_realistic_gold_sequence(num_bars=2000000)
    
    # Validar
    generator.validate_dataset(df)
    
    # Salvar
    output_file = f"data/GOLD_SYNTHETIC_REALISTIC_2M_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"\nSalvando dataset em: {output_file}")
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)
    
    print(f"Dataset salvo com sucesso!")
    print(f"Arquivo: {output_file}")
    print(f"Tamanho: {len(df):,} barras")
    print(f"Tamanho do arquivo: ~{os.path.getsize(output_file)/1024/1024:.1f} MB")
    
    print(f"\nPRONTO PARA TREINAR V7 COM DATASET REALISTA!")

if __name__ == "__main__":
    main()
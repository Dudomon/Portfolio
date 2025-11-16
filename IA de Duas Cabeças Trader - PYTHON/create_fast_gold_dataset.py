#!/usr/bin/env python3
"""
⚡ GERADOR RÁPIDO DE DATASET SINTÉTICO DE OURO
Versão otimizada para gerar 2M barras rapidamente
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

class FastGoldGenerator:
    def __init__(self):
        """Gerador otimizado"""
        
        # CONFIGURAÇÕES BASEADAS EM ANÁLISE REAL
        self.base_price = 2000.0
        self.base_vol = 0.0008  # Volatilidade base
        
        # REGIMES SIMPLIFICADOS (mais eficiente)
        self.regimes = {
            'low': {'prob': 0.45, 'vol_mult': 0.3},     # Baixa volatilidade
            'med': {'prob': 0.35, 'vol_mult': 1.0},     # Média volatilidade  
            'high': {'prob': 0.15, 'vol_mult': 3.0},    # Alta volatilidade
            'extreme': {'prob': 0.05, 'vol_mult': 8.0}  # Extrema volatilidade
        }
        
    def generate_fast(self, num_bars=2000000):
        """Geração vectorizada ultra-rápida"""
        
        print(f"GERANDO {num_bars:,} BARRAS - MODO RAPIDO")
        print("="*50)
        
        # PASSO 1: Gerar regimes de volatilidade em lotes
        print("Passo 1: Definindo regimes...")
        regime_names = list(self.regimes.keys())
        regime_probs = [self.regimes[r]['prob'] for r in regime_names]
        
        # Gerar regimes em chunks para eficiência
        chunk_size = 1000
        regimes = []
        
        for i in range(0, num_bars, chunk_size):
            chunk_regimes = np.random.choice(regime_names, size=min(chunk_size, num_bars-i), p=regime_probs)
            regimes.extend(chunk_regimes)
        
        regimes = np.array(regimes)
        
        # PASSO 2: Gerar volatilidades baseadas no regime
        print("Passo 2: Calculando volatilidades...")
        vol_multipliers = np.array([self.regimes[r]['vol_mult'] for r in regimes])
        volatilities = self.base_vol * vol_multipliers
        
        # PASSO 3: Gerar retornos
        print("Passo 3: Gerando retornos...")
        returns = np.random.normal(0, volatilities)
        
        # PASSO 4: Adicionar drift sutil
        drift = np.cumsum(np.random.normal(0.00001, 0.000001, num_bars))
        returns += drift / 1000  # Drift muito sutil
        
        # PASSO 5: Aplicar sazonalidade intraday (simplificada)
        print("Passo 4: Aplicando sazonalidade...")
        hours = np.tile(np.arange(24), num_bars // 24 + 1)[:num_bars]
        intraday_mult = 1 + 0.3 * np.sin(2 * np.pi * hours / 24)  # Padrão senoidal
        returns *= intraday_mult
        
        # PASSO 6: Gerar preços
        print("Passo 5: Calculando precos...")
        prices = np.zeros(num_bars)
        prices[0] = self.base_price
        
        # Cálculo vectorizado dos preços
        price_multipliers = 1 + returns
        prices = self.base_price * np.cumprod(price_multipliers)
        
        # PASSO 7: Gerar OHLC simplificado
        print("Passo 6: Gerando OHLC...")
        
        # Noise para high/low
        high_noise = np.random.uniform(0, 0.003, num_bars)  # 0-0.3%
        low_noise = np.random.uniform(0, 0.003, num_bars)
        
        opens = np.roll(prices, 1)  # Open = close anterior
        opens[0] = prices[0]
        
        highs = prices * (1 + high_noise)
        lows = prices * (1 - low_noise)
        closes = prices
        
        # Garantir consistência OHLC
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        
        # PASSO 8: Gerar volumes
        print("Passo 7: Gerando volumes...")
        base_volume = 5000
        volume_multipliers = vol_multipliers * np.random.uniform(0.5, 2.0, num_bars)
        volumes = (base_volume * volume_multipliers).astype(int)
        
        # PASSO 9: Criar timestamps
        print("Passo 8: Criando timestamps...")
        start_date = datetime(2023, 1, 1, 9, 0)
        timestamps = [start_date + timedelta(minutes=5*i) for i in range(num_bars)]
        
        print("Passo 9: Montando DataFrame...")
        
        # Criar DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs, 
            'low': lows,
            'close': closes,
            'volume': volumes,
            'regime': regimes
        })
        
        print("Geracao concluida!")
        return df
    
    def quick_validation(self, df):
        """Validação rápida"""
        print(f"\nVALIDACAO RAPIDA:")
        print("="*30)
        
        returns = df['close'].pct_change().dropna()
        
        print(f"Barras: {len(df):,}")
        print(f"Volatilidade: {returns.std()*100:.3f}%")
        print(f"Preco inicial: ${df['close'].iloc[0]:.2f}")
        print(f"Preco final: ${df['close'].iloc[-1]:.2f}")
        
        # Distribuição de regimes
        regime_dist = df['regime'].value_counts(normalize=True)
        print(f"\nRegimes:")
        for regime, pct in regime_dist.items():
            print(f"  {regime}: {pct*100:.1f}%")

def main():
    """Função principal otimizada"""
    
    print("GERADOR RAPIDO - DATASET SINTETICO DE OURO")
    print("="*50)
    
    # Criar gerador
    generator = FastGoldGenerator()
    
    # Gerar dataset
    start_time = datetime.now()
    df = generator.generate_fast(num_bars=2000000)
    end_time = datetime.now()
    
    generation_time = (end_time - start_time).total_seconds()
    print(f"\nTempo de geracao: {generation_time:.1f} segundos")
    
    # Validação rápida
    generator.quick_validation(df)
    
    # Salvar
    output_file = f"data/GOLD_SYNTHETIC_FAST_2M_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"\nSalvando: {output_file}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    file_size = os.path.getsize(output_file) / 1024 / 1024
    
    print(f"\nSUCESSO!")
    print(f"Arquivo: {output_file}")
    print(f"Tamanho: {file_size:.1f} MB")
    print(f"Velocidade: {len(df)/generation_time:.0f} barras/segundo")
    print(f"\nPRONTO PARA TREINAR!")

if __name__ == "__main__":
    main()
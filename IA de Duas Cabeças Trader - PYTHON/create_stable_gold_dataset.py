#!/usr/bin/env python3
"""
🎯 GERADOR ESTÁVEL DE DATASET SINTÉTICO DE OURO
Versão corrigida sem overflow, com volatilidade realista
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

class StableGoldGenerator:
    def __init__(self):
        """Gerador estável e realista"""
        
        # CONFIGURAÇÕES REALISTAS (baseado em análise do ouro real)
        self.base_price = 2000.0
        
        # VOLATILIDADES REALISTAS (por regime)
        self.regimes = {
            'consolidation': {'prob': 0.45, 'vol': 0.0003},  # 0.03% std
            'trending': {'prob': 0.35, 'vol': 0.0008},       # 0.08% std
            'breakout': {'prob': 0.15, 'vol': 0.0020},       # 0.20% std
            'extreme': {'prob': 0.05, 'vol': 0.0050}         # 0.50% std
        }
        
    def generate_stable(self, num_bars=2000000):
        """Geração estável sem overflow"""
        
        print(f"GERANDO {num_bars:,} BARRAS ESTAVEIS")
        print("="*40)
        
        # CHUNK PROCESSING para evitar overflow
        chunk_size = 100000  # 100k barras por chunk
        all_chunks = []
        
        current_price = self.base_price
        
        for chunk_start in range(0, num_bars, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_bars)
            chunk_bars = chunk_end - chunk_start
            
            progress = (chunk_start / num_bars) * 100
            print(f"Processando chunk: {progress:.0f}% ({chunk_start:,}-{chunk_end:,})")
            
            # Gerar regime para este chunk
            regime_names = list(self.regimes.keys())
            regime_probs = [self.regimes[r]['prob'] for r in regime_names]
            
            chunk_regimes = np.random.choice(regime_names, size=chunk_bars, p=regime_probs)
            
            # Gerar retornos baseados no regime
            chunk_returns = np.zeros(chunk_bars)
            
            for i, regime in enumerate(chunk_regimes):
                vol = self.regimes[regime]['vol']
                # Retorno pequeno e controlado
                ret = np.random.normal(0, vol)
                # Limitar retorno para evitar explosão
                ret = np.clip(ret, -0.01, 0.01)  # Max ±1%
                chunk_returns[i] = ret
            
            # Calcular preços do chunk
            chunk_prices = np.zeros(chunk_bars)
            chunk_prices[0] = current_price
            
            for i in range(1, chunk_bars):
                # Aplicar retorno de forma controlada
                chunk_prices[i] = chunk_prices[i-1] * (1 + chunk_returns[i])
                
                # Verificar limites de sanidade
                if chunk_prices[i] <= 0 or chunk_prices[i] > 10000:
                    chunk_prices[i] = chunk_prices[i-1]  # Manter preço anterior
            
            # Atualizar preço atual para próximo chunk
            current_price = chunk_prices[-1]
            
            # Gerar OHLC do chunk
            chunk_opens = np.roll(chunk_prices, 1)
            chunk_opens[0] = chunk_prices[0]
            
            # High/Low com noise controlado
            high_noise = np.random.uniform(0.0005, 0.002, chunk_bars)  # 0.05%-0.2%
            low_noise = np.random.uniform(0.0005, 0.002, chunk_bars)
            
            chunk_highs = chunk_prices * (1 + high_noise)
            chunk_lows = chunk_prices * (1 - low_noise)
            
            # Garantir consistência OHLC
            chunk_highs = np.maximum(chunk_highs, np.maximum(chunk_opens, chunk_prices))
            chunk_lows = np.minimum(chunk_lows, np.minimum(chunk_opens, chunk_prices))
            
            # Volumes baseados em volatilidade
            vol_multipliers = np.array([self.regimes[r]['vol'] for r in chunk_regimes])
            base_volume = 5000
            chunk_volumes = (base_volume * (1 + vol_multipliers * 100) * 
                           np.random.uniform(0.8, 1.5, chunk_bars)).astype(int)
            
            # Timestamps
            start_time = datetime(2023, 1, 1, 9, 0) + timedelta(minutes=5*chunk_start)
            chunk_timestamps = [start_time + timedelta(minutes=5*i) for i in range(chunk_bars)]
            
            # Criar DataFrame do chunk
            chunk_df = pd.DataFrame({
                'timestamp': chunk_timestamps,
                'open': chunk_opens,
                'high': chunk_highs,
                'low': chunk_lows, 
                'close': chunk_prices,
                'volume': chunk_volumes,
                'regime': chunk_regimes
            })
            
            all_chunks.append(chunk_df)
        
        # Combinar todos os chunks
        print("Combinando chunks...")
        df = pd.concat(all_chunks, ignore_index=True)
        
        print("Geracao concluida!")
        return df
    
    def validate_stable(self, df):
        """Validação robusta"""
        print(f"\nVALIDACAO DO DATASET:")
        print("="*30)
        
        print(f"Total barras: {len(df):,}")
        print(f"Periodo: {df['timestamp'].iloc[0]} ate {df['timestamp'].iloc[-1]}")
        
        # Verificar sanidade dos preços
        prices = df['close']
        print(f"Preco inicial: ${prices.iloc[0]:.2f}")
        print(f"Preco final: ${prices.iloc[-1]:.2f}")
        print(f"Preco min: ${prices.min():.2f}")
        print(f"Preco max: ${prices.max():.2f}")
        
        # Estatísticas dos retornos
        returns = prices.pct_change().dropna()
        valid_returns = returns[np.isfinite(returns)]
        
        if len(valid_returns) > 0:
            print(f"\nEstatisticas dos retornos:")
            print(f"  Volatilidade: {valid_returns.std()*100:.3f}%")
            print(f"  Retorno medio: {valid_returns.mean()*100:.6f}%")
            print(f"  Min return: {valid_returns.min()*100:.3f}%")
            print(f"  Max return: {valid_returns.max()*100:.3f}%")
        
        # Distribuição de regimes
        regime_dist = df['regime'].value_counts(normalize=True)
        print(f"\nDistribuicao de regimes:")
        for regime, pct in regime_dist.items():
            print(f"  {regime}: {pct*100:.1f}%")
        
        # Verificar dados problemáticos
        inf_count = np.sum(~np.isfinite(prices))
        zero_count = np.sum(prices <= 0)
        
        if inf_count > 0 or zero_count > 0:
            print(f"\nPROBLEMAS DETECTADOS:")
            print(f"  Valores infinitos: {inf_count}")
            print(f"  Valores <= 0: {zero_count}")
            return False
        
        print(f"\nDataset VALIDO!")
        return True

def main():
    """Função principal estável"""
    
    print("GERADOR ESTAVEL - DATASET SINTETICO OURO")
    print("="*45)
    
    generator = StableGoldGenerator()
    
    # Gerar dataset
    start_time = datetime.now()
    df = generator.generate_stable(num_bars=2000000)
    end_time = datetime.now()
    
    generation_time = (end_time - start_time).total_seconds()
    
    # Validar
    is_valid = generator.validate_stable(df)
    
    if is_valid:
        # Salvar
        output_file = f"data/GOLD_SYNTHETIC_STABLE_2M_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"\nSalvando: {output_file}")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        
        file_size = os.path.getsize(output_file) / 1024 / 1024
        
        print(f"\n*** SUCESSO! ***")
        print(f"Arquivo: {output_file}")
        print(f"Tamanho: {file_size:.1f} MB")
        print(f"Tempo: {generation_time:.1f} segundos")
        print(f"Velocidade: {len(df)/generation_time:.0f} barras/seg")
        
        # Mostrar distribuição final
        vol_analysis = []
        for regime in df['regime'].unique():
            regime_data = df[df['regime'] == regime]
            regime_returns = regime_data['close'].pct_change().dropna()
            if len(regime_returns) > 0:
                regime_vol = regime_returns.std() * 100
                vol_analysis.append(f"{regime}: {regime_vol:.3f}%")
        
        print(f"\nVolatilidade por regime:")
        for analysis in vol_analysis:
            print(f"  {analysis}")
        
        print(f"\n🚀 DATASET PRONTO PARA V7!")
        
    else:
        print(f"\n❌ DATASET INVÁLIDO - não foi salvo")

if __name__ == "__main__":
    main()
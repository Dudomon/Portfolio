#!/usr/bin/env python3
"""
Time Compression Seletiva - Dataset Yahoo Enhanced
Comprime períodos de baixa volatilidade mantendo eventos importantes
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TimeCompressor:
    def __init__(self, compression_rate=0.3, volatility_threshold_percentile=0.3, min_sequence_length=24):
        """
        Inicializa compressor de tempo
        
        Args:
            compression_rate: Taxa de compressão (0.3 = remove 30% das barras)
            volatility_threshold_percentile: Percentil abaixo do qual considera baixa volatilidade
            min_sequence_length: Mínimo de barras consecutivas para aplicar compressão
        """
        self.compression_rate = compression_rate
        self.volatility_threshold_percentile = volatility_threshold_percentile
        self.min_sequence_length = min_sequence_length
        
    def compress_time(self, df):
        """
        Aplica compressão temporal seletiva
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame comprimido
        """
        print(f"⚡ TIME COMPRESSION SELETIVA")
        print(f"   Taxa de compressão: {self.compression_rate*100:.1f}%")
        print(f"   Threshold volatilidade: P{self.volatility_threshold_percentile*100:.0f}")
        print(f"   Sequência mínima: {self.min_sequence_length} barras")
        
        df_compressed = df.copy()
        
        # Calcular volatilidade rolling
        returns = df['close'].pct_change()
        volatility = returns.rolling(20, min_periods=5).std()
        volatility = volatility.fillna(volatility.mean())
        
        # Determinar threshold de baixa volatilidade
        vol_threshold = volatility.quantile(self.volatility_threshold_percentile)
        low_vol_mask = volatility < vol_threshold
        
        print(f"   Threshold calculado: {vol_threshold*100:.4f}%")
        print(f"   Barras baixa volatilidade: {low_vol_mask.sum():,} ({low_vol_mask.mean()*100:.1f}%)")
        
        # Detectar eventos importantes (nunca comprimir)
        large_moves = abs(returns) > returns.quantile(0.95)  # Top 5% de movimentos
        gap_moves = abs(df['open'] / df['close'].shift(1) - 1) > 0.005  # Gaps > 0.5%
        
        # Expandir proteção ao redor de eventos importantes
        protection_window = 3  # Proteger 3 barras antes e depois
        protected_mask = large_moves.copy()
        
        for i in range(1, protection_window + 1):
            protected_mask |= large_moves.shift(i).fillna(False)
            protected_mask |= large_moves.shift(-i).fillna(False)
            protected_mask |= gap_moves.shift(i).fillna(False)
            protected_mask |= gap_moves.shift(-i).fillna(False)
        
        never_skip = large_moves | gap_moves | protected_mask
        
        print(f"   Eventos importantes protegidos: {never_skip.sum():,} barras")
        
        # Identificar sequências consecutivas de baixa volatilidade
        compression_candidates = low_vol_mask & ~never_skip
        sequences = self._find_consecutive_sequences(compression_candidates, self.min_sequence_length)
        
        print(f"   Sequências para compressão: {len(sequences)}")
        
        # Aplicar compressão nas sequências identificadas
        skip_indices = set()
        total_skipped = 0
        
        for start_idx, end_idx in sequences:
            sequence_length = end_idx - start_idx + 1
            n_to_skip = int(sequence_length * self.compression_rate)
            
            if n_to_skip > 0:
                # Escolher indices para skip uniformemente distribuídos
                skip_positions = np.linspace(start_idx, end_idx, n_to_skip + 2, dtype=int)[1:-1]
                skip_indices.update(skip_positions)
                total_skipped += n_to_skip
        
        print(f"   Total de barras a remover: {total_skipped:,}")
        
        # Criar dataset comprimido
        keep_mask = pd.Series(True, index=df.index)
        keep_mask.iloc[list(skip_indices)] = False
        
        df_compressed = df[keep_mask].reset_index(drop=True)
        
        # Validar compressão
        self._validate_compression(df, df_compressed, total_skipped)
        
        return df_compressed
        
    def _find_consecutive_sequences(self, mask, min_length):
        """Encontra sequências consecutivas de True no mask"""
        sequences = []
        current_start = None
        
        for i, value in enumerate(mask):
            if value and current_start is None:
                current_start = i
            elif not value and current_start is not None:
                sequence_length = i - current_start
                if sequence_length >= min_length:
                    sequences.append((current_start, i - 1))
                current_start = None
        
        # Verificar sequência no final
        if current_start is not None:
            sequence_length = len(mask) - current_start
            if sequence_length >= min_length:
                sequences.append((current_start, len(mask) - 1))
        
        return sequences
        
    def _validate_compression(self, df_original, df_compressed, expected_removed):
        """Valida resultado da compressão"""
        print(f"\n✅ VALIDAÇÃO TIME COMPRESSION:")
        
        original_count = len(df_original)
        compressed_count = len(df_compressed)
        actual_removed = original_count - compressed_count
        
        print(f"   Barras originais: {original_count:,}")
        print(f"   Barras comprimidas: {compressed_count:,}")
        print(f"   Barras removidas: {actual_removed:,} (esperado: {expected_removed:,})")
        print(f"   Taxa de compressão real: {actual_removed/original_count*100:.2f}%")
        
        # Verificar preservação de tendência
        trend_orig = df_original['close'].iloc[-1] / df_original['close'].iloc[0]
        trend_comp = df_compressed['close'].iloc[-1] / df_compressed['close'].iloc[0]
        
        print(f"   Tendência original: {(trend_orig-1)*100:+.2f}%")
        print(f"   Tendência comprimida: {(trend_comp-1)*100:+.2f}%")
        print(f"   Diferença: {abs((trend_comp-trend_orig)/trend_orig)*100:.2f}%")
        
        # Verificar preservação de volatilidade relativa
        vol_orig = df_original['close'].pct_change().std()
        vol_comp = df_compressed['close'].pct_change().std()
        
        print(f"   Volatilidade original: {vol_orig*100:.3f}%")
        print(f"   Volatilidade comprimida: {vol_comp*100:.3f}%")
        print(f"   Ratio: {vol_comp/vol_orig:.2f}x")

def apply_time_compression():
    """Aplica compressão temporal ao dataset enhanced"""
    
    print("⚡ APLICANDO TIME COMPRESSION")
    print("=" * 50)
    
    # Carregar dataset enhanced
    input_file = 'data/GC_YAHOO_VOLATILITY_ENHANCED_20250804_181338.csv'
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"📊 Dataset enhanced carregado:")
    print(f"   Arquivo: {input_file}")
    print(f"   Barras: {len(df):,}")
    print(f"   Período: {df['time'].min()} - {df['time'].max()}")
    
    # Aplicar compressão
    compressor = TimeCompressor(
        compression_rate=0.25,  # 25% de compressão
        volatility_threshold_percentile=0.35,  # Bottom 35%
        min_sequence_length=20  # Mínimo 20 barras consecutivas
    )
    
    df_compressed = compressor.compress_time(df)
    
    # Salvar dataset final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/GC_YAHOO_ENHANCED_COMPRESSED_{timestamp}.csv'
    df_compressed.to_csv(output_file, index=False)
    
    print(f"\n💾 DATASET COMPRIMIDO SALVO:")
    print(f"   Arquivo: {output_file}")
    print(f"   Barras finais: {len(df_compressed):,}")
    
    # Estatísticas comparativas
    compression_ratio = (len(df) - len(df_compressed)) / len(df)
    
    print(f"\n📈 RESULTADO FINAL:")
    print(f"   Compressão real: {compression_ratio*100:.2f}%")
    print(f"   Barras economizadas: {len(df) - len(df_compressed):,}")
    print(f"   Eficiência temporal: +{1/(1-compression_ratio):.1f}x mais rápido")
    
    return output_file, df_compressed

if __name__ == "__main__":
    try:
        output_file, df_compressed = apply_time_compression()
        print(f"\n✅ TIME COMPRESSION CONCLUÍDA!")
        print(f"   Dataset pronto: {output_file}")
        
    except Exception as e:
        print(f"❌ Erro na compressão: {e}")
        import traceback
        traceback.print_exc()
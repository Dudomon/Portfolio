"""
🔧 ADIÇÃO DE INDICADORES TÉCNICOS - V3.0 CRÍTICO
Adicionar indicadores essenciais para acelerar convergência RL
"""

import pandas as pd
import numpy as np
import ta
from tqdm import tqdm

def add_technical_indicators(df):
    """
    Adicionar indicadores técnicos essenciais ao dataset
    """
    print(f"🔧 Adicionando indicadores técnicos a {len(df)} barras...")
    
    # Backup das colunas originais
    original_columns = df.columns.tolist()
    
    # 1. MOVING AVERAGES (Trend Following)
    print("📊 Calculando Moving Averages...")
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    
    # 2. MOMENTUM INDICATORS
    print("📈 Calculando Momentum Indicators...")
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    
    # 3. MACD
    print("🔄 Calculando MACD...")
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['macd_histogram'] = ta.trend.macd(df['close']) - ta.trend.macd_signal(df['close'])
    
    # 4. BOLLINGER BANDS
    print("📊 Calculando Bollinger Bands...")
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # 5. VOLATILITY INDICATORS
    print("📊 Calculando Volatility Indicators...")
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['volatility_20'] = df['close'].rolling(window=20).std()
    
    # 6. VOLUME INDICATORS
    print("📊 Calculando Volume Indicators...")
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], window=20)
    df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # 7. SUPPORT/RESISTANCE LEVELS
    print("📊 Calculando Support/Resistance...")
    df['resistance_20'] = df['high'].rolling(window=20).max()
    df['support_20'] = df['low'].rolling(window=20).min()
    df['price_position'] = (df['close'] - df['support_20']) / (df['resistance_20'] - df['support_20'])
    
    # 8. TREND STRENGTH
    print("📊 Calculando Trend Strength...")
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['trend_strength'] = np.where(df['adx'] > 25, 1, np.where(df['adx'] < 20, -1, 0))
    
    # 9. PRICE ACTION PATTERNS
    print("📊 Calculando Price Action...")
    df['doji'] = np.where(abs(df['open'] - df['close']) / (df['high'] - df['low'] + 1e-8) < 0.1, 1, 0)
    df['hammer'] = np.where(
        (df['close'] > df['open']) & 
        ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) < 0.3) &
        ((df['low'] - df['open']) / (df['high'] - df['low'] + 1e-8) > 0.6), 1, 0
    )
    
    # 10. REGIME FEATURES (MELHORIA)
    print("📊 Melhorando Regime Features...")
    regime_mapping = {'bear': -1, 'bull': 1, 'sideways': 0, 'volatile': 2}
    df['regime_numeric'] = df['regime'].map(regime_mapping).fillna(0)
    
    # Rolling regime stability
    df['regime_stability'] = df['regime_numeric'].rolling(window=10).std().fillna(0)
    
    # 11. FILL MISSING VALUES
    print("🔧 Preenchendo valores missing...")
    
    # Forward fill first, then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # For any remaining NaN, fill with reasonable defaults
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # 12. QUALITY CHECK
    print("✅ Verificação de qualidade...")
    added_indicators = [col for col in df.columns if col not in original_columns]
    print(f"✅ {len(added_indicators)} indicadores adicionados:")
    for indicator in added_indicators:
        missing_pct = (df[indicator].isna().sum() / len(df)) * 100
        print(f"   {indicator}: {missing_pct:.2f}% missing")
    
    print(f"📊 Dataset final: {len(df)} barras × {len(df.columns)} features")
    
    return df

def main():
    """Processar dataset principal"""
    print("🚀 ADICIONANDO INDICADORES TÉCNICOS AO DATASET PRINCIPAL")
    print("=" * 60)
    
    # Carregar dataset
    dataset_path = 'data/GOLD_TRADING_READY_2M_20250803_222334.csv'
    print(f"📂 Carregando: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    print(f"📊 Dataset carregado: {len(df)} barras × {len(df.columns)} colunas")
    
    # Adicionar indicadores
    df_enhanced = add_technical_indicators(df)
    
    # Salvar dataset melhorado
    output_path = 'data/GOLD_TRADING_READY_2M_ENHANCED_INDICATORS.csv'
    print(f"💾 Salvando dataset melhorado: {output_path}")
    df_enhanced.to_csv(output_path, index=False)
    
    print("🎯 DATASET COM INDICADORES TÉCNICOS CRIADO COM SUCESSO!")
    print(f"📊 Original: {len(df.columns)} features")
    print(f"📊 Melhorado: {len(df_enhanced.columns)} features")
    print(f"📊 Indicadores adicionados: {len(df_enhanced.columns) - len(df.columns)}")
    
    # Atualizar daytrader.py para usar novo dataset
    update_daytrader_path()

def update_daytrader_path():
    """Atualizar caminho do dataset no daytrader.py"""
    print("\n🔧 Atualizando daytrader.py para usar dataset melhorado...")
    
    try:
        # Ler daytrader.py
        with open('daytrader.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Substituir caminho do dataset
        old_path = "dataset_path = 'data/GOLD_TRADING_READY_2M_20250803_222334.csv'"
        new_path = "dataset_path = 'data/GOLD_TRADING_READY_2M_ENHANCED_INDICATORS.csv'"
        
        if old_path in content:
            content = content.replace(old_path, new_path)
            
            # Salvar arquivo atualizado
            with open('daytrader.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ daytrader.py atualizado com novo dataset!")
        else:
            print("⚠️ Caminho do dataset não encontrado em daytrader.py")
            print("🔧 Será necessário atualizar manualmente")
            
    except Exception as e:
        print(f"❌ Erro ao atualizar daytrader.py: {e}")
        print("🔧 Será necessário atualizar manualmente")

if __name__ == "__main__":
    main()
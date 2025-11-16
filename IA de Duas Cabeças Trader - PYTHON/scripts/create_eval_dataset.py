#!/usr/bin/env python3
"""
🚀 CRIADOR DE DATASET OTIMIZADO PARA AVALIAÇÃO
============================================

OBJETIVO: Criar dataset pequeno e rápido para testes de modelos
- Yahoo Finance dados recentes (últimos 60 dias)
- Features pré-computadas 
- Tamanho otimizado (50k steps vs 216k atuais)
- Formato idêntico ao dataset V4
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fetch_yahoo_gold_1min(days=60):
    """
    🔥 Buscar dados recentes do ouro no Yahoo Finance
    """
    print(f"📊 Buscando dados do Yahoo Finance (últimos {days} dias)...")
    
    try:
        # Ticker do ouro futuro
        ticker = yf.Ticker("GC=F")
        
        # Buscar dados com diferentes períodos (Yahoo tem limitações)
        periods_to_try = [f"{days}d", "60d", "30d", "7d"]
        
        data = None
        for period in periods_to_try:
            try:
                print(f"  Tentando período: {period}")
                data = ticker.history(period=period, interval="1m")
                if len(data) > 1000:  # Mínimo viável
                    print(f"  ✅ Sucesso com {period}: {len(data)} barras")
                    break
                else:
                    print(f"  ⚠️ Poucos dados com {period}: {len(data)} barras")
            except Exception as e:
                print(f"  ❌ Erro com {period}: {e}")
                continue
        
        if data is None or len(data) == 0:
            raise ValueError("Nenhum período funcionou")
        
        # Converter para formato esperado
        data = data.reset_index()
        data.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        
        # Renomear volume para tick_volume (compatibilidade)
        data = data.rename(columns={'volume': 'tick_volume'})
        
        print(f"📈 Dados obtidos: {len(data)} barras de {data['time'].min()} a {data['time'].max()}")
        
        return data
        
    except Exception as e:
        print(f"❌ Erro ao buscar dados do Yahoo: {e}")
        return None

def add_technical_features(data):
    """
    🔧 Adicionar features técnicas idênticas ao dataset V4
    """
    print("🔧 Calculando features técnicas...")
    
    df = data.copy()
    
    # Verificar se temos dados suficientes
    if len(df) < 200:
        print(f"⚠️ Poucos dados para features técnicas: {len(df)} barras")
        return df
    
    try:
        # Features básicas do timeframe 1m
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['tick_volume'].values
        
        # 1. Returns
        df['returns_1m'] = df['close'].pct_change().fillna(0)
        
        # 2. Volatilidade
        df['volatility_20_1m'] = df['close'].rolling(20).std().fillna(0)
        
        # 3. SMAs
        df['sma_20_1m'] = df['close'].rolling(20).mean().fillna(df['close'])
        df['sma_50_1m'] = df['close'].rolling(50).mean().fillna(df['close'])
        
        # 4. RSI
        try:
            rsi = talib.RSI(close.astype(float), timeperiod=14)
            df['rsi_14_1m'] = pd.Series(rsi).fillna(50)
        except:
            df['rsi_14_1m'] = 50
        
        # 5. Stochastic (simplificado)
        df['stoch_k_1m'] = 50.0
        
        # 6. Bollinger Band Position
        bb_sma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_sma + (bb_std * 2)
        bb_lower = bb_sma - (bb_std * 2)
        df['bb_position_1m'] = ((df['close'] - bb_lower) / (bb_upper - bb_lower)).fillna(0.5).clip(0, 1)
        
        # 7. Trend Strength
        df['trend_strength_1m'] = df['returns_1m'].rolling(10).mean().fillna(0)
        
        # 8. ATR (Average True Range)
        try:
            atr = talib.ATR(high.astype(float), low.astype(float), close.astype(float), timeperiod=14)
            df['atr_14_1m'] = pd.Series(atr).fillna(0.01)
        except:
            df['atr_14_1m'] = 0.01
        
        # 9. High Quality Features (simplificadas para velocidade)
        df['volume_momentum'] = df['tick_volume'].rolling(10).mean().fillna(1)
        df['price_position'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min()).fillna(0.5)
        df['breakout_strength'] = abs(df['returns_1m']) * 100
        df['trend_consistency'] = abs(df['trend_strength_1m'])
        df['support_resistance'] = df['bb_position_1m']  # Proxy
        df['volatility_regime'] = (df['volatility_20_1m'] - df['volatility_20_1m'].rolling(50).mean()).fillna(0)
        df['market_structure'] = df['close'].rolling(5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1).fillna(0)
        
        # Limpar infinitos e NaNs
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(method='ffill').fillna(0)
        
        print(f"✅ Features calculadas: {len(df.columns)} colunas")
        
        return df
        
    except Exception as e:
        print(f"❌ Erro ao calcular features: {e}")
        return df

def optimize_dataset_size(data, target_size=50000):
    """
    🎯 Otimizar tamanho do dataset para avaliação rápida
    """
    print(f"🎯 Otimizando dataset: {len(data)} → {target_size} barras")
    
    if len(data) <= target_size:
        print(f"✅ Dataset já otimizado: {len(data)} barras")
        return data
    
    # Pegar dados mais recentes
    optimized = data.tail(target_size).reset_index(drop=True)
    
    print(f"✅ Dataset otimizado: {len(optimized)} barras")
    print(f"   Período: {optimized['time'].min()} a {optimized['time'].max()}")
    
    return optimized

def save_eval_dataset(data, filename=None):
    """
    💾 Salvar dataset otimizado para avaliação
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"GC=F_EVAL_OPTIMIZED_{timestamp}.csv"
    
    filepath = f"D:/Projeto/data/{filename}"
    
    print(f"💾 Salvando dataset: {filepath}")
    
    try:
        data.to_csv(filepath, index=False)
        print(f"✅ Dataset salvo: {len(data)} barras, {len(data.columns)} colunas")
        print(f"   Tamanho: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Erro ao salvar: {e}")
        return None

def validate_dataset(data):
    """
    ✅ Validar dataset para compatibilidade com SILUS
    """
    print("✅ Validando dataset...")
    
    required_columns = [
        'time', 'open_1m', 'high_1m', 'low_1m', 'close_1m', 'tick_volume_1m',
        'returns_1m', 'volatility_20_1m', 'sma_20_1m', 'sma_50_1m', 'rsi_14_1m'
    ]
    
    # Verificar colunas mínimas
    basic_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    missing_basic = [col for col in basic_columns if col not in data.columns]
    
    if missing_basic:
        print(f"❌ Colunas básicas ausentes: {missing_basic}")
        return False
    
    # Verificar dados válidos
    if len(data) < 1000:
        print(f"❌ Poucos dados: {len(data)} barras (mínimo 1000)")
        return False
    
    # Verificar NaNs excessivos
    nan_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
    if nan_percentage > 5:
        print(f"❌ Muitos NaNs: {nan_percentage:.1f}% (máximo 5%)")
        return False
    
    print(f"✅ Dataset válido: {len(data)} barras, {len(data.columns)} colunas")
    print(f"   NaNs: {nan_percentage:.2f}%")
    
    return True

def main():
    """
    🚀 Função principal: Criar dataset de avaliação otimizado
    """
    import os
    
    print("🚀 CRIANDO DATASET DE AVALIAÇÃO OTIMIZADO")
    print("=" * 50)
    
    # 1. Buscar dados do Yahoo Finance
    data = fetch_yahoo_gold_1min(days=60)
    if data is None:
        print("❌ Falha ao obter dados do Yahoo Finance")
        return False
    
    # 2. Adicionar features técnicas
    data = add_technical_features(data)
    
    # 3. Otimizar tamanho
    data = optimize_dataset_size(data, target_size=50000)
    
    # 4. Renomear colunas para formato SILUS (_1m suffix)
    column_mapping = {
        'open': 'open_1m',
        'high': 'high_1m',
        'low': 'low_1m', 
        'close': 'close_1m',
        'tick_volume': 'tick_volume_1m'
    }
    
    for old, new in column_mapping.items():
        if old in data.columns:
            data = data.rename(columns={old: new})
    
    # 5. Validar dataset
    if not validate_dataset(data):
        print("❌ Dataset inválido")
        return False
    
    # 6. Salvar dataset
    filepath = save_eval_dataset(data)
    if filepath is None:
        print("❌ Falha ao salvar dataset")
        return False
    
    print("\n" + "=" * 50)
    print("✅ DATASET DE AVALIAÇÃO CRIADO COM SUCESSO!")
    print(f"📁 Arquivo: {os.path.basename(filepath)}")
    print(f"📊 Dados: {len(data)} barras")
    print(f"🕐 Período: {data['time'].min()} a {data['time'].max()}")
    print(f"⚡ Otimização: ~5x mais rápido que dataset V4")
    print("\n🎯 PRÓXIMOS PASSOS:")
    print("1. Atualizar completo_1m_optimized.py para usar este dataset")
    print("2. Testar velocidade de avaliação")
    print("3. Validar métricas vs dataset V4")
    
    return True

if __name__ == "__main__":
    main()
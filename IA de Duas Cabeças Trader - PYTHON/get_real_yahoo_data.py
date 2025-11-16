#!/usr/bin/env python3
"""
📊 BAIXAR DADOS YAHOO FINANCE ORGÂNICOS
Dados reais do GC=F (Gold Futures) sem modificações artificiais
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def download_real_gold_data():
    """Baixa dados reais do Yahoo Finance para GC=F (Gold Futures)"""
    
    print("📊 Baixando dados reais do Yahoo Finance...")
    
    # Definir período - últimos 2 anos de dados intraday
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 anos
    
    try:
        # Baixar dados de 5 minutos (máximo disponível no Yahoo)
        ticker = yf.Ticker("GC=F")
        
        # Tentar diferentes intervalos
        intervals = ["5m", "15m", "1h"]
        df = None
        
        for interval in intervals:
            try:
                print(f"  Tentando intervalo {interval}...")
                df = ticker.history(
                    period="2y",
                    interval=interval,
                    auto_adjust=True,
                    prepost=True
                )
                if len(df) > 1000:  # Se conseguiu dados suficientes
                    print(f"  ✅ Sucesso com {interval}: {len(df)} barras")
                    break
            except Exception as e:
                print(f"  ❌ Erro com {interval}: {e}")
                continue
        
        if df is None or len(df) < 100:
            print("❌ Falha ao baixar dados do Yahoo")
            return None
            
        # Resetar índice para ter timestamp como coluna
        df.reset_index(inplace=True)
        df.rename(columns={'Datetime': 'time'}, inplace=True)
        
        # Renomear colunas para padrão esperado
        column_mapping = {
            'Open': 'open_5m',
            'High': 'high_5m', 
            'Low': 'low_5m',
            'Close': 'close_5m',
            'Volume': 'volume_5m'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Adicionar apenas features básicas REAIS
        df['returns_5m'] = df['close_5m'].pct_change()
        df['volatility_5m'] = df['returns_5m'].rolling(20).std()
        df['sma_20'] = df['close_5m'].rolling(20).mean()
        df['sma_50'] = df['close_5m'].rolling(50).mean()
        
        # RSI simples
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = calculate_rsi(df['close_5m'])
        
        # ATR simples
        df['tr'] = np.maximum(df['high_5m'] - df['low_5m'],
                             np.maximum(abs(df['high_5m'] - df['close_5m'].shift(1)),
                                       abs(df['low_5m'] - df['close_5m'].shift(1))))
        df['atr_14'] = df['tr'].rolling(14).mean()
        
        # Remover colunas auxiliares
        df.drop(['tr'], axis=1, inplace=True)
        
        # Remover linhas com NaN (das janelas móveis)
        df.dropna(inplace=True)
        
        # Salvar dataset orgânico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"D:/Projeto/data/GC_YAHOO_ORGANIC_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        
        print(f"✅ Dataset orgânico salvo: {filename}")
        print(f"📊 Período: {df['time'].min()} até {df['time'].max()}")
        print(f"📊 Total de barras: {len(df)}")
        print(f"📊 Colunas: {list(df.columns)}")
        
        # Estatísticas básicas
        print(f"\n📈 ESTATÍSTICAS BÁSICAS:")
        print(f"  Preço médio: ${df['close_5m'].mean():.2f}")
        print(f"  Volatilidade média: {df['volatility_5m'].mean()*100:.3f}%")
        print(f"  Volume médio: {df['volume_5m'].mean():,.0f}")
        print(f"  Range de preços: ${df['close_5m'].min():.2f} - ${df['close_5m'].max():.2f}")
        
        return filename
        
    except Exception as e:
        print(f"❌ Erro ao baixar dados: {e}")
        return None

if __name__ == "__main__":
    filename = download_real_gold_data()
    if filename:
        print(f"\n🎯 DATASET ORGÂNICO PRONTO: {filename}")
    else:
        print(f"\n❌ FALHA AO CRIAR DATASET ORGÂNICO")
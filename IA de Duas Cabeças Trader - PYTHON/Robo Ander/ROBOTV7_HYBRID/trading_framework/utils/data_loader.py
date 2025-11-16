#!/usr/bin/env python3
"""
📊 MÓDULO DE CARREGAMENTO DE DADOS INDEPENDENTE
Sistema de carregamento e processamento de dados de trading
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import pickle
import logging


class DataLoader:
    """📊 Sistema de carregamento de dados otimizado"""
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "data_cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_optimized_data(self, filename: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Carrega dados otimizados com cache
        Suporta arquivos .pkl (pickle) e .csv automaticamente.
        Se volume_5m não existir, tenta usar real_volume_5m como volume_5m.
        """
        cache_file = self.cache_dir / f"{Path(filename).stem}.pkl"
        data_file = self.data_dir / filename

        # Tentar carregar cache
        if use_cache and cache_file.exists():
            try:
                print(f"📊 Carregando cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Erro ao carregar cache: {e}")
                df = None
            else:
                # Ajuste de volume
                if 'volume_5m' not in df.columns and 'real_volume_5m' in df.columns:
                    df['volume_5m'] = df['real_volume_5m']
                return df

        if not data_file.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {data_file}")

        # Carregar arquivo conforme extensão
        if filename.endswith('.pkl'):
            print(f"📊 Carregando pickle: {data_file}")
            df = pd.read_pickle(data_file)
        elif filename.endswith('.csv'):
            print(f"📊 Carregando CSV: {data_file}")
            df = pd.read_csv(data_file)
            # Converter timestamp se existir
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {filename}")

        # Otimizar tipos e limpar se for CSV
        if filename.endswith('.csv'):
            df = self._optimize_dtypes(df)
            df = self._clean_data(df)
            df = self._add_technical_features(df)

        # Ajuste de volume
        if 'volume_5m' not in df.columns and 'real_volume_5m' in df.columns:
            df['volume_5m'] = df['real_volume_5m']

        # Salvar cache
        if use_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                print(f"💾 Cache salvo: {cache_file}")
            except Exception as e:
                print(f"⚠️ Erro ao salvar cache: {e}")

        return df
    
    def _process_csv_data(self, filepath: Path) -> pd.DataFrame:
        """Processa dados CSV com otimizações"""
        
        # Carregar CSV
        df = pd.read_csv(filepath)
        
        # Converter timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Otimizar tipos de dados
        df = self._optimize_dtypes(df)
        
        # Limpar dados
        df = self._clean_data(df)
        
        # Adicionar features técnicas
        df = self._add_technical_features(df)
        
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Otimiza tipos de dados para reduzir memória"""
        
        # Converter colunas numéricas para float32
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa dados removendo outliers e NaN"""
        
        # Remover linhas com NaN
        df = df.dropna()
        
        # Remover outliers extremos (99.9% percentile)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            q99 = df[col].quantile(0.999)
            q01 = df[col].quantile(0.001)
            df = df[(df[col] >= q01) & (df[col] <= q99)]
        
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features técnicas básicas"""
        
        # RSI
        if 'close_5m' in df.columns:
            df['rsi_5m'] = self._calculate_rsi(df['close_5m'], 14)
        
        # SMA
        if 'close_5m' in df.columns:
            df['sma_20_5m'] = df['close_5m'].rolling(20).mean()
            df['sma_50_5m'] = df['close_5m'].rolling(50).mean()
        
        # ATR
        if all(col in df.columns for col in ['high_5m', 'low_5m', 'close_5m']):
            df['atr_5m'] = self._calculate_atr(df['high_5m'], df['low_5m'], df['close_5m'], 14)
        
        # Bollinger Bands
        if 'close_5m' in df.columns:
            bb_upper, bb_lower = self._calculate_bollinger_bands(df['close_5m'], 20, 2)
            df['bb_upper_5m'] = bb_upper
            df['bb_lower_5m'] = bb_lower
            df['bb_position_5m'] = (df['close_5m'] - bb_lower) / (bb_upper - bb_lower)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calcula ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calcula Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def get_latest_processed_file(self, pattern: str = "*.csv") -> Optional[str]:
        """Encontra o arquivo mais recente processado"""
        csv_files = list(self.data_dir.glob(pattern))
        if not csv_files:
            return None
        
        # Ordenar por data de modificação
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        return latest_file.name
    
    def clear_cache(self) -> None:
        """Limpa todos os arquivos de cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                print(f"🗑️ Cache removido: {cache_file}")
            except Exception as e:
                print(f"⚠️ Erro ao remover cache {cache_file}: {e}")


class DataValidator:
    """🔍 Validador de dados"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> Dict[str, Any]:
        """Valida DataFrame e retorna relatório"""
        
        report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Verificar colunas obrigatórias
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                report['is_valid'] = False
                report['errors'].append(f"Colunas faltando: {missing_columns}")
        
        # Verificar dados
        if df.empty:
            report['is_valid'] = False
            report['errors'].append("DataFrame vazio")
        
        # Verificar NaN
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            report['warnings'].append(f"NaN encontrados: {nan_counts.to_dict()}")
        
        # Verificar duplicatas
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            report['warnings'].append(f"Linhas duplicadas: {duplicates}")
        
        # Informações básicas
        report['info'] = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        return report
    
    @staticmethod
    def print_validation_report(report: Dict[str, Any]) -> None:
        """Imprime relatório de validação"""
        
        print("🔍 RELATÓRIO DE VALIDAÇÃO DE DADOS")
        print("=" * 50)
        
        if report['is_valid']:
            print("✅ Dados válidos")
        else:
            print("❌ Dados inválidos")
        
        if report['errors']:
            print("\n❌ ERROS:")
            for error in report['errors']:
                print(f"   - {error}")
        
        if report['warnings']:
            print("\n⚠️ AVISOS:")
            for warning in report['warnings']:
                print(f"   - {warning}")
        
        print("\n📊 INFORMAÇÕES:")
        info = report['info']
        print(f"   Shape: {info['shape']}")
        print(f"   Colunas: {len(info['columns'])}")
        print(f"   Memória: {info['memory_usage'] / 1024 / 1024:.2f} MB")
        
        print("=" * 50) 
#!/usr/bin/env python3
"""
🛠️ FUNÇÕES AUXILIARES REUTILIZÁVEIS
Utilitários gerais para o trading framework
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
import gc
import time


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """Configura sistema de logging"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "trading_framework.log"),
            logging.StreamHandler()
        ]
    )


def setup_torch_optimizations(device: str = "auto", num_threads: int = 4) -> torch.device:
    """Configura otimizações do PyTorch"""
    
    # Configurar threads
    torch.set_num_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    
    # Detectar device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🚀 GPU detectada: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("💻 Usando CPU")
    else:
        device = torch.device(device)
    
    # Configurações específicas para RTX 4070Ti
    if device.type == "cuda":
        # Desabilitar TF32 para RTX 4070Ti (problemas conhecidos)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        # Configurar cuDNN
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Configurar memory pool
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.cuda.empty_cache()
    
    return device


def memory_cleanup() -> None:
    """Limpa memória do sistema"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def calculate_portfolio_metrics(portfolio_values: List[float], trades: List[Dict]) -> Dict[str, float]:
    """Calcula métricas básicas do portfólio"""
    
    if not portfolio_values:
        return {
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'total_trades': 0,
            'win_rate': 0.0
        }
    
    # Retornos
    portfolio_array = np.array(portfolio_values)
    returns = np.diff(portfolio_array) / portfolio_array[:-1]
    
    # Retorno total
    total_return = (portfolio_array[-1] - portfolio_array[0]) / portfolio_array[0]
    
    # Maximum drawdown
    peak = np.maximum.accumulate(portfolio_array)
    drawdown = (peak - portfolio_array) / peak
    max_drawdown = np.max(drawdown)
    
    # Volatilidade
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
    
    # Sharpe ratio
    if volatility > 0:
        sharpe_ratio = (np.mean(returns) * 252) / volatility
    else:
        sharpe_ratio = 0.0
    
    # Métricas de trading
    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade.get('pnl_usd', 0) > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    return {
        'total_return': float(total_return),
        'max_drawdown': float(max_drawdown),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'total_trades': int(total_trades),
        'win_rate': float(win_rate)
    }


def format_number(value: float, decimals: int = 2, as_percentage: bool = False) -> str:
    """Formata número para exibição"""
    if as_percentage:
        return f"{value * 100:.{decimals}f}%"
    else:
        return f"{value:.{decimals}f}"


def format_currency(value: float) -> str:
    """Formata valor como moeda"""
    return f"${value:,.2f}"


def format_duration(seconds: float) -> str:
    """Formata duração em segundos para formato legível"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "*_steps") -> Optional[str]:
    """Encontra o checkpoint mais recente"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return None
    
    checkpoints = list(checkpoint_path.glob(pattern))
    if not checkpoints:
        return None
    
    # Ordenar por data de modificação
    latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
    return str(latest)


def extract_steps_from_checkpoint(checkpoint_path: str) -> Optional[int]:
    """Extrai número de steps do nome do checkpoint"""
    try:
        filename = Path(checkpoint_path).name
        # Procurar por padrão *_steps
        if '_steps' in filename:
            steps_str = filename.split('_steps')[0].split('_')[-1]
            return int(steps_str)
    except (ValueError, IndexError):
        pass
    return None


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Cria barra de progresso"""
    progress = current / total
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    percentage = progress * 100
    return f"[{bar}] {percentage:.1f}% ({current:,}/{total:,})"


def print_training_status(step: int, total_steps: int, metrics: Dict[str, float]) -> None:
    """Imprime status do treinamento"""
    progress_bar = create_progress_bar(step, total_steps)
    
    print(f"\n🚀 TREINAMENTO - {progress_bar}")
    print(f"   💰 Portfolio: {format_currency(metrics.get('portfolio_value', 0))}")
    print(f"   📊 Trades: {metrics.get('total_trades', 0)}")
    print(f"   🏆 Win Rate: {format_number(metrics.get('win_rate', 0), as_percentage=True)}")
    print(f"   📈 Sharpe: {format_number(metrics.get('sharpe_ratio', 0))}")
    print(f"   📉 Max DD: {format_number(metrics.get('max_drawdown', 0), as_percentage=True)}")


def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """Valida se dataframe tem colunas necessárias"""
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            return False
    
    return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divisão segura com valor padrão para divisão por zero"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ValueError):
        return default


def clip_value(value: float, min_val: float, max_val: float) -> float:
    """Limita valor entre mínimo e máximo"""
    return max(min_val, min(value, max_val))


def calculate_rolling_metrics(data: np.ndarray, window: int = 20) -> Dict[str, np.ndarray]:
    """Calcula métricas rolantes"""
    if len(data) < window:
        return {
            'mean': np.array([]),
            'std': np.array([]),
            'min': np.array([]),
            'max': np.array([])
        }
    
    # Rolling mean
    mean = np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Rolling std
    std = np.array([np.std(data[i:i+window]) for i in range(len(data)-window+1)])
    
    # Rolling min/max
    min_vals = np.array([np.min(data[i:i+window]) for i in range(len(data)-window+1)])
    max_vals = np.array([np.max(data[i:i+window]) for i in range(len(data)-window+1)])
    
    return {
        'mean': mean,
        'std': std,
        'min': min_vals,
        'max': max_vals
    }


def time_function(func):
    """Decorator para medir tempo de execução"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ {func.__name__} executou em {end_time - start_time:.3f}s")
        return result
    return wrapper


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator para retry em caso de erro"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"⚠️ Tentativa {attempt + 1} falhou: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def print_system_info() -> None:
    """Imprime informações do sistema"""
    print("🖥️ INFORMAÇÕES DO SISTEMA")
    print("=" * 40)
    
    # CPU
    import multiprocessing
    print(f"💻 CPU: {multiprocessing.cpu_count()} cores")
    
    # GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("💻 GPU: Não disponível")
    
    # Memória
    import psutil
    memory = psutil.virtual_memory()
    print(f"💾 RAM: {memory.total / 1e9:.1f}GB total, {memory.available / 1e9:.1f}GB disponível")
    
    # Python
    import sys
    print(f"🐍 Python: {sys.version}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    print("=" * 40) 
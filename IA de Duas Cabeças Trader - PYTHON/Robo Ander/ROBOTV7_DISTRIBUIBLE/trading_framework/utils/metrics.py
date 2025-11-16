#!/usr/bin/env python3
"""
📊 MÓDULO DE MÉTRICAS INDEPENDENTE
Sistema de métricas de portfólio e trading reutilizável
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union
from collections import deque


class PortfolioMetrics:
    """📊 Sistema de métricas de portfólio robusto e reutilizável"""
    
    @staticmethod
    def calculate_metrics(returns: np.ndarray, trades: list, start_date: datetime, 
                         end_date: datetime, df: Optional[pd.DataFrame] = None, 
                         peak_portfolio: Optional[float] = None) -> Dict[str, float]:
        """
        Calcula métricas completas de portfólio com proteção robusta
        
        Args:
            returns: Array de retornos
            trades: Lista de trades
            start_date: Data inicial
            end_date: Data final
            df: DataFrame opcional para cálculo de dias de trading
            peak_portfolio: Valor máximo do portfólio (opcional)
            
        Returns:
            Dict com todas as métricas calculadas
        """
        if len(returns) == 0 or len(trades) == 0:
            return {
                'annual_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'trades_per_day': 0.0,
                'trading_days': 0,
                'peak_drawdown': 0.0,
                'lucro_medio_dia': 0.0,
                'peak_portfolio': 0.0,
            }

        # Calcular métricas básicas
        returns = np.array(returns)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calcular dias de trading
        if df is not None:
            unique_days = pd.to_datetime(df.index.normalize()).unique()
            trading_days = len(unique_days)
        else:
            trading_days = max((end_date - start_date).days, 1)

        # Calcular retorno total e anualizado
        total_return = np.sum(returns)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else total_return

        # Calcular volatilidade
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

        # Sharpe Ratio com proteção contra valores infinitos
        risk_free_rate = 0.0
        if volatility > 1e-8 and not np.isnan(volatility) and not np.isinf(volatility):
            sharpe_ratio = (annual_return - risk_free_rate) / volatility
            sharpe_ratio = np.clip(sharpe_ratio, -100, 100)
        else:
            sharpe_ratio = 0.0

        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / running_max
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 1.0
        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0.0

        # Calmar Ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0

        # Métricas de trading
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.get('pnl_usd', 0) > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        trades_per_day = total_trades / trading_days if trading_days > 0 else 0.0

        # Peak drawdown
        peak_drawdown = max_drawdown

        # Lucro médio por dia
        lucro_medio_dia = total_return / trading_days if trading_days > 0 else 0.0

        # Peak portfolio
        peak_portfolio_val = peak_portfolio if peak_portfolio is not None else 0.0

        # Proteção final: Limpar todos os valores de NaN/Inf
        def safe_float(val, default=0.0):
            if np.isnan(val) or np.isinf(val):
                return default
            return np.clip(float(val), -1e6, 1e6)

        return {
            'annual_return': safe_float(annual_return),
            'volatility': safe_float(volatility),
            'sharpe_ratio': safe_float(sharpe_ratio),
            'max_drawdown': safe_float(max_drawdown),
            'sortino_ratio': safe_float(sortino_ratio),
            'calmar_ratio': safe_float(calmar_ratio),
            'total_trades': int(total_trades),
            'win_rate': safe_float(win_rate),
            'trades_per_day': safe_float(trades_per_day),
            'trading_days': int(trading_days),
            'peak_drawdown': safe_float(peak_drawdown),
            'lucro_medio_dia': safe_float(lucro_medio_dia),
            'peak_portfolio': safe_float(peak_portfolio_val),
        }


class TradingMetrics:
    """📈 Métricas específicas de trading em tempo real"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.returns_buffer = deque(maxlen=window_size)
        self.portfolio_buffer = deque(maxlen=window_size)
        self.drawdown_buffer = deque(maxlen=window_size)
        self.trades_buffer = deque(maxlen=window_size)
        
    def update(self, portfolio_value: float, returns: Union[float, np.ndarray], 
               drawdown: float, trades: List[Dict], current_step: int) -> Dict[str, float]:
        """Atualiza métricas em tempo real"""
        
        # Processar retornos
        if isinstance(returns, (list, np.ndarray)):
            if len(returns) > 0:
                returns_scalar = float(returns[-1]) if hasattr(returns, '__len__') else float(returns)
            else:
                returns_scalar = 0.0
        else:
            returns_scalar = float(returns) if returns else 0.0
            
        self.returns_buffer.append(returns_scalar)
        self.portfolio_buffer.append(float(portfolio_value))
        self.drawdown_buffer.append(float(drawdown))
        
        if len(self.returns_buffer) >= 10:
            return self._calculate_advanced_metrics(portfolio_value, trades, current_step)
        else:
            return self._calculate_basic_metrics(trades, drawdown, portfolio_value)
    
    def _calculate_basic_metrics(self, trades: List[Dict], drawdown: float, 
                                portfolio_value: float) -> Dict[str, float]:
        """Calcula métricas básicas quando há poucos dados"""
        return {
            'sharpe_ratio': 0.0,
            'win_rate': len([t for t in trades if t.get('pnl_usd', 0) > 0]) / len(trades) if trades else 0.0,
            'profit_factor': 0.0,
            'risk_score': 0.5,
            'current_dd': drawdown,
            'max_dd': drawdown,
            'portfolio_value': portfolio_value,
            'data_points': len(self.returns_buffer)
        }
    
    def _calculate_advanced_metrics(self, portfolio_value: float, trades: List[Dict], 
                                   current_step: int) -> Dict[str, float]:
        """Calcula métricas avançadas com dados suficientes"""
        returns_array = np.array(list(self.returns_buffer))
        
        # Sharpe ratio
        if len(returns_array) > 1:
            volatility = np.std(returns_array) * np.sqrt(252)
            mean_return = np.mean(returns_array) * 252
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Win rate
        win_rate = len([t for t in trades if t.get('pnl_usd', 0) > 0]) / len(trades) if trades else 0.0
        
        # Profit factor
        if trades:
            gross_profit = sum(t.get('pnl_usd', 0) for t in trades if t.get('pnl_usd', 0) > 0)
            gross_loss = abs(sum(t.get('pnl_usd', 0) for t in trades if t.get('pnl_usd', 0) < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        else:
            profit_factor = 0.0
        
        # Risk score (0-1, onde 1 é mais arriscado)
        max_dd = max(self.drawdown_buffer) if self.drawdown_buffer else 0.0
        risk_score = min(max_dd, 1.0)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'risk_score': risk_score,
            'current_dd': max(self.drawdown_buffer) if self.drawdown_buffer else 0.0,
            'max_dd': max_dd,
            'portfolio_value': portfolio_value,
            'data_points': len(self.returns_buffer)
        }


class MetricsFormatter:
    """🎨 Formatador de métricas para exibição"""
    
    @staticmethod
    def format_metrics(metrics: Dict[str, float], format_type: str = "summary") -> str:
        """Formata métricas para exibição"""
        
        if format_type == "summary":
            return MetricsFormatter._format_summary(metrics)
        elif format_type == "detailed":
            return MetricsFormatter._format_detailed(metrics)
        elif format_type == "compact":
            return MetricsFormatter._format_compact(metrics)
        else:
            return str(metrics)
    
    @staticmethod
    def _format_summary(metrics: Dict[str, float]) -> str:
        """Formato resumido das métricas principais"""
        lines = []
        lines.append("📊 MÉTRICAS DE PORTFÓLIO")
        lines.append("=" * 40)
        
        # Portfolio
        if 'portfolio_value' in metrics:
            lines.append(f"💰 Portfolio: ${metrics['portfolio_value']:.2f}")
        if 'peak_portfolio' in metrics:
            lines.append(f"📈 Peak: ${metrics['peak_portfolio']:.2f}")
        
        # Trading
        if 'trades_per_day' in metrics:
            lines.append(f"🎯 Trades/dia: {metrics['trades_per_day']:.1f}")
        if 'win_rate' in metrics:
            lines.append(f"🏆 Win rate: {metrics['win_rate']:.1%}")
        if 'total_trades' in metrics:
            lines.append(f"📊 Total trades: {metrics['total_trades']}")
        
        # Performance
        if 'sharpe_ratio' in metrics:
            lines.append(f"📊 Sharpe: {metrics['sharpe_ratio']:.2f}")
        if 'max_drawdown' in metrics:
            lines.append(f"📉 Max DD: {metrics['max_drawdown']:.1%}")
        if 'annual_return' in metrics:
            lines.append(f"📈 Retorno anual: {metrics['annual_return']:.1%}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_detailed(metrics: Dict[str, float]) -> str:
        """Formato detalhado com todas as métricas"""
        lines = []
        lines.append("📊 MÉTRICAS DETALHADAS")
        lines.append("=" * 50)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'rate' in key or 'drawdown' in key:
                    lines.append(f"   {key}: {value:.1%}")
                elif 'portfolio' in key or 'pnl' in key:
                    lines.append(f"   {key}: ${value:.2f}")
                elif 'duration' in key:
                    lines.append(f"   {key}: {value:.1f}h")
                else:
                    lines.append(f"   {key}: {value:.2f}")
            else:
                lines.append(f"   {key}: {value}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_compact(metrics: Dict[str, float]) -> str:
        """Formato compacto para logs"""
        compact = []
        
        if 'portfolio_value' in metrics:
            compact.append(f"P:${metrics['portfolio_value']:.0f}")
        if 'trades_per_day' in metrics:
            compact.append(f"T:{metrics['trades_per_day']:.1f}")
        if 'win_rate' in metrics:
            compact.append(f"W:{metrics['win_rate']:.0%}")
        if 'sharpe_ratio' in metrics:
            compact.append(f"S:{metrics['sharpe_ratio']:.1f}")
        if 'max_drawdown' in metrics:
            compact.append(f"DD:{metrics['max_drawdown']:.0%}")
        
        return " | ".join(compact) 
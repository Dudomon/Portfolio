"""
🎯 ACTIVITY ENHANCEMENT SYSTEM
Sistema para aumentar atividade de trading durante o treinamento

Funcionalidades:
1. Position Timeout - Fechar posições após N candles
2. Dynamic SL/TP - Alvos baseados em volatilidade
3. Activity Monitoring - Tracking de atividade de trading
4. Exploration Incentives - Incentivar tentativas de entrada
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque

@dataclass
class ActivityConfig:
    """Configuração do sistema de activity enhancement"""
    position_timeout_candles: int = 60      # Timeout 5h para posições (48->60)
    target_activity_rate: float = 0.15      # Taxa alvo de atividade (15% do tempo em posição)
    dynamic_sl_factor: float = 1.5          # Multiplicador da volatilidade para SL
    dynamic_tp_factor: float = 2.5          # Multiplicador da volatilidade para TP
    min_sl_percent: float = 0.005           # SL mínimo (0.5%)
    max_sl_percent: float = 0.03            # SL máximo (3%)
    min_tp_percent: float = 0.01            # TP mínimo (1%)
    max_tp_percent: float = 0.05            # TP máximo (5%)
    volatility_window: int = 20             # Janela para cálculo de volatilidade
    progressive_timeout: bool = True        # Ativar sistema progressivo de timeout
    training_steps_total: int = 12000000    # Total de steps do treinamento

class ActivityEnhancementSystem:
    """
    Sistema para aumentar atividade de trading
    Integra com environment existente sem quebrar compatibilidade
    """
    
    def __init__(self, config: Optional[ActivityConfig] = None):
        self.config = config or ActivityConfig()
        
        # Tracking de atividade
        self.position_start_step = None
        self.position_steps = 0
        self.total_steps = 0
        self.trades_count = 0
        self.last_activity_check = 0
        
        # Sistema progressivo de timeout
        self.training_steps_global = 0  # Steps globais do treinamento
        self.current_timeout = self.config.position_timeout_candles
        
        # Volatilidade dinâmica
        self.price_history = deque(maxlen=self.config.volatility_window)
        self.current_volatility = 0.01  # 1% padrão
        
        # Métricas de activity
        self.activity_metrics = {
            'forced_closes': 0,
            'dynamic_targets_used': 0,
            'activity_rate_current': 0.0,
            'avg_position_duration': 0.0,
            'current_timeout_candles': self.current_timeout
        }
        
        logging.info("🎯 [ACTIVITY] Activity Enhancement System inicializado")
    
    def update_training_progress(self, global_steps: int):
        """
        🎯 Atualiza progresso do treinamento para timeout progressivo
        """
        self.training_steps_global = global_steps
        
        if self.config.progressive_timeout:
            # 🎯 Sistema automático baseado em % do progresso total
            progress = global_steps / self.config.training_steps_total
            
            if progress < 0.4:  # Primeiros 40% (início: timeout rígido para forçar atividade)
                self.current_timeout = 60  # 5h (rígido)
            elif progress < 0.8:  # 40-80% (meio: timeout flexível)  
                self.current_timeout = 90  # 7.5h (flexível)
            else:  # Últimos 20% (final: sem timeout - só SL/TP naturais)
                self.current_timeout = 999999  # Sem timeout (maduro)
            
            # Atualizar métricas
            self.activity_metrics['current_timeout_candles'] = self.current_timeout
            
            # Log mudanças significativas APENAS nos milestones (anti-spam)
            if global_steps % 100000 == 0 and not hasattr(self, f'_logged_milestone_{global_steps}'):
                timeout_status = "RÍGIDO" if self.current_timeout == 60 else "FLEXÍVEL" if self.current_timeout == 90 else "DESABILITADO"
                logging.info(f"🎯 [TIMEOUT PROGRESSIVO] Step {global_steps:,}: {self.current_timeout} candles ({timeout_status})")
                # Marcar milestone como logado para evitar spam
                setattr(self, f'_logged_milestone_{global_steps}', True)
    
    def on_step(self, env, action: np.ndarray) -> Dict[str, Any]:
        """
        Chamado a cada step do environment
        Retorna informações sobre ações tomadas
        """
        self.total_steps += 1
        info = {}
        
        # 1. Atualizar tracking de posição
        current_position = getattr(env, 'current_position', 0)
        
        if current_position != 0:
            if self.position_start_step is None:
                self.position_start_step = self.total_steps
            self.position_steps = self.total_steps - self.position_start_step
        else:
            self.position_start_step = None
            self.position_steps = 0
        
        # 2. Check position timeout
        timeout_action = self._check_position_timeout(env, current_position)
        if timeout_action:
            info['position_timeout'] = True
            info['timeout_reason'] = f"Position open for {self.position_steps} candles"
            self.activity_metrics['forced_closes'] += 1
        
        # 3. Atualizar volatilidade dinâmica
        self._update_volatility(env)
        
        # 4. Calcular SL/TP dinâmicos se necessário
        if current_position == 0 and self._is_entry_action(action):
            sl_tp = self._calculate_dynamic_sl_tp(env, action)
            if sl_tp:
                info['dynamic_targets'] = sl_tp
                self.activity_metrics['dynamic_targets_used'] += 1
                # Sinalizar para environment usar targets dinâmicos
                if hasattr(env, 'set_dynamic_targets'):
                    env.set_dynamic_targets(sl_tp['sl_percent'], sl_tp['tp_percent'])
                    env.using_dynamic_targets = True
        
        # 5. Activity monitoring (periódico)
        if self.total_steps % 100 == 0:
            activity_info = self._calculate_activity_metrics(env)
            info.update(activity_info)
        
        return info
    
    def _check_position_timeout(self, env, current_position: float) -> bool:
        """
        Verifica se posição deve ser fechada por timeout
        """
        # Usar timeout atual (progressivo) em vez do valor fixo do config
        timeout_threshold = self.current_timeout
        
        if current_position == 0 or self.position_steps < timeout_threshold:
            return False
        
        # Tentar fechar posição via environment
        if hasattr(env, 'force_close_position'):
            try:
                env.force_close_position(reason='timeout')
                logging.debug(f"🎯 [ACTIVITY] Posição fechada por timeout após {self.position_steps} candles")
                return True
            except Exception as e:
                logging.warning(f"🎯 [ACTIVITY] Erro ao fechar posição por timeout: {e}")
        
        # Fallback: sinalizar timeout sem forçar fechamento
        if hasattr(env, 'position_timeout_signal'):
            env.position_timeout_signal = True
        
        return False
    
    def _update_volatility(self, env):
        """
        Atualiza cálculo de volatilidade baseado em preços recentes
        """
        try:
            # Tentar obter preço atual de múltiplas fontes
            current_price = None
            
            if hasattr(env, 'current_price'):
                current_price = env.current_price
            elif hasattr(env, 'close_price'):
                current_price = env.close_price
            elif hasattr(env, 'price'):
                current_price = env.price
            elif hasattr(env, 'data') and hasattr(env.data, 'iloc'):
                # Assumir que tem dados OHLCV
                current_row = env.data.iloc[getattr(env, 'current_step', 0)]
                current_price = current_row.get('close', current_row.get('Close'))
            
            if current_price is not None and current_price > 0:
                self.price_history.append(float(current_price))
                
                # Calcular volatilidade se temos dados suficientes
                if len(self.price_history) >= 10:
                    prices = np.array(self.price_history)
                    returns = np.diff(np.log(prices))
                    self.current_volatility = np.std(returns) * np.sqrt(252)  # Anualized
                    
                    # Clamp volatilidade em range sensato
                    self.current_volatility = np.clip(self.current_volatility, 0.005, 0.1)
                    
        except Exception as e:
            logging.warning(f"🎯 [ACTIVITY] Erro no cálculo de volatilidade: {e}")
    
    def _is_entry_action(self, action: np.ndarray) -> bool:
        """
        Determina se a ação representa uma tentativa de entrada
        """
        try:
            if len(action) == 0:
                return False
            
            # Assumir que action[0] é entry decision
            entry_decision = float(action[0])
            
            # Threshold para considerar como tentativa de entrada
            # (pode ser ajustado baseado na arquitetura específica)
            return abs(entry_decision) > 0.1
            
        except Exception:
            return False
    
    def _calculate_dynamic_sl_tp(self, env, action: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Calcula SL/TP dinâmicos baseados na volatilidade atual
        """
        try:
            # SL baseado em volatilidade
            sl_percent = self.current_volatility * self.config.dynamic_sl_factor
            sl_percent = np.clip(sl_percent, self.config.min_sl_percent, self.config.max_sl_percent)
            
            # TP baseado em volatilidade
            tp_percent = self.current_volatility * self.config.dynamic_tp_factor
            tp_percent = np.clip(tp_percent, self.config.min_tp_percent, self.config.max_tp_percent)
            
            # Garantir que TP > SL
            if tp_percent <= sl_percent:
                tp_percent = sl_percent * 1.5
            
            return {
                'sl_percent': sl_percent,
                'tp_percent': tp_percent,
                'volatility_used': self.current_volatility,
                'sl_factor': self.config.dynamic_sl_factor,
                'tp_factor': self.config.dynamic_tp_factor
            }
            
        except Exception as e:
            logging.warning(f"🎯 [ACTIVITY] Erro no cálculo de SL/TP dinâmicos: {e}")
            return None
    
    def _calculate_activity_metrics(self, env) -> Dict[str, Any]:
        """
        Calcula métricas de atividade atual
        """
        try:
            current_trades = len(getattr(env, 'trades', []))
            new_trades = current_trades - self.trades_count
            steps_since_check = self.total_steps - self.last_activity_check
            
            # Taxa de atividade (trades por step)
            activity_rate = new_trades / max(steps_since_check, 1)
            self.activity_metrics['activity_rate_current'] = activity_rate
            
            # Duração média das posições
            if hasattr(env, 'trades') and env.trades:
                recent_trades = env.trades[-10:]  # Últimos 10 trades
                durations = []
                for trade in recent_trades:
                    duration = trade.get('duration_steps', trade.get('duration', 1))
                    if duration > 0:
                        durations.append(duration)
                
                if durations:
                    self.activity_metrics['avg_position_duration'] = np.mean(durations)
            
            # Update tracking
            self.trades_count = current_trades
            self.last_activity_check = self.total_steps
            
            # Status da atividade
            activity_status = "🟢 ATIVO" if activity_rate >= self.config.target_activity_rate else "🔴 BAIXO"
            
            return {
                'activity_rate': activity_rate,
                'activity_target': self.config.target_activity_rate,
                'activity_status': activity_status,
                'current_volatility': self.current_volatility,
                'position_steps': self.position_steps,
                'activity_metrics': self.activity_metrics.copy()
            }
            
        except Exception as e:
            logging.warning(f"🎯 [ACTIVITY] Erro no cálculo de métricas: {e}")
            return {}
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo do status atual do sistema
        """
        return {
            'total_steps': self.total_steps,
            'current_volatility': self.current_volatility,
            'position_active': self.position_start_step is not None,
            'position_duration': self.position_steps,
            'timeout_threshold': self.config.position_timeout_candles,
            'activity_metrics': self.activity_metrics.copy(),
            'config': {
                'timeout_candles': self.current_timeout,  # Usar timeout atual (progressivo)
                'timeout_base': self.config.position_timeout_candles,
                'progressive_enabled': self.config.progressive_timeout,
                'training_progress': self.training_steps_global / self.config.training_steps_total,
                'target_activity': self.config.target_activity_rate,
                'sl_factor': self.config.dynamic_sl_factor,
                'tp_factor': self.config.dynamic_tp_factor
            }
        }

def create_activity_enhancement_system(position_timeout: int = 30,
                                     target_activity: float = 0.15,
                                     dynamic_factors: Tuple[float, float] = (1.5, 2.5),
                                     progressive_timeout: bool = True,
                                     training_steps_total: int = 12000000) -> ActivityEnhancementSystem:
    """
    Factory function para criar sistema de activity enhancement
    
    Args:
        position_timeout: Timeout base em candles
        target_activity: Taxa alvo de atividade
        dynamic_factors: (SL_factor, TP_factor) para SL/TP dinâmicos
        progressive_timeout: Ativar sistema progressivo de timeout
        training_steps_total: Total de steps do treinamento para progressão
    """
    config = ActivityConfig(
        position_timeout_candles=position_timeout,
        target_activity_rate=target_activity,
        dynamic_sl_factor=dynamic_factors[0],
        dynamic_tp_factor=dynamic_factors[1],
        progressive_timeout=progressive_timeout,
        training_steps_total=training_steps_total
    )
    
    return ActivityEnhancementSystem(config)
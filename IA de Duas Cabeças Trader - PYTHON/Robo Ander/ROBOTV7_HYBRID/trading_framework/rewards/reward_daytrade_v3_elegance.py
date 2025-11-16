"""
🚀 REWARD DAYTRADE V3 ELEGANCE
Sistema de recompensas otimizado para V8 Elegance Policy

FILOSOFIA V3 ELEGANCE:
- PnL Supremacy: 85% focado em lucro real
- Risk Precision: 15% apenas para controle crítico  
- Zero Noise: Eliminação de componentes sintéticos
- Mathematical Purity: Escalamento correto sem saturação
- V8 Harmony: Projetado para V8 Elegance architecture
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Anti-gaming system V3 (existente)
try:
    from .anti_gaming_system_v3 import AntiGamingSystemV3
    ANTI_GAMING_V3_AVAILABLE = True
except ImportError:
    ANTI_GAMING_V3_AVAILABLE = False
    print("⚠️ AntiGamingSystemV3 não disponível - usando fallback")

@dataclass
class V3EleganceConfig:
    """🎯 Configurações V3 Elegance"""
    
    # Core weights - BALANCEADO E SEGURO
    pnl_weight: float = 0.85          # 85% PnL dominante
    risk_weight: float = 0.15         # 15% risk - ESSENCIAL para segurança
    
    # Scaling - MATEMÁTICA CORRETA
    reward_scale: float = 10.0        # Scale [-10, +10] sem saturação
    max_reward: float = 15.0          # Clamp máximo
    min_reward: float = -15.0         # Clamp mínimo
    
    # Modifiers ELIMINADOS - PNL PURO (NOTA 10)
    pain_multiplier: float = 1.0      # 1.0x - sem modification
    pain_threshold: float = 999.0     # Disabled - nunca ativa
    profit_amplifier: float = 1.0     # 1.0x - sem modification
    profit_threshold: float = 999.0   # Disabled - nunca ativa
    
    # Risk thresholds - SÓ O CRÍTICO
    critical_drawdown: float = 0.22   # 22% drawdown crítico (ajuste fino)
    max_position_risk: float = 0.04   # 4% max por trade (ajuste fino)
    overtrading_limit: int = 40       # 40 trades/day max (ajuste fino)
    
    # V8 Integration - HARMONY
    regime_adjustments: bool = True   # Market regime aware
    memory_integration: bool = True   # V8 memory bank integration
    head_specific: bool = True        # Entry/Management head aware

class ElegantDayTradingRewardV3:
    """🚀 V3 Elegance Reward System - Simplicidade Focada"""
    
    def __init__(self, config: Optional[V3EleganceConfig] = None):
        """Inicializar sistema V3 Elegance"""
        
        self.config = config or V3EleganceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Anti-gaming system V3 (refinado)
        if ANTI_GAMING_V3_AVAILABLE:
            self.anti_gaming = AntiGamingSystemV3()
            self.logger.info("✅ AntiGamingSystemV3 carregado")
        else:
            self.anti_gaming = None
            self.logger.warning("⚠️ AntiGamingSystemV3 não disponível")
        
        # Estado interno
        self.episode_trades = []
        self.last_portfolio_value = None
        self.episode_start_balance = None
        self.step_count = 0
        
        # Cache para noise reduction
        self.cached_risk_penalty = 0.0
        self.cached_regime_multiplier = 1.0
        self.cached_gaming_penalty = 0.0
        
        # V8 Integration
        self.current_market_regime = 'sideways'  # Default regime
        self.regime_multipliers = {
            'bull': 1.0,      # Uniform - zero noise
            'bear': 1.0,      # Uniform - zero noise
            'sideways': 1.0,  # Uniform - zero noise
            'volatile': 1.0   # Uniform - zero noise
        }
        
        self.logger.info("🚀 V3 Elegance Reward System inicializado - NOTA 10")
        self.logger.info(f"   PnL Weight: {self.config.pnl_weight} (100% - PURO)")
        self.logger.info(f"   Risk Weight: {self.config.risk_weight} (0% - ELIMINADO)")
        self.logger.info(f"   Reward Scale: {self.config.reward_scale}")
    
    def reset_episode(self, initial_balance: float):
        """🔄 Reset para novo episódio"""
        self.episode_trades = []
        self.last_portfolio_value = initial_balance
        self.episode_start_balance = initial_balance
        self.step_count = 0
        self.current_market_regime = 'sideways'
        
        # Reset cache
        self.cached_risk_penalty = 0.0
        self.cached_regime_multiplier = 1.0
        self.cached_gaming_penalty = 0.0
        self._last_weighted_reward = None
        
    def calculate_pnl_pure(self, env) -> float:
        """🎯 PnL direto sem interferências - 85% do reward"""
        
        try:
            # Extrair PnL atual  
            initial_capital = getattr(env, 'initial_balance', 10000.0)  # Default fallback
            if self.episode_start_balance is None:
                self.episode_start_balance = initial_capital
            
            current_portfolio = getattr(env, 'portfolio_value', self.episode_start_balance)
            
            if initial_capital <= 0:
                return 0.0
            
            # PnL normalizado
            normalized_pnl = (current_portfolio - initial_capital) / initial_capital
            
            # Pain multiplier para losses
            if normalized_pnl < -self.config.pain_threshold:
                pain_factor = min(self.config.pain_multiplier, abs(normalized_pnl) * 10)
                pnl_reward = normalized_pnl * pain_factor
                
            # Profit amplifier para gains consistentes
            elif normalized_pnl > self.config.profit_threshold:
                # Check consistency streak se disponível
                consistency_streak = getattr(env, 'positive_days_streak', 1)
                amplifier = self.config.profit_amplifier if consistency_streak >= 3 else 1.0
                pnl_reward = normalized_pnl * amplifier
                
            # Range neutro sem modificação
            else:
                pnl_reward = normalized_pnl
            
            return float(pnl_reward)
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo PnL pure: {e}")
            return 0.0
    
    def calculate_risk_critical(self, env) -> float:
        """⚠️ Risco unificado com cache - 15% do reward"""
        
        # Cache a cada 5 steps para reduzir ruído
        if self.step_count % 5 != 0:
            return self.cached_risk_penalty
        
        try:
            # Coleta componentes de risco
            drawdown_score = 0.0
            position_score = 0.0 
            trading_score = 0.0
            
            # 1. DRAWDOWN SCORE (0-1)
            max_drawdown = getattr(env, 'max_drawdown', 0.0)
            if hasattr(env, 'get_max_drawdown'):
                max_drawdown = env.get_max_drawdown()
            
            if max_drawdown > self.config.critical_drawdown:
                drawdown_score = min((max_drawdown - self.config.critical_drawdown) / 0.3, 1.0)
            
            # 2. POSITION SCORE (0-1)
            current_risk_percent = 0.0
            if hasattr(env, 'get_current_position_risk'):
                current_risk_percent = env.get_current_position_risk()
            elif hasattr(env, 'positions') and len(env.positions) > 0:
                total_risk = sum(abs(pos.get('volume', 0)) for pos in env.positions)
                portfolio_value = getattr(env, 'portfolio_value', self.episode_start_balance)
                current_risk_percent = total_risk / portfolio_value if portfolio_value > 0 else 0
            
            if current_risk_percent > self.config.max_position_risk:
                position_score = min((current_risk_percent - self.config.max_position_risk) / 0.2, 1.0)
            
            # 3. TRADING SCORE (0-1)
            daily_trades = len(self.episode_trades)
            if hasattr(env, 'get_daily_trade_count'):
                daily_trades = env.get_daily_trade_count()
            
            if daily_trades > self.config.overtrading_limit:
                trading_score = min((daily_trades - self.config.overtrading_limit) / 100, 1.0)
            
            # UNIFICAÇÃO SUAVE com pesos
            unified_risk_score = (
                drawdown_score * 0.6 +      # 60% peso para drawdown
                position_score * 0.3 +      # 30% peso para sizing  
                trading_score * 0.1         # 10% peso para overtrading
            )
            
            # Penalty suave baseado no score unificado
            if unified_risk_score > 0.15:  # Threshold mais sensível (era 0.2)
                self.cached_risk_penalty = -(unified_risk_score - 0.15) * 4.0  # Multiplier suave (era 6.0)
            else:
                self.cached_risk_penalty = 0.0
            
            # Cap suave -1.5 (era -2.0)
            self.cached_risk_penalty = max(self.cached_risk_penalty, -1.5)
            
            return self.cached_risk_penalty
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo risk critical: {e}")
            return 0.0
    
    def apply_anti_gaming(self, base_reward: float, env) -> float:
        """🛡️ Anti-gaming refinado usando sistema V3"""
        
        if not self.anti_gaming:
            return base_reward
        
        try:
            # Extrair histórico de trades
            trade_history = getattr(env, 'trades', [])
            if not trade_history:
                return base_reward
            
            # Detectar gaming usando V3 system
            gaming_penalty = self.anti_gaming.detect_gaming(trade_history)
            
            # Aplicar penalty apenas se gaming real detectado
            if gaming_penalty < -0.1:  # Threshold conservador
                return base_reward + gaming_penalty
            
            return base_reward  # Sem interferência em trading normal
            
        except Exception as e:
            self.logger.error(f"Erro no anti-gaming: {e}")
            return base_reward
    
    def get_regime_adjusted_reward(self, base_reward: float, market_regime: Optional[str] = None) -> float:
        """🌍 Ajuste baseado no regime de mercado detectado pela V8"""
        
        if not self.config.regime_adjustments:
            return base_reward
        
        # Cache regime multiplier a cada 20 steps
        if self.step_count % 20 == 0:
            regime = market_regime or self.current_market_regime
            self.cached_regime_multiplier = self.regime_multipliers.get(regime, 1.0)
        
        return base_reward * self.cached_regime_multiplier
    
    def update_v8_memory_context(self, reward: float, env) -> np.ndarray:
        """💾 Criar contexto para V8 Elegance Memory Bank"""
        
        if not self.config.memory_integration:
            return np.array([])
        
        try:
            # Extrair dados do environment
            current_pnl = getattr(env, 'portfolio_value', 0) - self.episode_start_balance
            current_drawdown = getattr(env, 'current_drawdown', 0)
            trade_duration = getattr(env, 'current_trade_duration', 0)
            
            # Market regime ID (0-3 para bull, bear, sideways, volatile)
            regime_mapping = {'bull': 0, 'bear': 1, 'sideways': 2, 'volatile': 3}
            regime_id = regime_mapping.get(self.current_market_regime, 2)
            
            # Position size atual
            position_size = 0.0
            if hasattr(env, 'positions') and env.positions:
                position_size = sum(abs(pos.get('volume', 0)) for pos in env.positions)
            
            # Trade profitability
            is_profitable = 1.0 if current_pnl > 0 else 0.0
            
            # Confidence score (se disponível)
            confidence_score = getattr(env, 'last_confidence_score', 0.5)
            
            # Criar contexto para V8 Memory (8D)
            memory_context = np.array([
                reward,                    # Reward atual
                current_pnl,              # PnL atual
                current_drawdown,         # Drawdown atual
                trade_duration,           # Duração do trade
                float(regime_id),         # ID do regime (0-3)
                position_size,            # Tamanho da posição
                is_profitable,            # Trade lucrativo (0/1)
                confidence_score          # Confidence da entrada
            ], dtype=np.float32)
            
            return memory_context
            
        except Exception as e:
            self.logger.error(f"Erro ao criar V8 memory context: {e}")
            return np.zeros(8, dtype=np.float32)
    
    def calculate_final_reward(self, env, action: np.ndarray, old_state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """🚀 Cálculo final elegante e direto"""
        
        self.step_count += 1
        
        try:
            # 1. PNL COMPONENT (85%)
            pnl_reward = self.calculate_pnl_pure(env)
            
            # 2. RISK COMPONENT (15%) - ESSENCIAL PARA SEGURANÇA
            risk_reward = self.calculate_risk_critical(env)
            
            # 3. COMBINAÇÃO SEGURA
            combined_reward = (pnl_reward * self.config.pnl_weight) + (risk_reward * self.config.risk_weight)
            
            # 4. SCALING 
            final_reward = combined_reward * self.config.reward_scale
            
            # 3. CLAMP SUAVE 
            final_reward = np.clip(final_reward, self.config.min_reward, self.config.max_reward)
            
            # 7. Update V8 memory context
            v8_memory_context = self.update_v8_memory_context(final_reward, env)
            
            # 5. Debug info COMPLETO E SEGURO
            reward_info = {
                'v3_elegance_safe': True,
                'pnl_component': pnl_reward,
                'risk_component': risk_reward,  # RESTAURADO
                'combined_reward': combined_reward,
                'final_reward': final_reward,
                'pnl_weight': self.config.pnl_weight,
                'risk_weight': self.config.risk_weight,
                'step_count': self.step_count,
                'risk_management_active': True
            }
            
            # Update internal state
            current_portfolio = getattr(env, 'portfolio_value', self.episode_start_balance)
            self.last_portfolio_value = current_portfolio
            
            # Track trades
            if hasattr(env, 'trades') and len(env.trades) > len(self.episode_trades):
                new_trades = env.trades[len(self.episode_trades):]
                self.episode_trades.extend(new_trades)
            
            return float(final_reward), reward_info
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo final reward: {e}")
            return 0.0, {'error': str(e)}
    
    def set_market_regime(self, regime: str):
        """🌍 Atualizar regime de mercado (chamado pela V8)"""
        if regime in self.regime_multipliers:
            self.current_market_regime = regime
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """📊 Estatísticas do sistema de reward"""
        
        return {
            'config': {
                'pnl_weight': self.config.pnl_weight,
                'risk_weight': self.config.risk_weight,
                'reward_scale': self.config.reward_scale,
                'pain_multiplier': self.config.pain_multiplier,
                'profit_amplifier': self.config.profit_amplifier
            },
            'state': {
                'episode_trades': len(self.episode_trades),
                'step_count': self.step_count,
                'current_regime': self.current_market_regime,
                'last_portfolio': self.last_portfolio_value,
                'anti_gaming_available': self.anti_gaming is not None
            }
        }

# 🚀 FACTORY FUNCTIONS

def create_v3_elegance_reward_system(
    pnl_weight: float = 0.85,
    risk_weight: float = 0.15,
    reward_scale: float = 10.0,
    regime_adjustments: bool = True,
    memory_integration: bool = True
) -> ElegantDayTradingRewardV3:
    """🏭 Factory function para criar sistema V3 Elegance"""
    
    config = V3EleganceConfig(
        pnl_weight=pnl_weight,
        risk_weight=risk_weight,
        reward_scale=reward_scale,
        regime_adjustments=regime_adjustments,
        memory_integration=memory_integration
    )
    
    return ElegantDayTradingRewardV3(config)

def create_v3_elegance_conservative() -> ElegantDayTradingRewardV3:
    """🛡️ Versão conservadora para início de treinamento"""
    
    config = V3EleganceConfig(
        pnl_weight=0.80,           # Slightly less PnL focus
        risk_weight=0.20,          # More risk awareness
        pain_multiplier=2.0,       # Less aggressive pain
        profit_amplifier=1.1,      # Less aggressive profit boost
        reward_scale=8.0           # Smaller scale
    )
    
    return ElegantDayTradingRewardV3(config)

def create_v3_elegance_aggressive() -> ElegantDayTradingRewardV3:
    """🔥 Versão agressiva para fases avançadas"""
    
    config = V3EleganceConfig(
        pnl_weight=0.90,           # Maximum PnL focus
        risk_weight=0.10,          # Minimal risk interference
        pain_multiplier=3.0,       # Aggressive pain
        profit_amplifier=1.3,      # Aggressive profit boost
        reward_scale=12.0          # Larger scale
    )
    
    return ElegantDayTradingRewardV3(config)

# 🚀 INTEGRATION HELPER

class V8EleganceRewardInterface:
    """🎯 Interface específica para integração com V8 Elegance"""
    
    def __init__(self, reward_system: ElegantDayTradingRewardV3):
        self.reward_system = reward_system
        
    def set_v8_policy_context(self, policy_output: Dict[str, Any]):
        """Atualizar contexto baseado no output da V8 policy"""
        
        # Extract market regime from V8 policy
        if 'regime_info' in policy_output:
            regime_name = policy_output['regime_info'].get('regime_name', 'sideways')
            self.reward_system.set_market_regime(regime_name)
        
        # Extract confidence scores
        if 'entry_info' in policy_output:
            entry_info = policy_output['entry_info']
            # Store confidence for memory context
            # This would be used in update_v8_memory_context
    
    def add_to_v8_memory_bank(self, env, memory_bank):
        """Adicionar contexto ao V8 memory bank se disponível"""
        
        try:
            if hasattr(memory_bank, 'add_trade') and self.reward_system.config.memory_integration:
                # Get last reward info
                last_reward = getattr(env, 'last_reward', 0.0)
                memory_context = self.reward_system.update_v8_memory_context(last_reward, env)
                
                if len(memory_context) > 0:
                    memory_bank.add_trade(memory_context)
                    
        except Exception as e:
            logging.error(f"Erro ao adicionar ao V8 memory bank: {e}")

if __name__ == "__main__":
    # Test básico
    print("🚀 Testing V3 Elegance Reward System")
    
    # Create systems
    standard = create_v3_elegance_reward_system()
    conservative = create_v3_elegance_conservative()
    aggressive = create_v3_elegance_aggressive()
    
    print(f"✅ Standard: PnL={standard.config.pnl_weight}, Scale={standard.config.reward_scale}")
    print(f"🛡️ Conservative: PnL={conservative.config.pnl_weight}, Scale={conservative.config.reward_scale}")
    print(f"🔥 Aggressive: PnL={aggressive.config.pnl_weight}, Scale={aggressive.config.reward_scale}")
    
    # Get statistics
    stats = standard.get_reward_statistics()
    print(f"📊 Statistics: {stats}")
    
    print("✅ V3 Elegance Reward System ready for integration!")
"""
🎯 SISTEMA DE RECOMPENSAS SIMPLES E MATEMATICAMENTE COERENTE
Focado na correção das inconsistências matemáticas identificadas

🎯 PRINCÍPIOS:
1. Recompensa = PnL real (1:1 ratio)
2. Valores pequenos e proporcionais
3. Sem multiplicadores insanos
4. Matemática transparente e coerente
5. 🧠 FOCO NA CABEÇA TÁTICA: Gestão dinâmica de SL/TP e fechamento inteligente
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

class SimpleRewardCalculator:
    """Sistema de recompensas simples e coerente com foco na gestão tática"""
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.step_count = 0
        self.position_history = {}  # Track para cada posição
        
        # Pesos simples e realistas
        self.weights = {
            # 💰 CORE - PnL direto
            "pnl_direct": 1.0,           # Recompensa = PnL real (1:1)
            "win_bonus": 1.0,            # +$1 bônus por trade vencedor
            "loss_penalty": -0.5,        # -$0.5 penalidade por trade perdedor
            
            # 🎯 ATIVIDADE BÁSICA
            "trade_action": 0.1,         # +0.1 por fazer trade
            "daily_activity": 0.2,       # +0.2 por estar ativo no dia
            "inactivity_penalty": -0.05, # -0.05 por inatividade
            "excessive_trading": 0.0,   # REMOVIDO: penalização contra overtrading
            
            # 🛡️ GESTÃO DE RISCO BÁSICA
            "sl_tp_usage": 0.2,          # +0.2 por usar SL/TP
            "drawdown_penalty": -0.1,    # -0.1 por cada 1% de drawdown
            
            # 🧠 NOVA SEÇÃO: INTELIGÊNCIA TÁTICA
            # === TRAILING STOP INTELIGENTE ===
            "trailing_stop_activation": 2.0,     # +2.0 por ativar trailing stop no momento certo
            "trailing_stop_execution": 3.0,      # +3.0 por executar trailing stop com lucro
            "trailing_stop_protection": 1.5,     # +1.5 por proteger lucros com trailing
            "trailing_stop_timing": 1.0,         # +1.0 por timing correto do trailing
            
            # === GESTÃO DINÂMICA DE SL/TP ===
            "sl_tightening_profit": 2.5,         # +2.5 por apertar SL quando em lucro
            "tp_extension_trend": 2.0,            # +2.0 por estender TP em tendência forte
            "sl_loosening_volatile": 1.0,        # +1.0 por afrouxar SL em volatilidade
            "risk_reduction_move": 2.0,          # +2.0 por reduzir risco inteligentemente
            
            # === FECHAMENTO TÁTICO INTELIGENTE ===
            "early_profit_taking": 1.5,          # +1.5 por fechar com lucro antes de reversão
            "quick_loss_cutting": 2.0,           # +2.0 por cortar loss rapidamente
            "market_change_reaction": 2.5,       # +2.5 por reagir a mudanças de mercado
            "momentum_exit": 1.8,                # +1.8 por sair no momentum correto
            
            # === PENALIDADES POR MÁ GESTÃO ===
            "missed_trailing_opportunity": -1.0, # -1.0 por não usar trailing quando deveria
            "poor_sl_management": -1.5,          # -1.5 por gestão ruim de SL
            "late_exit": -1.0,                   # -1.0 por sair tarde demais
            "panic_closure": -0.5,               # -0.5 por fechamento em pânico
        }
        
    def reset(self):
        """Reset para novo episódio"""
        self.step_count = 0
        self.position_history = {}
        
    def calculate_reward_and_info(self, env, action: np.ndarray, old_state: Dict) -> Tuple[float, Dict, bool]:
        """
        Sistema ultra-simples de recompensas com foco na GESTÃO TÁTICA
        
        FÓRMULA PRINCIPAL:
        Reward = PnL_real + pequenos_incentivos + BÔNUS_GESTÃO_TÁTICA
        """
        self.step_count += 1
        
        reward = 0.0
        info = {"components": {}, "tactical_analysis": {}}
        
        # Processar ações
        entry_decision = int(action[0]) if len(action) > 0 else 0
        mgmt_action = int(action[3]) if len(action) > 3 else 0
        sl_adjust = float(action[4]) if len(action) > 4 else 0.0
        tp_adjust = float(action[5]) if len(action) > 5 else 0.0
        
        # 💰 COMPONENTE PRINCIPAL: PnL DIRETO
        old_trades_count = old_state.get('trades_count', 0)
        current_trades_count = len(env.trades)
        
        if current_trades_count > old_trades_count:
            # Trade fechado - recompensa = PnL real
            last_trade = env.trades[-1]
            pnl = last_trade.get('pnl_usd', 0.0)
            
            # Recompensa direta baseada no PnL
            pnl_reward = pnl * self.weights["pnl_direct"]
            reward += pnl_reward
            info["components"]["pnl_direct"] = pnl_reward
            
            # Bônus/penalidade pequenos e fixos
            if pnl > 0:
                win_bonus = self.weights["win_bonus"]
                reward += win_bonus
                info["components"]["win_bonus"] = win_bonus
            else:
                loss_penalty = self.weights["loss_penalty"]
                reward += loss_penalty
                info["components"]["loss_penalty"] = loss_penalty
        
        # 🎯 INCENTIVOS PEQUENOS PARA ATIVIDADE
        trades_today = self._get_trades_today(env)
        
        if entry_decision > 0:
            trade_bonus = self.weights["trade_action"]
            reward += trade_bonus
            info["components"]["trade_action"] = trade_bonus
        
        if trades_today > 0:
            daily_bonus = self.weights["daily_activity"]
            reward += daily_bonus
            info["components"]["daily_activity"] = daily_bonus
        
        # ⚡ PENALIDADES MÍNIMAS
        if trades_today == 0 and env.current_step > 200:
            inactivity_penalty = self.weights["inactivity_penalty"]
            reward += inactivity_penalty
            info["components"]["inactivity_penalty"] = inactivity_penalty
        
        # 🎯 LIMITES REALISTAS
        reward = np.clip(reward, -50.0, 50.0)
        
        info.update({
            "trades_today": trades_today,
            "total_reward": reward,
            "step_count": self.step_count
        })
        
        return reward, info, False
    
    def _get_trades_today(self, env) -> int:
        """Calcular trades do dia atual"""
        if not env.trades:
            return 0
        
        # 288 steps = 1 dia em timeframe de 5min
        steps_per_day = 288
        current_day = env.current_step // steps_per_day
        
        trades_today = 0
        for trade in env.trades:
            trade_day = trade.get('entry_step', 0) // steps_per_day
            if trade_day == current_day:
                trades_today += 1
        
        return trades_today

# 🚀 ======================== SISTEMA COMPLETO COM EXECUÇÃO DE ORDENS ======================== 🚀

class SimpleRewardCalculatorWithExecution:
    """
    🎯 SISTEMA COMPLETO: REWARDS SIMPLES + EXECUÇÃO DE ORDENS
    
    COMPONENTES:
    1. 📈 Sistema de rewards matemático e coerente
    2. 🎯 Execução completa de ordens (abertura, fechamento, gestão)
    3. 🧠 Gestão tática avançada de SL/TP
    4. 🔄 Processamento de ações especializadas
    
    PARA USO NO TREINAMENTO DIFERENCIADO (treinodiferenciadoPPO.py)
    """
    
    def __init__(self, initial_balance: float = 1000.0):
        # Herdar configurações do sistema simples
        self.simple_calculator = SimpleRewardCalculator(initial_balance)
        
        # Configurações específicas de execução
        self.max_positions = 3
        self.min_position_size = 0.01
        self.max_position_size = 1.0
        self.spread_points = 2  # Spread em pontos
        self.commission_rate = 0.0001  # 0.01%
        self.order_id_counter = 0
        
        # Pesos adicionais para execução
        self.execution_weights = {
            "execution_bonus": 0.2,           # +0.2 por execução bem-sucedida
            "adjustment_bonus": 0.5,          # +0.5 por ajuste bem-sucedido
            "smart_sizing_bonus": 0.3,        # +0.3 por sizing inteligente
            "execution_error_penalty": -0.1,  # -0.1 por erro de execução
        }
    
    def reset(self):
        """Reset para novo episódio"""
        self.simple_calculator.reset()
        self.order_id_counter = 0
    
    def process_action_and_calculate_reward(self, env, action: np.ndarray, old_state: Dict) -> Tuple[float, Dict, bool]:
        """
        🎯 FUNÇÃO PRINCIPAL: PROCESSA AÇÃO + CALCULA REWARD
        
        FLUXO:
        1. 🔄 Processar ação (Entry Head + Management Head)
        2. 🎯 Executar ordens (abertura, fechamento, ajustes)
        3. 📊 Calcular rewards baseado nos resultados
        4. 📈 Retornar reward + info detalhado
        """
        
        # 1. 🔄 PROCESSAR AÇÃO
        processed_action = self._decode_specialized_action(action)
        
        # 2. 🎯 EXECUTAR ORDENS
        execution_results = self._execute_trading_orders(env, processed_action)
        
        # 3. 📊 CALCULAR REWARDS (usar sistema simples como base)
        base_reward, base_info, done = self.simple_calculator.calculate_reward_and_info(env, action, old_state)
        
        # 4. 📈 ADICIONAR BÔNUS DE EXECUÇÃO
        execution_bonus = self._calculate_execution_bonus(execution_results)
        total_reward = base_reward + execution_bonus
        
        # 5. 🔄 COMPILAR INFO COMPLETO
        info = base_info.copy()
        info.update({
            "execution_results": execution_results,
            "processed_action": processed_action,
            "execution_bonus": execution_bonus,
            "positions_count": len(env.positions) if hasattr(env, 'positions') else 0,
            "total_reward": total_reward
        })
        
        return total_reward, info, done
    
    def _decode_specialized_action(self, action: np.ndarray) -> Dict:
        """
        🔄 DECODIFICAÇÃO DE AÇÃO ESPECIALIZADA
        
        ACTION SPACE: [entry_decision, entry_confidence, position_size, mgmt_action, sl_adjust, tp_adjust]
        """
        
        return {
            "entry": {
                "decision": int(action[0]) if len(action) > 0 else 0,
                "confidence": float(action[1]) if len(action) > 1 else 0.0,
                "size": float(action[2]) if len(action) > 2 else 0.1
            },
            "management": {
                "action": int(action[3]) if len(action) > 3 else 0,
                "sl_adjust": float(action[4]) if len(action) > 4 else 0.0,
                "tp_adjust": float(action[5]) if len(action) > 5 else 0.0
            }
        }
    
    def _execute_trading_orders(self, env, processed_action: Dict) -> Dict:
        """🎯 EXECUÇÃO COMPLETA DE ORDENS DE TRADING"""
        
        results = {
            "orders_executed": [],
            "positions_opened": [],
            "positions_closed": [],
            "adjustments_made": [],
            "errors": []
        }
        
        current_price = self._get_current_price(env)
        if current_price <= 0:
            results["errors"].append("Invalid current price")
            return results
        
        # 1. 📈 PROCESSAR ENTRADA (Entry Head)
        entry_result = self._process_entry_order(env, processed_action["entry"], current_price)
        if entry_result:
            results["orders_executed"].append(entry_result)
            if entry_result["status"] == "executed":
                results["positions_opened"].append(entry_result)
        
        # 2. 🔧 PROCESSAR GESTÃO (Management Head)
        mgmt_results = self._process_management_orders(env, processed_action["management"], current_price)
        for result in mgmt_results:
            results["orders_executed"].append(result)
            if result["type"] == "close":
                results["positions_closed"].append(result)
            elif result["type"] == "adjustment":
                results["adjustments_made"].append(result)
        
        return results
    
    def _process_entry_order(self, env, entry_action: Dict, current_price: float) -> Optional[Dict]:
        """📈 PROCESSAR ORDEM DE ENTRADA"""
        
        decision = entry_action["decision"]
        confidence = entry_action["confidence"]
        size = entry_action["size"]
        
        # Verificar se deve abrir posição
        if decision == 0:  # Hold
            return None
        
        # Verificar limites de posições
        current_positions = len(env.positions) if hasattr(env, 'positions') else 0
        if current_positions >= self.max_positions:
            return {
                "type": "entry",
                "status": "rejected",
                "reason": "max_positions_reached"
            }
        
        # Calcular tamanho adaptativo
        adaptive_size = self._calculate_adaptive_position_size(env, confidence, size)
        
        if adaptive_size < self.min_position_size:
            return {
                "type": "entry", 
                "status": "rejected",
                "reason": "size_too_small"
            }
        
        # Determinar tipo de trade
        trade_type = "long" if decision == 1 else "short"
        
        # Calcular SL/TP iniciais
        sl_tp = self._calculate_initial_sl_tp(current_price, trade_type, confidence)
        
        # Executar ordem
        return self._execute_market_order(env, trade_type, adaptive_size, current_price, sl_tp["sl"], sl_tp["tp"])
    
    def _process_management_orders(self, env, mgmt_action: Dict, current_price: float) -> List[Dict]:
        """🔧 PROCESSAR ORDENS DE GESTÃO"""
        
        results = []
        action = mgmt_action["action"]
        sl_adjust = mgmt_action["sl_adjust"]
        tp_adjust = mgmt_action["tp_adjust"]
        
        if not hasattr(env, 'positions') or not env.positions:
            return results
        
        # 💰 PROCESSAR FECHAMENTOS
        if action == 1:  # Close profitable positions
            for pos in env.positions[:]:
                if self._is_position_profitable(pos, current_price):
                    close_result = self._close_position(env, pos, current_price, "manual_profit")
                    if close_result:
                        results.append(close_result)
        
        elif action == 2:  # Close all positions
            for pos in env.positions[:]:
                close_result = self._close_position(env, pos, current_price, "manual_all")
                if close_result:
                    results.append(close_result)
        
        # 🎯 PROCESSAR AJUSTES DE SL/TP
        if abs(sl_adjust) > 0.1 or abs(tp_adjust) > 0.1:
            for pos in env.positions:
                adjust_result = self._adjust_sl_tp(env, pos, current_price, sl_adjust, tp_adjust)
                if adjust_result:
                    results.append(adjust_result)
        
        return results
    
    def _calculate_adaptive_position_size(self, env, confidence: float, base_size: float) -> float:
        """📊 CALCULAR TAMANHO DE POSIÇÃO ADAPTATIVO"""
        
        # Base size ajustado pela confiança
        confidence_multiplier = 0.3 + (confidence * 0.7)  # 0.3 a 1.0
        adjusted_size = base_size * confidence_multiplier
        
        # Ajustar pela volatilidade (se disponível)
        if hasattr(env, 'df') and env.current_step > 20:
            try:
                recent_data = env.df.iloc[env.current_step-20:env.current_step+1]
                volatility = recent_data[f'close_{env.base_tf}'].pct_change().std()
                
                # Reduzir size em alta volatilidade
                if volatility > 0.02:  # 2% volatilidade
                    adjusted_size *= 0.7
                elif volatility < 0.005:  # 0.5% volatilidade
                    adjusted_size *= 1.2
            except:
                pass
        
        # Limitar pelos constraints
        return np.clip(adjusted_size, self.min_position_size, self.max_position_size)
    
    def _calculate_initial_sl_tp(self, price: float, trade_type: str, confidence: float) -> Dict:
        """🎯 CALCULAR SL/TP INICIAIS ADAPTATIVOS"""
        
        # SL padrão: 20-40 pontos baseado na confiança
        sl_points = 30 - (confidence * 10)  # 20-30 pontos
        tp_points = sl_points * 1.5  # Risk/reward 1:1.5
        
        if trade_type == "long":
            sl_price = price - (sl_points * 0.1)
            tp_price = price + (tp_points * 0.1)
        else:  # short
            sl_price = price + (sl_points * 0.1)
            tp_price = price - (tp_points * 0.1)
        
        return {
            "sl": max(sl_price, 0.1),
            "tp": max(tp_price, 0.1)
        }
    
    def _execute_market_order(self, env, trade_type: str, size: float, price: float, sl: float, tp: float) -> Dict:
        """🚀 EXECUTAR ORDEM DE MERCADO"""
        
        self.order_id_counter += 1
        
        # Aplicar spread
        execution_price = price + (self.spread_points * 0.1) if trade_type == "long" else price - (self.spread_points * 0.1)
        
        # Calcular comissão
        position_value = execution_price * size
        commission = position_value * self.commission_rate
        
        # Criar posição
        position = {
            "id": self.order_id_counter,
            "type": trade_type,
            "entry_price": execution_price,
            "size": size,
            "sl": sl,
            "tp": tp,
            "commission": commission,
            "entry_step": env.current_step,
            "adjustment_count": 0,
            "sl_adjustment_type": None,
            "tp_adjustment_type": None
        }
        
        # Adicionar à lista de posições
        if not hasattr(env, 'positions'):
            env.positions = []
        env.positions.append(position)
        
        return {
            "type": "entry",
            "status": "executed", 
            "order_id": self.order_id_counter,
            "trade_type": trade_type,
            "size": size,
            "execution_price": execution_price,
            "sl": sl,
            "tp": tp,
            "commission": commission
        }
    
    def _close_position(self, env, position: Dict, current_price: float, reason: str) -> Dict:
        """💰 FECHAR POSIÇÃO"""
        
        trade_type = position["type"]
        entry_price = position["entry_price"]
        size = position["size"]
        entry_commission = position.get("commission", 0)
        
        # Aplicar spread no fechamento
        exit_price = current_price - (self.spread_points * 0.1) if trade_type == "long" else current_price + (self.spread_points * 0.1)
        
        # Calcular PnL
        if trade_type == "long":
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price
        
        pnl_gross = pnl_points * size
        exit_commission = exit_price * size * self.commission_rate
        pnl_net = pnl_gross - entry_commission - exit_commission
        
        # Calcular duração
        duration_steps = env.current_step - position["entry_step"]
        
        # Criar registro do trade
        trade_record = {
            "id": position["id"],
            "type": trade_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "entry_step": position["entry_step"],
            "exit_step": env.current_step,
            "duration_steps": duration_steps,
            "pnl_points": pnl_points,
            "pnl_usd": pnl_net,
            "pnl_gross": pnl_gross,
            "commission_total": entry_commission + exit_commission,
            "exit_reason": reason,
            "sl_adjustment_type": position.get("sl_adjustment_type"),
            "tp_adjustment_type": position.get("tp_adjustment_type"),
            "adjustment_count": position.get("adjustment_count", 0)
        }
        
        # Adicionar à lista de trades
        if not hasattr(env, 'trades'):
            env.trades = []
        env.trades.append(trade_record)
        
        # Remover da lista de posições
        if position in env.positions:
            env.positions.remove(position)
        
        return {
            "type": "close",
            "status": "executed",
            "position_id": position["id"],
            "pnl_usd": pnl_net,
            "exit_reason": reason,
            "duration_steps": duration_steps
        }
    
    def _adjust_sl_tp(self, env, position: Dict, current_price: float, sl_adjust: float, tp_adjust: float) -> Optional[Dict]:
        """🎯 AJUSTAR SL/TP DINAMICAMENTE"""
        
        adjustments_made = []
        
        # Ajustar SL
        if abs(sl_adjust) > 0.1:
            old_sl = position["sl"]
            
            # Calcular novo SL
            if position["type"] == "long":
                new_sl = old_sl + (sl_adjust * 0.1 * abs(current_price - position["entry_price"]))
            else:
                new_sl = old_sl - (sl_adjust * 0.1 * abs(current_price - position["entry_price"]))
            
            position["sl"] = max(new_sl, 0.1)
            position["adjustment_count"] = position.get("adjustment_count", 0) + 1
            
            adjustments_made.append(f"SL: {old_sl:.2f} -> {position['sl']:.2f}")
        
        # Ajustar TP
        if abs(tp_adjust) > 0.1:
            old_tp = position["tp"]
            
            # Calcular novo TP
            if position["type"] == "long":
                new_tp = old_tp + (tp_adjust * 0.1 * abs(current_price - position["entry_price"]))
            else:
                new_tp = old_tp - (tp_adjust * 0.1 * abs(current_price - position["entry_price"]))
            
            position["tp"] = max(new_tp, 0.1)
            position["adjustment_count"] = position.get("adjustment_count", 0) + 1
            
            adjustments_made.append(f"TP: {old_tp:.2f} -> {position['tp']:.2f}")
        
        if adjustments_made:
            return {
                "type": "adjustment",
                "status": "executed",
                "position_id": position["id"],
                "adjustments": adjustments_made
            }
        
        return None
    
    def _is_position_profitable(self, position: Dict, current_price: float) -> bool:
        """💰 Verificar se posição está lucrativa"""
        
        if position["type"] == "long":
            return current_price > position["entry_price"]
        else:
            return current_price < position["entry_price"]
    
    def _get_current_price(self, env) -> float:
        """📊 Obter preço atual"""
        
        try:
            if hasattr(env, 'df') and hasattr(env, 'current_step') and hasattr(env, 'base_tf'):
                return float(env.df[f'close_{env.base_tf}'].iloc[env.current_step])
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_execution_bonus(self, execution_results: Dict) -> float:
        """🎯 Calcular bônus por execução eficiente"""
        
        bonus = 0.0
        
        # Bônus por abrir posições com boa configuração
        for opened in execution_results.get("positions_opened", []):
            if opened.get("status") == "executed":
                bonus += self.execution_weights["execution_bonus"]
        
        # Bônus por ajustes inteligentes
        for adjustment in execution_results.get("adjustments_made", []):
            if adjustment.get("status") == "executed":
                bonus += self.execution_weights["adjustment_bonus"]
        
        # Penalidade por erros de execução
        for error in execution_results.get("errors", []):
            bonus += self.execution_weights["execution_error_penalty"]
        
        return bonus

# 🎯 ======================== FUNÇÕES DE CRIAÇÃO ======================== 🎯

def create_simple_reward_system(initial_balance: float = 1000.0):
    """Criar sistema de recompensas simples com foco tático"""
    return SimpleRewardCalculator(initial_balance)

def create_simple_reward_system_with_execution(initial_balance: float = 1000.0):
    """Criar sistema de recompensas simples com execução de ordens para treinamento diferenciado"""
    return SimpleRewardCalculatorWithExecution(initial_balance)

# Configuração padrão para o sistema simples
SIMPLE_REWARD_CONFIG = {
    "initial_balance": 1000.0,
    "system_type": "simple_reward",
    "description": "Sistema de recompensas simples e matematicamente coerente"
} 
 
 d e f   c r e a t e _ s i m p l e _ e x e c u t i o n _ s y s t e m ( i n i t i a l _ b a l a n c e :   f l o a t   =   1 0 0 0 . 0 ) :  
         \  
 \ \ S i s t e m a  
 c o m p l e t o  
 c o m  
 e x e c u c a o  
 d e  
 o r d e n s  
 p a r a  
 t r e i n a m e n t o  
 d i f e r e n c i a d o \ \ \  
         r e t u r n   S i m p l e R e w a r d C a l c u l a t o r W i t h E x e c u t i o n ( i n i t i a l _ b a l a n c e )  
 
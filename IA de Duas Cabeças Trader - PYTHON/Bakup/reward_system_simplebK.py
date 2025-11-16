"""
💰 SISTEMA DE RECOMPENSAS SIMPLES E MATEMATICAMENTE COERENTE - V2 OTIMIZADO
Focado em guiar o modelo aos targets específicos e balanceamento inteligente

🎯 TARGETS ESPECÍFICOS:
- Trades/dia: 18 (otimizado)
- SL Range: 11-56 pontos  
- TP Range: 14-82 pontos
- Risk/Reward: 1.5-1.8 (ótimo)

🎯 PRINCÍPIOS V2:
1. PnL real dominante (60% do peso)
2. Expert SL/TP simplificado mas inteligente (25% do peso)
3. Atividade guiada aos targets (15% do peso)
4. Penalidades disciplinares moderadas
5. Matemática transparente e balanceada

🔥 NOVO: FILTROS DE QUALIDADE INTELIGENTES
- Confluência de sinais (Score 90/100)
- Contexto de mercado
- Risk/Reward dinâmico
- Filtro de correlação
- Memória de mercado
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class QualityFilter:
    """
    🧠 FILTRO DE QUALIDADE INTELIGENTE V3
    ENSINA o modelo sobre qualidade através de REWARDS GRADUAIS
    NÃO bloqueia trades - apenas dá mais reward para trades melhores
    """
    
    def __init__(self):
        # 🧠 SISTEMA DE ENSINO: Sem bloqueios, apenas educação
        self.teaching_mode = True  # Sempre ensinar, nunca bloquear
        
        # 🎯 CONFIGURAÇÕES INTELIGENTES DE RISCO/GANHO
        self.excellent_risk_reward = 2.0   # 1:2 = excelente
        self.good_risk_reward = 1.5        # 1:1.5 = bom
        self.acceptable_risk_reward = 1.0  # 1:1 = aceitável
        
    def calculate_trade_quality_score(self, env, action: np.ndarray) -> float:
        """
        🎯 SCORE DE QUALIDADE INTELIGENTE (0-100)
        Analisa qualidade SEM bloquear - apenas para educar o modelo
        """
        score = 20.0  # Base: Todo trade começa com 20 pontos
        
        try:
            if hasattr(env, 'df') and hasattr(env, 'current_step'):
                current_idx = min(env.current_step, len(env.df) - 1)
                
                # 1. CONFLUÊNCIA DE SINAIS (30 pontos máximo)
                confluence_score = self._analyze_market_confluence(env, current_idx)
                score += confluence_score * 0.3  # Max +9 pontos
                
                # 2. TIMING DE ENTRADA (25 pontos máximo)  
                timing_score = self._analyze_entry_timing(env, current_idx)
                score += timing_score * 0.25  # Max +6.25 pontos
                
                # 3. GESTÃO DE RISCO (25 pontos máximo)
                risk_score = self._analyze_risk_management(env, action)
                score += risk_score * 0.25  # Max +6.25 pontos
                
                # 4. CONTEXTO DE VOLATILIDADE (20 pontos máximo)
                vol_score = self._analyze_volatility_context(env, current_idx)
                score += vol_score * 0.2  # Max +4 pontos
                
        except Exception:
            # Em caso de erro, score neutro (não penalizar)
            score = 50.0
            
        return min(max(score, 0), 100)  # Entre 0-100
    
    def _analyze_market_confluence(self, env, current_idx: int) -> float:
        """Analisa confluência de múltiplos sinais (0-100) - MAIS SENSÍVEL"""
        signals = []
        
        try:
            # RSI extremo - ESCALA MAIS DIFERENCIADA
            if 'rsi_5m' in env.df.columns:
                rsi = env.df['rsi_5m'].iloc[current_idx]
                if rsi < 20 or rsi > 80:  # Extremo
                    signals.append(90)
                elif rsi < 25 or rsi > 75:  # Muito forte
                    signals.append(75)
                elif rsi < 30 or rsi > 70:  # Forte
                    signals.append(60)
                elif rsi < 35 or rsi > 65:  # Moderado
                    signals.append(45)
                elif rsi < 40 or rsi > 60:  # Fraco
                    signals.append(30)
                else:  # Neutro
                    signals.append(15)
            
            # Momentum - ESCALA MAIS SENSÍVEL
            momentum_cols = ['momentum_5_5m', 'returns_5m']
            for col in momentum_cols:
                if col in env.df.columns:
                    momentum = abs(env.df[col].iloc[current_idx])
                    if momentum > 0.02:  # Muito forte
                        signals.append(90)
                    elif momentum > 0.015:  # Forte
                        signals.append(75)
                    elif momentum > 0.01:  # Moderado
                        signals.append(60)
                    elif momentum > 0.007:  # Fraco
                        signals.append(45)
                    elif momentum > 0.003:  # Muito fraco
                        signals.append(30)
                    else:  # Praticamente zero
                        signals.append(15)
                    break
            
            # Volatilidade - MAIS ESPECÍFICA
            vol_cols = ['atr_5m', 'volatility_20_5m']
            for col in vol_cols:
                if col in env.df.columns:
                    vol = env.df[col].iloc[current_idx]
                    if 0.010 <= vol <= 0.020:  # Volatilidade perfeita
                        signals.append(90)
                    elif 0.008 <= vol <= 0.025:  # Volatilidade muito boa
                        signals.append(75)
                    elif 0.006 <= vol <= 0.030:  # Volatilidade boa
                        signals.append(60)
                    elif 0.004 <= vol <= 0.035:  # Volatilidade aceitável
                        signals.append(45)
                    elif 0.002 <= vol <= 0.040:  # Volatilidade questionável
                        signals.append(30)
                    else:  # Volatilidade ruim
                        signals.append(15)
                    break
            
            # Trend strength - MAIS DIFERENCIADO
            if 'sma_20' in env.df.columns and 'close_5m' in env.df.columns:
                price = env.df['close_5m'].iloc[current_idx]
                sma = env.df['sma_20'].iloc[current_idx]
                trend_strength = abs(price - sma) / sma if sma > 0 else 0
                
                if trend_strength > 0.025:  # Tendência muito forte
                    signals.append(90)
                elif trend_strength > 0.015:  # Tendência forte
                    signals.append(75)
                elif trend_strength > 0.010:  # Tendência moderada
                    signals.append(60)
                elif trend_strength > 0.005:  # Tendência fraca
                    signals.append(45)
                elif trend_strength > 0.002:  # Tendência muito fraca
                    signals.append(30)
                else:  # Sem tendência
                    signals.append(15)
        
        except Exception:
            signals = [30]  # Score mais baixo em caso de erro
        
        return np.mean(signals) if signals else 30
    
    def _analyze_entry_timing(self, env, current_idx: int) -> float:
        """Analisa timing de entrada (0-100)"""
        try:
            # Verificar se não está em área de alta atividade recente
            recent_activity_score = 70  # Base: timing neutro
            
            if hasattr(env, 'trades') and len(env.trades) > 0:
                recent_trades = env.trades[-5:]  # Últimos 5 trades
                
                # Se win rate recente for alto, timing é bom
                recent_wins = sum(1 for t in recent_trades if t.get('pnl_usd', 0) > 0)
                if len(recent_trades) >= 3:
                    win_rate = recent_wins / len(recent_trades)
                    if win_rate >= 0.6:
                        recent_activity_score = 90  # Timing excelente
                    elif win_rate >= 0.4:
                        recent_activity_score = 75  # Timing bom
                    else:
                        recent_activity_score = 60  # Timing questionável
            
            return recent_activity_score
            
        except Exception:
            return 70  # Timing neutro
    
    def _analyze_risk_management(self, env, action: np.ndarray) -> float:
        """Analisa gestão de risco baseada em SL/TP (0-100)"""
        try:
            if len(action) >= 6:
                sl_adjust = float(action[4])
                tp_adjust = float(action[5])
                
                # Converter para pontos - ESCALA MAIS SENSÍVEL
                sl_points = abs(sl_adjust * 15)  # Escala realista
                tp_points = abs(tp_adjust * 15)
                
                if sl_points > 0 and tp_points > 0:
                    risk_reward_ratio = tp_points / sl_points
                    
                    # Score baseado no ratio - MAIS DIFERENCIADO
                    if risk_reward_ratio >= 3.0:  # R:R >= 3:1
                        return 95  # Excepcional
                    elif risk_reward_ratio >= 2.5:  # R:R >= 2.5:1
                        return 85  # Excelente
                    elif risk_reward_ratio >= 2.0:  # R:R >= 2:1
                        return 75  # Muito bom
                    elif risk_reward_ratio >= 1.5:  # R:R >= 1.5:1
                        return 65  # Bom
                    elif risk_reward_ratio >= 1.0:  # R:R >= 1:1
                        return 50  # Aceitável
                    elif risk_reward_ratio >= 0.8:  # R:R >= 0.8:1
                        return 35  # Questionável
                    else:
                        return 20  # Ruim
                else:
                    return 15  # Sem SL/TP definido
            else:
                return 15  # Ação incompleta
                
        except Exception:
            return 30  # Score neutro
    
    def _analyze_volatility_context(self, env, current_idx: int) -> float:
        """Analisa contexto de volatilidade (0-100)"""
        try:
            vol_cols = ['atr_5m', 'volatility_20_5m']
            for col in vol_cols:
                if col in env.df.columns:
                    current_vol = env.df[col].iloc[current_idx]
                    
                    # Calcular volatilidade média dos últimos 10 períodos
                    start_idx = max(0, current_idx - 10)
                    avg_vol = env.df[col].iloc[start_idx:current_idx+1].mean()
                    
                    # Score baseado na relação com volatilidade média
                    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
                    
                    if 0.8 <= vol_ratio <= 1.5:  # Volatilidade normal
                        return 80
                    elif 0.6 <= vol_ratio <= 2.0:  # Volatilidade aceitável
                        return 65
                    else:  # Volatilidade extrema
                        return 45
                    break
            
            return 70  # Score neutro se não encontrar dados
            
        except Exception:
            return 70
    
    def get_quality_reward_bonus(self, quality_score: float) -> float:
        """
        🎯 CONVERTE SCORE DE QUALIDADE EM REWARD BONUS
        Sistema de ensino: Premia qualidade, não penaliza mediocridade
        """
        if quality_score >= 85:
            return 2.0    # Trade excepcional
        elif quality_score >= 75:
            return 1.5    # Trade de alta qualidade
        elif quality_score >= 65:
            return 1.0    # Trade de boa qualidade
        elif quality_score >= 50:
            return 0.5    # Trade médio
        else:
            return 0.0    # Trade questionável (sem bônus, mas sem penalidade)
    
    def should_allow_trade(self, env, action: np.ndarray) -> tuple[bool, float, dict]:
        """
        🧠 SISTEMA INTELIGENTE: SEMPRE PERMITE, APENAS EDUCA
        Retorna qualidade para fins educacionais, nunca bloqueia
        """
        entry_decision = int(action[0]) if len(action) > 0 else 0
        
        if entry_decision == 0:
            return True, 100.0, {"reason": "hold_action", "quality": "neutral"}
        
        # Calcular qualidade para fins educacionais
        quality_score = self.calculate_trade_quality_score(env, action)
        quality_bonus = self.get_quality_reward_bonus(quality_score)
        
        # Determinar categoria de qualidade
        if quality_score >= 85:
            quality_category = "EXCEPTIONAL"
        elif quality_score >= 75:
            quality_category = "HIGH_QUALITY"
        elif quality_score >= 65:
            quality_category = "GOOD_QUALITY"
        elif quality_score >= 50:
            quality_category = "AVERAGE"
        else:
            quality_category = "LEARNING_OPPORTUNITY"
        
        info = {
            "quality_score": quality_score,
            "quality_bonus": quality_bonus,
            "quality_category": quality_category,
            "teaching_mode": True,
            "always_allowed": True,
            "reason": "intelligent_teaching_system"
        }
        
        # 🧠 SEMPRE PERMITE - Sistema de ensino, não de bloqueio
        return True, quality_score, info

class SimpleRewardCalculator:
    """Sistema de recompensas V2 otimizado para targets específicos"""
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.step_count = 0
        self.position_history = {}
        
        # 🧠 FILTRO DE QUALIDADE INTELIGENTE - ENSINA EM VEZ DE BLOQUEAR
        self.quality_filter = QualityFilter()
        self.quality_filter_enabled = True  # 🧠 ATIVADO: Sistema inteligente que ensina qualidade
        
        # 🎯 TARGETS ESPECÍFICOS
        self.target_trades_per_day = 18
        self.sl_range_min = 11
        self.sl_range_max = 56
        self.tp_range_min = 14
        self.tp_range_max = 82
        self.optimal_risk_reward_min = 1.5
        self.optimal_risk_reward_max = 1.8
        
        # 🔥 PESOS V4 CORRIGIDOS - INCENTIVA FECHAMENTO DE TRADES
        self.weights = {
            # 💰 CORE DOMINANTE - PnL direto (90% do peso total)
            "pnl_direct": 1.0,           # Base: $1 reward = $1 PnL
            "win_bonus": 8.0,            # 🔥 MASSIVO: +$8 por trade vencedor FECHADO
            "loss_penalty": -1.0,        # 🔥 MODERADO: -$1 por trade perdedor FECHADO
            
            # 🎯 INCENTIVOS PARA FECHAMENTO (10% do peso total)
            "trade_completion_bonus": 2.0,  # 🔥 NOVO: +2.0 por FECHAR qualquer trade
            "position_management": 1.0,     # 🔥 NOVO: +1.0 por gestão ativa de posições
            "target_zone_bonus": 1.0,       # 🔥 REDUZIDO: +1.0 por estar na zona 16-20 trades/dia
            
            # 🎯 QUALIDADE ESCALADA OPCIONAL
            "excellent_trade_bonus": 2.0,    # +2.0 por trade >$15 lucro
            "quality_trade_bonus": 1.0,      # +1.0 por trade >$5 lucro
            "win_streak_bonus": 0.5,         # +0.5 por cada win consecutivo
            "quick_profit_bonus": 1.0,       # +1.0 por lucro <30 steps
            
            # 🧠 EXPERT SL/TP OPCIONAL
            "perfect_sltp_ratio": 1.0,       # +1.0 por ratio 1.5-1.8
            "target_sltp_ranges": 0.5,       # +0.5 por usar ranges corretos
            "adaptive_sltp": 0.5,            # +0.5 por adaptação contextual
            "smart_sltp_timing": 0.5,        # +0.5 por timing inteligente
            
            # 🛡️ PENALIDADES REMOVIDAS COMPLETAMENTE
            "flip_flop_penalty": 0.0,        # 🔥 REMOVIDO: Sem penalidade por flip-flop
            "micro_trade_penalty": 0.0,      # 🔥 REMOVIDO: Sem penalidade por micro-trades
            "poor_sltp_penalty": 0.0,        # 🔥 REMOVIDO: Sem penalidade por SL/TP ruins
            "excessive_drawdown": 0.0,       # 🔥 REMOVIDO: Sem penalidade por drawdown
            
            # 🎯 BÔNUS ESPECIAIS MANTIDOS
            "consistency_bonus": 0.5,        # +0.5 por manter padrão consistente
            "risk_management_bonus": 0.5,    # +0.5 por gestão de risco inteligente
        }
        
        # Tracking para análise comportamental
        self.recent_actions = []
        self.daily_trade_count = 0
        self.last_trade_type = None
        self.last_action_step = 0
        
    def reset(self):
        """Reset para novo episódio"""
        self.step_count = 0
        self.position_history = {}
        self.recent_actions = []
        self.daily_trade_count = 0
        self.last_trade_type = None
        self.last_action_step = 0
        
        # 🔥 NOVO: Reset do filtro de qualidade
        if hasattr(self, 'quality_filter'):
            self.quality_filter.recent_trades = []
            self.quality_filter.market_memory = {}
        
    def calculate_reward_and_info(self, env, action: np.ndarray, old_state: Dict) -> Tuple[float, Dict, bool]:
        """
        Calcula reward principal + info extra (incluindo trailing stop)
        """
        reward = 0.0
        info = {"components": {}, "target_analysis": {}, "behavioral_analysis": {}}
        done = False

        self.step_count += 1
        
        # Processar ações PRIMEIRO
        entry_decision = int(action[0]) if len(action) > 0 else 0
        mgmt_action = int(action[3]) if len(action) > 3 else 0
        sl_adjust = float(action[4]) if len(action) > 4 else 0.0
        tp_adjust = float(action[5]) if len(action) > 5 else 0.0
        
        # 🚨 CORREÇÃO CRÍTICA: REMOVER BÔNUS IMEDIATO QUE CAUSA MODELO TRAVADO
        # O problema era dar +1.0 por abrir trade mas só dar PnL quando fecha
        # Isso fazia o modelo aprender: "abrir = bom, manter = neutro" → TRAVADO
        
        # 🔥 NOVO: SÓ DAR RECOMPENSAS QUANDO TRADES SÃO FECHADOS OU GERENCIADOS
        immediate_reward = 0.0
        
        # 🧠 BÔNUS APENAS POR GESTÃO DE POSIÇÕES EXISTENTES
        if mgmt_action > 0 and hasattr(env, 'positions') and len(env.positions) > 0:
            # Recompensar gestão ativa de posições abertas
            management_bonus = self.weights["position_management"]
            immediate_reward += management_bonus
            info["components"]["position_management_bonus"] = management_bonus
            
            # 🧠 ADICIONAR BÔNUS DE QUALIDADE INTELIGENTE APENAS PARA GESTÃO
            if hasattr(self, 'quality_filter') and self.quality_filter_enabled:
                try:
                    allow_trade, quality_score, quality_info = self.quality_filter.should_allow_trade(env, action)
                    if mgmt_action > 0:  # Só para gestão real
                        quality_bonus = self.quality_filter.get_quality_reward_bonus(quality_score) * 0.5
                        immediate_reward += quality_bonus
                        info["components"]["quality_bonus"] = quality_bonus
                        info["quality_analysis"] = {
                            "score": quality_score,
                            "bonus": quality_bonus,
                            "category": quality_info.get("quality_category", "unknown"),
                            "teaching_mode": True,
                            "action_type": "management"
                        }
                except Exception as e:
                    info["quality_analysis"] = {"status": "error", "error": str(e)}
            else:
                info["quality_analysis"] = {"status": "disabled"}
        else:
            info["quality_analysis"] = {"status": "no_management_action"}
        
        reward += immediate_reward
        
        # 💰 COMPONENTE PRINCIPAL: PnL DIRETO (60% do peso)
        old_trades_count = old_state.get('trades_count', 0)
        current_trades_count = len(env.trades)
        
        if current_trades_count > old_trades_count:
            # Trade fechado - análise completa
            last_trade = env.trades[-1]
            pnl = last_trade.get('pnl_usd', 0.0)
            
            # 🔥 BÔNUS CRÍTICO POR FECHAR TRADE - ENSINA COMPLETAR TRADES
            completion_bonus = self.weights["trade_completion_bonus"]
            reward += completion_bonus
            info["components"]["trade_completion_bonus"] = completion_bonus
            
            # 🔥 PnL DIRETO - COMPONENTE DOMINANTE
            pnl_reward = pnl * self.weights["pnl_direct"]
            reward += pnl_reward
            info["components"]["pnl_direct"] = pnl_reward
            
            # 🔥 WIN/LOSS BALANCEADO
            if pnl > 0:
                win_bonus = self.weights["win_bonus"]
                reward += win_bonus
                info["components"]["win_bonus"] = win_bonus
                
                # 🎯 QUALIDADE ESCALADA
                if pnl > 15.0:  # Trade excelente
                    excellent_bonus = self.weights["excellent_trade_bonus"]
                    reward += excellent_bonus
                    info["components"]["excellent_trade_bonus"] = excellent_bonus
                elif pnl > 5.0:  # Trade de qualidade
                    quality_bonus = self.weights["quality_trade_bonus"]
                    reward += quality_bonus
                    info["components"]["quality_trade_bonus"] = quality_bonus
                
                # 🎯 WIN STREAK
                consecutive_wins = self._count_consecutive_wins(env.trades)
                if consecutive_wins > 1:
                    streak_bonus = self.weights["win_streak_bonus"] * min(consecutive_wins - 1, 5)
                    reward += streak_bonus
                    info["components"]["win_streak_bonus"] = streak_bonus
                
                # 🎯 LUCRO RÁPIDO
                trade_duration = last_trade.get('duration_steps', 0)
                if trade_duration < 30 and pnl > 3.0:
                    quick_bonus = self.weights["quick_profit_bonus"]
                    reward += quick_bonus
                    info["components"]["quick_profit_bonus"] = quick_bonus
                    
            else:
                loss_penalty = self.weights["loss_penalty"]
                reward += loss_penalty
                info["components"]["loss_penalty"] = loss_penalty
            
            # 🧠 EXPERT SL/TP SIMPLIFICADO (25% do peso)
            expert_sltp_reward = self._analyze_expert_sltp_v2(env, last_trade)
            reward += expert_sltp_reward
            info["target_analysis"]["expert_sltp"] = expert_sltp_reward
            
            # 🔍 ANÁLISE COMPORTAMENTAL
            behavioral_reward = self._analyze_trade_behavior(env, last_trade, entry_decision)
            reward += behavioral_reward
            info["behavioral_analysis"]["trade_behavior"] = behavioral_reward
        
        # 🎯 ATIVIDADE GUIADA AOS TARGETS (15% do peso)
        trades_today = self._get_trades_today(env)
        activity_reward = self._calculate_activity_reward(trades_today, entry_decision)
        reward += activity_reward
        info["target_analysis"]["activity"] = activity_reward
        
        # 🛡️ PENALIDADES DISCIPLINARES REMOVIDAS - CAUSAVAM HOLD ETERNO
        discipline_penalty = 0.0  # Completamente desabilitado
        info["behavioral_analysis"]["discipline"] = {"status": "disabled", "penalty": 0.0}
        
        # 🎯 GESTÃO DE RISCO E DRAWDOWN
        risk_reward = self._calculate_risk_management_reward(env)
        reward += risk_reward
        info["target_analysis"]["risk_management"] = risk_reward
        
        # --- INÍCIO: Lógica de trailing stop ---
        trailing_bonus = 0.0
        missed_trailing_penalty = 0.0
        trailing_activated = False
        trailing_executed = False
        trailing_protected = False
        trailing_timing = False

        # Exemplo: checar se trade foi fechado por trailing stop com lucro
        if 'trades' in env.__dict__ and len(env.trades) > 0:
            last_trade = env.trades[-1]
            if last_trade.get('exit_reason', '') == 'trailing_stop':
                if last_trade.get('pnl_usd', 0) > 0:
                    trailing_bonus += self.weights["trailing_stop_execution"]
                    trailing_executed = True
                # Ativação correta do trailing
                if last_trade.get('trailing_activated', False):
                    trailing_bonus += self.weights["trailing_stop_activation"]
                    trailing_activated = True
                # Proteção de lucro
                if last_trade.get('trailing_protected', False):
                    trailing_bonus += self.weights["trailing_stop_protection"]
                    trailing_protected = True
                # Timing correto
                if last_trade.get('trailing_timing', False):
                    trailing_bonus += self.weights["trailing_stop_timing"]
                    trailing_timing = True
            # Penalidade se perdeu oportunidade de trailing
            if last_trade.get('missed_trailing_opportunity', False):
                missed_trailing_penalty -= self.weights["missed_trailing_opportunity"]

        reward += trailing_bonus + missed_trailing_penalty
        info["trailing_stop_bonus"] = trailing_bonus
        info["missed_trailing_penalty"] = missed_trailing_penalty
        info["trailing_executed"] = trailing_executed
        info["trailing_activated"] = trailing_activated
        info["trailing_protected"] = trailing_protected
        info["trailing_timing"] = trailing_timing
        # --- FIM: Lógica de trailing stop ---
        
        # 🎯 LIMITES REALISTAS
        reward = np.clip(reward, -100.0, 100.0)
        
        # 📊 INFORMAÇÕES DETALHADAS
        info.update({
            "trades_today": trades_today,
            "target_trades": self.target_trades_per_day,
            "activity_zone": self._get_activity_zone(trades_today),
            "total_reward": reward,
            "step_count": self.step_count,
            "weight_distribution": {
                "pnl_core": "60%",
                "expert_sltp": "25%", 
                "activity_guided": "15%"
            },
            # 🧠 SISTEMA DE QUALIDADE INTELIGENTE
            "quality_system_status": {
                "enabled": getattr(self, 'quality_filter_enabled', False),
                "total_quality_reward": immediate_reward,
                "system_type": "intelligent_teaching_system",
                "always_allows_trades": True,
                "teaches_via_rewards": True
            }
        })
        
        return reward, info, done
    
    def _analyze_expert_sltp_v2(self, env, trade: Dict) -> float:
        """
        🧠 EXPERT SL/TP V2 - SIMPLIFICADO MAS INTELIGENTE
        Foca nos targets específicos e contexto básico
        """
        expert_reward = 0.0
        
        try:
            sl_points = abs(trade.get('sl_points', 0))
            tp_points = abs(trade.get('tp_points', 0))
            pnl = trade.get('pnl_usd', 0.0)
            
            if sl_points == 0 or tp_points == 0:
                # Penalizar falta de SL/TP
                expert_reward -= self.weights["poor_sltp_penalty"]
                return expert_reward
            
            # 🎯 ANÁLISE 1: RATIO RISK/REWARD PERFEITO
            risk_reward_ratio = tp_points / sl_points if sl_points > 0 else 0
            if self.optimal_risk_reward_min <= risk_reward_ratio <= self.optimal_risk_reward_max:
                expert_reward += self.weights["perfect_sltp_ratio"]
            elif 1.3 <= risk_reward_ratio <= 2.0:  # Próximo do ótimo
                expert_reward += self.weights["perfect_sltp_ratio"] * 0.6
            
            # 🎯 ANÁLISE 2: RANGES ALVO
            sl_in_range = self.sl_range_min <= sl_points <= self.sl_range_max
            tp_in_range = self.tp_range_min <= tp_points <= self.tp_range_max
            
            if sl_in_range and tp_in_range:
                expert_reward += self.weights["target_sltp_ranges"]
            elif sl_in_range or tp_in_range:  # Pelo menos um no range
                expert_reward += self.weights["target_sltp_ranges"] * 0.5
            else:
                # Penalizar por estar fora dos ranges
                expert_reward -= self.weights["poor_sltp_penalty"]
            
            # 🎯 ANÁLISE 3: ADAPTAÇÃO CONTEXTUAL BÁSICA
            volatility_context = self._get_market_volatility_simple(env)
            if volatility_context == "HIGH" and sl_points >= 25:  # SL maior em alta volatilidade
                expert_reward += self.weights["adaptive_sltp"]
            elif volatility_context == "LOW" and sl_points <= 25:  # SL menor em baixa volatilidade
                expert_reward += self.weights["adaptive_sltp"]
            elif volatility_context == "MEDIUM":  # Qualquer SL no range é bom
                expert_reward += self.weights["adaptive_sltp"] * 0.7
            
            # 🎯 ANÁLISE 4: TIMING INTELIGENTE
            if pnl > 0:  # Trade vencedor
                duration = trade.get('duration_steps', 0)
                if duration < 50 and sl_points <= 30:  # Scalping bem executado
                    expert_reward += self.weights["smart_sltp_timing"]
                elif 50 <= duration <= 200 and 20 <= sl_points <= 40:  # Swing bem executado
                    expert_reward += self.weights["smart_sltp_timing"]
                elif duration > 200 and sl_points >= 30:  # Position bem executado
                    expert_reward += self.weights["smart_sltp_timing"] * 0.8
            
        except Exception:
            # Em caso de erro, pequena penalidade
            expert_reward -= 0.5
        
        return expert_reward
    
    def _calculate_activity_reward(self, trades_today: int, entry_decision: int) -> float:
        """
        🎯 ATIVIDADE GUIADA AOS TARGETS - CORRIGIDO V2
        Recompensa APENAS por estar na zona alvo, SEM qualquer bônus para inatividade
        """
        activity_reward = 0.0
        
        # 🎯 ANÁLISE DA ZONA DE ATIVIDADE - APENAS ZONA ALVO RECEBE BÔNUS
        if 16 <= trades_today <= 20:  # Zona alvo (18±2) - ÚNICO CASO COM BÔNUS
            activity_reward += self.weights["target_zone_bonus"]
        
        # 🚨 REMOVIDO: Todos os outros incentivos que causavam reward inesperado
        # elif trades_today < 8:  # REMOVIDO
        # elif trades_today > 30:  # REMOVIDO
        
        return activity_reward
    
    def _analyze_trade_behavior(self, env, trade: Dict, entry_decision: int) -> float:
        """
        🔍 ANÁLISE COMPORTAMENTAL DO TRADE
        Detecta padrões inteligentes vs problemáticos
        """
        behavioral_reward = 0.0
        
        try:
            trade_type = trade.get('type', 'long')
            duration = trade.get('duration_steps', 0)
            pnl = trade.get('pnl_usd', 0.0)
            
            # Detectar micro-trades
            if duration < 5:
                behavioral_reward += self.weights["micro_trade_penalty"]
            
            # Detectar flip-flop (mudança rápida de direção)
            if (self.last_trade_type and 
                self.last_trade_type != trade_type and 
                self.step_count - self.last_action_step < 10):
                behavioral_reward += self.weights["flip_flop_penalty"]
            
            # Bônus por consistência
            if len(env.trades) >= 3:
                recent_trades = env.trades[-3:]
                win_rate = sum(1 for t in recent_trades if t.get('pnl_usd', 0) > 0) / len(recent_trades)
                if win_rate >= 0.6:  # 60%+ win rate
                    behavioral_reward += self.weights["consistency_bonus"]
            
            # Atualizar tracking
            self.last_trade_type = trade_type
            self.last_action_step = self.step_count
            
        except Exception:
            pass
        
        return behavioral_reward
    
    def _calculate_discipline_penalties(self, env, action: np.ndarray, entry_decision: int) -> float:
        """
        🛡️ PENALIDADES DISCIPLINARES MODERADAS
        Evita comportamento errático sem ser muito punitivo
        """
        penalty = 0.0
        
        # Tracking de ações recentes
        self.recent_actions.append(entry_decision)
        if len(self.recent_actions) > 20:
            self.recent_actions.pop(0)
        
        # Detectar padrões problemáticos
        if len(self.recent_actions) >= 10:
            # Flip-flop excessivo
            changes = sum(1 for i in range(1, len(self.recent_actions)) 
                         if self.recent_actions[i] != self.recent_actions[i-1])
            if changes > 7:  # Mais de 7 mudanças em 10 ações
                penalty += self.weights["flip_flop_penalty"] * 0.5
        
        return penalty
    
    def _calculate_risk_management_reward(self, env) -> float:
        """
        🎯 GESTÃO DE RISCO E DRAWDOWN
        Recompensa gestão inteligente de risco
        """
        risk_reward = 0.0
        
        try:
            # Drawdown penalty
            current_dd = getattr(env, 'current_drawdown', 0.0)
            if current_dd > 5.0:  # Acima de 5%
                risk_reward += self.weights["excessive_drawdown"] * (current_dd - 5.0)
            
            # Bônus por gestão de risco
            if hasattr(env, 'positions') and env.positions:
                positions_with_sltp = sum(1 for pos in env.positions 
                                        if pos.get('sl', 0) > 0 and pos.get('tp', 0) > 0)
                if positions_with_sltp == len(env.positions):  # Todas com SL/TP
                    risk_reward += self.weights["risk_management_bonus"]
            
        except Exception:
            pass
        
        return risk_reward
    
    def _get_activity_zone(self, trades_today: int) -> str:
        """Determina zona de atividade atual"""
        if trades_today < 12:
            return "UNDERTRADING"
        elif 16 <= trades_today <= 20:
            return "TARGET_ZONE"
        elif trades_today > 25:
            return "OVERTRADING"
        else:
            return "NORMAL"
    
    def _get_market_volatility_simple(self, env) -> str:
        """Análise simples de volatilidade"""
        try:
            if hasattr(env, 'df') and hasattr(env, 'current_step'):
                if env.current_step < 20:
                    return "MEDIUM"
                
                # Usar ATR se disponível
                for col in ['atr_5m', 'atr_14', 'atr']:
                    if col in env.df.columns:
                        current_idx = min(env.current_step, len(env.df) - 1)
                        recent_atr = env.df[col].iloc[max(0, current_idx-5):current_idx+1].mean()
                        
                        if recent_atr > 20:
                            return "HIGH"
                        elif recent_atr < 10:
                            return "LOW"
                        else:
                            return "MEDIUM"
        except Exception:
            pass
        return "MEDIUM"
    
    def _get_trades_today(self, env) -> int:
        """Conta trades realizados hoje"""
        try:
            if hasattr(env, 'trades') and env.trades:
                # Aproximação: assumir que cada episódio = 1 dia
                return len(env.trades)
            return 0
        except Exception:
            return 0
    
    def _count_consecutive_wins(self, trades: List[Dict]) -> int:
        """Conta wins consecutivos"""
        consecutive = 0
        for trade in reversed(trades):
            if trade.get('pnl_usd', 0.0) > 0:
                consecutive += 1
            else:
                break
        return consecutive

    def calculate_reward(self, action, portfolio_value, trades, current_step, old_state):
        """Método de compatibilidade para sistemas legados"""
        # Criar mock environment
        class MockEnv:
            def __init__(self, trades, current_step, portfolio_value):
                self.trades = trades
                self.current_step = current_step
                self.portfolio_value = portfolio_value
                self.positions = []
                self.current_drawdown = 0.0
        
        env = MockEnv(trades, current_step, portfolio_value)
        reward, info, done = self.calculate_reward_and_info(env, action, old_state)
        return reward

def create_simple_reward_system(initial_balance: float = 1000.0):
    """Factory function para criar sistema de rewards simples"""
    return SimpleRewardCalculator(initial_balance)

# Removido create_simple_reward_system_with_execution - usando apenas SimpleRewardCalculator

# Configuração padrão para o sistema simples
SIMPLE_REWARD_CONFIG = {
    "initial_balance": 1000.0,
    "system_type": "simple_reward",
    "description": "Sistema de recompensas simples e matematicamente coerente"
} 

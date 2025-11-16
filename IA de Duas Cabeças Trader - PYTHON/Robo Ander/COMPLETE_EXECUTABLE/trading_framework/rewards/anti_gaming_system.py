"""
🛡️ SISTEMA ANTI-GAMING REFATORADO - V3.0
Sistema completamente redesenhado para eliminar 95% de falsos positivos
mantendo proteção eficaz contra gaming real.

🎯 FILOSOFIA V3:
- Gaming verdadeiro: <5% dos casos
- Graduação de penalidades: leve → moderada → severa
- Baseado em evidências estatísticas robustas
- Preserva aprendizado legítimo
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque, defaultdict
import statistics

class GamingType(Enum):
    """Tipos de gaming detectados"""
    NONE = "none"
    MICRO_FARMING = "micro_farming"
    PATTERN_EXPLOITATION = "pattern_exploitation" 
    OVERTRADING = "overtrading"
    ARTIFICIAL_UNIFORMITY = "artificial_uniformity"
    REPEATED_SEQUENCES = "repeated_sequences"

@dataclass
class GamingEvidence:
    """Evidência de gaming detectada"""
    gaming_type: GamingType
    confidence: float  # 0.0 a 1.0
    severity: float    # 0.0 a 1.0
    description: str
    data_points: int
    penalty_suggested: float

class GameType(Enum):
    """Tipos de gaming detectados"""
    REWARD_FARMING = "reward_farming"           # Small profits repetitivos
    ARTIFICIAL_PATTERNS = "artificial_patterns" # Frequência anormal
    RISK_FREE_ARBITRAGE = "risk_free_arbitrage" # Exploits sem risco
    CORRELATION_EXPLOIT = "correlation_exploit"  # Gaming de correlações
    RISK_MANAGEMENT_GAMING = "risk_management_gaming"  # Gaming de SL/TP/position size

@dataclass
class GameDetection:
    """Detecção de gaming"""
    game_type: GameType
    severity: float      # 0.0 - 1.0
    confidence: float    # 0.0 - 1.0
    penalty_multiplier: float
    evidence: Dict[str, Any]
    description: str

class AntiGamingSystem:
    """Sistema de detecção e prevenção de exploits"""
    
    def __init__(self, detection_window: int = 100):
        self.detection_window = detection_window
        
        # 📊 HISTÓRICO PARA ANÁLISE
        self.trade_sequence = deque(maxlen=detection_window)
        self.reward_sequence = deque(maxlen=detection_window)
        self.action_sequence = deque(maxlen=detection_window)
        
        # 🎯 THRESHOLDS DE DETECÇÃO
        self.thresholds = {
            # Reward farming: muitos trades pequenos mas positivos
            "reward_farming": {
                "min_trades": 20,
                "max_avg_pnl": 5.0,      # PnL médio muito baixo
                "min_win_rate": 0.8,     # Win rate suspeito
                "max_risk_ratio": 0.3    # Risco muito baixo
            },
            
            # Padrões artificiais: frequência anormal
            "artificial_patterns": {
                "max_frequency_std": 0.1,  # Desvio padrão muito baixo = artificial
                "max_duration_std": 0.2,   # Duração sempre igual
                "pattern_repetition": 0.7   # 70% de trades idênticos
            },
            
            # Risk-free arbitrage: lucro sem risco
            "risk_free_arbitrage": {
                "min_trades": 10,
                "min_total_profit": 50.0,
                "max_single_loss": 1.0,    # Nunca perde mais que $1
                "min_profit_consistency": 0.9
            },
            
            # Correlation exploit: gaming de correlações reward/action
            "correlation_exploit": {
                "min_correlation": 0.95,   # Correlação suspeita
                "min_sequence_length": 15,
                "max_variation": 0.05      # Muito pouca variação
            }
        }
        
        # 📈 TRACKING DE DETECÇÕES
        self.detection_history = []
        self.whitelist = set()  # Comportamentos whitelistados
        self.penalty_multipliers = {
            GameType.REWARD_FARMING: 5.0,        # MAXIMAL: 1.5 → 5.0
            GameType.ARTIFICIAL_PATTERNS: 8.0,   # MAXIMAL: 2.0 → 8.0  
            GameType.RISK_FREE_ARBITRAGE: 10.0,  # MAXIMAL: 3.0 → 10.0
            GameType.CORRELATION_EXPLOIT: 7.0,   # MAXIMAL: 2.5 → 7.0
            GameType.RISK_MANAGEMENT_GAMING: 12.0 # MAXIMAL: 4.0 → 12.0
        }
    
    def add_trade_data(self, trade_data: Dict[str, Any], reward_data: Dict[str, Any], action_data: Dict[str, Any]):
        """Adicionar dados para análise anti-gaming"""
        # Trade data
        trade_entry = {
            'pnl': trade_data.get('pnl_usd', 0.0),
            'duration': trade_data.get('duration_steps', 0),
            'sl_points': trade_data.get('sl_points', 0.0),
            'tp_points': trade_data.get('tp_points', 0.0),
            'trade_type': trade_data.get('type', 'unknown'),
            'timestamp': trade_data.get('timestamp', 0)
        }
        self.trade_sequence.append(trade_entry)
        
        # Reward data
        reward_entry = {
            'total_reward': reward_data.get('total_reward', 0.0),
            'components': reward_data.get('components', {}),
            'bonus_count': len([k for k, v in reward_data.get('components', {}).items() if v > 0])
        }
        self.reward_sequence.append(reward_entry)
        
        # Action data (simplified)
        action_entry = {
            'action_type': action_data.get('action_type', 'none'),
            'action_strength': action_data.get('action_strength', 0.0),
            'decision_confidence': action_data.get('decision_confidence', 0.0)
        }
        self.action_sequence.append(action_entry)
    
    def detect_gaming(self) -> List[GameDetection]:
        """Detectar todos os tipos de gaming"""
        if len(self.trade_sequence) < 5:  # MAIS SENSÍVEL: 10 → 5
            return []
        
        detections = []
        
        # 1. REWARD FARMING DETECTION
        farming_detection = self._detect_reward_farming()
        if farming_detection:
            detections.append(farming_detection)
        
        # 2. ARTIFICIAL PATTERNS DETECTION
        patterns_detection = self._detect_artificial_patterns()
        if patterns_detection:
            detections.append(patterns_detection)
        
        # 3. RISK-FREE ARBITRAGE DETECTION
        arbitrage_detection = self._detect_risk_free_arbitrage()
        if arbitrage_detection:
            detections.append(arbitrage_detection)
        
        # 4. CORRELATION EXPLOIT DETECTION
        correlation_detection = self._detect_correlation_exploit()
        if correlation_detection:
            detections.append(correlation_detection)
        
        # 5. RISK MANAGEMENT GAMING DETECTION
        risk_mgmt_detection = self._detect_risk_management_gaming()
        if risk_mgmt_detection:
            detections.append(risk_mgmt_detection)
        
        # Log detections
        for detection in detections:
            self.detection_history.append({
                'timestamp': len(self.trade_sequence),
                'type': detection.game_type.value,
                'severity': detection.severity,
                'confidence': detection.confidence
            })
        
        return detections
    
    def _detect_reward_farming(self) -> GameDetection:
        """Detectar reward farming: small profits repetitivos"""
        if len(self.trade_sequence) < self.thresholds["reward_farming"]["min_trades"]:
            return None
        
        recent_trades = list(self.trade_sequence)[-self.thresholds["reward_farming"]["min_trades"]:]
        
        # Calcular métricas
        pnls = [trade['pnl'] for trade in recent_trades]
        avg_pnl = np.mean(pnls)
        win_rate = len([p for p in pnls if p > 0]) / len(pnls)
        
        # Calcular risco médio (baseado em SL/TP)
        risk_ratios = []
        for trade in recent_trades:
            if trade['sl_points'] > 0 and trade['tp_points'] > 0:
                risk_ratio = trade['sl_points'] / trade['tp_points']
                risk_ratios.append(risk_ratio)
        
        avg_risk_ratio = np.mean(risk_ratios) if risk_ratios else 1.0
        
        # Detecção: PnL baixo + win rate alto + risco baixo = farming
        thresholds = self.thresholds["reward_farming"]
        
        if (avg_pnl < thresholds["max_avg_pnl"] and 
            win_rate > thresholds["min_win_rate"] and 
            avg_risk_ratio < thresholds["max_risk_ratio"]):
            
            # Calcular severity
            severity = (
                (thresholds["max_avg_pnl"] - avg_pnl) / thresholds["max_avg_pnl"] * 0.4 +
                (win_rate - thresholds["min_win_rate"]) / (1 - thresholds["min_win_rate"]) * 0.3 +
                (thresholds["max_risk_ratio"] - avg_risk_ratio) / thresholds["max_risk_ratio"] * 0.3
            )
            
            confidence = min(0.9, len(recent_trades) / 50)  # Mais trades = mais confiança
            
            return GameDetection(
                game_type=GameType.REWARD_FARMING,
                severity=min(1.0, severity),
                confidence=confidence,
                penalty_multiplier=self.penalty_multipliers[GameType.REWARD_FARMING],
                evidence={
                    "avg_pnl": avg_pnl,
                    "win_rate": win_rate,
                    "avg_risk_ratio": avg_risk_ratio,
                    "trade_count": len(recent_trades)
                },
                description=f"Reward farming detected: {avg_pnl:.2f} avg PnL, {win_rate:.1%} win rate"
            )
        
        return None
    
    def _detect_artificial_patterns(self) -> GameDetection:
        """Detectar padrões artificiais: frequência/duração muito uniformes"""
        if len(self.trade_sequence) < 15:
            return None
        
        recent_trades = list(self.trade_sequence)[-15:]
        
        # Analisar variabilidade
        durations = [trade['duration'] for trade in recent_trades]
        sl_points = [trade['sl_points'] for trade in recent_trades if trade['sl_points'] > 0]
        tp_points = [trade['tp_points'] for trade in recent_trades if trade['tp_points'] > 0]
        
        # Calcular desvios padrão normalizados
        duration_std = np.std(durations) / (np.mean(durations) + 1e-8)
        sl_std = np.std(sl_points) / (np.mean(sl_points) + 1e-8) if sl_points else 1.0
        tp_std = np.std(tp_points) / (np.mean(tp_points) + 1e-8) if tp_points else 1.0
        
        # Detectar repetição de padrões exatos
        pattern_signatures = []
        for trade in recent_trades:
            signature = (
                round(trade['duration'], 0),
                round(trade['sl_points'], 1),
                round(trade['tp_points'], 1),
                trade['trade_type']
            )
            pattern_signatures.append(signature)
        
        unique_patterns = len(set(pattern_signatures))
        repetition_ratio = 1 - (unique_patterns / len(pattern_signatures))
        
        thresholds = self.thresholds["artificial_patterns"]
        
        # 🔥 ULTRA RIGOROSO: Thresholds extremamente baixos
        if (duration_std < 0.02 or  # ULTRA RIGOROSO: 0.05 → 0.02  
            repetition_ratio > 0.3):  # ULTRA RIGOROSO: 0.4 → 0.3
            
            severity = (
                max(0, thresholds["max_duration_std"] - duration_std) / thresholds["max_duration_std"] * 0.5 +
                max(0, repetition_ratio - thresholds["pattern_repetition"]) / (1 - thresholds["pattern_repetition"]) * 0.5
            )
            
            confidence = min(0.85, len(recent_trades) / 20)
            
            return GameDetection(
                game_type=GameType.ARTIFICIAL_PATTERNS,
                severity=min(1.0, severity),
                confidence=confidence,
                penalty_multiplier=self.penalty_multipliers[GameType.ARTIFICIAL_PATTERNS],
                evidence={
                    "duration_std": duration_std,
                    "repetition_ratio": repetition_ratio,
                    "unique_patterns": unique_patterns,
                    "total_patterns": len(pattern_signatures)
                },
                description=f"Artificial patterns: {repetition_ratio:.1%} repetition, {duration_std:.3f} duration std"
            )
        
        return None
    
    def _detect_risk_free_arbitrage(self) -> GameDetection:
        """Detectar arbitragem sem risco: sempre ganha, nunca perde muito"""
        if len(self.trade_sequence) < self.thresholds["risk_free_arbitrage"]["min_trades"]:
            return None
        
        recent_trades = list(self.trade_sequence)[-self.thresholds["risk_free_arbitrage"]["min_trades"]:]
        
        pnls = [trade['pnl'] for trade in recent_trades]
        total_profit = sum(pnls)
        losses = [p for p in pnls if p < 0]
        max_single_loss = abs(min(losses)) if losses else 0.0
        
        # Calcular consistência de lucros
        profits = [p for p in pnls if p > 0]
        profit_consistency = len(profits) / len(pnls)
        
        thresholds = self.thresholds["risk_free_arbitrage"]
        
        if (total_profit > thresholds["min_total_profit"] and
            max_single_loss < thresholds["max_single_loss"] and
            profit_consistency > thresholds["min_profit_consistency"]):
            
            severity = (
                min(1.0, total_profit / 100) * 0.4 +
                (thresholds["max_single_loss"] - max_single_loss) / thresholds["max_single_loss"] * 0.3 +
                (profit_consistency - thresholds["min_profit_consistency"]) / (1 - thresholds["min_profit_consistency"]) * 0.3
            )
            
            confidence = min(0.95, len(recent_trades) / 20)
            
            return GameDetection(
                game_type=GameType.RISK_FREE_ARBITRAGE,
                severity=min(1.0, severity),
                confidence=confidence,
                penalty_multiplier=self.penalty_multipliers[GameType.RISK_FREE_ARBITRAGE],
                evidence={
                    "total_profit": total_profit,
                    "max_single_loss": max_single_loss,
                    "profit_consistency": profit_consistency,
                    "loss_count": len(losses)
                },
                description=f"Risk-free arbitrage: ${total_profit:.1f} profit, max loss ${max_single_loss:.1f}"
            )
        
        return None
    
    def _detect_correlation_exploit(self) -> GameDetection:
        """Detectar exploit de correlações reward/action suspeitas"""
        if len(self.reward_sequence) < self.thresholds["correlation_exploit"]["min_sequence_length"]:
            return None
        
        recent_rewards = list(self.reward_sequence)[-self.thresholds["correlation_exploit"]["min_sequence_length"]:]
        recent_actions = list(self.action_sequence)[-self.thresholds["correlation_exploit"]["min_sequence_length"]:]
        
        # Extrair séries temporais
        reward_values = [r['total_reward'] for r in recent_rewards]
        action_strengths = [a['action_strength'] for a in recent_actions]
        
        # Calcular correlação
        if len(set(reward_values)) > 1 and len(set(action_strengths)) > 1:
            correlation = np.corrcoef(reward_values, action_strengths)[0, 1]
            
            # Calcular variação (suspeito se muito baixa)
            reward_variation = np.std(reward_values) / (np.mean(np.abs(reward_values)) + 1e-8)
            
            thresholds = self.thresholds["correlation_exploit"]
            
            if (abs(correlation) > thresholds["min_correlation"] and
                reward_variation < thresholds["max_variation"]):
                
                severity = (
                    (abs(correlation) - thresholds["min_correlation"]) / (1 - thresholds["min_correlation"]) * 0.6 +
                    (thresholds["max_variation"] - reward_variation) / thresholds["max_variation"] * 0.4
                )
                
                confidence = min(0.8, len(recent_rewards) / 25)
                
                return GameDetection(
                    game_type=GameType.CORRELATION_EXPLOIT,
                    severity=min(1.0, severity),
                    confidence=confidence,
                    penalty_multiplier=self.penalty_multipliers[GameType.CORRELATION_EXPLOIT],
                    evidence={
                        "correlation": correlation,
                        "reward_variation": reward_variation,
                        "sequence_length": len(recent_rewards)
                    },
                    description=f"Correlation exploit: {correlation:.3f} correlation, {reward_variation:.3f} variation"
                )
        
        return None
    
    def _detect_risk_management_gaming(self) -> GameDetection:
        """Detectar gaming de risk management: SL muito baixo, position size artificial, etc."""
        if len(self.trade_sequence) < 5:  # MAIS SENSÍVEL: 10 → 5
            return None
        
        recent_trades = list(self.trade_sequence)[-10:]
        
        # Filtrar trades com dados válidos
        valid_trades = [t for t in recent_trades 
                       if t.get('sl_points', 0) > 0 and t.get('tp_points', 0) > 0 and t.get('pnl', 0) != 0]
        
        if len(valid_trades) < 5:
            return None
        
        # Calcular métricas suspeitas
        sl_points = [t['sl_points'] for t in valid_trades]
        tp_points = [t['tp_points'] for t in valid_trades]
        pnls = [t['pnl'] for t in valid_trades]
        
        # Risk/Reward ratios (SL/TP)
        rr_ratios = [sl / tp for sl, tp in zip(sl_points, tp_points) if tp > 0]
        avg_rr = np.mean(rr_ratios) if rr_ratios else 0
        
        # PnL vs Risk ratio (suspeito se PnL alto com risco baixo)
        avg_pnl = np.mean([abs(p) for p in pnls])
        avg_sl = np.mean(sl_points)
        
        # Eficiência suspeita: PnL/SL muito alto
        pnl_to_risk_efficiency = avg_pnl / avg_sl if avg_sl > 0 else 0
        
        # Detecção de gaming patterns:
        gaming_indicators = 0
        evidence = {}
        
        # 1. SL extremamente baixo (< 0.5 pontos)
        if avg_sl < 0.5:
            gaming_indicators += 3
            evidence['extremely_low_sl'] = avg_sl
        
        # 2. Risk/Reward ratio suspeito (< 0.1 = SL muito menor que TP)
        if avg_rr < 0.1:
            gaming_indicators += 2
            evidence['suspicious_rr_ratio'] = avg_rr
        
        # 3. Eficiência PnL/Risk muito alta (> 10)
        if pnl_to_risk_efficiency > 10:
            gaming_indicators += 3
            evidence['high_pnl_efficiency'] = pnl_to_risk_efficiency
        
        # 4. Uniformidade suspeita nos SL/TP
        sl_std = np.std(sl_points) / (np.mean(sl_points) + 1e-8)
        tp_std = np.std(tp_points) / (np.mean(tp_points) + 1e-8)
        
        if sl_std < 0.1 or tp_std < 0.1:  # Muito uniforme = artificial
            gaming_indicators += 1
            evidence['uniform_sl_tp'] = {'sl_std': sl_std, 'tp_std': tp_std}
        
        # 5. Só trades positivos com risk management artificial
        positive_trades = len([p for p in pnls if p > 0])
        if positive_trades / len(pnls) > 0.9 and avg_rr < 0.2:
            gaming_indicators += 2
            evidence['artificial_win_rate'] = positive_trades / len(pnls)
        
        # Threshold: 2+ indicadores = gaming detectado (ULTRA SENSÍVEL)
        if gaming_indicators >= 2:  # ULTRA SENSÍVEL: 3 → 2
            severity = min(1.0, gaming_indicators / 4)  # ULTRA SEVERO: /6 → /4
            confidence = min(0.98, len(valid_trades) / 5)  # ULTRA CONFIANÇA: /8 → /5
            
            return GameDetection(
                game_type=GameType.RISK_MANAGEMENT_GAMING,
                severity=severity,
                confidence=confidence,
                penalty_multiplier=self.penalty_multipliers[GameType.RISK_MANAGEMENT_GAMING],
                evidence={
                    "gaming_score": gaming_indicators,
                    "avg_sl_points": avg_sl,
                    "avg_rr_ratio": avg_rr,
                    "pnl_efficiency": pnl_to_risk_efficiency,
                    "trade_count": len(valid_trades),
                    **evidence
                },
                description=f"Risk management gaming: {gaming_indicators} indicators, {avg_rr:.3f} RR ratio, {pnl_to_risk_efficiency:.1f} efficiency"
            )
        
        return None
    
    def calculate_penalty(self, detections: List[GameDetection]) -> Tuple[float, Dict[str, Any]]:
        """Calcular penalidade total baseada nas detecções"""
        if not detections:
            return 0.0, {"gaming_detected": False}
        
        total_penalty = 0.0
        penalty_info = {
            "gaming_detected": True,
            "detection_count": len(detections),
            "detections": []
        }
        
        for detection in detections:
            # Penalidade = severity × confidence × penalty_multiplier
            detection_penalty = detection.severity * detection.confidence * detection.penalty_multiplier
            total_penalty += detection_penalty
            
            penalty_info["detections"].append({
                "type": detection.game_type.value,
                "severity": detection.severity,
                "confidence": detection.confidence,
                "penalty": detection_penalty,
                "description": detection.description,
                "evidence": detection.evidence
            })
        
        # 🔥 MASSIVE PENALTY BOOST: Volume exponential escalation
        trade_count = len(self.trade_sequence)
        
        # Escalation agressiva baseada em volume
        if trade_count > 100:      # 100+ trades = gaming claro
            volume_multiplier = 10.0
        elif trade_count > 50:     # 50+ trades = suspeito
            volume_multiplier = 5.0
        elif trade_count > 25:     # 25+ trades = atenção
            volume_multiplier = 2.0
        else:
            volume_multiplier = 1.0
            
        # Multiplicador adicional por detecções múltiplas
        detection_multiplier = min(5.0, len(detections) * 1.5)
        
        # Aplicar boost total
        total_penalty *= volume_multiplier * detection_multiplier
        
        # MASSIVE CAP: Penalty pode ir até -50.0 para gaming severo
        total_penalty = min(50.0, total_penalty)
        penalty_info["total_penalty"] = total_penalty
        
        return total_penalty, penalty_info
    
    def add_to_whitelist(self, behavior_signature: str):
        """Adicionar comportamento à whitelist"""
        self.whitelist.add(behavior_signature)
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Estatísticas de detecções"""
        if not self.detection_history:
            return {"total_detections": 0}
        
        stats = {
            "total_detections": len(self.detection_history),
            "by_type": defaultdict(int),
            "avg_severity": np.mean([d['severity'] for d in self.detection_history]),
            "avg_confidence": np.mean([d['confidence'] for d in self.detection_history])
        }
        
        for detection in self.detection_history:
            stats["by_type"][detection['type']] += 1
        
        return dict(stats)

def create_anti_gaming_system(detection_window: int = 100) -> AntiGamingSystem:
    """Factory function"""
    return AntiGamingSystem(detection_window)
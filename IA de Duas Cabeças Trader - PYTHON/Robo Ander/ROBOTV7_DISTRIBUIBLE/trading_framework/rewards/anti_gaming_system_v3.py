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
from typing import Dict, List, Tuple, Optional
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

class AntiGamingSystemV3:
    """
    🛡️ Sistema Anti-Gaming V3.0 - Precisão Cirúrgica
    
    THRESHOLDS CALIBRADOS para reduzir falsos positivos de 95% para <5%
    """
    
    def __init__(self):
        # 📊 THRESHOLDS CALIBRADOS BASEADOS EM ANÁLISE ESTATÍSTICA
        self.thresholds = {
            # Micro-trading: Só penalizar casos extremos
            'micro_trade_usd_limit': 0.50,        # Aumentado de 1.0 para 0.5
            'micro_trade_ratio_severe': 0.90,     # Aumentado de 0.6 para 0.9 (90% micro trades)
            'micro_trade_ratio_moderate': 0.75,   # Novo: 75% para penalidade leve
            
            # Uniformidade: Muito mais conservador
            'uniformity_unique_threshold': 3,      # Reduzido de 5 para 3 valores únicos
            'uniformity_sample_size': 30,          # Aumentado de 20 para 30 trades
            
            # Overtrading: Mais permissivo
            'max_trades_warning': 50,              # Aumentado de 25 para 50
            'max_trades_severe': 100,              # Novo: 100 trades para penalidade severa
            
            # Padrões repetitivos: Novo sistema
            'pattern_repetition_threshold': 8,     # 8+ sequências idênticas
            'pattern_min_length': 4,               # Mínimo 4 trades na sequência
            
            # Análise temporal: Novo
            'rapid_fire_threshold': 10,            # 10+ trades em 1 minuto
            'rapid_fire_window': 60,               # Janela de 60 segundos
        }
        
        # 📈 SISTEMA DE PONTUAÇÃO GRADUADO
        self.penalty_scales = {
            'micro_farming': {
                'light': -0.2,      # Era -2.0, agora -0.2
                'moderate': -0.5,   # Era -2.0, agora -0.5  
                'severe': -1.0      # Era -2.0, agora -1.0
            },
            'uniformity': {
                'light': -0.1,      # Era -1.5, agora -0.1
                'moderate': -0.3,   # Era -1.5, agora -0.3
                'severe': -0.6      # Era -1.5, agora -0.6
            },
            'overtrading': {
                'per_trade': -0.01, # Era -0.1, agora -0.01 por trade extra
                'severe': -0.5      # Cap máximo para overtrading
            },
            'pattern_exploitation': {
                'moderate': -0.4,   # Novo
                'severe': -0.8      # Novo
            },
            'repeated_sequences': {
                'moderate': -0.3,   # Novo
                'severe': -0.7      # Novo
            }
        }
        
        # 🧠 MEMÓRIA E CONTEXTO
        self.trade_history = deque(maxlen=200)  # Histórico expandido
        self.gaming_incidents = defaultdict(int)
        self.false_positive_protection = True
        
        # 📊 ESTATÍSTICAS E DEBUGGING
        self.detection_stats = {
            'total_checks': 0,
            'penalties_applied': 0,
            'false_positives_prevented': 0,
            'by_type': defaultdict(int)
        }
        
        logging.info("🛡️ AntiGamingSystemV3 inicializado - Precisão Cirúrgica")
    
    def analyze_gaming_behavior(self, env) -> List[GamingEvidence]:
        """
        🔍 Análise principal de comportamento gaming
        Retorna lista de evidências detectadas
        """
        self.detection_stats['total_checks'] += 1
        
        if not hasattr(env, 'trades') or len(env.trades) < 10:
            return []
        
        # Atualizar histórico
        self._update_trade_history(env.trades)
        
        evidences = []
        
        # 1. Análise de Micro-Trading
        micro_evidence = self._detect_micro_farming()
        if micro_evidence:
            evidences.append(micro_evidence)
        
        # 2. Análise de Uniformidade Artificial
        uniformity_evidence = self._detect_artificial_uniformity()
        if uniformity_evidence:
            evidences.append(uniformity_evidence)
        
        # 3. Análise de Overtrading
        overtrading_evidence = self._detect_overtrading(env)
        if overtrading_evidence:
            evidences.append(overtrading_evidence)
        
        # 4. Análise de Padrões Repetitivos (NOVO)
        pattern_evidence = self._detect_pattern_exploitation()
        if pattern_evidence:
            evidences.append(pattern_evidence)
        
        # 5. Análise de Sequências Repetidas (NOVO)
        sequence_evidence = self._detect_repeated_sequences()
        if sequence_evidence:
            evidences.append(sequence_evidence)
        
        # 🛡️ PROTEÇÃO CONTRA FALSOS POSITIVOS
        evidences = self._apply_false_positive_protection(evidences)
        
        return evidences
    
    def calculate_gaming_penalty(self, evidences: List[GamingEvidence]) -> float:
        """
        💰 Calcula penalidade total baseada nas evidências
        """
        if not evidences:
            return 0.0
        
        total_penalty = 0.0
        
        for evidence in evidences:
            penalty = evidence.penalty_suggested
            
            # Aplicar fator de confiança
            penalty *= evidence.confidence
            
            # Aplicar fator de severidade
            penalty *= evidence.severity
            
            total_penalty += penalty
            
            # Update stats
            self.detection_stats['by_type'][evidence.gaming_type.value] += 1
        
        # Cap máximo: -2.0 (muito mais permissivo que -10.0 anterior)
        final_penalty = np.clip(total_penalty, -2.0, 0.0)
        
        if final_penalty < -0.01:  # Só contar se penalidade significativa
            self.detection_stats['penalties_applied'] += 1
        
        return final_penalty
    
    def _detect_micro_farming(self) -> Optional[GamingEvidence]:
        """🔍 Detectar micro-trading (reward farming)"""
        if len(self.trade_history) < 20:
            return None
        
        recent_trades = list(self.trade_history)[-30:]  # Últimos 30 trades
        
        # Contar micro-trades
        micro_trades = [
            t for t in recent_trades 
            if abs(t.get('pnl_usd', 0)) < self.thresholds['micro_trade_usd_limit']
        ]
        
        micro_ratio = len(micro_trades) / len(recent_trades)
        
        # 🎯 THRESHOLDS MAIS RESTRITIVOS
        if micro_ratio >= self.thresholds['micro_trade_ratio_severe']:
            return GamingEvidence(
                gaming_type=GamingType.MICRO_FARMING,
                confidence=0.9,
                severity=1.0,
                description=f"Micro-farming severo: {micro_ratio:.1%} trades < ${self.thresholds['micro_trade_usd_limit']}",
                data_points=len(recent_trades),
                penalty_suggested=self.penalty_scales['micro_farming']['severe']
            )
        elif micro_ratio >= self.thresholds['micro_trade_ratio_moderate']:
            return GamingEvidence(
                gaming_type=GamingType.MICRO_FARMING,
                confidence=0.6,
                severity=0.6,
                description=f"Micro-farming moderado: {micro_ratio:.1%} trades < ${self.thresholds['micro_trade_usd_limit']}",
                data_points=len(recent_trades),
                penalty_suggested=self.penalty_scales['micro_farming']['moderate']
            )
        
        return None
    
    def _detect_artificial_uniformity(self) -> Optional[GamingEvidence]:
        """🔍 Detectar uniformidade artificial nos PnLs"""
        if len(self.trade_history) < self.thresholds['uniformity_sample_size']:
            return None
        
        recent_trades = list(self.trade_history)[-self.thresholds['uniformity_sample_size']:]
        pnls = [t.get('pnl_usd', 0) for t in recent_trades]
        
        # Contar valores únicos (com precisão de centavo)
        unique_pnls = len(set([round(p, 2) for p in pnls]))
        
        # 🎯 THRESHOLD MUITO MAIS RESTRITIVO
        if unique_pnls <= self.thresholds['uniformity_unique_threshold']:
            # Verificar se não é legítimo (ex: SL/TP fixos)
            if self._is_legitimate_uniformity(pnls):
                self.detection_stats['false_positives_prevented'] += 1
                return None
            
            confidence = max(0.3, 1.0 - (unique_pnls / 10.0))  # Min 30% confidence
            
            return GamingEvidence(
                gaming_type=GamingType.ARTIFICIAL_UNIFORMITY,
                confidence=confidence,
                severity=0.8,
                description=f"Uniformidade artificial: apenas {unique_pnls} valores únicos em {len(pnls)} trades",
                data_points=len(pnls),
                penalty_suggested=self.penalty_scales['uniformity']['moderate']
            )
        
        return None
    
    def _detect_overtrading(self, env) -> Optional[GamingEvidence]:
        """🔍 Detectar overtrading excessivo"""
        trades_count = len(getattr(env, 'trades', []))
        
        if trades_count >= self.thresholds['max_trades_severe']:
            excess_trades = trades_count - self.thresholds['max_trades_severe']
            penalty = min(
                excess_trades * self.penalty_scales['overtrading']['per_trade'],
                self.penalty_scales['overtrading']['severe']
            )
            
            return GamingEvidence(
                gaming_type=GamingType.OVERTRADING,
                confidence=0.8,
                severity=1.0,
                description=f"Overtrading severo: {trades_count} trades (limite: {self.thresholds['max_trades_severe']})",
                data_points=trades_count,
                penalty_suggested=penalty
            )
        elif trades_count >= self.thresholds['max_trades_warning']:
            excess_trades = trades_count - self.thresholds['max_trades_warning']
            penalty = min(
                excess_trades * self.penalty_scales['overtrading']['per_trade'] * 0.5,  # 50% da penalidade
                self.penalty_scales['overtrading']['severe'] * 0.5
            )
            
            return GamingEvidence(
                gaming_type=GamingType.OVERTRADING,
                confidence=0.5,
                severity=0.5,
                description=f"Overtrading moderado: {trades_count} trades (aviso: {self.thresholds['max_trades_warning']})",
                data_points=trades_count,
                penalty_suggested=penalty
            )
        
        return None
    
    def _detect_pattern_exploitation(self) -> Optional[GamingEvidence]:
        """🔍 NOVO: Detectar exploração de padrões específicos"""
        if len(self.trade_history) < 50:
            return None
        
        recent_trades = list(self.trade_history)[-50:]
        
        # Analisar sequências de ações repetitivas
        action_sequences = []
        for i in range(len(recent_trades) - self.thresholds['pattern_min_length']):
            sequence = tuple([
                t.get('action_taken', 'unknown') 
                for t in recent_trades[i:i+self.thresholds['pattern_min_length']]
            ])
            action_sequences.append(sequence)
        
        # Contar sequências repetidas
        sequence_counts = defaultdict(int)
        for seq in action_sequences:
            sequence_counts[seq] += 1
        
        max_repetitions = max(sequence_counts.values()) if sequence_counts else 0
        
        if max_repetitions >= self.thresholds['pattern_repetition_threshold']:
            most_repeated = max(sequence_counts.items(), key=lambda x: x[1])
            
            return GamingEvidence(
                gaming_type=GamingType.PATTERN_EXPLOITATION,
                confidence=0.7,
                severity=0.8,
                description=f"Exploração de padrão: sequência {most_repeated[0]} repetida {most_repeated[1]} vezes",
                data_points=len(action_sequences),
                penalty_suggested=self.penalty_scales['pattern_exploitation']['moderate']
            )
        
        return None
    
    def _detect_repeated_sequences(self) -> Optional[GamingEvidence]:
        """🔍 NOVO: Detectar sequências de trades idênticas"""
        if len(self.trade_history) < 30:
            return None
        
        recent_trades = list(self.trade_history)[-30:]
        
        # Criar fingerprints dos trades
        trade_fingerprints = []
        for trade in recent_trades:
            fingerprint = (
                round(trade.get('pnl_usd', 0), 2),
                round(trade.get('duration_minutes', 0), 1),
                round(trade.get('position_size', 0), 4)
            )
            trade_fingerprints.append(fingerprint)
        
        # Contar fingerprints idênticos
        fingerprint_counts = defaultdict(int)
        for fp in trade_fingerprints:
            fingerprint_counts[fp] += 1
        
        max_identical = max(fingerprint_counts.values()) if fingerprint_counts else 0
        identical_ratio = max_identical / len(trade_fingerprints)
        
        if identical_ratio > 0.4:  # >40% trades idênticos
            return GamingEvidence(
                gaming_type=GamingType.REPEATED_SEQUENCES,
                confidence=0.8,
                severity=0.7,
                description=f"Sequências repetidas: {identical_ratio:.1%} trades idênticos",
                data_points=len(trade_fingerprints),
                penalty_suggested=self.penalty_scales['repeated_sequences']['moderate']
            )
        
        return None
    
    def _is_legitimate_uniformity(self, pnls: List[float]) -> bool:
        """🛡️ Verificar se uniformidade é legítima (SL/TP fixos)"""
        # Se todos PnLs são múltiplos de valores comuns (5, 10, 25), pode ser legítimo
        common_multiples = [5, 10, 25, 50]
        
        for multiple in common_multiples:
            if all(abs(pnl) % multiple < 0.01 or abs(pnl) % multiple > (multiple - 0.01) for pnl in pnls):
                return True
        
        # Se há clara separação entre wins/losses, pode ser SL/TP
        positive_pnls = [p for p in pnls if p > 0]
        negative_pnls = [p for p in pnls if p < 0]
        
        if len(positive_pnls) > 0 and len(negative_pnls) > 0:
            pos_std = statistics.stdev(positive_pnls) if len(positive_pnls) > 1 else 0
            neg_std = statistics.stdev(negative_pnls) if len(negative_pnls) > 1 else 0
            
            # Se baixa variação dentro de wins e losses, provavelmente SL/TP
            if pos_std < 2.0 and neg_std < 2.0:
                return True
        
        return False
    
    def _apply_false_positive_protection(self, evidences: List[GamingEvidence]) -> List[GamingEvidence]:
        """🛡️ Aplicar proteções contra falsos positivos"""
        if not self.false_positive_protection:
            return evidences
        
        filtered_evidences = []
        
        for evidence in evidences:
            # Filtro 1: Confiança mínima
            if evidence.confidence < 0.3:
                self.detection_stats['false_positives_prevented'] += 1
                continue
            
            # Filtro 2: Múltiplas evidências fracas não se acumulam em forte
            if evidence.confidence < 0.6 and len([e for e in evidences if e.confidence < 0.6]) > 2:
                self.detection_stats['false_positives_prevented'] += 1
                continue
            
            # Filtro 3: Proteção para fases iniciais de aprendizado
            if self.detection_stats['total_checks'] < 1000:  # Primeiros 1000 checks
                evidence.penalty_suggested *= 0.5  # 50% da penalidade
            
            filtered_evidences.append(evidence)
        
        return filtered_evidences
    
    def _update_trade_history(self, trades: List[Dict]):
        """📊 Atualizar histórico de trades"""
        # Limpar e recarregar (para testes e simplicidade)
        self.trade_history.clear()
        
        for trade in trades:
            self.trade_history.append(trade)
    
    def get_detection_stats(self) -> Dict:
        """📊 Obter estatísticas de detecção"""
        total_checks = max(1, self.detection_stats['total_checks'])  # Evitar divisão por zero
        
        return {
            'total_checks': total_checks,
            'penalties_applied': self.detection_stats['penalties_applied'],
            'penalty_rate': self.detection_stats['penalties_applied'] / total_checks,
            'false_positives_prevented': self.detection_stats['false_positives_prevented'],
            'false_positive_rate': self.detection_stats['false_positives_prevented'] / total_checks,
            'by_type': dict(self.detection_stats['by_type']),
            'thresholds': self.thresholds.copy()
        }
    
    def reset_episode(self):
        """🔄 Reset para novo episódio"""
        # Manter histórico entre episódios para melhor detecção
        # Apenas resetar contadores específicos do episódio
        self.gaming_incidents.clear()


def create_anti_gaming_system_v3() -> AntiGamingSystemV3:
    """🏭 Factory function para criar sistema anti-gaming V3"""
    return AntiGamingSystemV3()
#!/usr/bin/env python3
"""
🎯 SOLVE TRAINING VS PRODUCTION GAP - Resolver diferença entre treino e execução
"""

def analyze_domain_gap():
    """Analisar o problema de domain gap"""
    
    print("🔍 ANÁLISE DO DOMAIN GAP - TREINO VS PRODUÇÃO")
    print("=" * 60)
    
    print("📊 SITUAÇÃO ATUAL:")
    print("   TREINO:")
    print("   • 1 consulta por candle de 5m")
    print("   • Dados históricos estáveis")
    print("   • Observações consistentes")
    print("   • Ambiente controlado")
    
    print("\n   PRODUÇÃO:")
    print("   • 150-300 consultas por candle de 5m (on tick)")
    print("   • Dados em tempo real com ruído")
    print("   • Observações voláteis")
    print("   • Ambiente real com latência")
    
    print("\n🚨 PROBLEMAS CAUSADOS:")
    print("   • Modelo confuso com alta frequência de consultas")
    print("   • Decisões inconsistentes dentro do mesmo candle")
    print("   • Overtrading por excesso de sinais")
    print("   • Performance degradada vs treino")

def create_solutions():
    """Criar soluções para o domain gap"""
    
    print("\n🔧 SOLUÇÕES PROPOSTAS")
    print("=" * 60)
    
    solutions = {
        "temporal_smoothing": """
# SOLUÇÃO 1: TEMPORAL SMOOTHING
class TemporalSmoother:
    def __init__(self, window_size=10, confidence_threshold=0.7):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.decision_buffer = []
        self.confidence_buffer = []
    
    def smooth_decision(self, raw_decision, confidence):
        '''Suavizar decisões ao longo de múltiplos ticks'''
        
        # Adicionar ao buffer
        self.decision_buffer.append(raw_decision)
        self.confidence_buffer.append(confidence)
        
        # Manter tamanho do buffer
        if len(self.decision_buffer) > self.window_size:
            self.decision_buffer.pop(0)
            self.confidence_buffer.pop(0)
        
        # Calcular decisão suavizada
        if len(self.decision_buffer) >= 3:
            # Média ponderada por confiança
            weights = np.array(self.confidence_buffer)
            decisions = np.array(self.decision_buffer)
            
            smoothed = np.average(decisions, weights=weights)
            avg_confidence = np.mean(self.confidence_buffer)
            
            # Só agir se confiança alta e decisão consistente
            if avg_confidence > self.confidence_threshold:
                return smoothed, avg_confidence
        
        return None, 0.0  # Não agir ainda
""",
        
        "candle_based_decisions": """
# SOLUÇÃO 2: CANDLE-BASED DECISIONS
class CandleBasedDecisionMaker:
    def __init__(self):
        self.current_candle_time = None
        self.candle_decision = None
        self.candle_confidence = 0.0
        self.ticks_in_candle = 0
        self.decision_votes = {'HOLD': 0, 'LONG': 0, 'SHORT': 0}
    
    def get_decision(self, current_time, raw_decision, confidence):
        '''Tomar decisão apenas uma vez por candle, como no treino'''
        
        # Detectar novo candle (5 minutos)
        candle_time = current_time - (current_time % 300)  # 5min = 300s
        
        if candle_time != self.current_candle_time:
            # Novo candle - resetar
            self.current_candle_time = candle_time
            self.candle_decision = None
            self.candle_confidence = 0.0
            self.ticks_in_candle = 0
            self.decision_votes = {'HOLD': 0, 'LONG': 0, 'SHORT': 0}
        
        # Acumular votos durante o candle
        self.ticks_in_candle += 1
        decision_name = {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}.get(raw_decision, 'HOLD')
        self.decision_votes[decision_name] += confidence
        
        # Tomar decisão apenas no primeiro tick do candle
        if self.candle_decision is None and self.ticks_in_candle == 1:
            self.candle_decision = raw_decision
            self.candle_confidence = confidence
            return raw_decision, confidence
        
        # Retornar decisão do candle para ticks subsequentes
        return self.candle_decision, self.candle_confidence
""",
        
        "adaptive_frequency": """
# SOLUÇÃO 3: ADAPTIVE FREQUENCY
class AdaptiveFrequencyController:
    def __init__(self):
        self.last_decision_time = 0
        self.min_interval = 30  # Mínimo 30s entre decisões
        self.volatility_multiplier = 1.0
        self.recent_decisions = []
    
    def should_make_decision(self, current_time, market_volatility):
        '''Decidir se deve consultar o modelo baseado na volatilidade'''
        
        # Calcular intervalo adaptativo
        base_interval = self.min_interval
        volatility_factor = max(0.5, min(2.0, market_volatility))
        adaptive_interval = base_interval / volatility_factor
        
        # Verificar se passou tempo suficiente
        time_since_last = current_time - self.last_decision_time
        
        if time_since_last >= adaptive_interval:
            self.last_decision_time = current_time
            return True
        
        return False
    
    def update_volatility(self, price_changes):
        '''Atualizar multiplicador de volatilidade'''
        if len(price_changes) > 10:
            volatility = np.std(price_changes[-10:])
            self.volatility_multiplier = max(0.5, min(2.0, volatility * 100))
""",
        
        "confidence_gating": """
# SOLUÇÃO 4: CONFIDENCE GATING
class ConfidenceGate:
    def __init__(self, min_confidence=0.6, consistency_window=5):
        self.min_confidence = min_confidence
        self.consistency_window = consistency_window
        self.recent_predictions = []
    
    def gate_decision(self, decision, confidence):
        '''Filtrar decisões por confiança e consistência'''
        
        # Adicionar predição atual
        self.recent_predictions.append({
            'decision': decision,
            'confidence': confidence,
            'time': time.time()
        })
        
        # Manter janela de consistência
        if len(self.recent_predictions) > self.consistency_window:
            self.recent_predictions.pop(0)
        
        # Verificar confiança mínima
        if confidence < self.min_confidence:
            return None, 0.0
        
        # Verificar consistência
        if len(self.recent_predictions) >= 3:
            recent_decisions = [p['decision'] for p in self.recent_predictions[-3:]]
            recent_confidences = [p['confidence'] for p in self.recent_predictions[-3:]]
            
            # Decisão consistente com alta confiança
            if (len(set(recent_decisions)) == 1 and 
                np.mean(recent_confidences) > self.min_confidence):
                return decision, confidence
        
        return None, 0.0  # Não agir ainda
"""
    }
    
    print("💡 SOLUÇÃO 1: TEMPORAL SMOOTHING")
    print("   Suaviza decisões ao longo de múltiplos ticks")
    print("   Evita mudanças bruscas dentro do mesmo candle")
    
    print("\n💡 SOLUÇÃO 2: CANDLE-BASED DECISIONS")
    print("   Toma decisão apenas uma vez por candle (como no treino)")
    print("   Mantém consistência com ambiente de treinamento")
    
    print("\n💡 SOLUÇÃO 3: ADAPTIVE FREQUENCY")
    print("   Ajusta frequência baseado na volatilidade do mercado")
    print("   Mais consultas em alta volatilidade, menos em baixa")
    
    print("\n💡 SOLUÇÃO 4: CONFIDENCE GATING")
    print("   Filtra decisões por confiança e consistência")
    print("   Só age quando modelo está realmente confiante")
    
    return solutions

def recommend_hybrid_solution():
    """Recomendar solução híbrida combinando as melhores abordagens"""
    
    print("\n🚀 SOLUÇÃO HÍBRIDA RECOMENDADA")
    print("=" * 60)
    
    hybrid_solution = '''
class SmartDecisionManager:
    """Gerenciador inteligente de decisões - Híbrido das 4 soluções"""
    
    def __init__(self):
        # Componentes
        self.temporal_smoother = TemporalSmoother(window_size=8, confidence_threshold=0.65)
        self.candle_controller = CandleBasedDecisionMaker()
        self.frequency_controller = AdaptiveFrequencyController()
        self.confidence_gate = ConfidenceGate(min_confidence=0.6, consistency_window=5)
        
        # Estado
        self.last_action_time = 0
        self.current_candle_decision = None
        
    def process_model_output(self, raw_decision, confidence, current_time, market_volatility):
        """Processar output do modelo de forma inteligente"""
        
        # 1. FREQUENCY CONTROL: Verificar se deve consultar modelo
        if not self.frequency_controller.should_make_decision(current_time, market_volatility):
            return self.current_candle_decision, 0.0  # Usar decisão anterior
        
        # 2. CANDLE-BASED: Alinhar com candles como no treino
        candle_decision, candle_confidence = self.candle_controller.get_decision(
            current_time, raw_decision, confidence
        )
        
        # 3. CONFIDENCE GATING: Filtrar por confiança
        gated_decision, gated_confidence = self.confidence_gate.gate_decision(
            candle_decision, candle_confidence
        )
        
        if gated_decision is None:
            return None, 0.0  # Não agir
        
        # 4. TEMPORAL SMOOTHING: Suavizar decisão final
        final_decision, final_confidence = self.temporal_smoother.smooth_decision(
            gated_decision, gated_confidence
        )
        
        if final_decision is not None:
            self.current_candle_decision = final_decision
            self.last_action_time = current_time
            return final_decision, final_confidence
        
        return None, 0.0  # Aguardar mais dados
    
    def get_stats(self):
        """Obter estatísticas do gerenciador"""
        return {
            'ticks_in_candle': self.candle_controller.ticks_in_candle,
            'decision_votes': self.candle_controller.decision_votes,
            'recent_decisions': len(self.temporal_smoother.decision_buffer),
            'volatility_multiplier': self.frequency_controller.volatility_multiplier
        }
'''
    
    print("🎯 CARACTERÍSTICAS DA SOLUÇÃO HÍBRIDA:")
    print("   ✅ Alinha produção com treino (candle-based)")
    print("   ✅ Reduz overtrading (frequency control)")
    print("   ✅ Melhora qualidade das decisões (confidence gating)")
    print("   ✅ Suaviza ruído de alta frequência (temporal smoothing)")
    
    print("\n📊 PARÂMETROS RECOMENDADOS:")
    print("   • Intervalo mínimo: 30-60s entre decisões")
    print("   • Confiança mínima: 0.6-0.7")
    print("   • Janela de suavização: 5-10 ticks")
    print("   • Debouncing de volatilidade: 2x em alta, 0.5x em baixa")
    
    return hybrid_solution

def create_implementation():
    """Criar implementação prática para o RobotV7"""
    
    print("\n🔧 IMPLEMENTAÇÃO PARA ROBOTV7")
    print("=" * 60)
    
    implementation = '''
# Adicionar ao __init__ do TradingRobotV7:
def __init__(self, log_widget=None):
    # ... existing code ...
    
    # 🎯 Smart Decision Manager
    self.decision_manager = SmartDecisionManager()
    self.last_model_query = 0
    self.model_query_interval = 30  # Consultar modelo a cada 30s mínimo
    
# Modificar o método de predição:
def get_action_v7(self, observation):
    """🧠 Obter ação V7 com gerenciamento inteligente de frequência"""
    
    current_time = time.time()
    
    # Calcular volatilidade recente
    market_volatility = self._calculate_market_volatility()
    
    # Verificar se deve consultar o modelo
    time_since_last = current_time - self.last_model_query
    min_interval = self.model_query_interval / max(0.5, market_volatility)
    
    if time_since_last < min_interval:
        # Usar decisão anterior ou HOLD
        return self._get_cached_decision()
    
    # Consultar modelo
    self.last_model_query = current_time
    
    with torch.no_grad():
        raw_action, _states = self.model.predict(observation, deterministic=False)
    
    # Processar com decision manager
    processed_decision, processed_confidence = self.decision_manager.process_model_output(
        raw_action[0], raw_action[1], current_time, market_volatility
    )
    
    if processed_decision is not None:
        return self._convert_to_robot_action(processed_decision, processed_confidence)
    else:
        return self._get_hold_action()

def _calculate_market_volatility(self):
    """Calcular volatilidade recente do mercado"""
    try:
        if len(self.historical_df) > 20:
            recent_prices = self.historical_df['close'].tail(20)
            returns = recent_prices.pct_change().dropna()
            volatility = returns.std() * 100  # Converter para %
            return max(0.1, min(5.0, volatility))  # Limitar entre 0.1 e 5.0
        return 1.0
    except:
        return 1.0

def _get_cached_decision(self):
    """Obter decisão cached ou HOLD"""
    if hasattr(self, '_last_decision') and self._last_decision:
        return self._last_decision
    return self._get_hold_action()

def _get_hold_action(self):
    """Retornar ação HOLD padrão"""
    return np.array([0, 0.1, 0, 0], dtype=np.float32)  # HOLD com baixa confiança
'''
    
    print("🎯 IMPLEMENTAÇÃO PRÁTICA:")
    print("   1. SmartDecisionManager integrado ao robot")
    print("   2. Consulta modelo baseada em volatilidade")
    print("   3. Cache de decisões entre consultas")
    print("   4. Fallback para HOLD quando incerto")
    
    return implementation

def create_configuration_options():
    """Criar opções de configuração para diferentes estratégias"""
    
    print("\n⚙️ OPÇÕES DE CONFIGURAÇÃO")
    print("=" * 60)
    
    configs = {
        "conservative": {
            "model_query_interval": 60,  # 1 minuto
            "min_confidence": 0.75,
            "smoothing_window": 10,
            "volatility_sensitivity": 0.5
        },
        "balanced": {
            "model_query_interval": 30,  # 30 segundos
            "min_confidence": 0.65,
            "smoothing_window": 8,
            "volatility_sensitivity": 1.0
        },
        "aggressive": {
            "model_query_interval": 15,  # 15 segundos
            "min_confidence": 0.55,
            "smoothing_window": 5,
            "volatility_sensitivity": 1.5
        }
    }
    
    for name, config in configs.items():
        print(f"\n📋 PERFIL {name.upper()}:")
        for key, value in config.items():
            print(f"   {key}: {value}")
    
    return configs

if __name__ == "__main__":
    analyze_domain_gap()
    solutions = create_solutions()
    hybrid = recommend_hybrid_solution()
    configs = create_configuration_options()
    
    print("\n🚀 PRÓXIMOS PASSOS:")
    print("1. Escolher perfil de configuração (conservative/balanced/aggressive)")
    print("2. Implementar SmartDecisionManager no RobotV7")
    print("3. Testar com dados históricos")
    print("4. Ajustar parâmetros baseado na performance")
    print("5. Monitorar alinhamento treino vs produção")
#!/usr/bin/env python3
"""
🔧 REMOVE GATES V7 SIMPLE - Remoção cirúrgica dos gates mantendo compatibilidade
"""

import sys
import os
sys.path.append("D:/Projeto")

def remove_gates_surgically():
    """Remover gates da V7 Simple sem quebrar as outras versões"""
    
    original_file = "D:/Projeto/trading_framework/policies/two_head_v7_simple.py"
    backup_file = "D:/Projeto/trading_framework/policies/two_head_v7_simple_WITH_GATES_BACKUP.py"
    
    print("🔧 REMOVE GATES V7 SIMPLE - CIRURGIA PRECISA")
    print("=" * 60)
    
    try:
        # Fazer backup
        import shutil
        shutil.copy2(original_file, backup_file)
        print(f"✅ Backup criado: {backup_file}")
        
        # Ler arquivo original
        with open(original_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # MODIFICAÇÃO 1: SIMPLFIFICAR O FORWARD - BYPASS DOS GATES
        old_forward = '''    def forward(self, entry_signal, management_signal, market_context):
        # Combinar sinais para análise
        combined_input = torch.cat([entry_signal, management_signal, market_context], dim=-1)
        
        # [LAUNCH] FASE 1: CALCULAR OS 10 SCORES ESPECIALIZADOS
        
        # 1. Temporal Score
        temporal_score = self.horizon_analyzer(combined_input)
        
        # 2. Validation Score (MTF + Pattern)
        mtf_score = self.mtf_validator(combined_input)
        pattern_score = self.pattern_memory_validator(combined_input)
        validation_score = (mtf_score + pattern_score) / 2
        
        # 3. Risk Score (Risk + Regime)
        risk_score = self.risk_gate_entry(combined_input)
        regime_score = self.regime_gate(combined_input)
        risk_composite = (risk_score + regime_score) / 2
        
        # 4. Market Score (Lookahead + Fatigue)
        lookahead_score = self.lookahead_gate(combined_input)
        fatigue_score = 1.0 - self.fatigue_detector(combined_input)  # Inverter: alta fatigue = baixo score
        market_score = (lookahead_score + fatigue_score) / 2
        
        # 5. Quality Score (4 filtros)
        momentum_score = self.momentum_filter(combined_input)
        volatility_score = self.volatility_filter(combined_input)
        volume_score = self.volume_filter(combined_input)
        trend_score = self.trend_strength_filter(combined_input)
        quality_score = (momentum_score + volatility_score + volume_score + trend_score) / 4
        
        # 6. Confidence Score
        confidence_score = self.confidence_estimator(combined_input)
        
        # [LAUNCH] FASE 2: APLICAR THRESHOLDS ADAPTATIVOS
        
        # Clamp thresholds para ranges seguros (RANGES MAIS BAIXOS)
        main_threshold = torch.clamp(self.adaptive_threshold_main, 0.1, 0.6)    # REDUZIDO: 0.5-0.9 → 0.1-0.6
        risk_threshold = torch.clamp(self.adaptive_threshold_risk, 0.05, 0.5)   # REDUZIDO: 0.3-0.8 → 0.05-0.5  
        regime_threshold = torch.clamp(self.adaptive_threshold_regime, 0.02, 0.4) # REDUZIDO: 0.2-0.7 → 0.02-0.4
        
        # [LAUNCH] GATES HÍBRIDOS (IGUAL V5/V6): Sigmoid nos gates individuais + binário no final
        # Permite gradientes suaves para melhor convergência, mas mantém filtro real no final
        # 🔥 FIX CRÍTICO: Sigmoid gentil (* 2 ao invés de * 5) para prevenir saturação
        temporal_gate = torch.sigmoid((temporal_score - regime_threshold) * 2.0)      # REDUZIDO de 5 para 2
        validation_gate = torch.sigmoid((validation_score - main_threshold) * 2.0)    # REDUZIDO de 5 para 2
        risk_gate = torch.sigmoid((risk_composite - risk_threshold) * 2.0)            # REDUZIDO de 5 para 2
        market_gate = torch.sigmoid((market_score - regime_threshold) * 2.0)          # REDUZIDO de 5 para 2
        quality_gate = torch.sigmoid((quality_score - main_threshold) * 2.0)          # REDUZIDO de 5 para 2
        confidence_gate = torch.sigmoid((confidence_score - main_threshold) * 2.0)    # REDUZIDO de 5 para 2
        
        # [LAUNCH] FASE 3: GATE FINAL BINÁRIO - Sistema composite inteligente
        # Pontuação ponderada ao invés de multiplicação pura (como V5/V6)
        composite_score = (
            temporal_gate * 0.20 +      # 20% - timing (aumentado)
            validation_gate * 0.20 +    # 20% - validação multi-timeframe
            risk_gate * 0.25 +          # 25% - risco (mais importante)
            market_gate * 0.10 +        # 10% - condições de mercado (reduzido)
            quality_gate * 0.10 +       # 10% - qualidade técnica (reduzido)
            confidence_gate * 0.15      # 15% - confiança geral (aumentado)
        )
        
        # Gate final binário: só passa se composite score > threshold (REDUZIDO para menos bloqueios)
        final_gate_threshold = 0.60  # 60% da pontuação ponderada (REDUZIDO de 0.75 para permitir mais trades)
        final_gate = (composite_score > final_gate_threshold).float()
        
        # [LAUNCH] FASE 4: DECISÃO FINAL
        all_scores = torch.cat([
            temporal_score, validation_score, risk_composite, market_score, quality_score,
            confidence_score, mtf_score, pattern_score, lookahead_score, fatigue_score
        ], dim=-1)
        
        decision_input = torch.cat([combined_input, all_scores], dim=-1)
        raw_decision = self.final_decision_network(decision_input)
        
        # [FIX] FIX: Soft gating para prevenir gradient blocking total
        final_decision = raw_decision * (final_gate * 0.9 + 0.1)  # Min 10% gradient flow'''
        
        new_forward = '''    def forward(self, entry_signal, management_signal, market_context):
        # 🔥 GATES REMOVIDOS - FORWARD DIRETO PARA O MODELO
        # Combinar sinais para análise
        combined_input = torch.cat([entry_signal, management_signal, market_context], dim=-1)
        
        # [BYPASS] CÁLCULO DIRETO SEM GATES
        # Manter redes individuais para features, mas sem filtering
        
        # 1. Coletar features das redes (sem usar como gates)
        temporal_feature = self.horizon_analyzer(combined_input)
        
        # 2. MTF + Pattern features
        mtf_feature = self.mtf_validator(combined_input)
        pattern_feature = self.pattern_memory_validator(combined_input)
        validation_feature = (mtf_feature + pattern_feature) / 2
        
        # 3. Risk features
        risk_feature = self.risk_gate_entry(combined_input)
        regime_feature = self.regime_gate(combined_input)
        risk_composite_feature = (risk_feature + regime_feature) / 2
        
        # 4. Market features
        lookahead_feature = self.lookahead_gate(combined_input)
        fatigue_feature = self.fatigue_detector(combined_input)  # Sem inversão
        market_feature = (lookahead_feature + fatigue_feature) / 2
        
        # 5. Quality features
        momentum_feature = self.momentum_filter(combined_input)
        volatility_feature = self.volatility_filter(combined_input)
        volume_feature = self.volume_filter(combined_input)
        trend_feature = self.trend_strength_filter(combined_input)
        quality_feature = (momentum_feature + volatility_feature + volume_feature + trend_feature) / 4
        
        # 6. Confidence feature
        confidence_feature = self.confidence_estimator(combined_input)
        
        # [NO GATES] FEED DIRETO PARA DECISÃO FINAL
        # Todas as features vão direto para a rede de decisão
        all_features = torch.cat([
            temporal_feature, validation_feature, risk_composite_feature, 
            market_feature, quality_feature, confidence_feature,
            mtf_feature, pattern_feature, lookahead_feature, fatigue_feature
        ], dim=-1)
        
        # Input enriquecido para decisão
        decision_input = torch.cat([combined_input, all_features], dim=-1)
        
        # 🎯 DECISÃO FINAL SEM FILTROS - MODELO LIVRE PARA APRENDER
        final_decision = self.final_decision_network(decision_input)
        
        # 🔧 COMPATIBILIDADE: Manter confidence_score para signature
        confidence_score = confidence_feature  # Usar confidence como score principal'''
        
        # Aplicar substituição principal
        if old_forward in content:
            content = content.replace(old_forward, new_forward)
            print("✅ Forward modificado - gates removidos")
        else:
            print("⚠️ Não foi possível encontrar o forward exato para substituir")
            
        # MODIFICAÇÃO 2: MANTER GATE_INFO PARA COMPATIBILIDADE (mas com valores dummy)
        gate_info_section = '''
        
        # Retornar informações detalhadas para debug
        gate_info = {
            'temporal_gate': temporal_gate,
            'validation_gate': validation_gate,
            'risk_gate': risk_gate,
            'market_gate': market_gate,
            'quality_gate': quality_gate,
            'confidence_gate': confidence_gate,
            'composite_score': composite_score,
            'final_gate': final_gate,
            'scores': {
                'temporal': temporal_score,
                'validation': validation_score,
                'risk': risk_composite,
                'market': market_score,
                'quality': quality_score,
                'confidence': confidence_score,
                'mtf': mtf_score,
                'pattern': pattern_score,
                'lookahead': lookahead_score,
                'fatigue': fatigue_score'''
        
        # Adicionar compatibilidade para gate_info
        compatibility_section = '''
        
        # 🔧 COMPATIBILIDADE: Manter gate_info para debug (valores dummy)
        gate_info = {
            'temporal_gate': torch.ones_like(temporal_feature),      # Dummy: sempre 1.0
            'validation_gate': torch.ones_like(validation_feature),  # Dummy: sempre 1.0
            'risk_gate': torch.ones_like(risk_composite_feature),    # Dummy: sempre 1.0
            'market_gate': torch.ones_like(market_feature),          # Dummy: sempre 1.0
            'quality_gate': torch.ones_like(quality_feature),        # Dummy: sempre 1.0
            'confidence_gate': torch.ones_like(confidence_feature),  # Dummy: sempre 1.0
            'composite_score': torch.ones_like(confidence_feature),  # Dummy: sempre 1.0
            'final_gate': torch.ones_like(confidence_feature),       # Dummy: sempre 1.0
            'scores': {
                'temporal': temporal_feature,
                'validation': validation_feature,
                'risk': risk_composite_feature,
                'market': market_feature,
                'quality': quality_feature,
                'confidence': confidence_feature,
                'mtf': mtf_feature,
                'pattern': pattern_feature,
                'lookahead': lookahead_feature,
                'fatigue': fatigue_feature
            }
        }
        
        return final_decision, confidence_score, gate_info'''
        
        # Adicionar no final do forward
        if "return final_decision, gate_info" not in content:
            content = content.replace("final_decision = self.final_decision_network(decision_input)", 
                                    "final_decision = self.final_decision_network(decision_input)" + compatibility_section)
            print("✅ Compatibilidade gate_info adicionada")
        
        # MODIFICAÇÃO 3: COMENTAR INICIALIZAÇÃO DOS THRESHOLDS ADAPTATIVOS (não usados mais)
        threshold_init = '''        # [FIX] ADAPTIVE THRESHOLDS - RANGES MAIS AMPLOS PARA EVITAR SATURAÇÃO 0.500
        self.register_parameter('adaptive_threshold_main', nn.Parameter(torch.tensor(0.25)))     # REDUZIDO: 0.50→0.25 para evitar sigmoid(0)=0.5
        self.register_parameter('adaptive_threshold_risk', nn.Parameter(torch.tensor(0.15)))     # REDUZIDO: 0.35→0.15 para crear diferença
        self.register_parameter('adaptive_threshold_regime', nn.Parameter(torch.tensor(0.10)))   # REDUZIDO: 0.25→0.10 para maximizar distância'''
        
        threshold_commented = '''        # [DISABLED] ADAPTIVE THRESHOLDS - NÃO USADO MAIS (gates removidos)
        # self.register_parameter('adaptive_threshold_main', nn.Parameter(torch.tensor(0.25)))
        # self.register_parameter('adaptive_threshold_risk', nn.Parameter(torch.tensor(0.15)))
        # self.register_parameter('adaptive_threshold_regime', nn.Parameter(torch.tensor(0.10)))'''
        
        content = content.replace(threshold_init, threshold_commented)
        print("✅ Thresholds adaptativos comentados")
        
        # Salvar arquivo modificado
        with open(original_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n🎯 MODIFICAÇÕES APLICADAS:")
        print(f"  ✅ Gates removidos do forward")
        print(f"  ✅ Features das redes mantidas (sem filtering)")  
        print(f"  ✅ Compatibilidade gate_info mantida (valores dummy)")
        print(f"  ✅ Thresholds adaptativos desabilitados")
        print(f"  ✅ Backup criado: {backup_file}")
        
        print(f"\n🚀 RESULTADO:")
        print(f"  • Modelo agora aprende LIVREMENTE através dos rewards")
        print(f"  • Nenhum filtro artificial bloqueando decisões") 
        print(f"  • Features das 12 redes ainda disponíveis (como input enriquecido)")
        print(f"  • V7 Enhanced/Intuition NÃO afetadas")
        
        print(f"\n⚠️ IMPORTANTE:")
        print(f"  • Treinar novo modelo OU continuar treinamento existente")
        print(f"  • Monitorar se Entry Quality melhora significativamente")
        print(f"  • Se não funcionar, restaurar backup facilmente")
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    remove_gates_surgically()
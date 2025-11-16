#!/usr/bin/env python3
"""
🎯 ESTRATÉGIA PARA TRANSIÇÃO MULTI-TIMEFRAME
Preparação específica para quando "o bixo pega" na Fase 1
"""

# 📊 ANÁLISE DO PROBLEMA MULTI-TIMEFRAME
PHASE_ANALYSIS = {
    "Phase_0": {
        "dataset": "1m homogêneo (100k barras)",
        "features": "Timeframe único",
        "complexity": "Baixa - Padrões consistentes",
        "gradient_risk": "Mínimo",
        "lr_needed": "2.68e-05 a 5.0e-05",
        "problems": "Raros - LSTMs aprendem bem"
    },
    "Phase_1": {
        "dataset": "5m + 15m + features",
        "features": "Multi-timeframe heterogêneas",
        "complexity": "CRÍTICA - Escalas temporais conflitantes",
        "gradient_risk": "EXTREMO - 'O bixo pega'",
        "lr_needed": "6.0e-05 a 8.0e-05",
        "problems": [
            "Temporal confusion (LSTMs veem escalas misturadas)",
            "Feature scale mismatch (magnitudes diferentes)",
            "Pattern conflicts (sinais contraditórios)",
            "Gradient vanishing (complexidade explode)",
            "LSTM memory corruption (não sabe o que lembrar)"
        ]
    }
}

# 🎯 ESTRATÉGIA ESCALONADA
ESCALATION_STRATEGY = {
    "Phase_0_Start": {
        "lr": 5.0e-05,
        "grad_clip": 0.8,
        "thresholds": "Moderados",
        "monitoring": "Normal (2000 steps)"
    },
    "Phase_1_Start": {
        "lr": 6.0e-05,  # Preparação para 5m+15m+features
        "grad_clip": 1.0,
        "thresholds": {
            "main": 0.50,     # Mais permissivo para conflitos
            "risk": 0.35,     # Aceitar mais incerteza
            "regime": 0.25    # Flexível com mudanças
        },
        "monitoring": "Intensivo (1000 steps)",
        "entropy_coef": 0.025  # Mais exploração para padrões novos
    },
    "Phase_1_Emergency": {
        "lr": 8.0e-05,  # Se temporal confusion aparecer
        "grad_clip": 1.2,
        "thresholds": {
            "main": 0.45,     # Ainda mais permissivo
            "risk": 0.30,     # Aceitar alta incerteza
            "regime": 0.20    # Muito flexível
        },
        "monitoring": "Crítico (500 steps)",
        "entropy_coef": 0.035,  # Máxima exploração
        "batch_size": 32  # Reduzir para estabilidade
    }
}

# 🚨 SINAIS DE ALERTA PARA FASE 1
PHASE_1_WARNING_SIGNS = [
    "LSTM zeros > 50%",
    "Portfolio performance drop > 20%",
    "Win rate drop > 10%",
    "Gradient norm < 1e-05",
    "Explained variance < 0.1"
]

# 🛠️ CORREÇÕES AUTOMÁTICAS SUGERIDAS
AUTO_CORRECTIONS = {
    "gradient_vanishing": "Aumentar LR em 50%",
    "lstm_confusion": "Reduzir thresholds dos gates",
    "pattern_conflicts": "Aumentar entropy coefficient",
    "instability": "Reduzir batch size temporariamente"
}

print("🎯 ESTRATÉGIA MULTI-TIMEFRAME CARREGADA")
print("   - Fase 0: Preparação conservadora")
print("   - Fase 1: Monitoramento intensivo")
print("   - Emergência: Correções automáticas")
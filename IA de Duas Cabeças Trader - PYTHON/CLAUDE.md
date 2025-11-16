# Contexto do Projeto DayTrader - Sistema RL Trading

## Status Atual do Sistema
- **Versão Atual**: V7 Intuition
- **Checkpoint Mais Recente**: 5.85M steps
- **Sistema Principal**: daytrader.py (446KB)
- **Framework**: Stable-Baselines3 + Transformer Policy

## Componentes Críticos

### 1. Sistema de Debug de Zeros
- **Localização**: `zero_debug_callback.py`, `debug_zeros_extremos.py`
- **Função**: Monitora gradientes durante treinamento
- **Frequência**: A cada 1000 steps
- **Saída**: Arquivos `debug_zeros_report_step_*.txt`
- **⚠️ IMPLEMENTADO**: Limpeza automática de debug reports antigos na inicialização

### 2. Estrutura do Treinamento
- **Main Function**: `daytrader.py:8627`
- **Inicialização**: 
  - Testes obrigatórios V7
  - Limpeza automática de debug reports antigos (NEW)
  - Setup GPU otimizado
  - Zero debugger com threshold 5%

### 3. Arquivos de Avaliação
- **Pasta**: `avaliacao/` - Scripts de análise e teste
- **Pasta**: `avaliacoes/` - Resultados de avaliações
- **Sistema**: Avaliação on-demand integrada

## Problemas Conhecidos
1. **Memory/Header Loss**: Sessões longas podem perder contexto (limitação Claude Code)
2. **Debug Files**: Muitos arquivos debug acumulavam (RESOLVIDO com auto-cleanup)

## Comandos Úteis
- **Executar**: `python daytrader.py`
- **Análise**: Scripts em `avaliacao/`
- **Monitoramento**: Convergence monitors disponíveis

## Debug Zero System
- **Padrão de Arquivos**: `debug_zeros_report_step_*.txt`
- **Arquivos Finais**: `debug_zeros_FINAL_report_*_steps.txt`
- **Auto-cleanup**: ✅ Implementado na inicialização do main()

## Últimas Modificações
- **2025-08-01**: Implementada limpeza automática de debug reports antigos
- **Localização**: `daytrader.py:8680-8695`
- **Funcionalidade**: Remove todos os debug_zeros_report_*.txt na inicialização
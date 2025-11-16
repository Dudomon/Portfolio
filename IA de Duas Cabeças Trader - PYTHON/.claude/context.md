# Claude Context - DayTrader V7 System

## 🎯 STATUS CRÍTICO DO PROJETO

### Sistema Principal
- **Arquivo Principal**: `daytrader.py` (446KB, 8700+ linhas)
- **Versão**: V7 Intuition System
- **Checkpoint Atual**: 5.85M steps (melhor performance)
- **Framework**: Stable-Baselines3 + Custom Transformer Policy

### 🔧 MODIFICAÇÃO RECENTE IMPORTANTE
**Data**: 2025-08-01
**Implementado**: Sistema de limpeza automática de debug reports
**Localização**: `daytrader.py:8680-8695`
**Função**: Remove automaticamente arquivos `debug_zeros_report_step_*.txt` e `debug_zeros_FINAL_report_*.txt` na inicialização

```python
# 🧹 LIMPEZA AUTOMÁTICA DE DEBUG REPORTS ANTIGOS
debug_files = glob.glob("debug_zeros_report_step_*.txt")
final_reports = glob.glob("debug_zeros_FINAL_report_*_steps.txt")
all_debug_files = debug_files + final_reports
for file in all_debug_files:
    try:
        os.remove(file)
    except OSError:
        pass
```

## 🏗️ ARQUITETURA DO SISTEMA

### Componentes Críticos
1. **Zero Debug System**: `zero_debug_callback.py`, `debug_zeros_extremos.py`
2. **Main Training**: `daytrader.py:main()` função em linha 8627
3. **Advanced Training System**: Classe integrada no daytrader.py
4. **Convergence Optimization**: Pasta `convergence_optimization/`

### Sistema de Debug de Zeros
- **Callback**: `ZeroExtremeDebugCallback` em `zero_debug_callback.py`
- **Frequência**: A cada 1000 steps
- **Threshold**: 5% (mais sensível)
- **Outputs**: Relatórios em txt removidos automaticamente
- **Foco**: Gradientes críticos (MLP, Attention, Bias)

## 🎮 ESTRUTURA DE TREINAMENTO

### Inicialização (daytrader.py:main)
1. Testes obrigatórios V7
2. **[NOVO]** Limpeza automática debug reports
3. Setup GPU otimizado
4. Inicialização zero debugger
5. Sistema avançado de treinamento

### Avaliação
- **Pasta Principal**: `avaliacao/` - Scripts de análise
- **Resultados**: `avaliacoes/` - Relatórios salvos
- **Sistema**: On-demand evaluation integrado

## 📁 ESTRUTURA DE ARQUIVOS IMPORTANTES

### Scripts de Execução
- `daytrader.py` - Sistema principal
- `avaliar_automatico.bat` - Avaliação rápida
- `start_convergence_monitor.bat` - Monitor de convergência

### Dados
- `data/` - Datasets (1M, 2M sintéticos disponíveis)
- `logs/` - Histórico de treinamentos
- `avaliacoes/` - Resultados de performance

### Documentação
- `documentacao/` - Guias técnicos
- `CLAUDE.md` - Contexto básico do projeto
- `.claude/context.md` - Este arquivo (contexto completo)

## 🐛 PROBLEMAS CONHECIDOS E SOLUÇÕES

### 1. Debug Reports Acumulando
- **PROBLEMA**: Milhares de arquivos debug_zeros_report_step_*.txt
- **SOLUÇÃO**: ✅ Implementada limpeza automática na inicialização
- **STATUS**: RESOLVIDO (2025-08-01)

### 2. Perda de Contexto Claude
- **PROBLEMA**: Sessões longas perdem contexto/memória
- **CAUSA**: Limitação Claude Code com sessões extensas
- **SOLUÇÃO**: Este arquivo `.claude/context.md` mantém informações críticas

### 3. Memory/Header Issues
- **PROBLEMA**: "Perda de header" em sessões longas
- **CAUSA**: Acúmulo de contexto no Claude Code
- **SOLUÇÃO**: Usar informações em `.claude/` para recuperar contexto

## 🚀 COMANDOS ESSENCIAIS

### Executar Treinamento
```bash
python daytrader.py
```

### Avaliação Rápida
```bash
python avaliacao/avaliar_checkpoint_recente.py
```

### Monitor de Convergência
```bash
python convergence_monitor.py
```

## 📊 MÉTRICAS DE PERFORMANCE

### Checkpoint 5.85M Steps
- **Performance**: Melhor resultado até agora
- **Localização**: Pasta raiz do projeto
- **Avaliação**: Disponível em `avaliacoes/`

### Sistema V7 Features
- Gates V7 (filtros relaxados)
- Convergence Optimization ativo
- Gradient Accumulation
- Advanced LR Scheduler
- Data Augmentation com volatility enhancement

## 🔍 DEBUG E MONITORAMENTO

### Zero Extreme Debugger
- **Threshold**: 5% (DETALHADO)
- **Foco**: Gradientes críticos (MLP, Transformer, Attention, Bias)
- **Alertas**: >70% zeros = GRADIENT VANISHING
- **Cleanup**: ✅ Automático na inicialização

### Convergence Monitoring
- Scripts em `convergence_monitor*.py`
- Dashboard HTML gerado automaticamente
- Dados em formato JSONL

## 💡 ÚLTIMAS MODIFICAÇÕES

### 2025-08-01
1. **Implementada limpeza automática** de debug reports antigos
2. **Criado sistema de contexto** em `.claude/`
3. **Documentação atualizada** para recuperação de sessões

### Próximos Passos Sugeridos
- Monitorar performance pós-limpeza
- Continuar treinamento a partir de 5.85M steps
- Avaliar necessidade de retreinamento com novos dados

---
**IMPORTANTE**: Este arquivo mantém o contexto crítico para recuperação de sessões perdidas no Claude Code. Sempre consulte quando retomar trabalho no projeto.

## 🔄 MANUTENÇÃO DESTE ARQUIVO
**CRÍTICO**: Sempre atualizar este arquivo quando:
- Modificar código principal (daytrader.py)
- Implementar novas funcionalidades
- Resolver problemas importantes
- Alterar configurações de treinamento
- Criar/modificar scripts de avaliação
- Descobrir bugs ou soluções

**Comando para atualizar**: Sempre editar `.claude/context.md`, `.claude/recent_changes.md` e `.claude/commands.md` após mudanças significativas.

**SEM ATUALIZAÇÕES = CONTEXTO PERDIDO = PROBLEMA REPETIDO**
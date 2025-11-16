# Comandos Essenciais - DayTrader V7

## 🚀 EXECUÇÃO PRINCIPAL

### Treinar Sistema
```bash
python daytrader.py
```
- Executa treinamento completo V7
- Limpa debug reports automaticamente
- Sistema otimizado para GPU

### Treinar com Instance ID
```bash
python daytrader.py 0
```
- Instance ID = 0 (padrão)
- Útil para múltiplas instâncias

## 📊 AVALIAÇÃO E ANÁLISE

### Avaliação Rápida
```bash
python avaliacao/avaliar_checkpoint_recente.py
```
- Avalia último checkpoint salvo
- Gera relatório de performance

### Avaliação Específica
```bash
python avaliacao/avaliar_checkpoint_especifico.py
```
- Permite escolher checkpoint específico
- Análise detalhada de performance

### Dashboard Completo
```bash
python avaliacao/dashboard_completo.py
```
- Gera dashboard HTML interativo
- Visualização completa de métricas

## 🔍 MONITORAMENTO

### Monitor de Convergência
```bash
python convergence_monitor.py
```
- Monitora progresso em tempo real
- Detecta problemas de convergência

### Monitor de Gradientes
```bash
python avaliacao/analisador_gradientes_atual.py
```
- Analisa health dos gradientes
- Detecta gradient vanishing/explosion

### Verificação Rápida
```bash
python quick_convergence_check.py
```
- Check rápido de status
- Últimas métricas disponíveis

## 🧹 LIMPEZA E MANUTENÇÃO

### Limpar Cache
```bash
python clear_all_cache.py
```
- Remove cache de data
- Limpa arquivos temporários

### Limpar Logs Antigos
```bash
python test_clean_logs_simple.py
```
- Remove logs antigos
- Mantém apenas recentes

## 🔧 DESENVOLVIMENTO E DEBUG

### Teste de Checkpoint
```bash
python test_current_checkpoint.py
```
- Verifica integridade do checkpoint
- Testa carregamento/salvamento

### Debug de Zeros
```bash
python debug_zeros_extremos.py
```
- Debug manual do sistema de zeros
- Análise detalhada de gradientes

### Verificação de Ambiente
```bash
python avaliacao/test_environment_daytrader.py
```
- Testa ambiente de trading
- Verifica compatibilidade

## 📁 ESTRUTURA DE ARQUIVOS

### Localizar Checkpoints
```bash
# Checkpoints ficam na pasta raiz
ls -la *.zip *.pkl
```

### Ver Debug Reports (se houver)
```bash
# Normalmente limpos automaticamente
ls debug_zeros_report_step_*.txt
```

### Logs de Treinamento
```bash
# Pasta logs/
ls logs/ppo_optimization_*.log
```

## 🎯 WORKFLOWS COMUNS

### 1. Iniciar Novo Treinamento
```bash
# 1. Limpar ambiente (opcional)
python clear_all_cache.py

# 2. Executar treinamento
python daytrader.py

# 3. Monitorar (em terminal separado)
python convergence_monitor.py
```

### 2. Avaliar Resultado
```bash
# 1. Avaliação rápida
python avaliacao/avaliar_checkpoint_recente.py

# 2. Dashboard completo
python avaliacao/dashboard_completo.py

# 3. Análise de gradientes
python avaliacao/analisador_gradientes_atual.py
```

### 3. Debug de Problemas
```bash
# 1. Verificar checkpoint
python test_current_checkpoint.py

# 2. Análise de zeros
python debug_zeros_extremos.py

# 3. Monitor detalhado
python convergence_monitor.py
```

## 🚨 COMANDOS DE EMERGÊNCIA

### Parar Treinamento Seguro
- `Ctrl+C` - Interrompe e salva checkpoint final
- Sistema salva automaticamente a cada intervalo

### Recuperar de Crash
```bash
# 1. Verificar último checkpoint
python test_current_checkpoint.py

# 2. Continuar treinamento
python daytrader.py
# Sistema detecta checkpoint e continua automaticamente
```

### Limpeza Completa (Reset)
```bash
# CUIDADO: Remove todos os checkpoints!
rm *.zip *.pkl debug_zeros_*.txt
python clear_all_cache.py
```

## 📋 BATCHES ÚTEIS (Windows)

### Avaliar Automaticamente
```batch
avaliar_automatico.bat
```

### Iniciar Monitor
```batch
start_convergence_monitor.bat
```

### Verificar Dependências
```batch
VerificarDependencias.bat
```
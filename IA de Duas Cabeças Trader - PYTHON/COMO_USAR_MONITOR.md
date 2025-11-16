# 📊 MONITOR EM TEMPO REAL: COM FILTRO vs SEM FILTRO

## 🎯 O que faz?

Monitora continuamente 2 instâncias do Robot_cherry rodando em paralelo:
- **Instância 1:** COM filtro de atividade ativado
- **Instância 2:** SEM filtro de atividade

Atualiza automaticamente as estatísticas conforme novos trades acontecem.

---

## 🚀 Como usar:

### **1. Executar o monitor:**

```bash
cd D:\Projeto
python monitor_filtro_vs_sem_filtro.py
```

Por padrão, atualiza a cada **10 segundos**.

### **2. Ajustar intervalo de atualização:**

```bash
# Atualizar a cada 5 segundos (mais frequente)
python monitor_filtro_vs_sem_filtro.py 5

# Atualizar a cada 30 segundos (menos frequente)
python monitor_filtro_vs_sem_filtro.py 30
```

### **3. Parar o monitor:**

Pressione **Ctrl+C** para interromper.

---

## 📈 O que o monitor mostra:

### **Tabela Principal:**
- Total de trades (cada instância)
- Win Rate (%)
- Net PnL ($)
- PnL por trade ($)
- **Diferença entre COM e SEM filtro**

### **Análise dos Horários Bloqueados:**
- Performance nos horários bloqueados [8, 9, 10, 11, 17, 21]
- Performance nos horários permitidos
- Comparação de Win Rate

### **Validação do Filtro:**
- Ganho/perda de WR ao usar filtro
- PnL evitado (horários bloqueados)
- Diferença percentual de performance
- **Veredicto: filtro é benéfico ou não?**

---

## 🎯 Exemplo de Output:

```
====================================================================================================
📊 COMPARAÇÃO EM TEMPO REAL: COM FILTRO vs SEM FILTRO
====================================================================================================
Atualizado em: 2025-10-31 16:30:00

MÉTRICA                        COM FILTRO           SEM FILTRO           DIFERENÇA
----------------------------------------------------------------------------------------------------
Total de Trades                47                   50                   -3
Win Rate                       45.8%                34.0%                +11.8%
Net PnL                        💰 $250.74           💰 $52.85            🟢 $+197.89
PnL por Trade                  $5.33                $1.06                $+4.27

====================================================================================================
🎯 ANÁLISE DOS HORÁRIOS BLOQUEADOS [8, 9, 10, 11, 17, 21]
====================================================================================================

📊 SEM FILTRO - Performance nos horários bloqueados:
   • Trades: 11
   • Win Rate: 28.3%
   • Net PnL: $-52.14

📊 SEM FILTRO - Performance nos horários permitidos:
   • Trades: 39
   • Win Rate: 41.6%
   • Net PnL: $104.99

====================================================================================================
✅ VALIDAÇÃO DO FILTRO
====================================================================================================

🎯 Ganho de Win Rate ao evitar bloqueados: +13.2%
💰 PnL evitado (horários bloqueados): $-52.14

📈 Resultado final COM FILTRO vs SEM FILTRO:
   • Diferença de WR: +11.8%
   • Diferença de PnL: $+197.89
   • Diferença %: +374.5%

✅ FILTRO ESTÁ SENDO BENÉFICO! (+11.8% WR, +$197.89 PnL)
```

---

## ⚠️ IMPORTANTE:

1. **Certifique-se de ter 2 instâncias rodando:**
   - Uma COM filtro (ativado via GUI)
   - Uma SEM filtro

2. **Logs corretos:**
   - Edite o script se os nomes dos logs mudarem
   - Linhas 9-10 em `monitor_filtro_vs_sem_filtro.py`

3. **Deixe rodar por tempo suficiente:**
   - Mínimo: 100-200 trades em cada
   - Ideal: 500+ trades para conclusões definitivas

---

## 🔧 Troubleshooting:

### Problema: "Aguardando dados dos logs..."
**Solução:** Verifique se os caminhos dos logs estão corretos no script.

### Problema: Atualização muito lenta
**Solução:** Reduza o intervalo: `python monitor_filtro_vs_sem_filtro.py 5`

### Problema: Tela piscando muito
**Solução:** Aumente o intervalo: `python monitor_filtro_vs_sem_filtro.py 30`

---

## 📝 Logs monitorados:

- **COM FILTRO:** `trading_session_20251031_160231_42780_590145c4.txt`
- **SEM FILTRO:** `trading_session_20251031_160208_43368_8fcc7702.txt`

---

**Criado por Claude Code - 31/10/2025**

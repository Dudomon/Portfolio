# Requirements Document

## Introduction

O sistema de treinamento PPO apresenta problemas críticos de convergência que persistem mesmo após múltiplas correções agressivas: KL divergence extremamente baixo (2.4e-05, quando deveria estar entre 1e-03 e 1e-02), clip fraction permanece zero, e detecção persistente de pesos congelados. Há conflito sistêmico entre learning rates (current_lr: 4.98e-05 vs configurado: 2.0e-04), indicando que existe um sistema oculto ou hardcoded que continua reduzindo o LR, impedindo que a policy faça mudanças significativas e resultando em convergência inadequada. É necessário implementar correções para restaurar a capacidade de aprendizado do modelo.

## Requirements

### Requirement 1

**User Story:** Como um desenvolvedor de RL, eu quero que o PPO tenha KL divergence adequado, para que a policy faça mudanças significativas durante o treinamento.

#### Acceptance Criteria

1. WHEN o treinamento PPO executa THEN o approx_kl SHALL estar entre 1e-03 e 1e-02 (atualmente 7e-05 é muito baixo)
2. WHEN o KL divergence está abaixo de 1e-04 THEN o sistema SHALL aumentar learning rate e clip range
3. WHEN o KL divergence está acima de 5e-02 THEN o sistema SHALL diminuir learning rate
4. IF o KL divergence permanece abaixo de 1e-04 por 5 iterações THEN o sistema SHALL aplicar perturbação nos pesos

### Requirement 2

**User Story:** Como um desenvolvedor de RL, eu quero que o clip fraction seja maior que zero, para que o PPO esteja efetivamente limitando mudanças excessivas na policy.

#### Acceptance Criteria

1. WHEN o treinamento PPO executa THEN o clip_fraction SHALL estar entre 0.05 e 0.3
2. WHEN clip_fraction é zero THEN o sistema SHALL aumentar o clip_range
3. WHEN clip_fraction é muito alto (>0.5) THEN o sistema SHALL diminuir o clip_range
4. IF clip_fraction permanece zero por 3 iterações consecutivas THEN o sistema SHALL reinicializar parâmetros críticos

### Requirement 3

**User Story:** Como um desenvolvedor de RL, eu quero que a detecção de pesos congelados seja mais precisa, para que não haja falsos positivos durante treinamento normal.

#### Acceptance Criteria

1. WHEN os pesos estão mudando normalmente THEN o status SHALL mostrar "PESOS OK" ou "PESOS ATIVOS"
2. WHEN a mudança média dos pesos é maior que 1e-5 THEN o sistema SHALL classificar como "PESOS NORMAIS"
3. WHEN a mudança relativa dos pesos é maior que 0.1% THEN o sistema SHALL classificar como "PESOS ATIVOS"
4. IF os pesos realmente estão congelados THEN o sistema SHALL aplicar perturbação nos parâmetros

### Requirement 4

**User Story:** Como um desenvolvedor de RL, eu quero que o learning rate seja adaptativo, para que o modelo possa escapar de mínimos locais e convergir adequadamente.

#### Acceptance Criteria

1. WHEN o KL divergence está muito baixo THEN o learning rate SHALL ser aumentado em 50%
2. WHEN o clip fraction é zero THEN o learning rate SHALL ser aumentado em 25%
3. WHEN os gradientes são muito pequenos THEN o learning rate SHALL ser aumentado progressivamente
4. IF múltiplos indicadores sugerem convergência prematura THEN o sistema SHALL aplicar "learning rate boost"

### Requirement 5

**User Story:** Como um desenvolvedor de RL, eu quero que haja mecanismos de recuperação automática, para que o treinamento continue mesmo quando detectados problemas de convergência.

#### Acceptance Criteria

1. WHEN problemas de convergência são detectados THEN o sistema SHALL aplicar correções automáticas
2. WHEN pesos estão realmente congelados THEN o sistema SHALL adicionar ruído aos parâmetros
3. WHEN a entropy loss está muito baixa THEN o sistema SHALL aumentar o coeficiente de entropia
4. IF múltiplas correções falharam THEN o sistema SHALL sugerir reinicialização parcial do modelo
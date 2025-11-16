#!/usr/bin/env python3
"""
📊 ACTION DISTRIBUTION CALLBACK - 1 linha de métricas

Captura distribuição HOLD/LONG/SHORT durante treino
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict

class ActionDistributionCallback(BaseCallback):
    """📊 Callback simples para capturar distribuição de ações"""
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.action_counts = defaultdict(int)
        self.step_count = 0
    
    def _on_step(self) -> bool:
        """📊 Captura ação a cada step"""
        
        # Capturar última ação executada - MÚLTIPLAS FONTES
        actions_captured = False
        
        # Método 1: self.locals
        if hasattr(self, 'locals') and self.locals and 'actions' in self.locals:
            actions = self.locals['actions']
            if actions is not None and hasattr(actions, 'shape') and len(actions.shape) > 0:
                entry_decisions = actions[:, 0] if len(actions.shape) > 1 else [actions[0]]
                for decision in entry_decisions:
                    self.action_counts[int(decision)] += 1
                actions_captured = True
        
        # Método 2: training_env
        if not actions_captured and hasattr(self, 'training_env') and self.training_env:
            try:
                last_actions = self.training_env.get_attr('last_action')
                if last_actions and len(last_actions) > 0 and last_actions[0] is not None:
                    action = last_actions[0]
                    if hasattr(action, '__len__') and len(action) > 0:
                        entry_decision = int(action[0])
                        self.action_counts[entry_decision] += 1
                        actions_captured = True
            except:
                pass
        
        # Método 3: FORÇAR PRINT A CADA LOG_FREQ mesmo sem dados
        if not actions_captured and self.step_count % self.log_freq == 0:
            print("⚠️ Action Distribution: SEM DADOS - callback não está capturando ações")
        
        self.step_count += 1
        
        # Log distribuição periodicamente
        if self.step_count % self.log_freq == 0:
            total = sum(self.action_counts.values())
            if total > 0:
                hold_pct = (self.action_counts[0] / total) * 100
                long_pct = (self.action_counts[1] / total) * 100  
                short_pct = (self.action_counts[2] / total) * 100
                
                # 📊 PRINT OBRIGATÓRIO DA DISTRIBUIÇÃO
                print(f"📊 Action Distribution: HOLD={hold_pct:.1f}% | LONG={long_pct:.1f}% | SHORT={short_pct:.1f}%")
                
                # Reset para próximo período
                self.action_counts.clear()
        
        return True

# Uso: model.learn(callback=ActionDistributionCallback())
#!/usr/bin/env python3
"""
📊 SATURATION MONITOR CALLBACK - Monitor para V7 sem sigmoids

Monitora se os valores estão saturando nos extremos (0.0 ou 1.0) 
mesmo usando torch.clamp ao invés de sigmoid
"""

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict, deque

class SaturationMonitorCallback(BaseCallback):
    """📊 Monitor de saturação para V7 sem sigmoids"""
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.step_count = 0
        
        # Histórico de saturação
        self.saturation_history = defaultdict(lambda: deque(maxlen=100))
        self.extreme_counts = defaultdict(int)
        
        # Thresholds para detectar saturação
        self.saturation_threshold = 0.95  # >95% = quase saturado
        self.extreme_threshold = 0.99     # >99% = extremamente saturado
        
    def _on_step(self) -> bool:
        """📊 Monitor saturação a cada step"""
        self.step_count += 1
        
        if self.step_count % self.log_freq != 0:
            return True
            
        try:
            # Verificar se é V7 Intuition
            if hasattr(self.model.policy, 'entry_head'):
                self._monitor_v7_saturation()
            
        except RuntimeError as cuda_error:
            if "CUDA" in str(cuda_error):
                print(f"⚠️ [SATURAÇÃO] CUDA error detectado, limpando cache: {cuda_error}")
                torch.cuda.empty_cache()
            else:
                print(f"❌ ERRO SaturationMonitor: {cuda_error}")
        except Exception as e:
            print(f"❌ ERRO SaturationMonitor: {e}")
        
        return True
    
    def _monitor_v7_saturation(self):
        """🔍 Monitor DESATIVADO - estava reportando falsos positivos"""
        # 🚨 MONITOR DESATIVADO!
        # Este monitor estava reportando FALSOS POSITIVOS porque:
        # 1. Checava se PESOS estavam próximos de 1.0 (não faz sentido!)
        # 2. Contava bias=0 como problema (é normal!)
        # 3. Contava LayerNorm.weight=1 como problema (é normal!)
        # 4. Contava log_std=-0.5 como problema (é normal!)
        # 
        # TODOS os "problemas" reportados eram inicializações NORMAIS!
        return
    
    def _monitor_actions_if_available(self):
        """Monitor de ações - separado e opcional"""
        try:
            # MÉTODO ALTERNATIVO: Monitorar ações do último rollout se disponível
            if hasattr(self.training_env, 'get_attr'):
                try:
                    last_actions = self.training_env.get_attr('last_action')
                    if last_actions and len(last_actions) > 0 and last_actions[0] is not None:
                        action = last_actions[0]
                        if hasattr(action, '__len__') and len(action) > 0:
                            # Monitor action[0] (entry decision)
                            entry_val = float(action[0]) if hasattr(action[0], '__float__') else action[0]
                            
                            if entry_val <= 0.01:
                                print(f"⚠️ [SATURAÇÃO BAIXA] Entry Decision: {entry_val:.4f} (próximo de 0)")
                            elif entry_val >= 0.99:
                                print(f"⚠️ [SATURAÇÃO ALTA] Entry Decision: {entry_val:.4f} (próximo de 1)")
                            
                            # Adicionar ao histórico
                            self.saturation_history['entry_decision'].append(entry_val)
                            
                except Exception:
                    pass  # Ignorar se não conseguir acessar
            
            # GERAÇÃO DE SUMMARY RÁPIDA a cada 10 chamadas
            if self.step_count % (self.log_freq * 10) == 0:
                total_extremes = sum(self.extreme_counts.values())
                if total_extremes > 0:
                    print(f"📊 [SATURAÇÃO SUMMARY] {total_extremes} saturações extremas detectadas nos últimos steps")
                else:
                    print("✅ [SATURAÇÃO OK] Nenhuma saturação extrema detectada recentemente")
            
        except Exception as e:
            print(f"❌ ERRO monitor V7: {e}")
    
    def _analyze_gate_saturation(self, gate_info):
        """🔍 Analisar saturação das gates V7"""
        gate_names = [
            'temporal_gate', 'validation_gate', 'confidence_gate', 
            'risk_gate', 'composite_score'
        ]
        
        saturated_gates = []
        extreme_gates = []
        
        for gate_name in gate_names:
            if gate_name in gate_info:
                gate_value = gate_info[gate_name]
                
                if torch.is_tensor(gate_value):
                    gate_value = gate_value.item()
                
                # Verificar saturação
                if gate_value >= self.saturation_threshold:
                    saturated_gates.append(f"{gate_name}={gate_value:.3f}")
                    
                if gate_value >= self.extreme_threshold:
                    extreme_gates.append(f"{gate_name}={gate_value:.3f}")
                    self.extreme_counts[gate_name] += 1
                
                # Adicionar ao histórico
                self.saturation_history[gate_name].append(gate_value)
        
        # Report se houver saturação
        if saturated_gates:
            print(f"⚠️ [SATURAÇÃO V7] Gates > {self.saturation_threshold}: {', '.join(saturated_gates)}")
            
        if extreme_gates:
            print(f"🚨 [EXTREMA SATURAÇÃO] Gates > {self.extreme_threshold}: {', '.join(extreme_gates)}")
    
    def _analyze_action_saturation(self, action_tensor, action_name):
        """🔍 Analisar saturação das ações"""
        if action_tensor is None:
            return
            
        if torch.is_tensor(action_tensor):
            action_value = action_tensor.item() if action_tensor.numel() == 1 else action_tensor.mean().item()
        else:
            action_value = float(action_tensor)
        
        # Verificar saturação nos extremos (clamp vai para 0.0 ou 1.0)
        if action_value <= 0.01:  # Quase 0
            print(f"⚠️ [SATURAÇÃO BAIXA] {action_name}: {action_value:.4f} (próximo de 0.0)")
            self.extreme_counts[f"{action_name}_low"] += 1
            
        elif action_value >= 0.99:  # Quase 1
            print(f"⚠️ [SATURAÇÃO ALTA] {action_name}: {action_value:.4f} (próximo de 1.0)")
            self.extreme_counts[f"{action_name}_high"] += 1
        
        # Adicionar ao histórico
        self.saturation_history[action_name].append(action_value)
    
    def _generate_saturation_report(self):
        """📊 Gerar relatório de saturação"""
        print(f"\n📊 RELATÓRIO SATURAÇÃO - Step {self.step_count}")
        print("="*50)
        
        # Contar extremos por componente
        total_extremes = sum(self.extreme_counts.values())
        
        if total_extremes > 0:
            print(f"🚨 TOTAL DE SATURAÇÕES EXTREMAS: {total_extremes}")
            
            for component, count in self.extreme_counts.items():
                if count > 0:
                    print(f"   {component}: {count} vezes")
        else:
            print("✅ NENHUMA SATURAÇÃO EXTREMA DETECTADA")
        
        # Análise do histórico recente
        print(f"\n📈 ANÁLISE HISTÓRICO RECENTE:")
        for component, history in self.saturation_history.items():
            if len(history) > 0:
                recent_values = list(history)[-10:]  # Últimos 10 valores
                
                avg_value = np.mean(recent_values)
                max_value = np.max(recent_values) 
                min_value = np.min(recent_values)
                
                # Status baseado na média
                if avg_value >= 0.95 or avg_value <= 0.05:
                    status = "🚨 SATURADO"
                elif avg_value >= 0.85 or avg_value <= 0.15:
                    status = "⚠️ RISCO"
                else:
                    status = "✅ OK"
                
                print(f"   {component}: avg={avg_value:.3f} (min={min_value:.3f}, max={max_value:.3f}) {status}")
        
        print("="*50)
        
        return total_extremes
    
    def get_saturation_summary(self):
        """📋 Retornar resumo de saturação"""
        total_extremes = sum(self.extreme_counts.values())
        
        summary = {
            'total_extreme_saturations': total_extremes,
            'components_affected': len([k for k, v in self.extreme_counts.items() if v > 0]),
            'extreme_counts': dict(self.extreme_counts),
            'current_step': self.step_count
        }
        
        # Adicionar médias recentes
        recent_averages = {}
        for component, history in self.saturation_history.items():
            if len(history) > 0:
                recent_averages[component] = np.mean(list(history)[-10:])
        
        summary['recent_averages'] = recent_averages
        
        return summary

# Uso: model.learn(callback=SaturationMonitorCallback())
import numpy as np
import torch
import time
from collections import defaultdict, deque
import logging

class ZeroExtremeDebugger:
    """
    🔍 DEBUGGER DE ZEROS EXTREMOS
    
    Sistema para detectar e debugar zeros extremos em tensores durante treinamento
    """
    
    def __init__(self, threshold=1e-8):
        self.threshold = float(threshold)
        self.detection_history = deque(maxlen=1000)
        self.zero_counts = defaultdict(int)
        self.total_checks = 0
        self.alert_threshold = 0.05  # 5% de zeros = alerta
        self.step = 0  # Add step counter
    
    def increment_step(self):
        """Incrementa contador de steps para tracking"""
        self.step += 1
        
    def debug_tensor(self, tensor, name="tensor", step=None):
        """Debug um tensor específico"""
        try:
            if tensor is None:
                return
                
            if isinstance(tensor, torch.Tensor):
                tensor_np = tensor.detach().cpu().numpy()
            else:
                tensor_np = np.array(tensor)
            
            # Detectar zeros extremos
            zero_mask = np.abs(tensor_np) < float(self.threshold)
            zero_count = np.sum(zero_mask)
            total_elements = tensor_np.size
            zero_percentage = (zero_count / total_elements) * 100 if total_elements > 0 else 0
            
            # Registrar detecção
            self.detection_history.append({
                'name': name,
                'step': step,
                'zero_count': zero_count,
                'total_elements': total_elements,
                'zero_percentage': zero_percentage,
                'timestamp': time.time()
            })
            
            self.zero_counts[name] += zero_count
            self.total_checks += 1
            
            # Alerta APENAS se MUITOS zeros (>20% para reduzir spam)
            if zero_percentage > 20.0:  # Apenas alertas críticos
                print(f"🚨 [CRÍTICO] {name}: {zero_percentage:.1f}% zeros")
                
            return {
                'zero_count': zero_count,
                'total_elements': total_elements,
                'zero_percentage': zero_percentage,
                'has_extreme_zeros': zero_percentage > self.alert_threshold * 100
            }
            
        except Exception as e:
            print(f"⚠️ [ZERO DEBUG] Erro ao debugar {name}: {e}")
            return None
    
    def debug_model_params(self, model):
        """Debug parâmetros do modelo"""
        try:
            total_zeros = 0
            total_params = 0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    result = self.debug_tensor(param.grad, f"grad_{name}")
                    if result:
                        total_zeros += result['zero_count']
                        total_params += result['total_elements']
                
                result = self.debug_tensor(param, f"param_{name}")
                if result:
                    total_zeros += result['zero_count']
                    total_params += result['total_elements']
            
            if total_params > 0:
                overall_zero_percentage = (total_zeros / total_params) * 100
                if overall_zero_percentage > 15.0:  # Apenas se >15% zeros críticos
                    print(f"🚨 [MODEL CRÍTICO] {overall_zero_percentage:.1f}% zeros totais")
                    
        except Exception as e:
            print(f"⚠️ [MODEL DEBUG] Erro: {e}")
    
    def debug_observation_features(self, observations, step=None):
        """
        🔍 NOVO: Debug features de observação (input do modelo)
        Detecta features mortas e preprocessing issues
        """
        try:
            if observations is None:
                return None
                
            # Convert to numpy if needed
            if isinstance(observations, torch.Tensor):
                obs_np = observations.detach().cpu().numpy()
            else:
                obs_np = np.array(observations)
            
            # Análise detalhada das features
            batch_size = obs_np.shape[0] if len(obs_np.shape) > 1 else 1
            if len(obs_np.shape) == 1:
                obs_np = obs_np.reshape(1, -1)
            
            feature_count = obs_np.shape[1]
            
            # Debug geral
            result = self.debug_tensor(obs_np, f"observations_input", step)
            
            # Análise por feature individual (detectar features mortas)
            dead_features = []
            constant_features = []
            
            for i in range(feature_count):
                feature_column = obs_np[:, i]
                
                # Features completamente zero
                if np.all(np.abs(feature_column) < self.threshold):
                    dead_features.append(i)
                
                # Features com valores constantes
                elif batch_size > 1 and np.std(feature_column) < 1e-6:
                    constant_features.append(i)
            
            # Alertas críticos apenas
            dead_pct = (len(dead_features) / feature_count) * 100
            constant_pct = (len(constant_features) / feature_count) * 100
            
            if dead_pct > 10.0:  # >10% features mortas
                print(f"🚨 [FEATURES CRÍTICO] {dead_pct:.1f}% features completamente mortas ({len(dead_features)}/{feature_count})")
                
            if constant_pct > 20.0:  # >20% features constantes
                print(f"⚠️ [FEATURES AVISO] {constant_pct:.1f}% features constantes")
            
            # Preparar return dict
            return_dict = {
                'dead_features_count': len(dead_features),
                'dead_features_indices': dead_features[:10],  # Primeiros 10 apenas
                'constant_features_count': len(constant_features),
                'dead_features_percentage': dead_pct,
                'constant_features_percentage': constant_pct,
                'total_features': feature_count,
                'batch_size': batch_size
            }
            
            # Adicionar result se disponível
            if result:
                return_dict.update(result)
                
            return return_dict
            
        except Exception as e:
            print(f"⚠️ [OBSERVATION DEBUG] Erro: {e}")
            return None
    
    def debug_intermediate_outputs(self, model, observations, step=None):
        """
        🔍 NOVO: Debug outputs de layers intermediárias
        Detecta dead neurons e vanishing activations
        """
        try:
            if not hasattr(model, 'policy') or observations is None:
                return None
            
            intermediate_outputs = {}
            
            # Hook para capturar outputs intermediários
            def create_hook(name):
                def hook(module, input, output):
                    if output is not None:
                        intermediate_outputs[name] = output
                return hook
            
            # Registrar hooks em layers críticas
            hooks = []
            policy = model.policy
            
            # Backbone layers
            if hasattr(policy, 'unified_backbone'):
                if hasattr(policy.unified_backbone, 'input_projection'):
                    hooks.append(policy.unified_backbone.input_projection.register_forward_hook(
                        create_hook('backbone_input_projection')))
                if hasattr(policy.unified_backbone, 'shared_feature_processor'):
                    hooks.append(policy.unified_backbone.shared_feature_processor.register_forward_hook(
                        create_hook('backbone_shared_features')))
            
            # Actor/Critic LSTM outputs
            if hasattr(policy, 'actor_lstm'):
                hooks.append(policy.actor_lstm.register_forward_hook(
                    create_hook('actor_lstm_output')))
            if hasattr(policy, 'critic_lstm'):
                hooks.append(policy.critic_lstm.register_forward_hook(
                    create_hook('critic_lstm_output')))
            
            try:
                # Forward pass para capturar outputs
                with torch.no_grad():
                    if hasattr(policy, 'predict_values'):
                        # Minimal forward pass
                        obs_tensor = torch.FloatTensor(observations).to(next(policy.parameters()).device)
                        _ = policy.forward_actor(obs_tensor, None, torch.ones(obs_tensor.shape[0], dtype=torch.bool))
                
                # Debug cada output capturado
                results = {}
                critical_issues = []
                
                for layer_name, output in intermediate_outputs.items():
                    if output is not None:
                        result = self.debug_tensor(output, f"layer_{layer_name}", step)
                        if result:
                            results[layer_name] = result
                            
                            # Detectar problemas críticos
                            if result['zero_percentage'] > 50.0:
                                critical_issues.append(f"{layer_name}: {result['zero_percentage']:.1f}% zeros")
                
                # Alertar apenas problemas críticos
                if critical_issues:
                    print(f"🚨 [LAYERS CRÍTICO] Dead neurons detectados:")
                    for issue in critical_issues[:3]:  # Max 3 alertas
                        print(f"     {issue}")
                
                return results
                
            finally:
                # Limpar hooks
                for hook in hooks:
                    hook.remove()
                    
        except Exception as e:
            print(f"⚠️ [INTERMEDIATE DEBUG] Erro: {e}")
            return None
    
    def get_stats(self):
        """Obter estatísticas de debug"""
        if not self.detection_history:
            return {}
            
        recent_detections = list(self.detection_history)[-10:]  # Últimas 10
        avg_zero_percentage = np.mean([d['zero_percentage'] for d in recent_detections])
        
        return {
            'total_checks': self.total_checks,
            'recent_avg_zeros': avg_zero_percentage,
            'zero_counts_by_name': dict(self.zero_counts),
            'alert_count': sum(1 for d in recent_detections if d['zero_percentage'] > self.alert_threshold * 100)
        }
    
    def reset_stats(self):
        """Resetar estatísticas"""
        self.detection_history.clear()
        self.zero_counts.clear()
        self.total_checks = 0
        self.step = 0  # Reset step counter

    def increment_step(self):
        """Incrementa o contador de steps"""
        self.step += 1
    
    def analyze_numpy_zeros(self, array, name="array"):
        """Analisa zeros extremos em array numpy e retorna estatísticas"""
        try:
            if array is None:
                return {'zero_extreme_ratio': 0.0}
            
            # Converter para numpy se necessário
            if isinstance(array, torch.Tensor):
                array = array.detach().cpu().numpy()
            elif not isinstance(array, np.ndarray):
                array = np.array(array)
            
            # Detectar zeros extremos
            zero_mask = np.abs(array) < float(self.threshold)
            total_elements = array.size
            zero_count = np.sum(zero_mask)
            zero_ratio = zero_count / total_elements if total_elements > 0 else 0.0
            
            # Usar debug_tensor interno para logging
            result = self.debug_tensor(array, name)
            
            # Retornar estatísticas adicionais
            return {
                'zero_extreme_ratio': zero_ratio,
                'zero_count': zero_count,
                'total_elements': total_elements,
                'zero_percentage': zero_ratio * 100,
                'has_extreme_zeros': zero_ratio > self.alert_threshold
            }
            
        except Exception as e:
            print(f"⚠️ [ANALYZE ZEROS] Erro ao analisar {name}: {e}")
            return {'zero_extreme_ratio': 0.0}
    
    def generate_report(self):
        """Gerar relatório completo de zeros extremos"""
        stats = self.get_stats()
        
        if not stats:
            return "📊 ZERO EXTREME DEBUG REPORT: Nenhum dado coletado ainda"
        
        report = []
        report.append("📊 ZERO EXTREME DEBUG REPORT")
        report.append("-" * 40)
        report.append(f"Total checks: {stats.get('total_checks', 0)}")
        report.append(f"Recent avg zeros: {stats.get('recent_avg_zeros', 0):.2f}%")
        report.append(f"Alert count: {stats.get('alert_count', 0)}")
        
        # Top componentes com mais zeros
        zero_counts = stats.get('zero_counts_by_name', {})
        if zero_counts:
            report.append("\n🔥 TOP COMPONENTES COM ZEROS:")
            sorted_counts = sorted(zero_counts.items(), key=lambda x: x[1], reverse=True)
            for i, (name, count) in enumerate(sorted_counts[:5]):
                report.append(f"  {i+1}. {name}: {count} zeros")
        
        report.append(f"\n⚠️ Threshold: {self.threshold}")
        report.append(f"🚨 Alert threshold: {self.alert_threshold * 100}%")
        
        return "\n".join(report)
    
    def save_debug_report(self, filepath):
        """Salvar relatório de debug em arquivo"""
        try:
            report = self.generate_report()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
                f.write(f"\n\n📅 Generated: {time.time()}")
            print(f"💾 Debug report saved: {filepath}")
        except Exception as e:
            print(f"⚠️ Erro ao salvar debug report: {e}")

def create_zero_extreme_debugger(threshold=1e-8):
    """Factory para criar debugger"""
    return ZeroExtremeDebugger(threshold=threshold)

def debug_zeros_extreme(tensor, name="tensor", debugger=None, step=None):
    """Função standalone para debug rápido"""
    if debugger is None:
        # Usar debugger global temporário
        if not hasattr(debug_zeros_extreme, '_global_debugger'):
            debug_zeros_extreme._global_debugger = ZeroExtremeDebugger()
        debugger = debug_zeros_extreme._global_debugger
    
    return debugger.debug_tensor(tensor, name, step)

# Criar instância global padrão
_global_debugger = ZeroExtremeDebugger()

def get_global_debugger():
    """Obter debugger global"""
    return _global_debugger
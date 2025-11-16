
def _analyze_weight_changes_improved(self, model):
    """Versão melhorada da análise de mudanças de peso"""
    try:
        if not hasattr(model, 'policy'):
            return
        
        # Coletar mais parâmetros para análise mais robusta
        weight_norms = []
        param_count = 0
        
        for name, param in model.policy.named_parameters():
            # Incluir pesos importantes e alguns bias críticos
            if ('weight' in name or 
                'bias' in name and ('attention' in name or 'lstm' in name)):
                
                if param.data.numel() > 1:  # Evitar parâmetros escalares
                    weight_norms.append(param.data.norm().item())
                    param_count += 1
                    
                    # Analisar até 15 parâmetros (mais robusto)
                    if param_count >= 15:
                        break
        
        if len(weight_norms) > 0:
            current_weight_norm = np.mean(weight_norms)
            
            if self.last_weights is not None:
                # Usar mudança relativa para ser mais robusto
                if self.last_weights > 1e-10:  # Evitar divisão por zero
                    relative_change = abs(current_weight_norm - self.last_weights) / self.last_weights
                    self.weight_changes.append(relative_change)
                else:
                    # Fallback para mudança absoluta se peso anterior muito pequeno
                    absolute_change = abs(current_weight_norm - self.last_weights)
                    self.weight_changes.append(absolute_change)
            
            self.last_weights = current_weight_norm
            
            # Manter histórico limitado para eficiência
            if len(self.weight_changes) > 20:
                self.weight_changes = self.weight_changes[-20:]
                
    except Exception as e:
        print(f"[WEIGHT_MONITOR] Erro: {e}")

def _classify_weight_status_improved(self, avg_change):
    """Classificação melhorada do status dos pesos"""
    
    # Thresholds mais realistas
    FROZEN_THRESHOLD = 1e-6      # Aumentado de 1e-8
    UNSTABLE_THRESHOLD = 0.05    # Diminuído de 0.1
    
    if avg_change < FROZEN_THRESHOLD:
        return "❌ PESOS CONGELADOS"
    elif avg_change > UNSTABLE_THRESHOLD:
        return "⚠️ PESOS INSTÁVEIS"
    elif avg_change < 1e-5:
        return "🔶 PESOS LENTOS"
    elif avg_change < 1e-4:
        return "✅ PESOS NORMAIS"
    else:
        return "🚀 PESOS ATIVOS"

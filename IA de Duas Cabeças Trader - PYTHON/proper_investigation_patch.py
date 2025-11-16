
# 🔍 INVESTIGAÇÃO ADEQUADA - Adicionar na política V7

def debug_raw_actions(self, raw_actions):
    """Debug das ações brutas antes do processamento"""
    
    print(f"🔍 [RAW ACTIONS DEBUG]:")
    print(f"   Shape: {raw_actions.shape}")
    print(f"   Min: {raw_actions.min().item():.4f}")
    print(f"   Max: {raw_actions.max().item():.4f}")
    print(f"   Mean: {raw_actions.mean().item():.4f}")
    print(f"   Std: {raw_actions.std().item():.4f}")
    
    # Verificar distribuição
    values = raw_actions.detach().cpu().numpy().flatten()
    
    ranges = [
        ("< -2", np.sum(values < -2)),
        ("-2 a -1", np.sum((values >= -2) & (values < -1))),
        ("-1 a 0", np.sum((values >= -1) & (values < 0))),
        ("0 a 1", np.sum((values >= 0) & (values < 1))),
        ("1 a 2", np.sum((values >= 1) & (values < 2))),
        ("> 2", np.sum(values >= 2)),
    ]
    
    total = len(values)
    print(f"   Distribuição:")
    for range_name, count in ranges:
        pct = (count / total) * 100
        print(f"     {range_name}: {count} ({pct:.1f}%)")
    
    return raw_actions

# Usar no forward_actor:
# raw_actions = self.actor_head(actor_input)
# raw_actions = self.debug_raw_actions(raw_actions)  # ADICIONAR ESTA LINHA

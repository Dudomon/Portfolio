
# 🔧 ENHANCED DEBUG PATCH - Adicionar no método step() após a correção scalar

# Log mais detalhado da correção
if np.isscalar(action) or (hasattr(action, 'shape') and action.shape == ()):
    original_action = action
    print(f"🚨 [ENHANCED DEBUG] SCALAR DETECTADO!")
    print(f"   Original: {original_action} (type: {type(original_action)})")
    
    # Aplicar conversão
    entry_val = max(0, min(2, int(float(action))))
    action = np.array([entry_val, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    print(f"   Convertido: {action}")
    print(f"   Entry Decision: {entry_val} ({'HOLD' if entry_val == 0 else 'LONG' if entry_val == 1 else 'SHORT'})")
    print(f"   Shape: {action.shape}")
    
    # Estatísticas de conversão
    if not hasattr(self, '_scalar_conversion_stats'):
        self._scalar_conversion_stats = {'total': 0, 'hold': 0, 'long': 0, 'short': 0}
    
    self._scalar_conversion_stats['total'] += 1
    if entry_val == 0:
        self._scalar_conversion_stats['hold'] += 1
    elif entry_val == 1:
        self._scalar_conversion_stats['long'] += 1
    else:
        self._scalar_conversion_stats['short'] += 1
    
    # Log estatísticas a cada 100 conversões
    if self._scalar_conversion_stats['total'] % 100 == 0:
        stats = self._scalar_conversion_stats
        total = stats['total']
        print(f"📊 [CONVERSION STATS] {total} conversões:")
        print(f"   HOLD: {stats['hold']} ({stats['hold']/total*100:.1f}%)")
        print(f"   LONG: {stats['long']} ({stats['long']/total*100:.1f}%)")
        print(f"   SHORT: {stats['short']} ({stats['short']/total*100:.1f}%)")

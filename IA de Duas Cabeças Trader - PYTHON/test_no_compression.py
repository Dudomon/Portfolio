"""
🧪 Teste Transformer NO-COMPRESSION 450D → 450D
"""
import torch
import sys
sys.path.insert(0, "D:/Projeto")

from trading_framework.extractors.transformer_no_compression import TradingTransformerNoCompression
from trading_framework.policies.two_head_v11_sigmoid import get_v11_sigmoid_no_compression_kwargs
from gym import spaces
import numpy as np

print("=" * 60)
print("🧪 TESTE TRANSFORMER NO-COMPRESSION")
print("=" * 60)

# 1. Testar Transformer No-Compression
print("\n1️⃣ Testando TradingTransformerNoCompression...")
obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(450,), dtype=np.float32)
transformer = TradingTransformerNoCompression(obs_space, features_dim=450)

# Input batch
batch_size = 32
test_obs = torch.randn(batch_size, 450)

print(f"   Input shape: {test_obs.shape}")
output = transformer(test_obs)
print(f"   Output shape: {output.shape}")
print(f"   Expected: torch.Size([{batch_size}, 450])")

assert output.shape == (batch_size, 450), f"Output shape mismatch! Got {output.shape}"
print("   ✅ Shape correto: 450D → 450D (zero compression)")

# 2. Verificar dimensões internas
print(f"\n2️⃣ Verificando arquitetura interna...")
print(f"   d_model: {transformer.d_model} (esperado: 450)")
print(f"   n_heads: {transformer.n_heads} (esperado: 10)")
print(f"   features_dim: {transformer.features_dim} (esperado: 450)")

assert transformer.d_model == 450
assert transformer.features_dim == 450
print("   ✅ Dimensões internas corretas")

# 3. Testar Policy Kwargs
print(f"\n3️⃣ Testando get_v11_sigmoid_no_compression_kwargs()...")
kwargs = get_v11_sigmoid_no_compression_kwargs()

print(f"   v8_features_dim: {kwargs['v8_features_dim']} (esperado: 450)")
print(f"   v8_lstm_hidden: {kwargs['v8_lstm_hidden']} (esperado: 512)")
print(f"   v8_context_dim: {kwargs['v8_context_dim']} (esperado: 128)")
print(f"   features_extractor_kwargs: {kwargs['features_extractor_kwargs']}")

assert kwargs['v8_features_dim'] == 450
assert kwargs['v8_lstm_hidden'] == 512
assert kwargs['v8_context_dim'] == 128
assert kwargs['features_extractor_kwargs']['features_dim'] == 450
print("   ✅ Policy kwargs corretos para NO-COMPRESSION")

# 4. Testar gradient flow
print(f"\n4️⃣ Testando gradient flow...")
transformer.train()
test_obs.requires_grad = True
output = transformer(test_obs)
loss = output.mean()
loss.backward()

grad_norm = test_obs.grad.norm().item()
print(f"   Gradient norm: {grad_norm:.4f}")
assert grad_norm > 0, "Gradients estão zerados!"
print("   ✅ Gradients fluindo corretamente")

# 5. Comparação de parâmetros
print(f"\n5️⃣ Comparação com versão comprimida...")
from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor

transformer_compressed = TradingTransformerFeatureExtractor(obs_space, features_dim=64)
params_compressed = sum(p.numel() for p in transformer_compressed.parameters())
params_no_compression = sum(p.numel() for p in transformer.parameters())

print(f"   Parâmetros comprimido (64D): {params_compressed:,}")
print(f"   Parâmetros NO-COMPRESSION (450D): {params_no_compression:,}")
print(f"   Aumento: {params_no_compression/params_compressed:.2f}x")

# 6. Teste de memória (estimado)
print(f"\n6️⃣ Estimativa de uso de memória...")
memory_compressed = params_compressed * 4 / (1024**2)  # 4 bytes per float32, MB
memory_no_compression = params_no_compression * 4 / (1024**2)

print(f"   Memória comprimido: ~{memory_compressed:.1f} MB")
print(f"   Memória NO-COMPRESSION: ~{memory_no_compression:.1f} MB")
print(f"   Aumento: {memory_no_compression/memory_compressed:.2f}x")

print("\n" + "=" * 60)
print("✅ TODOS OS TESTES PASSARAM!")
print("=" * 60)
print("\n📋 RESUMO:")
print(f"   • Transformer: 450D → 450D (100% fidelidade)")
print(f"   • LSTM: 512D (processamento robusto)")
print(f"   • Context: 128D (mais capacidade)")
print(f"   • Parâmetros: {params_no_compression/params_compressed:.2f}x mais")
print(f"   • Memória: {memory_no_compression/memory_compressed:.2f}x mais")
print(f"\n🚀 Sistema pronto para treinar SEM COMPRESSÃO!")

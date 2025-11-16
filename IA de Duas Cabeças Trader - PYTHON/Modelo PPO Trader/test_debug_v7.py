#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

print("🧪 Testando Debug V7 - Composite Gate + Anomaly Detection")
print("=" * 60)

# Simular múltiplas predições para ativar os debugs
print("📊 Simulando 25 predições para testar debug a cada 20 steps...")
print("🔍 E 250 predições para testar debug de anomalias a cada 200 steps...")
print("")
print("⏳ Aguarde enquanto os debugs são executados...")
print("=" * 60)
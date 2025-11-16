#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 EXEMPLO DE USO DO DAYPROFILE
Demonstra como integrar o profiling em tempo real no treinamento
"""

import time
import numpy as np
import torch
from dayprofile import start_profiling, stop_profiling, ProfileContext, profile_training_step, get_profiling_stats

# Simular imports do sistema de trading
# from daytrader import TradingEnv
# from stable_baselines3 import PPO

def simulate_data_loading():
    """Simular carregamento de dados"""
    print("📊 Carregando dados...")
    time.sleep(0.5)  # Simular I/O
    data = np.random.random((10000, 100))
    return data

def simulate_model_forward(data):
    """Simular forward pass do modelo"""
    print("🧠 Forward pass...")
    time.sleep(0.2)  # Simular computação
    
    # Simular uso de GPU
    if torch.cuda.is_available():
        tensor = torch.tensor(data).cuda()
        result = torch.nn.functional.relu(tensor)
        return result.cpu().numpy()
    else:
        return np.maximum(data, 0)

def simulate_model_backward():
    """Simular backward pass"""
    print("⚡ Backward pass...")
    time.sleep(0.3)  # Simular computação de gradientes

@profile_training_step
def simulate_training_step(step_num):
    """Simular um step de treinamento completo"""
    print(f"🎯 Training Step {step_num}")
    
    # Carregar dados
    with ProfileContext("data_loading"):
        data = simulate_data_loading()
    
    # Forward pass
    with ProfileContext("forward_pass"):
        output = simulate_model_forward(data)
    
    # Backward pass
    with ProfileContext("backward_pass"):
        simulate_model_backward()
    
    # Simular atualização de pesos
    with ProfileContext("optimizer_step"):
        time.sleep(0.1)
    
    return {"loss": np.random.random(), "reward": np.random.random() * 100}

def main():
    """Exemplo principal de uso do profiling"""
    print("🚀 INICIANDO EXEMPLO DE PROFILING")
    print("=" * 60)
    
    # Iniciar profiling em tempo real
    start_profiling()
    
    try:
        # Simular treinamento
        for step in range(20):
            # Executar step de treinamento
            results = simulate_training_step(step)
            
            # Obter estatísticas atuais
            if step % 5 == 0:
                stats = get_profiling_stats()
                print(f"\n📊 Stats Step {step}:")
                print(f"   CPU: {stats.get('cpu_percent', 0):.1f}%")
                print(f"   Memory: {stats.get('memory_mb', 0):.1f}MB")
                print(f"   GPU: {stats.get('gpu_utilization', 0):.1f}%")
                print(f"   Elapsed: {stats.get('elapsed_time', 0):.1f}s")
            
            # Simular intervalo entre steps
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n⚠️ Treinamento interrompido pelo usuário")
    
    finally:
        # Parar profiling e gerar relatório
        stop_profiling()
    
    print("\n✅ Exemplo concluído!")
    print("📋 Verifique a pasta 'profiling_results' para os relatórios")

def example_with_real_training():
    """Exemplo de integração com treinamento real"""
    print("🔍 EXEMPLO COM TREINAMENTO REAL")
    print("=" * 60)
    
    # Iniciar profiling
    start_profiling()
    
    try:
        # Aqui você integraria com seu código real:
        
        # # Carregar ambiente
        # with ProfileContext("env_creation"):
        #     env = TradingEnv()
        
        # # Criar modelo
        # with ProfileContext("model_creation"):
        #     model = PPO("MlpPolicy", env, verbose=1)
        
        # # Treinamento
        # with ProfileContext("training"):
        #     model.learn(total_timesteps=10000)
        
        # Para este exemplo, apenas simular
        with ProfileContext("simulated_real_training"):
            for i in range(100):
                # Simular step mais pesado
                data = np.random.random((5000, 200))
                if torch.cuda.is_available():
                    tensor = torch.tensor(data).cuda()
                    result = torch.matmul(tensor, tensor.T)
                    del tensor, result
                
                time.sleep(0.05)
                
                if i % 20 == 0:
                    stats = get_profiling_stats()
                    print(f"Step {i}: CPU={stats.get('cpu_percent', 0):.1f}% "
                          f"MEM={stats.get('memory_mb', 0):.1f}MB")
    
    finally:
        stop_profiling()

if __name__ == "__main__":
    print("🔍 DAYPROFILE - EXEMPLOS DE USO")
    print("=" * 60)
    print("1. Exemplo básico")
    print("2. Exemplo com treinamento simulado")
    print("=" * 60)
    
    choice = input("Escolha (1 ou 2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        example_with_real_training()
    else:
        print("Executando exemplo básico...")
        main()
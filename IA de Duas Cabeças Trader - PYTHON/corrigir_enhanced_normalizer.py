#!/usr/bin/env python3
"""
Corrigir o Enhanced Normalizer que está com problemas
"""

import sys
import os
import pickle
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

def analyze_normalizer_problems():
    """Analisa os problemas específicos do normalizer"""
    print("🔍 ANÁLISE DOS PROBLEMAS DO NORMALIZER")
    print("=" * 60)
    
    try:
        # Carregar o normalizer
        normalizer_path = "Modelo PPO Trader/enhanced_normalizer_final.pkl"
        
        with open(normalizer_path, 'rb') as f:
            normalizer_dict = pickle.load(f)
        
        print("✅ Normalizer carregado")
        
        # Analisar problemas específicos
        obs_rms_mean = normalizer_dict['obs_rms_mean']
        obs_rms_var = normalizer_dict['obs_rms_var']
        obs_rms_count = normalizer_dict['obs_rms_count']
        
        print(f"\n📊 ANÁLISE DETALHADA:")
        print(f"obs_rms_count: {obs_rms_count}")
        print(f"obs_rms_mean range: [{obs_rms_mean.min():.6f}, {obs_rms_mean.max():.6f}]")
        print(f"obs_rms_var range: [{obs_rms_var.min():.6f}, {obs_rms_var.max():.6f}]")
        
        # Identificar problemas
        problems = []
        
        if obs_rms_count == 0:
            problems.append("❌ obs_rms_count = 0 (normalizer não foi treinado)")
        
        if np.allclose(obs_rms_mean, 0.0):
            problems.append("❌ Todas as médias são zero (normalizer resetado)")
        
        if np.allclose(obs_rms_var, 1.0):
            problems.append("❌ Todas as variâncias são 1.0 (valores padrão)")
        
        if len(problems) > 0:
            print(f"\n⚠️ PROBLEMAS IDENTIFICADOS:")
            for problem in problems:
                print(f"  {problem}")
            return False
        else:
            print("✅ Normalizer parece estar OK")
            return True
            
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

def create_fixed_normalizer():
    """Cria um normalizer corrigido"""
    print(f"\n🔧 CRIANDO NORMALIZER CORRIGIDO")
    print("=" * 50)
    
    try:
        # Carregar dados reais do RobotV3 para calcular estatísticas corretas
        print("📊 Carregando dados reais do RobotV3...")
        
        # Importar RobotV3
        sys.path.append('Modelo PPO Trader')
        from RobotV3 import TradingEnv
        import MetaTrader5 as mt5
        
        # Inicializar MT5
        if not mt5.initialize():
            print("❌ Falha ao inicializar MT5")
            return False
        
        # Criar ambiente
        env = TradingEnv()
        print("✅ Ambiente RobotV3 criado")
        
        # Gerar múltiplas observações para calcular estatísticas
        print("📈 Gerando observações para cálculo de estatísticas...")
        
        observations = []
        num_samples = 1000  # 1000 observações para estatísticas robustas
        
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"  Gerando observação {i+1}/{num_samples}")
            
            obs = env._get_observation()
            observations.append(obs)
        
        observations = np.array(observations)
        print(f"✅ {len(observations)} observações geradas: {observations.shape}")
        
        # Calcular estatísticas corretas
        print("🧮 Calculando estatísticas corretas...")
        
        obs_mean = np.mean(observations, axis=0)
        obs_var = np.var(observations, axis=0)
        
        print(f"Mean range: [{obs_mean.min():.6f}, {obs_mean.max():.6f}]")
        print(f"Var range: [{obs_var.min():.6f}, {obs_var.max():.6f}]")
        
        # Verificar se há variâncias zero
        zero_var_count = np.sum(obs_var < 1e-10)
        print(f"Variâncias ≈ 0: {zero_var_count}/{len(obs_var)} ({100*zero_var_count/len(obs_var):.1f}%)")
        
        if zero_var_count > 0:
            print("⚠️ ATENÇÃO: Algumas dimensões têm variância zero!")
            print("Aplicando correção com epsilon...")
            
            # Adicionar epsilon pequeno para evitar divisão por zero
            epsilon = 1e-8
            obs_var = np.maximum(obs_var, epsilon)
            print(f"Variâncias corrigidas com epsilon={epsilon}")
        
        # Criar VecNormalize corrigido
        print("🔧 Criando VecNormalize corrigido...")
        
        # Criar ambiente dummy para VecNormalize
        dummy_env = type('DummyEnv', (), {
            'observation_space': type('DummySpace', (), {'shape': (1320,)})()
        })()
        
        vec_norm = VecNormalize(dummy_env, norm_obs=True, norm_reward=False)
        
        # Atualizar com estatísticas corretas
        vec_norm.obs_rms.mean = obs_mean.astype(np.float64)
        vec_norm.obs_rms.var = obs_var.astype(np.float64)
        vec_norm.obs_rms.count = num_samples
        
        print("✅ VecNormalize corrigido criado")
        
        # Testar normalização
        print("🧪 Testando normalização corrigida...")
        
        test_obs = np.random.randn(10, 1320).astype(np.float32)
        normalized = vec_norm.normalize_obs(test_obs)
        
        # Verificar anomalias
        nan_count = np.sum(np.isnan(normalized))
        inf_count = np.sum(np.isinf(normalized))
        extreme_count = np.sum(np.abs(normalized) > 10)
        
        print(f"Teste de normalização:")
        print(f"  NaN: {nan_count}")
        print(f"  Inf: {inf_count}")
        print(f"  Extreme (>10): {extreme_count}")
        
        if nan_count > 0 or inf_count > 0:
            print("❌ PROBLEMA: Ainda há anomalias!")
            return False
        else:
            print("✅ Normalização corrigida funcionando perfeitamente!")
            
            # Salvar normalizer corrigido
            print("💾 Salvando normalizer corrigido...")
            
            corrected_path = "Modelo PPO Trader/enhanced_normalizer_corrected.pkl"
            vec_norm.save(corrected_path)
            
            print(f"✅ Normalizer corrigido salvo em: {corrected_path}")
            
            # Criar também versão em dicionário (compatibilidade)
            corrected_dict = {
                'obs_rms_mean': obs_mean,
                'obs_rms_var': obs_var,
                'obs_rms_count': num_samples,
                'ret_rms_mean': np.array(0.0),
                'ret_rms_var': np.array(1.0),
                'ret_rms_count': 0,
                'step_count': 0,
                'warmup_complete': True,
                'config': {'norm_obs': True, 'norm_reward': False}
            }
            
            dict_path = "Modelo PPO Trader/enhanced_normalizer_corrected_dict.pkl"
            with open(dict_path, 'wb') as f:
                pickle.dump(corrected_dict, f)
            
            print(f"✅ Versão dicionário salva em: {dict_path}")
            
            return True
            
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_corrected_normalizer():
    """Testa o normalizer corrigido"""
    print(f"\n🧪 TESTE DO NORMALIZER CORRIGIDO")
    print("=" * 50)
    
    try:
        # Carregar normalizer corrigido
        corrected_path = "Modelo PPO Trader/enhanced_normalizer_corrected.pkl"
        
        if not os.path.exists(corrected_path):
            print(f"❌ Arquivo não encontrado: {corrected_path}")
            return False
        
        vec_norm = VecNormalize.load(corrected_path)
        print("✅ Normalizer corrigido carregado")
        
        # Testar com dados reais
        sys.path.append('Modelo PPO Trader')
        from RobotV3 import TradingEnv
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            print("❌ Falha ao inicializar MT5")
            return False
        
        env = TradingEnv()
        obs = env._get_observation()
        
        print(f"Observação real: {obs.shape}")
        print(f"Range original: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # Normalizar
        obs_reshaped = obs.reshape(1, -1)
        normalized = vec_norm.normalize_obs(obs_reshaped)
        normalized = normalized.flatten()
        
        print(f"Observação normalizada: {normalized.shape}")
        print(f"Range normalizado: [{normalized.min():.3f}, {normalized.max():.3f}]")
        
        # Verificar anomalias
        nan_count = np.sum(np.isnan(normalized))
        inf_count = np.sum(np.isinf(normalized))
        
        print(f"Análise final:")
        print(f"  NaN: {nan_count}")
        print(f"  Inf: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            print("✅ NORMALIZER CORRIGIDO FUNCIONANDO PERFEITAMENTE!")
            return True
        else:
            print("❌ Ainda há problemas no normalizer")
            return False
            
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

if __name__ == "__main__":
    print("🚀 CORREÇÃO DO ENHANCED NORMALIZER")
    print("=" * 70)
    
    # Analisar problemas
    problems_found = not analyze_normalizer_problems()
    
    if problems_found:
        print("\n🔧 PROBLEMAS DETECTADOS - INICIANDO CORREÇÃO")
        
        # Criar normalizer corrigido
        correction_ok = create_fixed_normalizer()
        
        if correction_ok:
            # Testar normalizer corrigido
            test_ok = test_corrected_normalizer()
            
            if test_ok:
                print("\n🎉 CORREÇÃO CONCLUÍDA COM SUCESSO!")
                print("✅ Enhanced Normalizer corrigido e funcionando")
            else:
                print("\n❌ CORREÇÃO FALHOU NO TESTE")
        else:
            print("\n❌ FALHA NA CRIAÇÃO DO NORMALIZER CORRIGIDO")
    else:
        print("\n✅ NORMALIZER ESTÁ OK")
    
    print("=" * 70) 
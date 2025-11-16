#!/usr/bin/env python3
"""
🧪 BATERIA DE TESTES V11 HÍBRIDA - PRÉ-TREINO
Testes completos para validar arquitetura LSTM+GRU antes do treinamento
"""

import sys
import os
sys.path.append("D:/Projeto")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

# Imports necessários
from trading_framework.policies.two_head_v11_sigmoid import (
    TwoHeadV11Sigmoid, 
    get_v8_elegance_kwargs,
    validate_v8_elegance_policy
)
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from daytrader8dim import TradingEnv, make_wrapped_env

class V11HybridTester:
    """🧪 Classe para testes da arquitetura V11 Híbrida"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_results = {}
        print(f"🧪 V11 Hybrid Tester iniciado - Device: {self.device}")
    
    def create_test_data(self, periods=100):
        """📊 Criar dados de teste completos para o trading environment"""
        timestamps = pd.date_range('2024-01-01', periods=periods, freq='1min')
        base_price = 100
        prices = np.random.randn(periods).cumsum() * 0.1 + base_price
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices + np.random.randn(periods) * 0.01,
            'high': prices + np.abs(np.random.randn(periods)) * 0.02,
            'low': prices - np.abs(np.random.randn(periods)) * 0.02,
            'close': prices,
            'volume': np.random.randint(1000, 10000, periods)
        })
        
        # Adicionar todas as features necessárias que o sistema espera
        df['close_5m'] = df['close'].rolling(5).mean().fillna(df['close'])
        df['close_15m'] = df['close'].rolling(15).mean().fillna(df['close'])
        df['close_1h'] = df['close'].rolling(60).mean().fillna(df['close'])
        df['volume_5m'] = df['volume'].rolling(5).mean().fillna(df['volume'])
        df['volume_15m'] = df['volume'].rolling(15).mean().fillna(df['volume'])
        df['returns_1m'] = df['close'].pct_change().fillna(0)
        df['returns_5m'] = df['close_5m'].pct_change().fillna(0)
        df['volatility'] = df['returns_1m'].rolling(20).std().fillna(0.01)
        
        return df
    
    def test_architecture_components(self):
        """🏗️ Teste 1: Componentes da arquitetura"""
        print("\n" + "="*60)
        print("🏗️ TESTE 1: COMPONENTES DA ARQUITETURA")
        print("="*60)
        
        try:
            # Criar dados de teste completos
            df = self.create_test_data(100)
            
            env = DummyVecEnv([lambda: Monitor(make_wrapped_env(df, 10, True))])
            
            # Criar modelo V11
            kwargs = get_v8_elegance_kwargs()
            model = RecurrentPPO(
                policy=TwoHeadV11Sigmoid,
                env=env,
                learning_rate=1e-4,
                n_steps=64,
                batch_size=16,
                n_epochs=1,
                policy_kwargs=kwargs,
                device=self.device,
                verbose=0
            )
            
            policy = model.policy
            
            # Verificar componentes híbridos
            tests = {
                'v8_shared_lstm': hasattr(policy, 'v8_shared_lstm'),
                'v11_shared_gru': hasattr(policy, 'v11_shared_gru'),
                'hybrid_fusion': hasattr(policy, 'hybrid_fusion'),
                'market_context': hasattr(policy, 'market_context'),
                'entry_head': hasattr(policy, 'entry_head'),
                'management_head': hasattr(policy, 'management_head'),
                'memory_bank': hasattr(policy, 'memory_bank'),
                'v8_critic': hasattr(policy, 'v8_critic')
            }
            
            # Contar parâmetros
            if tests['v8_shared_lstm']:
                lstm_params = sum(p.numel() for p in policy.v8_shared_lstm.parameters())
                print(f"✅ LSTM compartilhada: {lstm_params:,} parâmetros")
            
            if tests['v11_shared_gru']:
                gru_params = sum(p.numel() for p in policy.v11_shared_gru.parameters())
                print(f"✅ GRU paralela: {gru_params:,} parâmetros")
            
            if tests['hybrid_fusion']:
                fusion_params = sum(p.numel() for p in policy.hybrid_fusion.parameters())
                print(f"✅ Sistema fusão: {fusion_params:,} parâmetros")
            
            total_params = sum(p.numel() for p in policy.parameters())
            print(f"📊 Total de parâmetros: {total_params:,}")
            
            # Validar arquitetura
            validate_v8_elegance_policy(policy)
            
            self.test_results['architecture'] = {
                'passed': all(tests.values()),
                'components': tests,
                'total_params': total_params
            }
            
            print("✅ TESTE 1 APROVADO: Todos os componentes presentes")
            return True
            
        except Exception as e:
            print(f"❌ TESTE 1 FALHOU: {e}")
            self.test_results['architecture'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_forward_pass(self):
        """🔄 Teste 2: Forward pass híbrido"""
        print("\n" + "="*60)
        print("🔄 TESTE 2: FORWARD PASS HÍBRIDO")
        print("="*60)
        
        try:
            # Criar dados de teste completos
            df = self.create_test_data(100)
            
            env = DummyVecEnv([lambda: Monitor(make_wrapped_env(df, 10, True))])
            
            kwargs = get_v8_elegance_kwargs()
            model = RecurrentPPO(
                policy=TwoHeadV11Sigmoid,
                env=env,
                learning_rate=1e-4,
                n_steps=64,
                batch_size=16,
                n_epochs=1,
                policy_kwargs=kwargs,
                device=self.device,
                verbose=0
            )
            
            # Teste de forward pass
            obs = env.reset()
            print(f"📊 Observation shape: {obs.shape}")
            
            # Forward pass actor
            with torch.no_grad():
                actions, _, _ = model.policy.predict(obs, deterministic=False)
            
            print(f"✅ Actions shape: {actions.shape}")
            print(f"📊 Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
            
            # Verificar dimensões esperadas
            expected_action_dim = 4  # entry, confidence, pos1_mgmt, pos2_mgmt
            if actions.shape[-1] != expected_action_dim:
                raise ValueError(f"Action dim esperada: {expected_action_dim}, obtida: {actions.shape[-1]}")
            
            # Forward pass value
            values = model.policy.predict_values(obs)
            print(f"✅ Values shape: {values.shape}")
            print(f"📊 Values range: [{values.min():.3f}, {values.max():.3f}]")
            
            self.test_results['forward_pass'] = {
                'passed': True,
                'action_shape': actions.shape,
                'value_shape': values.shape,
                'action_range': [float(actions.min()), float(actions.max())],
                'value_range': [float(values.min()), float(values.max())]
            }
            
            print("✅ TESTE 2 APROVADO: Forward pass funcionando")
            return True
            
        except Exception as e:
            print(f"❌ TESTE 2 FALHOU: {e}")
            self.test_results['forward_pass'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_gradient_flow(self):
        """📈 Teste 3: Fluxo de gradientes"""
        print("\n" + "="*60)
        print("📈 TESTE 3: FLUXO DE GRADIENTES")
        print("="*60)
        
        try:
            # Criar dados de teste completos
            df = self.create_test_data(100)
            
            env = DummyVecEnv([lambda: Monitor(make_wrapped_env(df, 10, True))])
            
            kwargs = get_v8_elegance_kwargs()
            model = RecurrentPPO(
                policy=TwoHeadV11Sigmoid,
                env=env,
                learning_rate=1e-4,
                n_steps=64,
                batch_size=16,
                n_epochs=1,
                policy_kwargs=kwargs,
                device=self.device,
                verbose=0
            )
            
            # Simular um step de treinamento
            obs = env.reset()
            
            # Forward pass com gradientes
            model.policy.train()
            
            # Simular loss
            actions, values, log_probs = model.policy.forward(obs)
            fake_loss = actions.mean() + values.mean() + log_probs.mean()
            
            # Backward pass
            fake_loss.backward()
            
            # Verificar gradientes em componentes críticos
            gradient_stats = {}
            
            # LSTM gradientes
            for name, param in model.policy.v8_shared_lstm.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_zeros = (param.grad == 0).float().mean().item()
                    gradient_stats[f'lstm_{name}'] = {
                        'norm': grad_norm,
                        'zeros_ratio': grad_zeros
                    }
            
            # GRU gradientes
            for name, param in model.policy.v11_shared_gru.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_zeros = (param.grad == 0).float().mean().item()
                    gradient_stats[f'gru_{name}'] = {
                        'norm': grad_norm,
                        'zeros_ratio': grad_zeros
                    }
            
            # Fusion gradientes
            for i, layer in enumerate(model.policy.hybrid_fusion):
                if hasattr(layer, 'weight') and layer.weight.grad is not None:
                    grad_norm = layer.weight.grad.norm().item()
                    grad_zeros = (layer.weight.grad == 0).float().mean().item()
                    gradient_stats[f'fusion_layer_{i}'] = {
                        'norm': grad_norm,
                        'zeros_ratio': grad_zeros
                    }
            
            # Analisar resultados
            healthy_gradients = 0
            total_gradients = 0
            
            for name, stats in gradient_stats.items():
                total_gradients += 1
                norm = stats['norm']
                zeros = stats['zeros_ratio']
                
                # Gradiente saudável: norm > 1e-6 e zeros < 0.9
                if norm > 1e-6 and zeros < 0.9:
                    healthy_gradients += 1
                    status = "✅"
                else:
                    status = "⚠️"
                
                print(f"{status} {name}: norm={norm:.2e}, zeros={zeros:.1%}")
            
            health_ratio = healthy_gradients / total_gradients if total_gradients > 0 else 0
            print(f"📊 Gradientes saudáveis: {healthy_gradients}/{total_gradients} ({health_ratio:.1%})")
            
            # Limpar gradientes
            model.policy.zero_grad()
            
            self.test_results['gradient_flow'] = {
                'passed': health_ratio >= 0.7,  # 70% dos gradientes devem ser saudáveis
                'health_ratio': health_ratio,
                'gradient_stats': gradient_stats
            }
            
            if health_ratio >= 0.7:
                print("✅ TESTE 3 APROVADO: Fluxo de gradientes saudável")
                return True
            else:
                print("⚠️ TESTE 3 PARCIAL: Alguns gradientes problemáticos")
                return False
            
        except Exception as e:
            print(f"❌ TESTE 3 FALHOU: {e}")
            self.test_results['gradient_flow'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_training_step(self):
        """🎯 Teste 4: Step de treinamento completo"""
        print("\n" + "="*60)
        print("🎯 TESTE 4: STEP DE TREINAMENTO COMPLETO")
        print("="*60)
        
        try:
            # Criar dados de teste com mais períodos
            df = self.create_test_data(500)
            
            env = DummyVecEnv([lambda: Monitor(make_wrapped_env(df, 10, True))])
            
            kwargs = get_v8_elegance_kwargs()
            model = RecurrentPPO(
                policy=TwoHeadV11Sigmoid,
                env=env,
                learning_rate=1e-4,
                n_steps=64,
                batch_size=16,
                n_epochs=1,
                policy_kwargs=kwargs,
                device=self.device,
                verbose=0
            )
            
            print("🚀 Executando step de treinamento...")
            
            # Executar um step de treinamento
            initial_params = {name: param.clone() for name, param in model.policy.named_parameters()}
            
            model.learn(total_timesteps=64, log_interval=None)
            
            # Verificar se parâmetros foram atualizados
            param_changes = 0
            total_params = 0
            
            for name, param in model.policy.named_parameters():
                total_params += 1
                if not torch.equal(param, initial_params[name]):
                    param_changes += 1
            
            change_ratio = param_changes / total_params
            print(f"📊 Parâmetros atualizados: {param_changes}/{total_params} ({change_ratio:.1%})")
            
            # Verificar se modelo consegue fazer predições após treinamento
            obs = env.reset()
            actions, _, _ = model.policy.predict(obs, deterministic=True)
            
            print(f"✅ Predições pós-treinamento: {actions.shape}")
            
            self.test_results['training_step'] = {
                'passed': change_ratio > 0.5,  # >50% dos parâmetros devem ter mudado
                'param_change_ratio': change_ratio,
                'final_action_shape': actions.shape
            }
            
            if change_ratio > 0.5:
                print("✅ TESTE 4 APROVADO: Treinamento funcionando")
                return True
            else:
                print("⚠️ TESTE 4 FALHOU: Poucos parâmetros atualizados")
                return False
            
        except Exception as e:
            print(f"❌ TESTE 4 FALHOU: {e}")
            self.test_results['training_step'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_memory_efficiency(self):
        """💾 Teste 5: Eficiência de memória"""
        print("\n" + "="*60)
        print("💾 TESTE 5: EFICIÊNCIA DE MEMÓRIA")
        print("="*60)
        
        try:
            if not torch.cuda.is_available():
                print("⚠️ CUDA não disponível, pulando teste de memória GPU")
                self.test_results['memory_efficiency'] = {'passed': True, 'skipped': True}
                return True
            
            # Limpeza inicial
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Criar dados de teste completos
            df = self.create_test_data(100)
            
            env = DummyVecEnv([lambda: Monitor(make_wrapped_env(df, 10, True))])
            
            kwargs = get_v8_elegance_kwargs()
            model = RecurrentPPO(
                policy=TwoHeadV11Sigmoid,
                env=env,
                learning_rate=1e-4,
                n_steps=64,
                batch_size=16,
                n_epochs=1,
                policy_kwargs=kwargs,
                device=self.device,
                verbose=0
            )
            
            model_memory = torch.cuda.memory_allocated() - initial_memory
            
            # Executar forward pass múltiplos
            obs = env.reset()
            for _ in range(10):
                actions, _, _ = model.policy.predict(obs, deterministic=False)
            
            forward_memory = torch.cuda.memory_allocated() - initial_memory - model_memory
            
            print(f"📊 Memória do modelo: {model_memory / 1024**2:.1f} MB")
            print(f"📊 Memória forward pass: {forward_memory / 1024**2:.1f} MB")
            print(f"📊 Memória total: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            
            # Limpeza
            del model
            torch.cuda.empty_cache()
            
            # Verificar se memória foi liberada
            final_memory = torch.cuda.memory_allocated()
            memory_leak = final_memory - initial_memory
            
            print(f"📊 Memory leak: {memory_leak / 1024**2:.1f} MB")
            
            self.test_results['memory_efficiency'] = {
                'passed': model_memory < 500 * 1024**2,  # <500MB
                'model_memory_mb': model_memory / 1024**2,
                'forward_memory_mb': forward_memory / 1024**2,
                'memory_leak_mb': memory_leak / 1024**2
            }
            
            if model_memory < 500 * 1024**2:
                print("✅ TESTE 5 APROVADO: Memória eficiente")
                return True
            else:
                print("⚠️ TESTE 5 FALHOU: Uso excessivo de memória")
                return False
            
        except Exception as e:
            print(f"❌ TESTE 5 FALHOU: {e}")
            self.test_results['memory_efficiency'] = {'passed': False, 'error': str(e)}
            return False
    
    def run_full_battery(self):
        """🧪 Executar bateria completa de testes"""
        print("🧪 INICIANDO BATERIA COMPLETA DE TESTES V11 HÍBRIDA")
        print("="*60)
        
        tests = [
            ('Arquitetura', self.test_architecture_components),
            ('Forward Pass', self.test_forward_pass),
            ('Gradientes', self.test_gradient_flow),
            ('Treinamento', self.test_training_step),
            ('Memória', self.test_memory_efficiency)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🎯 Executando {test_name}...")
            if test_func():
                passed_tests += 1
        
        # Relatório final
        print("\n" + "="*60)
        print("📊 RELATÓRIO FINAL DOS TESTES")
        print("="*60)
        
        success_rate = passed_tests / total_tests
        print(f"✅ Testes aprovados: {passed_tests}/{total_tests} ({success_rate:.1%})")
        
        if success_rate >= 0.8:
            print("🎉 ARQUITETURA V11 HÍBRIDA APROVADA PARA PRÉ-TREINO!")
            recommendation = "RECOMENDADO"
        elif success_rate >= 0.6:
            print("⚠️ ARQUITETURA V11 HÍBRIDA PARCIALMENTE APROVADA")
            recommendation = "APROVADO_COM_RESERVAS"
        else:
            print("❌ ARQUITETURA V11 HÍBRIDA NÃO APROVADA")
            recommendation = "NÃO_RECOMENDADO"
        
        # Salvar relatório
        self.save_test_report(recommendation, success_rate)
        
        return recommendation == "RECOMENDADO"
    
    def save_test_report(self, recommendation, success_rate):
        """💾 Salvar relatório de testes"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = f"""
🧪 RELATÓRIO DE TESTES V11 HÍBRIDA
═══════════════════════════════════════════════════════════

📅 Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🖥️ Device: {self.device}
📊 Taxa de Sucesso: {success_rate:.1%}
🎯 Recomendação: {recommendation}

RESULTADOS DETALHADOS:
═══════════════════════════════════════════════════════════

"""
        
        for test_name, result in self.test_results.items():
            status = "✅ APROVADO" if result.get('passed', False) else "❌ FALHOU"
            report += f"{test_name.upper()}: {status}\n"
            
            if 'error' in result:
                report += f"  Erro: {result['error']}\n"
            
            # Adicionar detalhes específicos
            if test_name == 'architecture' and 'total_params' in result:
                report += f"  Parâmetros: {result['total_params']:,}\n"
            elif test_name == 'forward_pass' and 'action_shape' in result:
                report += f"  Action Shape: {result['action_shape']}\n"
            elif test_name == 'gradient_flow' and 'health_ratio' in result:
                report += f"  Gradientes Saudáveis: {result['health_ratio']:.1%}\n"
            elif test_name == 'training_step' and 'param_change_ratio' in result:
                report += f"  Parâmetros Atualizados: {result['param_change_ratio']:.1%}\n"
            elif test_name == 'memory_efficiency' and 'model_memory_mb' in result:
                report += f"  Memória do Modelo: {result['model_memory_mb']:.1f} MB\n"
            
            report += "\n"
        
        # Salvar arquivo
        report_path = f"test_v11_hybrid_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Relatório salvo: {report_path}")

def main():
    """🚀 Função principal"""
    print("🧪 INICIANDO TESTES DA ARQUITETURA V11 HÍBRIDA LSTM+GRU")
    
    try:
        tester = V11HybridTester()
        success = tester.run_full_battery()
        
        if success:
            print("\n🎉 TODOS OS TESTES APROVADOS!")
            print("🚀 V11 Híbrida pronta para pré-treino!")
            return 0
        else:
            print("\n⚠️ ALGUNS TESTES FALHARAM!")
            print("🔧 Revisar arquitetura antes do pré-treino")
            return 1
    
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO NOS TESTES: {e}")
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit_code = main()
    input(f"\n⏸️ Pressione Enter para sair (código: {exit_code})...")
    exit(exit_code)
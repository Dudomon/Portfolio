"""
🎯 TESTE DE BALANCEAMENTO REWARD V4 ADAPTIVE
Sistema para testar e analisar balanceamento entre componentes dos rewards V4
Analisa proporções de PnL vs Shaping vs Seletividade/Activity
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
import sys
from typing import Dict, List, Tuple, Any

# Adicionar path do projeto
sys.path.append('D:\\Projeto')

from trading_framework.rewards.reward_daytrade_v4_selective import SelectiveTradingReward
from trading_framework.rewards.reward_daytrade_v4_inno import InnovativeMoneyReward

class RewardBalanceAnalyzer:
    """
    Analisador de balanceamento de componentes dos rewards V4
    Testa diferentes cenários e mede a contribuição de cada componente
    """
    
    def __init__(self):
        self.results = {
            'selective': [],
            'inno': [],
            'scenarios': []
        }
        
    def create_mock_env(self, scenario_name: str) -> Dict[str, Any]:
        """
        Cria ambiente mock para diferentes cenários de teste
        """
        scenarios = {
            'winning_trade': {
                'portfolio_value': 1050.0,  # +5% gain
                'positions': [{'id': 1, 'pnl': 50.0, 'unrealized_pnl': 0}],
                'last_pnl': 50.0,
                'drawdown': 0.02,
                'volatility': 0.005,
                'trend_strength': 0.003,
                'cooldown_counter': 0
            },
            'losing_trade': {
                'portfolio_value': 970.0,  # -3% loss
                'positions': [],
                'last_pnl': -30.0,
                'drawdown': 0.05,
                'volatility': 0.002,
                'trend_strength': -0.001,
                'cooldown_counter': 5
            },
            'overtrading': {
                'portfolio_value': 995.0,  # Small loss from overtrading
                'positions': [{'id': 1, 'pnl': 2.0, 'unrealized_pnl': 3.0}],
                'last_pnl': 2.0,
                'drawdown': 0.01,
                'volatility': 0.001,
                'trend_strength': 0.0005,
                'cooldown_counter': 0,
                'time_in_position': 0.8  # 80% do tempo posicionado (overtrading)
            },
            'patient_waiting': {
                'portfolio_value': 1000.0,  # No change, but good patience
                'positions': [],
                'last_pnl': 0.0,
                'drawdown': 0.0,
                'volatility': 0.0001,  # Very low volatility
                'trend_strength': 0.0001,  # Very weak trend
                'cooldown_counter': 0
            },
            'high_volatility': {
                'portfolio_value': 1025.0,  # Good gain in volatile market
                'positions': [{'id': 1, 'pnl': 25.0, 'unrealized_pnl': 5.0}],
                'last_pnl': 25.0,
                'drawdown': 0.08,
                'volatility': 0.012,  # High volatility
                'trend_strength': 0.008,
                'cooldown_counter': 0
            }
        }
        
        return scenarios.get(scenario_name, scenarios['winning_trade'])
    
    def create_mock_actions(self) -> Dict[str, np.ndarray]:
        """
        Cria diferentes tipos de ações para teste
        """
        return {
            'aggressive_long': np.array([1, 0.9, 0.5, 0.0]),    # Strong long entry
            'conservative_long': np.array([1, 0.3, 0.2, 0.0]),  # Weak long entry
            'short_entry': np.array([2, 0.7, 0.0, 0.3]),        # Short entry
            'hold': np.array([0, 0.1, 0.0, 0.0]),               # Hold/wait
            'close_positions': np.array([0, 0.0, -0.8, -0.8])   # Close all positions
        }
    
    def simulate_reward_scenario(self, reward_system, env_data: Dict, action: np.ndarray, 
                                scenario_name: str, action_name: str) -> Dict:
        """
        Simula um cenário específico e retorna componentes detalhados
        """
        # Mock do ambiente
        class MockEnv:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        env = MockEnv(env_data)
        
        # Para V4 Selective, simular tracking de tempo em posição
        if hasattr(reward_system, 'steps_in_position'):
            if env_data.get('time_in_position', 0.2) > 0.5:  # Se overtrading
                reward_system.steps_in_position = 800
                reward_system.total_steps = 1000
            else:
                reward_system.steps_in_position = 200
                reward_system.total_steps = 1000
        
        # Calcular reward
        try:
            reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
            
            return {
                'scenario': scenario_name,
                'action': action_name,
                'total_reward': reward,
                'info': info,
                'done': done,
                'components': self._extract_reward_components(info)
            }
        except Exception as e:
            return {
                'scenario': scenario_name,
                'action': action_name,
                'error': str(e),
                'total_reward': 0.0
            }
    
    def _extract_reward_components(self, info: Dict) -> Dict:
        """
        Extrai componentes individuais do reward do info dict
        """
        components = {}
        
        # Componentes comuns
        for key in ['pnl_reward', 'shaping_reward', 'activity_bonus', 
                   'selectivity_reward', 'quality_reward']:
            if key in info:
                components[key] = info[key]
        
        # Componentes específicos
        if 'pnl_component' in info:
            components['pnl_component'] = info['pnl_component']
        if 'shaping_component' in info:
            components['shaping_component'] = info['shaping_component']
        if 'activity_component' in info:
            components['activity_component'] = info['activity_component']
        
        return components
    
    def run_comprehensive_test(self) -> Dict:
        """
        Executa teste abrangente de balanceamento para ambos os sistemas V4
        """
        print("🎯 INICIANDO TESTE DE BALANCEAMENTO V4 ADAPTIVE")
        print("=" * 60)
        
        # Inicializar sistemas de reward
        selective_reward = SelectiveTradingReward(initial_balance=1000.0)
        inno_reward = InnovativeMoneyReward(initial_balance=1000.0)
        
        # Cenários e ações de teste
        scenarios = ['winning_trade', 'losing_trade', 'overtrading', 'patient_waiting', 'high_volatility']
        actions = self.create_mock_actions()
        
        results = {
            'selective': {},
            'inno': {},
            'summary': {}
        }
        
        print("\\n📊 TESTANDO V4 SELECTIVE...")
        # Testar V4 Selective
        for scenario in scenarios:
            results['selective'][scenario] = {}
            env_data = self.create_mock_env(scenario)
            
            for action_name, action in actions.items():
                result = self.simulate_reward_scenario(
                    selective_reward, env_data, action, scenario, action_name
                )
                results['selective'][scenario][action_name] = result
                
                if 'error' not in result:
                    print(f"  {scenario:15} + {action_name:15} = {result['total_reward']:8.4f}")
        
        print("\\n📊 TESTANDO V4 INNO...")
        # Testar V4 Inno
        for scenario in scenarios:
            results['inno'][scenario] = {}
            env_data = self.create_mock_env(scenario)
            
            for action_name, action in actions.items():
                result = self.simulate_reward_scenario(
                    inno_reward, env_data, action, scenario, action_name
                )
                results['inno'][scenario][action_name] = result
                
                if 'error' not in result:
                    print(f"  {scenario:15} + {action_name:15} = {result['total_reward']:8.4f}")
        
        # Análise comparativa
        results['summary'] = self._generate_comparative_analysis(results)
        
        return results
    
    def _generate_comparative_analysis(self, results: Dict) -> Dict:
        """
        Gera análise comparativa entre os dois sistemas
        """
        analysis = {
            'selective_stats': {},
            'inno_stats': {},
            'comparison': {}
        }
        
        # Calcular estatísticas para cada sistema
        for system_name in ['selective', 'inno']:
            rewards = []
            for scenario_data in results[system_name].values():
                for result in scenario_data.values():
                    if 'error' not in result:
                        rewards.append(result['total_reward'])
            
            if rewards:
                analysis[f'{system_name}_stats'] = {
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'min_reward': np.min(rewards),
                    'max_reward': np.max(rewards),
                    'reward_range': np.max(rewards) - np.min(rewards)
                }
        
        return analysis
    
    def save_results(self, results: Dict, filename: str = None):
        """
        Salva resultados em arquivo JSON
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"teste_balanceamento_v4_{timestamp}.txt"
        
        # Converter numpy arrays para listas para JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_converted = convert_numpy(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("🎯 RELATÓRIO DE BALANCEAMENTO V4 ADAPTIVE\\n")
            f.write("=" * 60 + "\\n\\n")
            
            # Summary estatístico
            if 'summary' in results:
                f.write("📈 ANÁLISE ESTATÍSTICA:\\n")
                f.write("-" * 30 + "\\n")
                for key, stats in results['summary'].items():
                    if isinstance(stats, dict):
                        f.write(f"\\n{key.upper()}:\\n")
                        for stat_name, value in stats.items():
                            f.write(f"  {stat_name}: {value:.4f}\\n")
            
            f.write("\\n\\n📊 RESULTADOS DETALHADOS:\\n")
            f.write("-" * 40 + "\\n")
            f.write(json.dumps(results_converted, indent=2, ensure_ascii=False))
        
        print(f"\\n💾 Resultados salvos em: {filename}")
        return filename

def main():
    """
    Função principal do teste de balanceamento
    """
    analyzer = RewardBalanceAnalyzer()
    
    try:
        # Executar teste abrangente
        results = analyzer.run_comprehensive_test()
        
        # Salvar resultados
        filename = analyzer.save_results(results)
        
        print("\\n🎯 TESTE DE BALANCEAMENTO V4 CONCLUÍDO!")
        print(f"📁 Arquivo: {filename}")
        
        # Exibir resumo
        if 'summary' in results:
            print("\\n📈 RESUMO ESTATÍSTICO:")
            print("-" * 25)
            for system, stats in results['summary'].items():
                if isinstance(stats, dict) and 'mean_reward' in stats:
                    print(f"{system:15}: Média={stats['mean_reward']:7.4f}, "
                          f"Range={stats['reward_range']:7.4f}")
        
    except Exception as e:
        print(f"❌ ERRO durante teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
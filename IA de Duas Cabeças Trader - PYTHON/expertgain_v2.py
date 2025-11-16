#!/usr/bin/env python3
"""
🎯 EXPERTGAIN V2 - FINE-TUNING INTELIGENTE
Sistema especializado que REALMENTE funciona para melhorar Entry Quality
"""

import sys
import os
sys.path.append("D:/Projeto")

# Copiar todo o código base do expertgain.py
# MAS com estas configurações especializadas:

# ========== CONFIGURAÇÃO EXPERTGAIN V2 ==========

# 🔥 HIPERPARÂMETROS OTIMIZADOS PARA FINE-TUNING
EXPERTGAIN_V2_PARAMS = {
    # 🎯 LR com WARM-UP e DECAY
    "learning_rate": 3.5e-04,  # Começa ALTO para escapar do mínimo local
    "lr_schedule": {
        "warmup_steps": 50000,      # Warm-up gradual
        "decay_rate": 0.95,          # Decay a cada milestone
        "milestones": [500000, 1000000, 1500000]  # Pontos de redução
    },
    
    # 🎯 BATCH E EPOCHS OTIMIZADOS
    "n_steps": 2048,
    "batch_size": 128,              # MAIOR para estabilidade
    "n_epochs": 8,                  # MAIS epochs para exploração
    
    # 🎯 PPO PARAMETERS AJUSTADOS
    "gamma": 0.99,                  # Mantém visão de longo prazo
    "gae_lambda": 0.95,             # GAE padrão
    "clip_range": 0.25,             # MAIS liberdade para explorar
    "clip_range_vf": None,          # Sem clip no value function
    
    # 🎯 ENTROPY PROGRESSIVO
    "ent_coef": 0.02,               # COMEÇA com mais exploração
    "ent_coef_schedule": {
        "initial": 0.02,
        "final": 0.005,
        "decay_steps": 1000000
    },
    
    # 🎯 VALUE FUNCTION
    "vf_coef": 0.5,                 # Balanceado
    "max_grad_norm": 1.0,           # Permite gradientes maiores
    
    # 🎯 TARGET KL DINÂMICO
    "target_kl": 0.03,              # Permite mudanças maiores
    "target_kl_schedule": {
        "initial": 0.05,            # Começa permitindo grandes mudanças
        "final": 0.01,              # Termina conservador
        "decay_steps": 1500000
    }
}

# 🎯 FASES ESPECIALIZADAS COM OBJETIVOS CLAROS
EXPERTGAIN_V2_PHASES = [
    {
        "name": "Phase_1_Unlock_Gates",
        "steps": 500000,  # 500k steps apenas
        "objective": "Desbloquear gates travadas em 0.038",
        "config": {
            "learning_rate": 4.0e-04,  # LR ALTO para quebrar inércia
            "ent_coef": 0.03,          # MUITA exploração
            "clip_range": 0.3,         # Liberdade máxima
            "target_entry_quality": 0.15  # Meta modesta inicial
        },
        "success_metrics": {
            "entry_quality_min": 0.10,
            "trades_per_episode": 1
        }
    },
    {
        "name": "Phase_2_Calibrate_Quality",
        "steps": 750000,  # 750k steps
        "objective": "Elevar Entry Quality para 0.30+",
        "config": {
            "learning_rate": 2.5e-04,  # LR médio
            "ent_coef": 0.015,         # Exploração moderada
            "clip_range": 0.2,         # Liberdade controlada
            "target_entry_quality": 0.30
        },
        "success_metrics": {
            "entry_quality_min": 0.25,
            "trades_per_episode": 3,
            "win_rate_min": 0.45
        }
    },
    {
        "name": "Phase_3_Optimize_Trading",
        "steps": 750000,  # 750k steps
        "objective": "Atingir Entry Quality 0.50+ com trades consistentes",
        "config": {
            "learning_rate": 1.5e-04,  # LR conservador
            "ent_coef": 0.008,         # Pouca exploração
            "clip_range": 0.15,        # Mais focado
            "target_entry_quality": 0.55
        },
        "success_metrics": {
            "entry_quality_min": 0.45,
            "trades_per_episode": 5,
            "win_rate_min": 0.50,
            "positive_return": True
        }
    }
]

# 🎯 SISTEMA DE REWARD MODIFICADO PARA EXPERTGAIN
class ExpertGainRewardShaper:
    """
    Sistema de reward especializado para aumentar Entry Quality
    """
    def __init__(self, target_quality=0.5):
        self.target_quality = target_quality
        self.quality_history = []
        
    def shape_reward(self, original_reward, action, info):
        """
        Adiciona bonus/penalidade baseado em Entry Quality
        """
        entry_quality = action[1] if len(action) > 1 else 0.0
        
        # Histórico para suavização
        self.quality_history.append(entry_quality)
        if len(self.quality_history) > 100:
            self.quality_history.pop(0)
        
        # Reward shaping baseado em quality
        quality_bonus = 0.0
        
        # 1. BONUS por quality alto
        if entry_quality > self.target_quality:
            quality_bonus += 0.5 * (entry_quality - self.target_quality)
        
        # 2. PENALIDADE por quality muito baixo
        if entry_quality < 0.1:
            quality_bonus -= 0.3
        
        # 3. BONUS por MELHORIA
        if len(self.quality_history) > 10:
            recent_avg = sum(self.quality_history[-10:]) / 10
            old_avg = sum(self.quality_history[:10]) / 10
            if recent_avg > old_avg:
                quality_bonus += 0.2 * (recent_avg - old_avg)
        
        # 4. BONUS por executar trades com quality alto
        if info.get("trade_executed") and entry_quality > 0.4:
            quality_bonus += 1.0 * entry_quality
        
        # 5. PENALIDADE por 100% HOLD
        if info.get("episode_done"):
            if info.get("total_trades", 0) == 0:
                quality_bonus -= 2.0  # Forte penalidade por não tradear
        
        return original_reward + quality_bonus

# 🎯 CALLBACK ESPECIALIZADO PARA MONITORAMENTO
class ExpertGainMonitor:
    """
    Monitor especializado para acompanhar Entry Quality
    """
    def __init__(self):
        self.entry_qualities = []
        self.trade_counts = []
        self.phase_start_quality = None
        
    def on_step(self, action, reward, done, info):
        """
        Monitora cada step
        """
        entry_quality = action[1] if len(action) > 1 else 0.0
        self.entry_qualities.append(entry_quality)
        
        if done:
            avg_quality = sum(self.entry_qualities) / len(self.entry_qualities)
            print(f"📊 Episode Entry Quality Avg: {avg_quality:.3f}")
            
            # Alert se quality está travado
            if len(set(self.entry_qualities[-100:])) < 5:
                print("⚠️ ALERTA: Entry Quality TRAVADO! Aumentar LR ou entropy!")
            
            self.entry_qualities = []
    
    def check_phase_progress(self, current_quality):
        """
        Verifica progresso da fase
        """
        if self.phase_start_quality is None:
            self.phase_start_quality = current_quality
            return
        
        improvement = current_quality - self.phase_start_quality
        
        if improvement < 0.05:  # Menos de 5% de melhoria
            print("⚠️ PROGRESSO LENTO: Considere aumentar LR ou entropy")
            return "adjust_lr"
        elif improvement > 0.15:  # Mais de 15% de melhoria
            print("✅ EXCELENTE PROGRESSO: Pode reduzir LR para consolidar")
            return "reduce_lr"
        
        return "continue"

# 🎯 DYNAMIC LEARNING RATE ADJUSTER
class DynamicLRAdjuster:
    """
    Ajusta LR dinamicamente baseado em performance
    """
    def __init__(self, model):
        self.model = model
        self.quality_history = []
        self.lr_history = []
        self.stagnation_counter = 0
        
    def update(self, current_quality):
        """
        Atualiza LR baseado em Entry Quality
        """
        self.quality_history.append(current_quality)
        
        if len(self.quality_history) > 10:
            # Detecta estagnação
            recent = self.quality_history[-10:]
            if max(recent) - min(recent) < 0.01:  # Variação < 1%
                self.stagnation_counter += 1
                
                if self.stagnation_counter > 5:
                    # AUMENTA LR para escapar do mínimo local
                    current_lr = self.model.learning_rate
                    new_lr = min(current_lr * 1.5, 5e-04)  # Cap em 5e-04
                    
                    print(f"🔥 ESTAGNAÇÃO DETECTADA! LR: {current_lr:.2e} → {new_lr:.2e}")
                    self.model.learning_rate = new_lr
                    self.stagnation_counter = 0
            else:
                self.stagnation_counter = 0
        
        # Se quality muito alto, reduz LR para refinar
        if current_quality > 0.5 and len(self.quality_history) > 50:
            recent_std = np.std(self.quality_history[-50:])
            if recent_std < 0.05:  # Estável e bom
                current_lr = self.model.learning_rate
                new_lr = max(current_lr * 0.9, 5e-05)  # Floor em 5e-05
                print(f"✅ PERFORMANCE BOA! LR: {current_lr:.2e} → {new_lr:.2e}")
                self.model.learning_rate = new_lr

# 🎯 EARLY STOPPING INTELIGENTE
class SmartEarlyStopping:
    """
    Para treinamento se não houver progresso REAL
    """
    def __init__(self, patience=100000, min_improvement=0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_quality = 0
        self.steps_without_improvement = 0
        
    def should_stop(self, current_quality, current_trades):
        """
        Decide se deve parar
        """
        # Se não está tradeando, não vale a pena continuar
        if current_trades == 0 and self.steps_without_improvement > 50000:
            print("❌ MODELO TRAVADO EM HOLD! Parando treinamento.")
            return True
        
        # Verifica melhoria em quality
        if current_quality > self.best_quality + self.min_improvement:
            self.best_quality = current_quality
            self.steps_without_improvement = 0
            print(f"✅ Nova melhor Entry Quality: {self.best_quality:.3f}")
        else:
            self.steps_without_improvement += 1
        
        if self.steps_without_improvement > self.patience:
            print(f"⚠️ Sem melhoria há {self.patience} steps. Parando.")
            return True
        
        return False

# 🎯 MAIN FUNCTION ESPECIALIZADA
def train_expertgain_v2():
    """
    Treina ExpertGain V2 com todas as otimizações
    """
    print("🚀 EXPERTGAIN V2 - FINE-TUNING INTELIGENTE")
    print("=" * 60)
    
    # 1. Carregar checkpoint base do DayTrader (8M steps)
    base_checkpoint = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_phase4stresstesting_8000000_steps_20250808_173027.zip"
    if not os.path.exists(base_checkpoint):
        print(f"❌ Checkpoint DayTrader 8M não encontrado: {base_checkpoint}")
        return
    
    print(f"✅ Usando checkpoint DayTrader 8M: {os.path.basename(base_checkpoint)}")
    
    # 2. Criar ambiente com reward shaping
    env = create_expertgain_env(reward_shaper=ExpertGainRewardShaper())
    
    # 3. Carregar modelo com novos hiperparâmetros
    model = load_with_new_params(base_checkpoint, EXPERTGAIN_V2_PARAMS)
    
    # 4. Inicializar sistemas de monitoramento
    monitor = ExpertGainMonitor()
    lr_adjuster = DynamicLRAdjuster(model)
    early_stopping = SmartEarlyStopping()
    
    # 5. Treinar por fases
    for phase in EXPERTGAIN_V2_PHASES:
        print(f"\n🎯 INICIANDO {phase['name']}")
        print(f"   Objetivo: {phase['objective']}")
        print(f"   Steps: {phase['steps']:,}")
        
        # Aplicar configurações da fase
        apply_phase_config(model, phase['config'])
        
        # Treinar
        for step in range(phase['steps']):
            model.learn(total_timesteps=1000, callback=monitor)
            
            # Ajustes dinâmicos
            if step % 10000 == 0:
                current_quality = get_average_entry_quality(model)
                lr_adjuster.update(current_quality)
                
                if early_stopping.should_stop(current_quality, get_trade_count(model)):
                    break
        
        # Verificar sucesso da fase
        if check_phase_success(phase['success_metrics']):
            print(f"✅ {phase['name']} CONCLUÍDA COM SUCESSO!")
        else:
            print(f"⚠️ {phase['name']} não atingiu todas as metas")
    
    print("\n🏆 EXPERTGAIN V2 TREINAMENTO CONCLUÍDO!")

if __name__ == "__main__":
    train_expertgain_v2()
# 🎯 CRÍTICO V7 FIXES - GENERATED

# Learning Rates Diferenciados

# 🔧 FIX 1: LEARNING RATES DIFERENCIADOS
def create_optimizer_groups(policy):
    '''Criar grupos de otimização com LRs diferentes'''
    
    actor_params = []
    critic_params = []
    
    for name, param in policy.named_parameters():
        if 'critic' in name.lower() or 'value' in name.lower():
            critic_params.append(param)
        else:
            actor_params.append(param)
    
    # Crítico com LR 2-3x maior
    optimizer_groups = [
        {'params': actor_params, 'lr': 3e-4},      # LR normal para actor
        {'params': critic_params, 'lr': 1e-3}      # LR maior para crítico
    ]
    
    return optimizer_groups

# Aplicar no daytrader.py na criação do modelo:
# model.policy.optimizer = torch.optim.Adam(create_optimizer_groups(model.policy))
        

# Arquitetura Crítico Melhorada

# 🔧 FIX 2: ARQUITETURA CRÍTICO MELHORADA
class ImprovedCriticHead(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        
        # Crítico mais robusto com skip connections
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            
            nn.Linear(128, 1)  # Output final
        )
        
        # Skip connection para estabilidade
        self.skip_connection = nn.Linear(input_dim, 1)
        
    def forward(self, features):
        main_output = self.value_net(features)
        skip_output = self.skip_connection(features)
        # Combinar com peso
        return 0.8 * main_output + 0.2 * skip_output
        

# Gradient Clipping Diferenciado

# 🔧 FIX 3: GRADIENT CLIPPING DIFERENCIADO
def apply_differential_grad_clipping(model):
    '''Aplicar clipping diferenciado por componente'''
    
    # Clipping mais conservador para crítico
    critic_params = []
    actor_params = []
    
    for name, param in model.policy.named_parameters():
        if param.grad is not None:
            if 'critic' in name.lower() or 'value' in name.lower():
                critic_params.append(param)
            else:
                actor_params.append(param)
    
    # Clipping diferenciado
    if critic_params:
        torch.nn.utils.clip_grad_norm_(critic_params, max_norm=0.5)  # Mais conservador
    if actor_params:
        torch.nn.utils.clip_grad_norm_(actor_params, max_norm=1.0)   # Normal
        
# Adicionar no callback de treinamento
        

# Value Loss Scaling Adaptativo

# 🔧 FIX 4: VALUE LOSS SCALING ADAPTATIVO
class AdaptiveValueLossCallback:
    def __init__(self):
        self.value_loss_history = []
        self.exp_var_history = []
        self.loss_scale = 1.0
        
    def update_loss_scaling(self, value_loss, explained_variance):
        self.value_loss_history.append(value_loss)
        self.exp_var_history.append(explained_variance)
        
        # Manter histórico limitado
        if len(self.value_loss_history) > 100:
            self.value_loss_history.pop(0)
            self.exp_var_history.pop(0)
        
        # Ajustar scaling baseado na explained variance
        if len(self.exp_var_history) > 10:
            recent_exp_var = np.mean(self.exp_var_history[-10:])
            
            if recent_exp_var < -0.1:  # Muito negativo
                self.loss_scale = min(3.0, self.loss_scale * 1.1)  # Aumentar peso
            elif recent_exp_var > 0.1:  # Muito positivo
                self.loss_scale = max(0.5, self.loss_scale * 0.95)  # Reduzir peso
                
        return self.loss_scale

# Integrar no loop de treinamento
        

# Target Network para Estabilidade

# 🔧 FIX 5: TARGET NETWORK PARA ESTABILIDADE
class CriticWithTarget:
    def __init__(self, critic_network):
        self.main_critic = critic_network
        self.target_critic = copy.deepcopy(critic_network)
        self.update_freq = 1000  # Update target a cada 1000 steps
        self.tau = 0.005  # Soft update rate
        
    def update_target(self, step):
        if step % self.update_freq == 0:
            # Soft update
            for target_param, main_param in zip(
                self.target_critic.parameters(), 
                self.main_critic.parameters()
            ):
                target_param.data.copy_(
                    self.tau * main_param.data + (1.0 - self.tau) * target_param.data
                )
                
    def compute_target_values(self, next_states):
        with torch.no_grad():
            return self.target_critic(next_states)
        


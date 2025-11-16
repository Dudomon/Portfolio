# 🔒 Sistema de Proteção de Modelos Trading

Sistema completo para proteger modelos PPO/RecurrentPPO contra uso não autorizado, extração de pesos e reverse engineering.

## 🎯 Características

- **Criptografia AES-256**: Arquivos completamente criptografados
- **Obfuscação de Pesos**: Transformações matemáticas reversíveis nos tensors
- **Hardware Lock**: Modelos travados no hardware específico
- **Proteção por Chave**: Chave mestra necessária para descriptografar
- **Conversão Sem Retreino**: Protege modelos existentes

## 🚀 Uso Rápido

### Proteger Todos os Modelos Cherry
```bash
cd D:/Projeto
python scripts/protect_models.py --all-cherry
```

### Proteger Apenas os Melhores Modelos
```bash
python scripts/protect_models.py --best-models
```

### Proteger Modelo Específico
```bash
python scripts/protect_models.py --single "path/to/model.zip"
```

### Para Distribuição (sem hardware lock)
```bash
python scripts/protect_models.py --best-models --no-hardware-lock
```

## 🧪 Testar o Sistema

### Demo Completa
```bash
python scripts/secure_model_demo.py
```

### Verificar Modelo Protegido
```bash
python scripts/protect_models.py --verify "model.secure"
```

## 📋 Modelos Recomendados para Proteção

### Melhores Performers (baseado em análises):
- `Cherry_550000_steps` - Baseline vencedor
- `Cherry_3100000_steps` - Mid-training forte  
- `Cherry_15650000_steps` - Late training maduro

## 💻 Uso Programático

### Proteção Individual
```python
from trading_framework.security.secure_model_system import ModelSecurityManager

manager = ModelSecurityManager("sua_chave_secreta")

# Proteger modelo
success = manager.convert_checkpoint(
    input_path="modelo.zip",
    output_path="modelo.secure", 
    hardware_lock=True
)

# Carregar modelo protegido
model_info = manager.wrapper.load_secure(
    secure_path="modelo.secure",
    master_key="sua_chave_secreta"
)
```

### Proteção em Batch
```python
# Proteger todos Cherry
results = manager.batch_convert(
    input_pattern="D:/Projeto/Otimizacao/treino_principal/models/Cherry/*.zip",
    hardware_lock=True
)
```

## 🔐 Segurança

### Níveis de Proteção:

1. **Nível 1 - Criptografia**: Arquivo completamente criptografado com AES-256
2. **Nível 2 - Obfuscação**: Pesos dos tensors matematicamente transformados  
3. **Nível 3 - Hardware Lock**: Modelo trava no hardware específico
4. **Nível 4 - Chave Mestra**: Proteção adicional por senha

### Hardware Fingerprint:
- CPU + GPU information
- Unique per machine
- SHA-256 hash truncated to 16 chars
- Example: `a1b2c3d4e5f6g7h8`

## 📁 Estrutura dos Arquivos

### Modelo Original (.zip)
```
modelo_original.zip
├── policy.pth
├── policy.optimizer.pth 
├── data (pickled SB3 data)
└── ...
```

### Modelo Protegido (.secure)
```python
{
    'format': 'SecureTradingModel',
    'version': '1.0',
    'encrypted_payload': b'...'  # Conteúdo criptografado:
                                 # ├── obfuscated_weights
                                 # ├── hardware_fingerprint  
                                 # ├── metadata
                                 # └── additional_info
}
```

## ⚡ Performance

### Overhead de Proteção:
- **Conversão**: ~10-30s por modelo (dependendo do tamanho)
- **Carregamento**: +2-5s overhead vs modelo normal
- **Tamanho**: Arquivo protegido ~5-10% maior
- **RAM**: Mesma utilização após carregamento

### Otimizações:
- Obfuscação in-place quando possível
- Criptografia com Fernet (rápida)
- Cache de hardware fingerprint
- Validações lazy

## 🛠️ Troubleshooting

### Erro: "Hardware não autorizado"
- Modelo foi criado em outra máquina com hardware lock
- Solução: Recriar modelo sem `--no-hardware-lock`

### Erro: "Chave mestra inválida" 
- Chave incorreta ou corrupted
- Verificar se usa mesma chave da proteção

### Erro: "Formato inválido"
- Arquivo pode estar corrompido
- Re-proteger modelo original

## 🔄 Migração de Modelos Existentes

### Backup Recomendado:
```bash
# 1. Backup dos originais
cp -r "D:/Projeto/Otimizacao/treino_principal/models/Cherry" "Cherry_backup"

# 2. Proteger modelos
python scripts/protect_models.py --all-cherry

# 3. Verificar alguns modelos protegidos
python scripts/protect_models.py --verify "Cherry_550000_steps.secure"
```

### Rollback se Necessário:
```bash
# Restaurar originais se algo der errado
rm -f *.secure
cp -r "Cherry_backup/*" "D:/Projeto/Otimizacao/treino_principal/models/Cherry/"
```

## 📞 Suporte

Para problemas com o sistema de proteção:

1. **Teste o demo primeiro**: `python scripts/secure_model_demo.py`
2. **Verificar logs**: Mensagens detalhadas durante proteção/carregamento
3. **Testar com modelo pequeno**: Use modelos de 50k-100k steps primeiro
4. **Verificar dependências**: `cryptography`, `torch`, `stable-baselines3`

## 🔮 Futuras Melhorias

- [ ] Suporte a múltiplas chaves por modelo
- [ ] Sistema de expiração temporal
- [ ] Logs de acesso e auditoria
- [ ] Proteção contra debug/memory dumps
- [ ] Integração com serviços de licenciamento online
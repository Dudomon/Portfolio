# Radio Chart Tool 🎵

Ferramenta gratuita para monitorar charts musicais e gerenciar o acervo da sua rádio.

## ✨ Funcionalidades

- **Charts Brasileiros**: Billboard Brasil, Last.fm Brasil
- **Metadados Automáticos**: Via Spotify API (BPM, gênero, duração)
- **Previews de 30s**: Player integrado para análise
- **Sistema de Aprovação**: Aprove/rejeite músicas facilmente
- **Filtros Avançados**: Por gênero, BPM, título/artista
- **Export CSV**: Para integração com sistemas de rádio
- **100% Gratuito**: Sem custos ou limitações

## 🚀 Instalação Rápida

1. **Clone/baixe o projeto**
```bash
cd radio-chart-tool
```

2. **Instale as dependências**
```bash
pip install -r requirements.txt
```

3. **Configure Spotify (Opcional)**
```bash
# Copie e configure o arquivo de ambiente
cp .env.example .env

# Edite .env com suas credenciais do Spotify
# Crie uma app em: https://developer.spotify.com/
```

4. **Execute a aplicação**
```bash
python app.py
```

5. **Acesse no navegador**
```
http://localhost:5000
```

## 📱 Como Usar

1. **Atualize Charts**: Clique em "Atualizar Agora" na página inicial
2. **Analise Músicas**: Vá para o Dashboard e ouça os previews
3. **Aprove/Rejeite**: Decide quais músicas adicionar ao acervo
4. **Exporte**: Baixe a lista em CSV para usar na rádio

## ⚙️ Configuração do Spotify

Para ter metadados completos (BPM, gênero, previews):

1. Acesse [Spotify for Developers](https://developer.spotify.com/)
2. Crie uma nova aplicação
3. Copie Client ID e Client Secret
4. Configure no arquivo `.env`

**Sem Spotify**: A ferramenta funciona normalmente, mas com metadados limitados.

## 🎯 Fontes de Dados

- **Billboard Brasil**: Top 100 músicas brasileiras
- **Last.fm Brasil**: Trending nacional
- **Spotify**: Metadados e previews (opcional)

## 📊 Export e Integração

O CSV exportado contém:
- Título da música
- Artista
- Gênero
- BPM
- Duração
- URL do preview

Perfeito para importar em sistemas como:
- Winamp/AIMP
- Virtual DJ
- Serato
- Sistemas de automação de rádio

## 🛠️ Requisitos

- Python 3.8+
- Conexão com internet
- Navegador web moderno

## 📝 Licença

Projeto open-source para uso livre em rádios e DJs.

## 🆘 Problemas?

Se encontrar algum erro:
1. Verifique sua conexão com internet
2. Certifique-se que as dependências estão instaladas
3. Verifique se a porta 5000 está livre

---

**Feito com ❤️ para a comunidade de rádios brasileiras**
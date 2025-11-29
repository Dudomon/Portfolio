# Instruções para Subir o Projeto no GitHub

## Passo a Passo

### 1. Criar um repositório no GitHub

1. Acesse github.com e faça login
2. Clique no botão "+" no canto superior direito e selecione "New repository"
3. Configure o repositório:
   - Repository name: jitterbit-order-api (ou outro nome de sua preferência)
   - Description: API de Gerenciamento de Pedidos - Teste Técnico Jitterbit
   - Visibility: Public (para que a Jitterbit possa acessar)
   - NÃO marque "Add a README file" (já temos um README)
   - NÃO adicione .gitignore (já temos um)
4. Clique em "Create repository"

### 2. Inicializar Git e fazer o primeiro commit

Abra o terminal na pasta do projeto e execute os comandos abaixo:

```bash
# Navegar até a pasta do projeto
cd "C:\teste teorico\order-api"

# Inicializar o repositório Git
git init

# Adicionar todos os arquivos
git add .

# Criar o primeiro commit
git commit -m "feat: implementar API de gerenciamento de pedidos

- Implementar CRUD completo de pedidos
- Adicionar validação de dados com Mongoose
- Implementar transformação de dados (mapping)
- Adicionar tratamento de erros robusto
- Implementar paginação na listagem
- Adicionar logs de requisições
- Criar documentação completa no README

Teste técnico para Jitterbit - Professional Services"

# Renomear branch para main (se necessário)
git branch -M main
```

### 3. Conectar ao repositório remoto e fazer push

Substitua SEU_USUARIO pelo seu nome de usuário do GitHub:

```bash
# Adicionar o repositório remoto
git remote add origin https://github.com/SEU_USUARIO/jitterbit-order-api.git

# Fazer o push do código
git push -u origin main
```

### 4. Verificar se tudo está no GitHub

1. Acesse o repositório no navegador
2. Verifique se todos os arquivos foram enviados
3. Verifique se o README.md está sendo exibido corretamente
4. Copie a URL do repositório (exemplo: `https://github.com/SEU_USUARIO/jitterbit-order-api`)

### 5. Adicionar o link do GitHub no arquivo de respostas

Edite o arquivo `respostas_teste_jitterbit.md` e adicione o link do GitHub na seção do Desafio:

```markdown
**Link do GitHub:** https://github.com/SEU_USUARIO/jitterbit-order-api
```

---

## Comandos Git Úteis

### Se precisar fazer alterações depois:

```bash
# Ver status dos arquivos
git status

# Adicionar alterações
git add .

# Fazer commit
git commit -m "Descrição da alteração"

# Enviar para o GitHub
git push
```

### Se o arquivo .env foi commitado por engano:

```bash
# Remover .env do repositório (mas manter localmente)
git rm --cached .env

# Fazer commit da remoção
git commit -m "chore: remover arquivo .env do repositório"

# Fazer push
git push
```

---

## Checklist Final

Antes de enviar o teste, verifique:

- Repositório criado no GitHub com visibilidade pública
- Código completo foi enviado (git push)
- README.md está visível e formatado corretamente
- Arquivo .env NÃO foi commitado (está no .gitignore)
- Link do GitHub foi adicionado no respostas_teste_jitterbit.md
- Testou a API localmente e está funcionando
- Todos os endpoints obrigatórios estão implementados

## Arquivo para Enviar à Jitterbit

Você deve enviar dois itens:

1. Arquivo PDF ou Documento com as respostas
   - Converta o arquivo respostas_teste_jitterbit.md para PDF ou DOCX
   - Certifique-se de que o link do GitHub está incluído

2. Link do Repositório GitHub
   - Exemplo: https://github.com/SEU_USUARIO/jitterbit-order-api

## Dica Final

Adicione um arquivo .npmrc ou .nvmrc se quiser especificar a versão do Node.js:

```bash
# Criar arquivo .nvmrc
echo "18.17.0" > .nvmrc
```

Isso ajuda a garantir que todos usem a mesma versão do Node.js.

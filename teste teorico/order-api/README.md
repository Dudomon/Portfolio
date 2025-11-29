# API de Gerenciamento de Pedidos - Jitterbit

API REST desenvolvida em Node.js para gerenciamento de pedidos, criada como parte do teste técnico para a posição de Professional Services na Jitterbit.

## Autor

**Eduardo Peiter**
- Telefone: 49 988270076
- LinkedIn: [eduardo-lara-peiter-7347a935a](https://www.linkedin.com/in/eduardo-lara-peiter-7347a935a/)

## Tecnologias Utilizadas

- **Node.js** - Runtime JavaScript
- **Express** - Framework web
- **MongoDB** - Banco de dados NoSQL
- **Mongoose** - ODM para MongoDB
- **dotenv** - Gerenciamento de variáveis de ambiente
- **cors** - Middleware para habilitar CORS

## Funcionalidades

A API implementa um CRUD completo para gerenciamento de pedidos com os seguintes endpoints:

### Endpoints Obrigatórios

- `POST /order` - Criar um novo pedido
- `GET /order/:numeroPedido` - Obter dados de um pedido específico

### Endpoints Opcionais

- `GET /order/list` - Listar todos os pedidos (com paginação)
- `PUT /order/:numeroPedido` - Atualizar um pedido existente
- `DELETE /order/:numeroPedido` - Deletar um pedido

## Estrutura do Projeto

```
order-api/
├── src/
│   ├── config/
│   │   └── database.js          # Configuração do MongoDB
│   ├── controllers/
│   │   └── orderController.js   # Lógica de negócio dos pedidos
│   ├── middleware/
│   │   ├── errorHandler.js      # Tratamento centralizado de erros
│   │   ├── notFound.js          # Middleware para rotas não encontradas
│   │   └── requestLogger.js     # Log de requisições
│   ├── models/
│   │   └── Order.js             # Model do pedido (Mongoose)
│   └── routes/
│       └── orderRoutes.js       # Rotas da API
├── .env                          # Variáveis de ambiente (não commitado)
├── .env.example                  # Exemplo de variáveis de ambiente
├── .gitignore                    # Arquivos ignorados pelo Git
├── package.json                  # Dependências e scripts
├── server.js                     # Arquivo principal da aplicação
└── README.md                     # Documentação (este arquivo)
```

## Instalação

### Pré-requisitos

- Node.js >= 16.0.0
- MongoDB instalado e rodando localmente OU conta no MongoDB Atlas

### Passo a Passo

1. **Clone o repositório**
```bash
git clone <url-do-repositorio>
cd order-api
```

2. **Instale as dependências**
```bash
npm install
```

3. **Configure as variáveis de ambiente**

Copie o arquivo `.env.example` para `.env`:
```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas configurações:
```env
PORT=3000
NODE_ENV=development
MONGODB_URI=mongodb://localhost:27017/order-api
```

Para usar MongoDB Atlas, substitua a `MONGODB_URI`:
```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/order-api?retryWrites=true&w=majority
```

4. **Inicie o servidor**

Modo desenvolvimento (com auto-reload):
```bash
npm run dev
```

Modo produção:
```bash
npm start
```

5. **Verifique se a API está rodando**

Acesse: http://localhost:3000

Você deve ver uma mensagem de boas-vindas com a lista de endpoints disponíveis.

## Uso da API

### Base URL
```
http://localhost:3000
```

### 1. Criar um Novo Pedido

**Endpoint:** `POST /order`

**Request Body:**
```json
{
  "numeroPedido": "v10089015vdb-01",
  "valorTotal": 10000,
  "dataCriacao": "2023-07-19T12:24:11.5299601+00:00",
  "items": [
    {
      "idItem": "2434",
      "quantidadeItem": 1,
      "valorItem": 1000
    }
  ]
}
```

**Exemplo com cURL:**
```bash
curl --location 'http://localhost:3000/order' \
--header 'Content-Type: application/json' \
--data '{
  "numeroPedido": "v10089015vdb-01",
  "valorTotal": 10000,
  "dataCriacao": "2023-07-19T12:24:11.5299601+00:00",
  "items": [
    {
      "idItem": "2434",
      "quantidadeItem": 1,
      "valorItem": 1000
    }
  ]
}'
```

**Response (201 Created):**
```json
{
  "message": "Pedido criado com sucesso",
  "data": {
    "numeroPedido": "v10089015vdb-01",
    "valorTotal": 10000,
    "dataCriacao": "2023-07-19T12:24:11.529Z",
    "items": [
      {
        "idItem": "2434",
        "quantidadeItem": 1,
        "valorItem": 1000
      }
    ]
  }
}
```

---

### 2. Obter um Pedido Específico

**Endpoint:** `GET /order/:numeroPedido`

**Exemplo com cURL:**
```bash
curl --location 'http://localhost:3000/order/v10089015vdb-01'
```

**Response (200 OK):**
```json
{
  "message": "Pedido encontrado",
  "data": {
    "numeroPedido": "v10089015vdb-01",
    "valorTotal": 10000,
    "dataCriacao": "2023-07-19T12:24:11.529Z",
    "items": [
      {
        "idItem": "2434",
        "quantidadeItem": 1,
        "valorItem": 1000
      }
    ]
  }
}
```

---

### 3. Listar Todos os Pedidos

**Endpoint:** `GET /order/list`

**Query Parameters (opcionais):**
- `page` - Número da página (padrão: 1)
- `limit` - Itens por página (padrão: 10)

**Exemplo com cURL:**
```bash
curl --location 'http://localhost:3000/order/list?page=1&limit=10'
```

**Response (200 OK):**
```json
{
  "message": "Pedidos listados com sucesso",
  "data": [
    {
      "numeroPedido": "v10089015vdb-01",
      "valorTotal": 10000,
      "dataCriacao": "2023-07-19T12:24:11.529Z",
      "items": [
        {
          "idItem": "2434",
          "quantidadeItem": 1,
          "valorItem": 1000
        }
      ]
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 1,
    "totalPages": 1
  }
}
```

---

### 4. Atualizar um Pedido

**Endpoint:** `PUT /order/:numeroPedido`

**Request Body:**
```json
{
  "numeroPedido": "v10089015vdb-01",
  "valorTotal": 15000,
  "dataCriacao": "2023-07-19T12:24:11.5299601+00:00",
  "items": [
    {
      "idItem": "2434",
      "quantidadeItem": 2,
      "valorItem": 1500
    }
  ]
}
```

**Exemplo com cURL:**
```bash
curl --location --request PUT 'http://localhost:3000/order/v10089015vdb-01' \
--header 'Content-Type: application/json' \
--data '{
  "numeroPedido": "v10089015vdb-01",
  "valorTotal": 15000,
  "dataCriacao": "2023-07-19T12:24:11.5299601+00:00",
  "items": [
    {
      "idItem": "2434",
      "quantidadeItem": 2,
      "valorItem": 1500
    }
  ]
}'
```

**Response (200 OK):**
```json
{
  "message": "Pedido atualizado com sucesso",
  "data": {
    "numeroPedido": "v10089015vdb-01",
    "valorTotal": 15000,
    "dataCriacao": "2023-07-19T12:24:11.529Z",
    "items": [
      {
        "idItem": "2434",
        "quantidadeItem": 2,
        "valorItem": 1500
      }
    ]
  }
}
```

---

### 5. Deletar um Pedido

**Endpoint:** `DELETE /order/:numeroPedido`

**Exemplo com cURL:**
```bash
curl --location --request DELETE 'http://localhost:3000/order/v10089015vdb-01'
```

**Response (200 OK):**
```json
{
  "message": "Pedido deletado com sucesso",
  "data": {
    "numeroPedido": "v10089015vdb-01"
  }
}
```

---

## Mapeamento de Dados

A API realiza transformação automática dos dados entre o formato de entrada/saída e o formato do banco de dados:

### Formato de Entrada/Saída (API)
```json
{
  "numeroPedido": "v10089015vdb-01",
  "valorTotal": 10000,
  "dataCriacao": "2023-07-19T12:24:11.5299601+00:00",
  "items": [
    {
      "idItem": "2434",
      "quantidadeItem": 1,
      "valorItem": 1000
    }
  ]
}
```

### Formato do Banco de Dados (MongoDB)
```json
{
  "orderId": "v10089015vdb-01",
  "value": 10000,
  "creationDate": "2023-07-19T12:24:11.529Z",
  "items": [
    {
      "productId": 2434,
      "quantity": 1,
      "price": 1000
    }
  ]
}
```

## Tratamento de Erros

A API possui tratamento robusto de erros com mensagens claras:

### Exemplo: Pedido não encontrado (404)
```json
{
  "error": "Pedido não encontrado",
  "message": "O pedido v10089999vdb não existe"
}
```

### Exemplo: Dados inválidos (400)
```json
{
  "error": "Erro de validação",
  "message": "Validation failed",
  "details": [
    "numeroPedido, valorTotal, dataCriacao e items são obrigatórios"
  ]
}
```

### Exemplo: Pedido duplicado (409)
```json
{
  "error": "Pedido já existe",
  "message": "O pedido v10089015vdb-01 já está cadastrado"
}
```

### Exemplo: Rota não encontrada (404)
```json
{
  "error": "Rota não encontrada",
  "message": "A rota GET /invalid não existe",
  "availableRoutes": {
    "POST /order": "Criar novo pedido",
    "GET /order/:numeroPedido": "Obter pedido específico",
    "GET /order/list": "Listar todos os pedidos",
    "PUT /order/:numeroPedido": "Atualizar pedido",
    "DELETE /order/:numeroPedido": "Deletar pedido"
  }
}
```

## Recursos Implementados

### Obrigatórios
- Endpoint para criar pedido
- Endpoint para obter pedido por número
- Conexão com MongoDB
- Mapeamento de dados (transformação de campos)
- Validação de dados
- Tratamento de erros

### Opcionais
- Endpoint para listar todos os pedidos
- Endpoint para atualizar pedido
- Endpoint para deletar pedido
- Paginação na listagem
- Logs de requisições
- Middleware de validação
- Documentação completa
- Código organizado e comentado
- Mensagens de erro compreensíveis
- Respostas HTTP adequadas

### Não Implementados (Sugestões Futuras)
- Autenticação JWT
- Documentação com Swagger
- Testes automatizados
- Docker/Docker Compose
- CI/CD Pipeline

## Códigos de Status HTTP

A API utiliza os seguintes códigos de status:

- `200 OK` - Requisição bem-sucedida
- `201 Created` - Recurso criado com sucesso
- `400 Bad Request` - Dados inválidos ou incompletos
- `404 Not Found` - Recurso não encontrado
- `409 Conflict` - Conflito (ex: pedido duplicado)
- `500 Internal Server Error` - Erro interno do servidor

## Boas Práticas Implementadas

1. **Separação de responsabilidades** - Controllers, Models, Routes, Middleware
2. **Validação de dados** - Mongoose schema validation
3. **Tratamento de erros centralizado** - Error handler middleware
4. **Logging** - Request logger middleware
5. **Variáveis de ambiente** - Configuração via .env
6. **Código limpo e comentado** - JSDoc comments
7. **RESTful API** - Padrões REST seguidos
8. **Segurança** - CORS habilitado, validação de entrada
9. **Escalabilidade** - Estrutura modular e organizada

## Testando a API

Você pode testar a API usando:

### 1. cURL (linha de comando)
Exemplos fornecidos na seção "Uso da API" acima.

### 2. Postman
Importe os exemplos de cURL no Postman ou crie uma nova collection.

### 3. Insomnia
Similar ao Postman, copie os exemplos de requisição.

### 4. Thunder Client (VS Code Extension)
Extensão do VS Code para testar APIs.

## Troubleshooting

### Erro: "Cannot connect to MongoDB"
- Verifique se o MongoDB está rodando: `mongod --version`
- Verifique a string de conexão no arquivo `.env`
- Se estiver usando MongoDB Atlas, verifique se seu IP está na whitelist

### Erro: "Port 3000 is already in use"
- Altere a porta no arquivo `.env`: `PORT=3001`
- Ou finalize o processo que está usando a porta 3000

### Erro: "Module not found"
- Execute `npm install` para instalar todas as dependências

## Licença

MIT

---

**Desenvolvido por Eduardo Peiter como parte do teste técnico para Jitterbit - Professional Services**

Data: 29/11/2025

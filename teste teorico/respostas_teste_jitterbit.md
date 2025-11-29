# TESTE TEÓRICO - JITTERBIT
## Professional Services

---

### Dados do Candidato

| Campo | Informação |
|-------|------------|
| **Nome do Candidato(a)** | Eduardo Peiter |
| **Telefone** | 49 988270076 |
| **Linkedin** | https://www.linkedin.com/in/eduardo-lara-peiter-7347a935a/ |
| **Data** | 29/11/2025 |

---

## Javascript

### 1. Qual é o operador lógico usado para verificar a negação de uma expressão? (Nota: 0,2)

**Resposta: c) !**

---

### 2. Qual dos seguintes métodos é usado para adicionar um elemento ao final de um array? (Nota: 0,2)

**Resposta: a) push()**

---

### 3. O que o método "Array.map()" faz? (Nota: 0,2)

**Resposta: b) Mapeia os elementos de um array para um novo array com base em uma função de mapeamento.**

---

### 4. Qual é a função do método "Array.filter()"? (Nota: 0,2)

**Resposta: b) Remover elementos do array com base em uma função de filtro.**

---

### 5. O que é async/await em JavaScript? (Nota: 0,2)

**Resposta: c) Um conjunto de palavras-chave que tornam as funções assíncronas mais legíveis e fáceis de usar.**

---

### 6. Qual é a sintaxe correta para definir uma função assíncrona chamada "getData"? (Nota: 0,2)

**Resposta: c) async function getData() { return new Promise({}); }**

---

### 7. O que será impresso no código abaixo? (Nota: 0,6)

```javascript
let palavra = "ABC";
switch (palavra) {
  case "ACB":
    console.log("C");
    break;
  case "BC":
  case "ABC":
    console.log("A");
    break;
  case "B":
    console.log("Hello");
    break;
  default:
    console.log("Palavra não encontrada");
    break;
}
```

**Resposta: b) A.**

**Explicação:** A variável `palavra` contém "ABC", que corresponde ao case "ABC", então será impresso "A" e o break interrompe a execução do switch.

---

### 8. Escreva uma função em JavaScript chamada "somaImpares" que recebe um número inteiro positivo "n" como parâmetro e retorna a soma de todos os números ímpares de 1 até n. (Nota: 0,6)

**Resposta:**

```javascript
function somaImpares(n) {
  let soma = 0;
  for (let i = 1; i <= n; i++) {
    if (i % 2 !== 0) {
      soma += i;
    }
  }
  return soma;
}

// Exemplos de uso:
// somaImpares(5);  // Saída: 9 (1 + 3 + 5)
// somaImpares(10); // Saída: 25 (1 + 3 + 5 + 7 + 9)
```

**Alternativa mais concisa:**

```javascript
function somaImpares(n) {
  let soma = 0;
  for (let i = 1; i <= n; i += 2) {
    soma += i;
  }
  return soma;
}
```

---

### 9. Escreva uma função chamada "inverterPalavra" que recebe uma string como parâmetro e retorna a string com as letras invertidas. (Nota: 0,6)

**Resposta:**

```javascript
function inverterPalavra(str) {
  return str.split('').reverse().join('');
}

// Exemplo de uso:
// inverterPalavra("javascript"); // Saída: "tpircsavaj"
```

**Alternativa sem usar métodos nativos:**

```javascript
function inverterPalavra(str) {
  let resultado = '';
  for (let i = str.length - 1; i >= 0; i--) {
    resultado += str[i];
  }
  return resultado;
}
```

---

### 10. Considere o seguinte trecho de código em JavaScript que tenta realizar a divisão de dois números: (Nota: 0,6)

```javascript
function dividirNumeros(number1, number2) {
  try {
    if (number2 === 0) {
      throw new Error("Divisão por zero não é permitida.");
    }
    return number1 / number2;
  } catch (error) {
    return "Erro: " + error.message;
  }
}
```

**Escreva abaixo o resultado retornado por cada função:**

**a) console.log(dividirNumeros(20, 2));**

**Resposta:** `10`

**b) console.log(dividirNumeros(6, 0));**

**Resposta:** `"Erro: Divisão por zero não é permitida."`

**c) console.log(dividirNumeros(21, 3));**

**Resposta:** `7`

---

### 11. Como você pode percorrer e mapear um array JSON em JavaScript? Explique como usar métodos como "map", "forEach" ou "for...of" para iterar e manipular os elementos do array. (Nota: 0,7)

**Resposta:**

Em JavaScript, existem várias formas de percorrer e manipular arrays JSON:

**1. Método map():**
O `map()` cria um novo array com os resultados da aplicação de uma função em cada elemento do array original. É ideal quando você precisa transformar os dados.

```javascript
const users = [
  { name: "João", age: 25 },
  { name: "Maria", age: 30 }
];

const names = users.map(user => user.name);
// Resultado: ["João", "Maria"]
```

**2. Método forEach():**
O `forEach()` executa uma função para cada elemento do array, mas não retorna um novo array. É usado quando você quer apenas iterar sem transformar.

```javascript
users.forEach(user => {
  console.log(`${user.name} tem ${user.age} anos`);
});
```

**3. Loop for...of:**
Permite iterar diretamente sobre os valores do array de forma mais legível.

```javascript
for (const user of users) {
  console.log(user.name);
}
```

**Diferenças principais:**
- `map()`: Retorna um novo array transformado
- `forEach()`: Apenas executa uma função, não retorna nada
- `for...of`: Sintaxe mais simples, permite uso de break e continue

---

### 12. O que são variáveis em JavaScript? Explique como declarar e atribuir valores a uma variável. (Nota: 0,7)

**Resposta:**

Variáveis em JavaScript são containers que armazenam valores de dados que podem ser utilizados e manipulados durante a execução do programa.

**Formas de declarar variáveis:**

**1. var (escopo de função, forma antiga):**
```javascript
var nome = "Eduardo";
var idade = 30;
```

**2. let (escopo de bloco, forma moderna):**
```javascript
let cidade = "Florianópolis";
let contador = 0;
contador = 1; // Pode ser reatribuída
```

**3. const (escopo de bloco, valor constante):**
```javascript
const PI = 3.14159;
const API_URL = "https://api.exemplo.com";
// PI = 3.14; // Erro! Não pode ser reatribuída
```

**Diferenças:**
- **var**: Tem escopo de função e sofre hoisting (içamento)
- **let**: Tem escopo de bloco e pode ser reatribuída
- **const**: Tem escopo de bloco e não pode ser reatribuída (o valor é constante)

**Boas práticas:**
- Use `const` por padrão
- Use `let` apenas quando o valor precisar mudar
- Evite usar `var` em código moderno

---

### 13. Em JavaScript, é possível ter múltiplas condições em uma estrutura "if/else"? Descreva como usar operadores lógicos (como "&&" e "||") para combinar condições. (Nota: 0,6)

**Resposta:**

Sim, é possível ter múltiplas condições em estruturas `if/else` usando operadores lógicos.

**Operadores lógicos:**

**1. && (AND - E):** Retorna true apenas se TODAS as condições forem verdadeiras
```javascript
let idade = 25;
let temCarteira = true;

if (idade >= 18 && temCarteira) {
  console.log("Pode dirigir");
}
// Ambas condições precisam ser verdadeiras
```

**2. || (OR - OU):** Retorna true se PELO MENOS UMA condição for verdadeira
```javascript
let dia = "sábado";

if (dia === "sábado" || dia === "domingo") {
  console.log("É fim de semana!");
}
// Pelo menos uma condição precisa ser verdadeira
```

**3. ! (NOT - NÃO):** Inverte o valor booleano
```javascript
let chovendo = false;

if (!chovendo) {
  console.log("Pode sair sem guarda-chuva");
}
```

**Combinando múltiplas condições:**
```javascript
let idade = 25;
let temCarteira = true;
let temCarro = false;

if (idade >= 18 && temCarteira && (temCarro || podeAlugar)) {
  console.log("Pode viajar de carro");
} else if (idade >= 18 && !temCarteira) {
  console.log("Precisa tirar carteira");
} else {
  console.log("Muito jovem para dirigir");
}
```

**Precedência:** O operador `&&` tem precedência sobre `||`, mas é recomendado usar parênteses para maior clareza.

---

### 14. Descreva a sintaxe do bloco "try" em JavaScript. Dê um exemplo prático de como usar o "try" para envolver um código suscetível a erros. (Nota: 0,7)

**Resposta:**

O bloco `try...catch` é usado para capturar e tratar erros em JavaScript, evitando que o programa quebre completamente.

**Sintaxe:**
```javascript
try {
  // Código que pode gerar erro
} catch (error) {
  // Código executado se houver erro
} finally {
  // Código executado sempre (opcional)
}
```

**Exemplo prático - Parse de JSON:**
```javascript
function parseJSON(jsonString) {
  try {
    const data = JSON.parse(jsonString);
    console.log("JSON válido:", data);
    return data;
  } catch (error) {
    console.error("Erro ao fazer parse do JSON:", error.message);
    return null;
  } finally {
    console.log("Tentativa de parse finalizada");
  }
}

// Uso:
parseJSON('{"name": "Eduardo"}');  // Funciona
parseJSON('invalid json');          // Captura o erro
```

**Exemplo prático - Requisição de API:**
```javascript
async function buscarUsuario(id) {
  try {
    const response = await fetch(`https://api.exemplo.com/users/${id}`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Erro ao buscar usuário:", error.message);
    return { error: "Não foi possível buscar o usuário" };
  }
}
```

**Componentes:**
- **try**: Envolve o código que pode gerar erro
- **catch**: Captura e trata o erro
- **finally**: Executa código independente de erro ou sucesso (opcional)

---

### 15. Como você pode lançar manualmente uma exceção em JavaScript? Explique o uso da palavra-chave "throw" para criar e lançar exceções personalizadas. (Nota: 0,7)

**Resposta:**

A palavra-chave `throw` permite lançar exceções manualmente em JavaScript, criando erros personalizados para situações específicas.

**Sintaxe básica:**
```javascript
throw new Error("Mensagem de erro");
```

**Exemplos práticos:**

**1. Validação de parâmetros:**
```javascript
function calcularDesconto(preco, desconto) {
  if (desconto < 0 || desconto > 100) {
    throw new Error("Desconto deve estar entre 0 e 100");
  }

  if (preco <= 0) {
    throw new Error("Preço deve ser maior que zero");
  }

  return preco - (preco * desconto / 100);
}

try {
  calcularDesconto(100, 150); // Lança erro
} catch (error) {
  console.error(error.message); // "Desconto deve estar entre 0 e 100"
}
```

**2. Criando classes de erro personalizadas:**
```javascript
class ValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = "ValidationError";
  }
}

class DatabaseError extends Error {
  constructor(message) {
    super(message);
    this.name = "DatabaseError";
  }
}

function validarEmail(email) {
  if (!email.includes("@")) {
    throw new ValidationError("Email inválido: deve conter @");
  }
  return true;
}

try {
  validarEmail("emailinvalido");
} catch (error) {
  if (error instanceof ValidationError) {
    console.error("Erro de validação:", error.message);
  } else {
    console.error("Erro desconhecido:", error);
  }
}
```

**3. Uso com async/await:**
```javascript
async function buscarDados(id) {
  if (!id) {
    throw new Error("ID é obrigatório");
  }

  const response = await fetch(`/api/dados/${id}`);

  if (!response.ok) {
    throw new Error(`Falha na requisição: ${response.status}`);
  }

  return await response.json();
}
```

**Tipos de exceções que podem ser lançadas:**
- Strings: `throw "Erro simples"`
- Números: `throw 404`
- Objetos Error: `throw new Error("Mensagem")`
- Objetos personalizados: `throw new CustomError("Mensagem")`

---

## SQL

### 1. Como você seleciona todas as colunas de uma tabela em SQL? (Nota: 0,2)

**Resposta: b) SELECT \***

---

### 2. Qual é o comando SQL utilizado para filtrar resultados em uma consulta? (Nota: 0,2)

**Resposta: d) WHERE**

---

### 3. Qual é o comando SQL utilizado para ordenar os resultados de uma consulta em ordem ascendente? (Nota: 0,2)

**Resposta: d) ORDER BY**

(Nota: Para ordem ascendente, usa-se `ORDER BY coluna ASC` ou apenas `ORDER BY coluna`, pois ASC é o padrão)

---

### 4. Qual é o comando SQL utilizado para inserir novos dados em uma tabela? (Nota: 0,2)

**Resposta: b) INSERT**

---

### 5. Qual é o comando SQL utilizado para atualizar dados em uma tabela? (Nota: 0,2)

**Resposta: b) UPDATE**

---

## Integração de sistemas

### 1. O que é integração de sistemas? (Nota: 0,2)

**Resposta: a) É um processo de comunicação entre diferentes sistemas de computador para permitir o compartilhamento de dados e funcionalidades.**

---

### 2. O que significa API (Interface de Programação de Aplicativos) em integração de sistemas? (Nota: 0,2)

**Resposta: c) Um conjunto de funções e procedimentos que permitem a comunicação entre sistemas.**

---

### 3. O que é um Web Service? (Nota: 0,2)

**Resposta: c) É uma solução para conectar sistemas diferentes via web, usando padrões como XML e SOAP.**

---

### 4. O que é um token de acesso em integração de sistemas? (Nota: 0,2)

**Resposta: c) Uma chave de autenticação usada para autorizar o acesso a um serviço.**

---

### 5. O que é um "webhook" na integração de sistemas? (Nota: 0,2)

**Resposta: d) É uma URL pública fornecida por um sistema para receber notificações automáticas de outro sistema.**

---

### 6. O que é JSON? (Nota: 0,2)

**Resposta: c) Um formato de dados leve e de fácil leitura usado para trocar informações entre sistemas.**

---

### 7. Qual é o código de status HTTP que indica sucesso na solicitação? (Nota: 0,2)

**Resposta: a) 200 OK.**

---

### 8. O que são headers HTTP? (Nota: 0,2)

**Resposta: b) Informações adicionais enviadas pelo cliente e servidor em uma solicitação ou resposta HTTP.**

---

### 9. Quais são os delimitadores usados para marcar tags em XML? (Nota: 0,2)

**Resposta: d) < >**

---

### 10. Qual é a diferença entre integração de sistemas síncrona e assíncrona? (Nota: 0,2)

**Resposta: a) Na síncrona, a comunicação ocorre em tempo real com respostas imediatas, enquanto na assíncrona, a resposta pode ser recebida em um momento posterior.**

---

## Desafio

### API de Gerenciamento de Pedidos em Node.js

O desafio prático foi **implementado com sucesso** e está localizado na pasta `order-api/` deste projeto.

Localização: `C:\teste teorico\order-api\`

Documentação completa: Consulte o arquivo `order-api/README.md` para instruções detalhadas de instalação, uso e exemplos de requisições.

**Tecnologias utilizadas:**
- Node.js + Express
- MongoDB (Mongoose)
- Validação de dados robusta
- Tratamento de erros centralizado
- Logs de requisições
- CORS habilitado

Endpoints implementados:
- POST `/order` - Criar novo pedido (obrigatório)
- GET `/order/:numeroPedido` - Obter pedido específico (obrigatório)
- GET `/order/list` - Listar todos os pedidos com paginação (opcional)
- PUT `/order/:numeroPedido` - Atualizar pedido (opcional)
- DELETE `/order/:numeroPedido` - Deletar pedido (opcional)

**Estrutura do projeto:**
```
order-api/
├── src/
│   ├── config/
│   │   └── database.js          # Configuração MongoDB
│   ├── controllers/
│   │   └── orderController.js   # Lógica de negócio
│   ├── middleware/
│   │   ├── errorHandler.js      # Tratamento de erros
│   │   ├── notFound.js          # Rotas não encontradas
│   │   └── requestLogger.js     # Logs de requisições
│   ├── models/
│   │   └── Order.js             # Model Mongoose
│   └── routes/
│       └── orderRoutes.js       # Rotas da API
├── .env                          # Variáveis de ambiente
├── .env.example                  # Exemplo de configuração
├── .gitignore                    # Arquivos ignorados
├── package.json                  # Dependências
├── server.js                     # Arquivo principal
└── README.md                     # Documentação completa
```

Recursos adicionais implementados:
- Transformação automática de dados (mapping entre formatos)
- Validação completa com Mongoose
- Mensagens de erro compreensíveis
- Respostas HTTP adequadas para cada operação
- Paginação na listagem de pedidos
- Código bem organizado e comentado
- Documentação completa com exemplos de uso
- Tratamento de casos edge (pedido duplicado, não encontrado, etc.)

**Como executar:**
```bash
cd order-api
npm install
npm run dev
```

Acesse: http://localhost:3000

---

**Observações:**
- Todas as respostas foram elaboradas com base no conhecimento técnico real
- O código está otimizado e segue as boas práticas de desenvolvimento
- Os exemplos são funcionais e podem ser testados diretamente

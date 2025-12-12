# Fleet Management System

[English](#english) | [Portugues](#portugues)

---

## English

### Overview

Enterprise-grade Fleet Management System built with ASP.NET Core 8 MVC for transportation and logistics operations. Designed for bulk hauling companies to manage vehicles, drivers, routes, shipments, and maintenance schedules with full audit trails and real-time tracking capabilities.

### Tech Stack

- **Framework**: ASP.NET Core 8 MVC / Web API
- **Language**: C# 12
- **Architecture**: Clean Architecture (Domain, Application, Infrastructure, Presentation)
- **ORM**: Entity Framework Core 8
- **Database**: SQL Server 2022
- **Authentication**: ASP.NET Core Identity + JWT
- **Validation**: FluentValidation
- **Mapping**: AutoMapper
- **Logging**: Serilog
- **Testing**: xUnit, Moq, FluentAssertions, AutoFixture
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Documentation**: Swagger/OpenAPI

### Key Features

#### 1. Vehicle Management
- Complete CRUD operations for fleet vehicles
- Vehicle status tracking (Available, In Use, Under Maintenance, Out of Service)
- Support for multiple vehicle types (Trucks, Vans, Tractor Trailers, Tankers, etc.)
- Mileage tracking and fuel consumption monitoring
- Document management (registration, insurance, inspection certificates)
- Maintenance alerts for expiring documents

#### 2. Driver Management
- Driver profiles with license validation
- License category tracking (A, B, C, D, E)
- Driver status management (Active, On Leave, Suspended)
- Emergency contact information
- Document expiration tracking and alerts
- Driver assignment to vehicles and routes

#### 3. Route Planning and Tracking
- Route creation with origin and destination coordinates
- Google Maps API integration for GPS tracking
- Real-time route status updates (Scheduled, In Progress, Completed, Cancelled)
- Checkpoint tracking along routes
- Estimated vs actual distance and costs comparison
- Fuel and toll cost tracking

#### 4. Shipment Management
- Multiple shipments per route
- Cargo type classification (General, Perishable, Hazardous, Bulk, etc.)
- Weight and volume tracking
- Customer information management
- Delivery confirmation with signature capture
- Shipment status tracking throughout lifecycle

#### 5. Maintenance Management
- Preventive and corrective maintenance scheduling
- Maintenance history per vehicle
- Cost tracking for maintenance operations
- Service provider management
- Parts replacement tracking
- Next maintenance date and mileage predictions

### Architecture

```
FleetManagement-ASPNET-Core/
├── src/
│   ├── FleetManagement.Web/              # ASP.NET Core MVC + API Controllers
│   ├── FleetManagement.Application/      # Business Logic, Services, DTOs, Validators
│   ├── FleetManagement.Domain/           # Entities, Interfaces, Domain Logic
│   ├── FleetManagement.Infrastructure/   # EF Core, Repositories, Data Access
│   └── FleetManagement.Shared/           # Shared Constants, Helpers, Extensions
├── tests/
│   ├── FleetManagement.UnitTests/        # xUnit Unit Tests
│   └── FleetManagement.IntegrationTests/ # Integration Tests
└── docs/                                 # Architecture documentation
```

### Design Patterns

- **Repository Pattern**: Abstraction over data access logic
- **Unit of Work Pattern**: Transaction management across repositories
- **Dependency Injection**: Constructor injection throughout application
- **CQRS Principles**: Separation of read and write operations
- **Service Layer Pattern**: Business logic encapsulation
- **DTO Pattern**: Data transfer between layers
- **Specification Pattern**: Reusable query logic

### API Endpoints

#### Vehicles
- `GET /api/vehicles` - Get all vehicles
- `GET /api/vehicles/{id}` - Get vehicle by ID
- `GET /api/vehicles/available` - Get available vehicles
- `GET /api/vehicles/status/{status}` - Get vehicles by status
- `GET /api/vehicles/plate/{plateNumber}` - Get vehicle by plate number
- `GET /api/vehicles/maintenance-needed` - Get vehicles needing maintenance
- `POST /api/vehicles` - Create new vehicle
- `PUT /api/vehicles/{id}` - Update vehicle
- `DELETE /api/vehicles/{id}` - Delete vehicle (soft delete)

#### Drivers
- `GET /api/drivers` - Get all drivers
- `GET /api/drivers/{id}` - Get driver by ID
- `GET /api/drivers/active` - Get active drivers
- `POST /api/drivers` - Create new driver
- `PUT /api/drivers/{id}` - Update driver
- `DELETE /api/drivers/{id}` - Delete driver

#### Routes
- `GET /api/routes` - Get all routes
- `GET /api/routes/{id}` - Get route with full details
- `GET /api/routes/active` - Get active routes
- `GET /api/routes/status/{status}` - Get routes by status
- `POST /api/routes` - Create new route
- `PUT /api/routes/{id}/status` - Update route status
- `DELETE /api/routes/{id}` - Cancel route

### Getting Started

#### Prerequisites
- .NET 8 SDK
- SQL Server 2022 or Docker
- Visual Studio 2022 or Rider

#### Running Locally

1. Clone the repository
```bash
git clone https://github.com/yourusername/FleetManagement-ASPNET-Core.git
cd FleetManagement-ASPNET-Core
```

2. Update database connection string in `appsettings.json`
```json
"ConnectionStrings": {
  "DefaultConnection": "Server=localhost;Database=FleetManagementDb;Trusted_Connection=True;"
}
```

3. Run database migrations
```bash
dotnet ef database update --project src/FleetManagement.Infrastructure --startup-project src/FleetManagement.Web
```

4. Run the application
```bash
dotnet run --project src/FleetManagement.Web
```

5. Access Swagger UI at `https://localhost:5001`

#### Running with Docker

```bash
docker-compose up -d
```

Access the API at `http://localhost:5000` and Swagger UI at `http://localhost:5000/swagger`

### Testing

Run all tests:
```bash
dotnet test
```

Run tests with coverage:
```bash
dotnet test /p:CollectCoverage=true /p:CoverageReportFormat=opencover
```

### Security Features

- **JWT Authentication**: Secure token-based authentication
- **Role-Based Authorization**: Granular access control
- **Password Policies**: Enforced strong password requirements
- **SQL Injection Protection**: Parameterized queries via EF Core
- **HTTPS Enforcement**: TLS 1.2+ required
- **CORS Configuration**: Configurable cross-origin policies
- **Soft Deletes**: Data preservation with audit trails
- **Logging**: Comprehensive request/response logging with Serilog

### Performance Optimizations

- **Async/Await**: Non-blocking I/O operations throughout
- **Connection Pooling**: Efficient database connection management
- **Retry Logic**: Resilient database operations with exponential backoff
- **Query Optimization**: Eager loading to prevent N+1 queries
- **Indexing**: Strategic database indexes on frequently queried fields
- **Caching Ready**: Infrastructure prepared for Redis/Memory cache

### CI/CD Pipeline

GitHub Actions workflow includes:
- Automated builds on push/PR
- Unit and integration test execution
- Code coverage reporting
- Docker image building
- Security vulnerability scanning with Trivy
- Artifact generation for deployments

### Future Enhancements

- Real-time GPS tracking with SignalR
- Advanced reporting and analytics dashboard
- Mobile app for drivers (Xamarin/MAUI)
- Integration with telematics devices
- Route optimization algorithms
- Predictive maintenance using ML models
- Multi-tenant support for fleet management companies

### License

This project is for portfolio demonstration purposes.

### Contact

Eduardo Lara Peiter
- Email: dudu.peiter@gmail.com
- GitHub: [@Dudomon](https://github.com/Dudomon)

---

## Portugues

### Visao Geral

Sistema de Gestao de Frotas de nivel empresarial construido com ASP.NET Core 8 MVC para operacoes de transporte e logistica. Projetado para empresas de transporte de cargas para gerenciar veiculos, motoristas, rotas, carregamentos e cronogramas de manutencao com trilhas de auditoria completas e recursos de rastreamento em tempo real.

### Stack Tecnologica

- **Framework**: ASP.NET Core 8 MVC / Web API
- **Linguagem**: C# 12
- **Arquitetura**: Clean Architecture (Domain, Application, Infrastructure, Presentation)
- **ORM**: Entity Framework Core 8
- **Banco de Dados**: SQL Server 2022
- **Autenticacao**: ASP.NET Core Identity + JWT
- **Validacao**: FluentValidation
- **Mapeamento**: AutoMapper
- **Logging**: Serilog
- **Testes**: xUnit, Moq, FluentAssertions, AutoFixture
- **Containerizacao**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Documentacao**: Swagger/OpenAPI

### Funcionalidades Principais

#### 1. Gestao de Veiculos
- Operacoes CRUD completas para veiculos da frota
- Rastreamento de status de veiculos (Disponivel, Em Uso, Em Manutencao, Fora de Servico)
- Suporte para multiplos tipos de veiculos (Caminhoes, Vans, Carretas, Tanques, etc.)
- Rastreamento de quilometragem e monitoramento de consumo de combustivel
- Gestao de documentos (registro, seguro, certificados de inspecao)
- Alertas de manutencao para documentos expirando

#### 2. Gestao de Motoristas
- Perfis de motoristas com validacao de licenca
- Rastreamento de categoria de licenca (A, B, C, D, E)
- Gestao de status de motorista (Ativo, De Ferias, Suspenso)
- Informacoes de contato de emergencia
- Rastreamento de expiracao de documentos e alertas
- Atribuicao de motoristas a veiculos e rotas

#### 3. Planejamento e Rastreamento de Rotas
- Criacao de rotas com coordenadas de origem e destino
- Integracao com Google Maps API para rastreamento GPS
- Atualizacoes de status de rota em tempo real (Agendada, Em Progresso, Completa, Cancelada)
- Rastreamento de pontos de verificacao ao longo das rotas
- Comparacao de distancia e custos estimados vs reais
- Rastreamento de custos de combustivel e pedagio

#### 4. Gestao de Carregamentos
- Multiplos carregamentos por rota
- Classificacao de tipo de carga (Geral, Perecivel, Perigosa, Granel, etc.)
- Rastreamento de peso e volume
- Gestao de informacoes de clientes
- Confirmacao de entrega com captura de assinatura
- Rastreamento de status de carregamento ao longo do ciclo de vida

#### 5. Gestao de Manutencao
- Agendamento de manutencao preventiva e corretiva
- Historico de manutencao por veiculo
- Rastreamento de custos para operacoes de manutencao
- Gestao de prestadores de servicos
- Rastreamento de substituicao de pecas
- Previsoes de proxima data e quilometragem de manutencao

### Arquitetura

```
FleetManagement-ASPNET-Core/
├── src/
│   ├── FleetManagement.Web/              # ASP.NET Core MVC + API Controllers
│   ├── FleetManagement.Application/      # Logica de Negocio, Services, DTOs, Validators
│   ├── FleetManagement.Domain/           # Entidades, Interfaces, Logica de Dominio
│   ├── FleetManagement.Infrastructure/   # EF Core, Repositories, Acesso a Dados
│   └── FleetManagement.Shared/           # Constantes Compartilhadas, Helpers, Extensions
├── tests/
│   ├── FleetManagement.UnitTests/        # Testes Unitarios xUnit
│   └── FleetManagement.IntegrationTests/ # Testes de Integracao
└── docs/                                 # Documentacao de Arquitetura
```

### Padroes de Design

- **Repository Pattern**: Abstracao sobre logica de acesso a dados
- **Unit of Work Pattern**: Gestao de transacoes entre repositories
- **Dependency Injection**: Injecao por construtor em toda aplicacao
- **CQRS Principles**: Separacao de operacoes de leitura e escrita
- **Service Layer Pattern**: Encapsulamento de logica de negocio
- **DTO Pattern**: Transferencia de dados entre camadas
- **Specification Pattern**: Logica de consulta reutilizavel

### Endpoints da API

#### Veiculos
- `GET /api/vehicles` - Obter todos os veiculos
- `GET /api/vehicles/{id}` - Obter veiculo por ID
- `GET /api/vehicles/available` - Obter veiculos disponiveis
- `GET /api/vehicles/status/{status}` - Obter veiculos por status
- `GET /api/vehicles/plate/{plateNumber}` - Obter veiculo por placa
- `GET /api/vehicles/maintenance-needed` - Obter veiculos precisando manutencao
- `POST /api/vehicles` - Criar novo veiculo
- `PUT /api/vehicles/{id}` - Atualizar veiculo
- `DELETE /api/vehicles/{id}` - Deletar veiculo (soft delete)

#### Motoristas
- `GET /api/drivers` - Obter todos os motoristas
- `GET /api/drivers/{id}` - Obter motorista por ID
- `GET /api/drivers/active` - Obter motoristas ativos
- `POST /api/drivers` - Criar novo motorista
- `PUT /api/drivers/{id}` - Atualizar motorista
- `DELETE /api/drivers/{id}` - Deletar motorista

#### Rotas
- `GET /api/routes` - Obter todas as rotas
- `GET /api/routes/{id}` - Obter rota com detalhes completos
- `GET /api/routes/active` - Obter rotas ativas
- `GET /api/routes/status/{status}` - Obter rotas por status
- `POST /api/routes` - Criar nova rota
- `PUT /api/routes/{id}/status` - Atualizar status da rota
- `DELETE /api/routes/{id}` - Cancelar rota

### Executando o Projeto

#### Pre-requisitos
- .NET 8 SDK
- SQL Server 2022 ou Docker
- Visual Studio 2022 ou Rider

#### Executando Localmente

1. Clonar o repositorio
```bash
git clone https://github.com/yourusername/FleetManagement-ASPNET-Core.git
cd FleetManagement-ASPNET-Core
```

2. Atualizar string de conexao em `appsettings.json`
```json
"ConnectionStrings": {
  "DefaultConnection": "Server=localhost;Database=FleetManagementDb;Trusted_Connection=True;"
}
```

3. Executar migrations do banco de dados
```bash
dotnet ef database update --project src/FleetManagement.Infrastructure --startup-project src/FleetManagement.Web
```

4. Executar a aplicacao
```bash
dotnet run --project src/FleetManagement.Web
```

5. Acessar Swagger UI em `https://localhost:5001`

#### Executando com Docker

```bash
docker-compose up -d
```

Acessar a API em `http://localhost:5000` e Swagger UI em `http://localhost:5000/swagger`

### Testes

Executar todos os testes:
```bash
dotnet test
```

Executar testes com cobertura:
```bash
dotnet test /p:CollectCoverage=true /p:CoverageReportFormat=opencover
```

### Recursos de Seguranca

- **JWT Authentication**: Autenticacao segura baseada em tokens
- **Role-Based Authorization**: Controle de acesso granular
- **Password Policies**: Requisitos de senha forte obrigatorios
- **SQL Injection Protection**: Consultas parametrizadas via EF Core
- **HTTPS Enforcement**: TLS 1.2+ obrigatorio
- **CORS Configuration**: Politicas de origem cruzada configuraveis
- **Soft Deletes**: Preservacao de dados com trilhas de auditoria
- **Logging**: Logging abrangente de requisicoes/respostas com Serilog

### Otimizacoes de Performance

- **Async/Await**: Operacoes de I/O nao bloqueantes em toda aplicacao
- **Connection Pooling**: Gestao eficiente de conexoes de banco de dados
- **Retry Logic**: Operacoes resilientes de banco de dados com backoff exponencial
- **Query Optimization**: Eager loading para prevenir consultas N+1
- **Indexing**: Indices estrategicos em campos frequentemente consultados
- **Caching Ready**: Infraestrutura preparada para cache Redis/Memory

### Pipeline CI/CD

Workflow do GitHub Actions inclui:
- Builds automatizados em push/PR
- Execucao de testes unitarios e de integracao
- Relatorios de cobertura de codigo
- Construcao de imagens Docker
- Varredura de vulnerabilidades de seguranca com Trivy
- Geracao de artefatos para deployments

### Melhorias Futuras

- Rastreamento GPS em tempo real com SignalR
- Dashboard avancado de relatorios e analytics
- App mobile para motoristas (Xamarin/MAUI)
- Integracao com dispositivos telematicos
- Algoritmos de otimizacao de rotas
- Manutencao preditiva usando modelos ML
- Suporte multi-tenant para empresas de gestao de frotas

### Licenca

Este projeto e para fins de demonstracao de portfolio.

### Contato

Eduardo Lara Peiter
- Email: dudu.peiter@gmail.com
- GitHub: [@Dudomon](https://github.com/Dudomon)

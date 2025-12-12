# Warehouse Management System - ASPNetZero

[English](#english) | [Portugues](#portugues)

---

## English

### Overview

Enterprise-grade Warehouse Management System (WMS) built with ASPNetZero framework for logistics and distribution operations. Implements multi-tenant SaaS architecture with comprehensive inventory management, order processing, picking/packing workflows, and real-time notifications. Based on ASP.NET Boilerplate (ABP) framework providing production-ready infrastructure.

### Tech Stack

- **Framework**: ASPNetZero 9.0 (based on ABP Framework)
- **Language**: C# 12
- **ORM**: Entity Framework Core 8
- **Database**: SQL Server 2022
- **Authentication**: ASP.NET Identity + JWT Tokens
- **Authorization**: Permission-based with role management
- **Multi-Tenancy**: Database-per-tenant and shared database options
- **Background Jobs**: Hangfire integration via ABP
- **Real-time**: SignalR for live updates
- **Mapping**: AutoMapper
- **Testing**: xUnit, Shouldly, NSubstitute
- **API**: RESTful with Swagger/OpenAPI

### Key Features

#### 1. Multi-Tenant Architecture
- Complete tenant isolation with ABP multi-tenancy
- Tenant-specific data segregation
- Separate databases or shared database with tenant filtering
- Tenant management and configuration
- Per-tenant branding and localization

#### 2. Warehouse Management
- Multiple warehouse support per tenant
- Warehouse zones (Receiving, Storage, Picking, Packing, Shipping, Quarantine)
- Location hierarchy (Warehouse > Zone > Aisle > Rack > Shelf > Bin)
- Capacity management and space optimization
- Barcode scanning integration ready

#### 3. Inventory Management
- Real-time inventory tracking with multi-location support
- SKU management with batch/lot tracking
- Quantity on hand, reserved, and available calculations
- Automatic reorder point alerts
- Inventory adjustments with audit trails
- Expiration date tracking for perishables
- FIFO/LIFO inventory valuation

#### 4. Order Processing
- Inbound orders (receiving)
- Outbound orders (shipping)
- Transfer orders between warehouses
- Order status workflow (Pending > Confirmed > Picking > Packing > Shipped > Delivered)
- Order line item management
- Customer information and shipping address tracking

#### 5. Picking and Packing
- Automated picking task generation
- Task assignment to warehouse workers
- Priority-based task queue
- Real-time picking progress tracking
- Barcode verification during picking
- Packing workflows with shipping label generation

#### 6. Background Jobs
- Automated reorder notifications
- Inventory reconciliation jobs
- Report generation
- Data archival and cleanup
- Integration with external systems

#### 7. Authorization and Security
- Granular permission system
- Role-based access control (RBAC)
- Feature permissions (Create, Edit, Delete, Process)
- Tenant-specific user management
- Audit logging for all operations
- Secure API endpoints with JWT authentication

### Architecture

```
WarehouseManagement-ASPNetZero/
├── src/
│   ├── WarehouseManagement.Core/              # Domain Layer (Entities, Authorization, Jobs)
│   ├── WarehouseManagement.Application/       # Application Services, DTOs, Business Logic
│   ├── WarehouseManagement.EntityFrameworkCore/ # Data Access, DbContext, Migrations
│   └── WarehouseManagement.Web.Mvc/           # Presentation Layer (Controllers, Views)
└── tests/
    └── WarehouseManagement.Tests/             # Unit and Integration Tests
```

### ABP Framework Features Utilized

- **FullAuditedEntity**: Automatic tracking of CreationTime, CreatorUserId, LastModificationTime, etc.
- **IMustHaveTenant**: Enforces tenant context on all entities
- **ISoftDelete**: Soft deletion with automatic query filtering
- **AsyncCrudAppService**: Base class for CRUD operations with built-in authorization
- **IPermissionDefinitionProvider**: Centralized permission management
- **BackgroundJob**: Reliable background task execution
- **UnitOfWork**: Automatic transaction management
- **AbpAuthorize**: Declarative authorization on services
- **IRepository**: Generic repository pattern implementation

### Permission Structure

```
Pages.Warehouses
├── Pages.Warehouses.Create
├── Pages.Warehouses.Edit
└── Pages.Warehouses.Delete

Pages.Inventory
├── Pages.Inventory.Create
├── Pages.Inventory.Edit
├── Pages.Inventory.Delete
└── Pages.Inventory.AdjustQuantity

Pages.Orders
├── Pages.Orders.Create
├── Pages.Orders.Edit
├── Pages.Orders.Cancel
└── Pages.Orders.Process

Pages.Picking
├── Pages.Picking.Assign
└── Pages.Picking.Complete

Pages.Reports
├── Pages.Reports.Inventory
├── Pages.Reports.Orders
└── Pages.Reports.Performance
```

### Application Services

#### InventoryAppService
- `GetAllAsync`: Paginated inventory list with filtering
- `GetItemsNeedingReorderAsync`: Items below reorder point
- `GetItemsByWarehouseAsync`: Filter by warehouse
- `AdjustQuantityAsync`: Inventory adjustments with audit trail
- `GetItemBySKUAsync`: Lookup by SKU

Follows ABP patterns:
- Inherits from `AsyncCrudAppService`
- Uses `IRepository<T>` for data access
- Decorated with `[AbpAuthorize]` for permission checking
- Returns DTOs mapped via AutoMapper
- Automatic UnitOfWork transaction management

### Database Design

Multi-tenant aware entities with:
- `TenantId` on all entities (enforced by `IMustHaveTenant`)
- Composite unique indexes including `TenantId`
- Soft delete support via `IsDeleted` flag
- Full audit fields (CreationTime, CreatorUserId, etc.)
- Cascading rules respecting tenant boundaries

### Getting Started

#### Prerequisites
- .NET 8 SDK
- SQL Server 2022
- Visual Studio 2022 or Rider
- ASPNetZero license (or ABP Framework for open-source version)

#### Running Locally

1. Clone the repository
```bash
git clone https://github.com/yourusername/WarehouseManagement-ASPNetZero.git
cd WarehouseManagement-ASPNetZero
```

2. Update connection string in `appsettings.json`
```json
"ConnectionStrings": {
  "Default": "Server=localhost;Database=WarehouseManagementDb;Trusted_Connection=True;"
}
```

3. Apply migrations
```bash
dotnet ef database update --project src/WarehouseManagement.EntityFrameworkCore --startup-project src/WarehouseManagement.Web.Mvc
```

4. Run the application
```bash
dotnet run --project src/WarehouseManagement.Web.Mvc
```

5. Default credentials
- Username: `admin`
- Password: `123qwe`

### Testing

```bash
dotnet test
```

Tests use ABP's `AbpIntegratedTestBase` for integration testing with real DbContext.

### Multi-Tenancy Configuration

Enable/disable in `WarehouseManagementConsts.cs`:
```csharp
public const bool MultiTenancyEnabled = true;
```

Tenant resolution:
- Subdomain (tenant1.yourapp.com)
- HTTP header
- Cookie
- Query parameter

### Background Jobs

Reorder inventory job example:
```csharp
BackgroundJob.Enqueue<ReorderInventoryJob>(job =>
    job.Execute(new ReorderInventoryJobArgs { TenantId = currentTenantId }));
```

### Deployment Considerations

- Host database vs tenant databases strategy
- Redis for distributed caching and SignalR backplane
- Hangfire dashboard for background job monitoring
- Elasticsearch for advanced search capabilities
- Docker containerization with multi-stage builds

### ASPNetZero vs Open-Source ABP

This project demonstrates ASPNetZero patterns. Key differences from open-source ABP:
- Pre-built admin UI with Angular/MVC
- Advanced multi-tenancy features
- Rapid application development templates
- Commercial support and regular updates
- Additional modules (chat, payment integration, etc.)

### Future Enhancements

- Mobile app for warehouse workers (MAUI)
- Advanced wave picking strategies
- Slotting optimization algorithms
- Integration with WMS hardware (RF scanners, printers)
- Machine learning for demand forecasting
- IoT sensor integration for cold storage monitoring

### License

This project is for portfolio demonstration purposes.

### Contact

Eduardo Lara Peiter
- Email: dudu.peiter@gmail.com
- GitHub: [@Dudomon](https://github.com/Dudomon)

---

## Portugues

### Visao Geral

Sistema de Gestao de Armazem (WMS) de nivel empresarial construido com framework ASPNetZero para operacoes de logistica e distribuicao. Implementa arquitetura SaaS multi-tenant com gestao abrangente de inventario, processamento de pedidos, workflows de picking/packing, e notificacoes em tempo real. Baseado no framework ASP.NET Boilerplate (ABP) fornecendo infraestrutura pronta para producao.

### Stack Tecnologica

- **Framework**: ASPNetZero 9.0 (baseado em ABP Framework)
- **Linguagem**: C# 12
- **ORM**: Entity Framework Core 8
- **Banco de Dados**: SQL Server 2022
- **Autenticacao**: ASP.NET Identity + JWT Tokens
- **Autorizacao**: Baseada em permissoes com gestao de roles
- **Multi-Tenancy**: Banco por tenant ou banco compartilhado
- **Background Jobs**: Integracao Hangfire via ABP
- **Tempo Real**: SignalR para atualizacoes ao vivo
- **Mapeamento**: AutoMapper
- **Testes**: xUnit, Shouldly, NSubstitute
- **API**: RESTful com Swagger/OpenAPI

### Funcionalidades Principais

#### 1. Arquitetura Multi-Tenant
- Isolamento completo de tenant com multi-tenancy ABP
- Segregacao de dados especifica por tenant
- Bancos de dados separados ou banco compartilhado com filtragem de tenant
- Gestao e configuracao de tenants
- Branding e localizacao por tenant

#### 2. Gestao de Armazens
- Suporte a multiplos armazens por tenant
- Zonas de armazem (Recebimento, Armazenamento, Picking, Packing, Expedicao, Quarentena)
- Hierarquia de localizacao (Armazem > Zona > Corredor > Estante > Prateleira > Gaveta)
- Gestao de capacidade e otimizacao de espaco
- Integracao com leitura de codigo de barras pronta

#### 3. Gestao de Inventario
- Rastreamento de inventario em tempo real com suporte multi-localizacao
- Gestao de SKU com rastreamento de lote/batch
- Calculos de quantidade disponivel, reservada e em maos
- Alertas automaticos de ponto de reabastecimento
- Ajustes de inventario com trilha de auditoria
- Rastreamento de data de validade para pereciveis
- Avaliacao de inventario FIFO/LIFO

#### 4. Processamento de Pedidos
- Pedidos de entrada (recebimento)
- Pedidos de saida (expedicao)
- Pedidos de transferencia entre armazens
- Workflow de status de pedido (Pendente > Confirmado > Picking > Packing > Expedido > Entregue)
- Gestao de linhas de pedido
- Rastreamento de informacoes de cliente e endereco de entrega

#### 5. Picking e Packing
- Geracao automatizada de tarefas de picking
- Atribuicao de tarefas a trabalhadores do armazem
- Fila de tarefas baseada em prioridade
- Rastreamento de progresso de picking em tempo real
- Verificacao de codigo de barras durante picking
- Workflows de packing com geracao de etiquetas de envio

#### 6. Background Jobs
- Notificacoes automaticas de reabastecimento
- Jobs de reconciliacao de inventario
- Geracao de relatorios
- Arquivamento e limpeza de dados
- Integracao com sistemas externos

#### 7. Autorizacao e Seguranca
- Sistema de permissoes granular
- Controle de acesso baseado em roles (RBAC)
- Permissoes de features (Criar, Editar, Deletar, Processar)
- Gestao de usuarios especifica por tenant
- Logging de auditoria para todas operacoes
- Endpoints de API seguros com autenticacao JWT

### Arquitetura

```
WarehouseManagement-ASPNetZero/
├── src/
│   ├── WarehouseManagement.Core/              # Camada de Dominio (Entidades, Autorizacao, Jobs)
│   ├── WarehouseManagement.Application/       # Application Services, DTOs, Logica de Negocio
│   ├── WarehouseManagement.EntityFrameworkCore/ # Acesso a Dados, DbContext, Migrations
│   └── WarehouseManagement.Web.Mvc/           # Camada de Apresentacao (Controllers, Views)
└── tests/
    └── WarehouseManagement.Tests/             # Testes Unitarios e de Integracao
```

### Recursos do ABP Framework Utilizados

- **FullAuditedEntity**: Rastreamento automatico de CreationTime, CreatorUserId, LastModificationTime, etc.
- **IMustHaveTenant**: Impoe contexto de tenant em todas entidades
- **ISoftDelete**: Delecao logica com filtragem automatica de consultas
- **AsyncCrudAppService**: Classe base para operacoes CRUD com autorizacao embutida
- **IPermissionDefinitionProvider**: Gestao centralizada de permissoes
- **BackgroundJob**: Execucao confiavel de tarefas em background
- **UnitOfWork**: Gestao automatica de transacoes
- **AbpAuthorize**: Autorizacao declarativa em services
- **IRepository**: Implementacao do padrao repository generico

### Estrutura de Permissoes

```
Pages.Warehouses
├── Pages.Warehouses.Create
├── Pages.Warehouses.Edit
└── Pages.Warehouses.Delete

Pages.Inventory
├── Pages.Inventory.Create
├── Pages.Inventory.Edit
├── Pages.Inventory.Delete
└── Pages.Inventory.AdjustQuantity

Pages.Orders
├── Pages.Orders.Create
├── Pages.Orders.Edit
├── Pages.Orders.Cancel
└── Pages.Orders.Process

Pages.Picking
├── Pages.Picking.Assign
└── Pages.Picking.Complete

Pages.Reports
├── Pages.Reports.Inventory
├── Pages.Reports.Orders
└── Pages.Reports.Performance
```

### Application Services

#### InventoryAppService
- `GetAllAsync`: Lista paginada de inventario com filtragem
- `GetItemsNeedingReorderAsync`: Itens abaixo do ponto de reabastecimento
- `GetItemsByWarehouseAsync`: Filtrar por armazem
- `AdjustQuantityAsync`: Ajustes de inventario com trilha de auditoria
- `GetItemBySKUAsync`: Busca por SKU

Segue padroes ABP:
- Herda de `AsyncCrudAppService`
- Usa `IRepository<T>` para acesso a dados
- Decorado com `[AbpAuthorize]` para verificacao de permissoes
- Retorna DTOs mapeados via AutoMapper
- Gestao automatica de transacao UnitOfWork

### Design do Banco de Dados

Entidades multi-tenant com:
- `TenantId` em todas entidades (imposto por `IMustHaveTenant`)
- Indices unicos compostos incluindo `TenantId`
- Suporte a soft delete via flag `IsDeleted`
- Campos de auditoria completos (CreationTime, CreatorUserId, etc.)
- Regras de cascata respeitando limites de tenant

### Executando o Projeto

#### Pre-requisitos
- .NET 8 SDK
- SQL Server 2022
- Visual Studio 2022 ou Rider
- Licenca ASPNetZero (ou ABP Framework para versao open-source)

#### Executando Localmente

1. Clonar o repositorio
```bash
git clone https://github.com/yourusername/WarehouseManagement-ASPNetZero.git
cd WarehouseManagement-ASPNetZero
```

2. Atualizar string de conexao em `appsettings.json`
```json
"ConnectionStrings": {
  "Default": "Server=localhost;Database=WarehouseManagementDb;Trusted_Connection=True;"
}
```

3. Aplicar migrations
```bash
dotnet ef database update --project src/WarehouseManagement.EntityFrameworkCore --startup-project src/WarehouseManagement.Web.Mvc
```

4. Executar a aplicacao
```bash
dotnet run --project src/WarehouseManagement.Web.Mvc
```

5. Credenciais padrao
- Usuario: `admin`
- Senha: `123qwe`

### Testes

```bash
dotnet test
```

Testes usam `AbpIntegratedTestBase` do ABP para testes de integracao com DbContext real.

### Configuracao Multi-Tenancy

Habilitar/desabilitar em `WarehouseManagementConsts.cs`:
```csharp
public const bool MultiTenancyEnabled = true;
```

Resolucao de tenant:
- Subdominio (tenant1.seuapp.com)
- Header HTTP
- Cookie
- Parametro de query

### Background Jobs

Exemplo de job de reabastecimento de inventario:
```csharp
BackgroundJob.Enqueue<ReorderInventoryJob>(job =>
    job.Execute(new ReorderInventoryJobArgs { TenantId = currentTenantId }));
```

### Consideracoes de Deploy

- Estrategia de banco de dados host vs bancos de tenants
- Redis para cache distribuido e backplane SignalR
- Dashboard Hangfire para monitoramento de background jobs
- Elasticsearch para capacidades avancadas de busca
- Containerizacao Docker com builds multi-stage

### ASPNetZero vs ABP Open-Source

Este projeto demonstra padroes ASPNetZero. Principais diferencas do ABP open-source:
- UI admin pre-construida com Angular/MVC
- Recursos avancados de multi-tenancy
- Templates de desenvolvimento rapido de aplicacoes
- Suporte comercial e atualizacoes regulares
- Modulos adicionais (chat, integracao de pagamento, etc.)

### Melhorias Futuras

- App mobile para trabalhadores de armazem (MAUI)
- Estrategias avancadas de wave picking
- Algoritmos de otimizacao de slotting
- Integracao com hardware WMS (scanners RF, impressoras)
- Machine learning para previsao de demanda
- Integracao de sensores IoT para monitoramento de armazenamento refrigerado

### Licenca

Este projeto e para fins de demonstracao de portfolio.

### Contato

Eduardo Lara Peiter
- Email: dudu.peiter@gmail.com
- GitHub: [@Dudomon](https://github.com/Dudomon)

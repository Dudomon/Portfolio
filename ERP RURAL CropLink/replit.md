# CropLink - Sistema de Gestão Agrícola

## Overview
CropLink is a comprehensive agricultural management system built with Flask, designed to provide farmers with tools to manage daily operations. Key capabilities include inventory management (supplies and agricultural inputs), machinery tracking, employee management, grain storage (silos), rainfall tracking, and operational dashboards. The system supports multi-tenant SaaS operations with isolated user workspaces, ensuring security and scalability for commercial deployment.

## User Preferences
Preferred communication style: Simple, everyday language.
Visual Design: Clean, professional white theme with green accents, simplified collapsible sidebar navigation, and fully responsive design. Menu structure with functional dropdown system - main topics expand to show sub-items when clicked, with fixed icon positioning and smooth animations. Consistent card sizing and Material Design-inspired elements focused on usability for rural producers.
Security: Follow security best practices including SAST, DAST, and IAST principles.

## Recent Changes - October 01, 2025

### Financial Module Implementation - Accounts Payable
- **ContasPagar Model**: Complete database model with multi-tenant isolation (fornecedor, descricao, valor, vencimento, status, categoria, observacoes)
- **CRUD Operations**: Full create, read, update, delete functionality for accounts payable management
- **Payment Tracking**: Comprehensive payment workflow with status management (pendente, pago, cancelado, vencido)
- **Responsive Interface**: Modal-based forms with mobile-first design, statistics cards, and data tables
- **Security Enhancement**: Fixed stored XSS vulnerability by refactoring inline JavaScript to use secure data-* attributes with proper HTML escaping
- **Event Handlers**: Implemented secure event delegation using data attributes instead of inline onclick handlers
- **Navigation Integration**: Financial module added to sidebar menu with dropdown for future expansion (contas receber, fluxo caixa, contas bancárias)

## Previous Changes - September 25, 2025

### Clean Project Architecture Reorganization
- **Modular Structure**: Restructured entire project with clean architecture patterns
- **Application Module**: Moved main application logic to `/app` folder with proper module organization
- **Entry Point**: Created `run.py` at project root as the standardized application entry point
- **Configuration Separation**: Moved configuration files to dedicated `/config` directory
- **Import Optimization**: Fixed all relative imports and template/static folder paths
- **Template Integration**: Properly configured Flask template and static folder paths for new structure
- **Legacy Cleanup**: Removed old `app.py` file after successful migration
- **System Integrity**: Maintained all existing functionality during reorganization

## Previous Changes - September 24, 2025

### Sistema de Cache Hierárquico L2 Implementado
- **Cache L2 com Redis/SimpleCache**: Sistema inteligente de cache implementado para otimização de performance
- **Multi-tenant Seguro**: Cache com isolamento completo por tenant (user_id) garantindo segurança
- **TTL Inteligente**: Configuração automática baseada na volatilidade dos dados (30s-15min)
- **Cache Dashboard**: Estatísticas principais cacheadas (cartões de resumo, contadores)  
- **Cache Silos**: Ocupação, capacidade disponível e estoque total com cache otimizado
- **Invalidação por Eventos**: Sistema automático de limpeza de cache após movimentações
- **Decorators Automáticos**: `@cached()` aplicado em funções críticas de performance
- **Fallback Seguro**: SimpleCache para desenvolvimento, Redis para produção
### Mobile App-like Experience (Refined) + Trial System Removal
- **Bottom Navigation**: Implemented native app-style bottom navigation (non-fixed positioning) with subtle animations
- **App-like Design System**: Enhanced cards with rounded corners, subtle shadows, and clean design patterns
- **Touch Interactions**: Added discrete ripple effects with neutral colors and minimal haptic feedback
- **Professional Typography**: Optimized text hierarchy with improved contrast and readability
- **Loading States**: Elegant loading animations and shimmer effects for better UX
- **Subtle Animations**: Page transitions and gentle button feedback without excessive color effects
- **Clean Interface**: Simplified interactions removing green color highlights and fixed positioning elements
- **Trial System Removal**: Completely removed 5-day trial period functionality including dashboard alerts, middleware checks, templates, and user interface elements
- **Mobile Color Contrast Fix**: Updated stat cards from dark green background to light gray gradient with dark text, improving readability and content visibility on mobile devices
- **Critical Menu Access Fix**: Corrected restrictive template condition that was hiding main navigation menu from non-admin users. Changed from role-specific access to authenticated user access, restoring full menu functionality for all clients

## System Architecture

### Frontend Architecture
- **Template Engine**: Jinja2 with Bootstrap 5 for responsive UI.
- **CSS Framework**: Bootstrap 5.3.0 with a modern clean theme using CSS variables.
- **Navigation**: Left sidebar with animated transitions and dropdown functionality.
- **Typography**: Inter font family.
- **JavaScript**: Vanilla JavaScript with Bootstrap components.
- **Icon Library**: Font Awesome 6.4.0.
- **Layout Pattern**: Base template inheritance with sidebar and main content area.

### Backend Architecture
- **Web Framework**: Flask 3.1.1 using blueprints.
- **Authentication**: Flask-Login with bcrypt for password hashing.
- **Session Management**: Flask's built-in session handling.
- **Database ORM**: SQLAlchemy 2.0.41 with Flask-SQLAlchemy.
- **Migration Support**: Flask-Migrate for database schema versioning.
- **Environment Configuration**: python-dotenv for environment variables.

### Data Storage Solutions
- **Primary Database**: PostgreSQL with psycopg2-binary.
- **ORM Models**: SQLAlchemy models for various agricultural entities.
- **Database Configuration**: Supports `DATABASE_URL` (cloud) and local parameters.
- **Connection Pooling**: SQLAlchemy's built-in pooling.

### Authentication and Authorization
- **User Authentication**: Flask-Login with `UserMixin`.
- **Password Security**: Flask-Bcrypt.
- **Access Control**: Login-required decorators and role-based admin authorization (`@admin_required`).
- **Session Security**: Secure session configuration.
- **Role Management**: `is_admin` database field and `user_role` for granular permissions.
- **User Approval System**: Administrative approval required for new user registrations.
- **CSRF Protection**: Implemented for all forms.

### Multi-Tenancy and Data Isolation
- **SaaS Architecture**: Complete data isolation per user with `user_id` foreign keys across 12 operational tables.
- **Security**: Critical privilege escalation vulnerabilities fixed; real-time ownership validation.
- **Automatic Filtering**: `filtrar_por_usuario()`, `criar_com_usuario()` functions and secure validators.
- **Administrative Bypass**: Admins maintain a global view of all data.

### Production Deployment
- **WSGI Server**: Gunicorn 21.2.0.
- **Process Management**: Multi-worker configuration.
- **Logging**: Structured logging to stdout/stderr.
- **Health Checks**: Built-in endpoints.
- **Environment Adaptation**: Automatic detection for cloud vs. local environments.

### Feature Specifications
- **Inventory Management**: Real-time stock display and atomic transactions for applying inputs.
- **Grain Management**: Optimized grain system with free-text input and dynamic AJAX loading, including API for grain by silo.
- **User Management**: Admin interface for listing, adding, editing, and approving users; includes super administrator controls for user roles and permissions.
- **ERP Data Import**: Secure upload system for Excel/TXT files with validation.
- **Direct Developer Contact**: WhatsApp contact button for non-admin users.

## External Dependencies

### Core Framework Dependencies
- **Flask**: Web framework.
- **SQLAlchemy**: Database ORM.
- **Bootstrap**: Frontend CSS framework (CDN).
- **Font Awesome**: Icon library (CDN).

### Database Services
- **PostgreSQL**: Primary production database.
- **SQLite**: Development database.
- **Flask-Migrate/Alembic**: Database schema management.

### Security Libraries
- **bcrypt**: Password hashing.
- **Flask-Login**: User session management.
- **python-dotenv**: Environment variable management.

### Production Infrastructure
- **Gunicorn**: WSGI HTTP server.
- **Cloud Platform**: Designed for Render or similar PaaS.

### Development Tools
- **Flask Development Server**: Local testing.
- **Logging**: Debugging and monitoring.
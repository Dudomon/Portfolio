// CropLink - Main JavaScript Functions

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize form validations
    initializeFormValidations();
    
    // Initialize data tables
    initializeDataTables();
    
    // Initialize auto-refresh for dashboard
    initializeDashboardRefresh();
    
    // Initialize notification system
    initializeNotifications();
});

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize form validations
 */
function initializeFormValidations() {
    // Bootstrap form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.prototype.slice.call(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
    
    // Custom validations
    setupCustomValidations();
}

/**
 * Setup custom form validations
 */
function setupCustomValidations() {
    // CPF validation
    const cpfInputs = document.querySelectorAll('input[name="cpf"]');
    cpfInputs.forEach(function(input) {
        input.addEventListener('blur', function() {
            if (this.value && !validateCPF(this.value)) {
                this.setCustomValidity('CPF inválido');
            } else {
                this.setCustomValidity('');
            }
        });
    });
    
    // Quantidade validation (não pode ser negativa)
    const quantidadeInputs = document.querySelectorAll('input[name="quantidade"], input[name="quantidade_kg"]');
    quantidadeInputs.forEach(function(input) {
        input.addEventListener('input', function() {
            if (parseFloat(this.value) < 0) {
                this.setCustomValidity('Quantidade não pode ser negativa');
            } else {
                this.setCustomValidity('');
            }
        });
    });
}

/**
 * Validate CPF
 */
function validateCPF(cpf) {
    cpf = cpf.replace(/[^\d]+/g, '');
    if (cpf.length !== 11 || !!cpf.match(/(\d)\1{10}/)) return false;
    cpf = cpf.split('').map(el => +el);
    const rest = (count) => (cpf.slice(0, count-12)
        .reduce((soma, el, index) => (soma + el * (count-index)), 0)*10) % 11 % 10;
    return rest(10) === cpf[9] && rest(11) === cpf[10];
}

/**
 * Initialize enhanced data tables
 */
function initializeDataTables() {
    // Add search functionality to tables
    const tables = document.querySelectorAll('.table');
    tables.forEach(function(table) {
        if (table.rows.length > 10) {
            addTableSearch(table);
        }
    });
    
    // Add sorting functionality
    addTableSorting();
}

/**
 * Add search functionality to table
 */
function addTableSearch(table) {
    const tableContainer = table.parentNode;
    if (tableContainer.querySelector('.table-search')) return; // Already added
    
    const searchContainer = document.createElement('div');
    searchContainer.className = 'mb-3 table-search';
    searchContainer.innerHTML = `
        <div class="input-group">
            <span class="input-group-text"><i class="fas fa-search"></i></span>
            <input type="text" class="form-control" placeholder="Buscar na tabela...">
        </div>
    `;
    
    const searchInput = searchContainer.querySelector('input');
    searchInput.addEventListener('keyup', function() {
        filterTable(table, this.value);
    });
    
    tableContainer.insertBefore(searchContainer, table);
}

/**
 * Filter table rows based on search term
 */
function filterTable(table, searchTerm) {
    const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');
    searchTerm = searchTerm.toLowerCase();
    
    Array.from(rows).forEach(function(row) {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(searchTerm) ? '' : 'none';
    });
}

/**
 * Add sorting functionality to tables
 */
function addTableSorting() {
    const tables = document.querySelectorAll('.table');
    tables.forEach(function(table) {
        const headers = table.querySelectorAll('th');
        headers.forEach(function(header, index) {
            if (header.textContent.trim()) {
                header.style.cursor = 'pointer';
                header.innerHTML += ' <i class="fas fa-sort text-muted"></i>';
                header.addEventListener('click', function() {
                    sortTable(table, index);
                });
            }
        });
    });
}

/**
 * Sort table by column
 */
function sortTable(table, column) {
    const tbody = table.getElementsByTagName('tbody')[0];
    const rows = Array.from(tbody.getElementsByTagName('tr'));
    
    const isAscending = table.dataset.sortOrder !== 'asc';
    table.dataset.sortOrder = isAscending ? 'asc' : 'desc';
    
    rows.sort(function(a, b) {
        const aVal = a.getElementsByTagName('td')[column].textContent.trim();
        const bVal = b.getElementsByTagName('td')[column].textContent.trim();
        
        const aNum = parseFloat(aVal.replace(/[^\d.-]/g, ''));
        const bNum = parseFloat(bVal.replace(/[^\d.-]/g, ''));
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return isAscending ? aNum - bNum : bNum - aNum;
        }
        
        return isAscending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    });
    
    rows.forEach(function(row) {
        tbody.appendChild(row);
    });
    
    // Update sort icons
    const headers = table.querySelectorAll('th i.fa-sort, th i.fa-sort-up, th i.fa-sort-down');
    headers.forEach(function(icon, index) {
        if (index === column) {
            icon.className = isAscending ? 'fas fa-sort-up' : 'fas fa-sort-down';
        } else {
            icon.className = 'fas fa-sort text-muted';
        }
    });
}

/**
 * Initialize dashboard auto-refresh (OTIMIZADO - só quando necessário)
 */
function initializeDashboardRefresh() {
    // Removido refresh automático para melhorar performance
    // O dashboard já é rápido agora, não precisa de refresh constante
    if (window.location.pathname === '/dashboard') {
        // Dashboard carregado - otimizado sem refresh automático
    }
}

/**
 * Formatação numérica para aceitar valores sem pontos (1000000 = 1 milhão)
 */
function formatarNumero(input) {
    // Remove tudo que não for número
    let valor = input.value.replace(/[^\d]/g, '');
    
    // Define o valor limpo no input
    input.value = valor;
    
    // Atualiza o display com formatação visual
    if (valor) {
        const numeroFormatado = parseInt(valor).toLocaleString('pt-BR');
        input.setAttribute('title', `Valor: ${numeroFormatado}`);
    }
}

/**
 * Converte string formatada para número
 */
function converterParaNumero(valorString) {
    if (!valorString) return 0;
    return parseFloat(valorString.replace(/[^\d]/g, '')) || 0;
}

/**
 * Calcula a tara automaticamente no modal de entrada
 */
function calcularTara() {
    const pesoEntrada = document.getElementById('peso_entrada');
    const pesoSaida = document.getElementById('peso_saida');
    const infoTara = document.getElementById('infoTara');
    const resultadoTara = document.getElementById('resultadoTara');

    if (pesoEntrada && pesoSaida && infoTara && resultadoTara && pesoEntrada.value && pesoSaida.value) {
        const entrada = converterParaNumero(pesoEntrada.value);
        const saida = converterParaNumero(pesoSaida.value);

        // LANÇAMENTO DE GRÃOS (ENTRADA NO SILO):
        // Peso de Entrada = Caminhão CHEIO
        // Peso de Saída = Caminhão VAZIO (tara do caminhão)
        // Peso Líquido = Entrada - Saída
        const pesoLiquido = entrada - saida;

        if (pesoLiquido < 0) {
            resultadoTara.textContent = 'ERRO: Peso de entrada deve ser maior que peso de saída!';
            infoTara.className = 'alert alert-danger';
            infoTara.style.display = 'block';
            return;
        }

        resultadoTara.textContent = `${pesoLiquido.toLocaleString('pt-BR')} kg (${(pesoLiquido / 60).toFixed(1)} sacas)`;
        infoTara.className = 'alert alert-info';
        infoTara.style.display = 'block';

        // Atualiza automaticamente o campo quantidade_kg (readonly)
        const quantidadeInput = document.getElementById('quantidade_lancamento');
        if (quantidadeInput) {
            quantidadeInput.value = pesoLiquido;
            // Adicionar destaque visual quando calculado
            quantidadeInput.classList.add('bg-success', 'bg-opacity-25');
            setTimeout(() => {
                quantidadeInput.classList.remove('bg-success', 'bg-opacity-25');
            }, 2000);

            if (typeof calcularSacasLancamento === 'function') {
                calcularSacasLancamento();
            }
        }
    } else {
        if (infoTara) infoTara.style.display = 'none';
    }
}

/**
 * Calcula a tara automaticamente no modal de saída
 */
function calcularTaraSaida() {
    const pesoEntrada = document.getElementById('peso_entrada_saida');
    const pesoSaida = document.getElementById('peso_saida_saida');
    const infoTara = document.getElementById('infoTaraSaida');
    const resultadoTara = document.getElementById('resultadoTaraSaida');

    if (pesoEntrada && pesoSaida && infoTara && resultadoTara && pesoEntrada.value && pesoSaida.value) {
        const entrada = converterParaNumero(pesoEntrada.value);
        const saida = converterParaNumero(pesoSaida.value);

        // SAÍDA DE GRÃOS (SAÍDA DO SILO):
        // Peso de Entrada = Caminhão VAZIO (tara do caminhão)
        // Peso de Saída = Caminhão CHEIO
        // Peso Líquido = Saída - Entrada
        const pesoLiquido = saida - entrada;

        if (pesoLiquido < 0) {
            resultadoTara.textContent = 'ERRO: Peso de saída deve ser maior que peso de entrada!';
            infoTara.className = 'alert alert-danger';
            infoTara.style.display = 'block';
            return;
        }

        resultadoTara.textContent = `${pesoLiquido.toLocaleString('pt-BR')} kg (${(pesoLiquido / 60).toFixed(1)} sacas)`;
        infoTara.className = 'alert alert-info';
        infoTara.style.display = 'block';

        // Atualiza automaticamente o campo quantidade_kg (readonly)
        const quantidadeInput = document.getElementById('quantidade_saida');
        if (quantidadeInput) {
            quantidadeInput.value = pesoLiquido;
            // Adicionar destaque visual quando calculado
            quantidadeInput.classList.add('bg-success', 'bg-opacity-25');
            setTimeout(() => {
                quantidadeInput.classList.remove('bg-success', 'bg-opacity-25');
            }, 2000);

            if (typeof calcularSacasSaida === 'function') {
                calcularSacasSaida();
            }
        }
    } else {
        if (infoTara) infoTara.style.display = 'none';
    }
}


/**
 * Refresh dashboard data
 */
function refreshDashboardData() {
    // Refresh specific dashboard elements via AJAX
    // Refreshing dashboard data...
    
    // This would typically make AJAX calls to update specific dashboard components
    // For now, we'll just log the action
}

/**
 * Initialize notification system
 */
function initializeNotifications() {
    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            if (alert && alert.parentNode) {
                alert.style.transition = 'opacity 0.5s ease-out';
                alert.style.opacity = '0';
                setTimeout(function() {
                    if (alert.parentNode) {
                        alert.parentNode.removeChild(alert);
                    }
                }, 500);
            }
        }, 5000);
    });
}

/**
 * Show notification
 */
function showNotification(message, type = 'success') {
    const alertClass = type === 'error' ? 'danger' : type;
    const alertHTML = `
        <div class="alert alert-${alertClass} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    const container = document.querySelector('.container-fluid') || document.body;
    const alertDiv = document.createElement('div');
    alertDiv.innerHTML = alertHTML;
    container.insertBefore(alertDiv.firstElementChild, container.firstChild);
}

/**
 * Format number with thousands separator
 */
function formatNumber(num) {
    return new Intl.NumberFormat('pt-BR').format(num);
}

/**
 * Format currency
 */
function formatCurrency(amount) {
    return new Intl.NumberFormat('pt-BR', {
        style: 'currency',
        currency: 'BRL'
    }).format(amount);
}

/**
 * Format date
 */
function formatDate(date, format = 'short') {
    const options = format === 'short' 
        ? { day: '2-digit', month: '2-digit', year: 'numeric' }
        : { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    
    return new Intl.DateTimeFormat('pt-BR', options).format(new Date(date));
}

/**
 * Debounce function
 */
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction() {
        const context = this;
        const args = arguments;
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
}

/**
 * Local storage helpers
 */
const Storage = {
    set: function(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            console.warn('Could not save to localStorage:', e);
        }
    },
    
    get: function(key) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : null;
        } catch (e) {
            console.warn('Could not read from localStorage:', e);
            return null;
        }
    },
    
    remove: function(key) {
        try {
            localStorage.removeItem(key);
        } catch (e) {
            console.warn('Could not remove from localStorage:', e);
        }
    }
};

/**
 * AJAX helper functions
 */
const Ajax = {
    get: function(url, callback, errorCallback) {
        fetch(url)
            .then(response => response.json())
            .then(data => callback(data))
            .catch(error => {
                console.error('Ajax GET error:', error);
                if (errorCallback) errorCallback(error);
            });
    },
    
    post: function(url, data, callback, errorCallback) {
        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => callback(data))
        .catch(error => {
            console.error('Ajax POST error:', error);
            if (errorCallback) errorCallback(error);
        });
    }
};

/**
 * Utility functions for forms
 */
const FormUtils = {
    /**
     * Clear all form fields
     */
    clearForm: function(formId) {
        const form = document.getElementById(formId);
        if (form) {
            form.reset();
            form.classList.remove('was-validated');
        }
    },
    
    /**
     * Serialize form data to object
     */
    serializeForm: function(form) {
        const formData = new FormData(form);
        const object = {};
        formData.forEach((value, key) => {
            object[key] = value;
        });
        return object;
    },
    
    /**
     * Populate form with data
     */
    populateForm: function(formId, data) {
        const form = document.getElementById(formId);
        if (!form) return;
        
        Object.keys(data).forEach(key => {
            const input = form.querySelector(`[name="${key}"]`);
            if (input) {
                input.value = data[key];
            }
        });
    }
};

/**
 * Chart utilities
 */
const ChartUtils = {
    /**
     * Default chart colors
     */
    colors: [
        '#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6f42c1',
        '#e83e8c', '#fd7e14', '#20c997', '#6c757d', '#343a40'
    ],
    
    /**
     * Get responsive chart options
     */
    getResponsiveOptions: function() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
            }
        };
    }
};

/**
 * Export functions for global use
 */
window.FazendaRebelato = {
    showNotification,
    formatNumber,
    formatCurrency,
    formatDate,
    Storage,
    Ajax,
    FormUtils,
    ChartUtils,
    validateCPF
};

// ===== MOBILE APP-LIKE INTERACTIONS =====

document.addEventListener('DOMContentLoaded', function() {
    // Adicionar classes app-like aos elementos existentes
    enhanceAppLikeExperience();
    
    // Gerenciar navegação inferior
    setupBottomNavigation();
    
    // Adicionar feedback tátil
    addTouchFeedback();
    
    // Smooth scrolling para mobile
    enableSmoothScrolling();
});

function enhanceAppLikeExperience() {
    // Adicionar loading states (simplificado)
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('click', function() {
            // Feedback visual muito suave
            this.style.opacity = '0.9';
            setTimeout(() => {
                this.style.opacity = '';
            }, 100);
        });
    });
    
    // Melhorar botões com ripple effect sutil
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            createRippleEffect(e, this);
        });
    });
}

function createRippleEffect(e, element) {
    // Ripple effect muito suave e discreto
    const ripple = document.createElement('span');
    const rect = element.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = e.clientX - rect.left - size / 2;
    const y = e.clientY - rect.top - size / 2;
    
    ripple.style.cssText = `
        position: absolute;
        width: ${size}px;
        height: ${size}px;
        left: ${x}px;
        top: ${y}px;
        background: rgba(0, 0, 0, 0.05);
        border-radius: 50%;
        transform: scale(0);
        animation: ripple 0.2s ease-out;
        pointer-events: none;
    `;
    
    element.style.position = 'relative';
    element.style.overflow = 'hidden';
    element.appendChild(ripple);
    
    setTimeout(() => {
        ripple.remove();
    }, 200);
}

function setupBottomNavigation() {
    const bottomNavItems = document.querySelectorAll('.bottom-nav-item');
    const currentPath = window.location.pathname;
    
    // Atualizar item ativo baseado na URL atual
    bottomNavItems.forEach(item => {
        const href = item.getAttribute('href');
        
        // Lógica para determinar item ativo
        if (href === currentPath) {
            item.classList.add('active');
        } else if (href === '/insumos-agricolas' && 
                  (currentPath.includes('/insumos') || currentPath.includes('/aplicacao'))) {
            item.classList.add('active');
        } else if (href === '/funcionarios' && 
                  (currentPath.includes('/funcionarios') || currentPath.includes('/diaristas'))) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
        
        // Adicionar feedback sutil sem animação
        item.addEventListener('click', function(e) {
            // Feedback visual muito suave
            this.style.opacity = '0.8';
            setTimeout(() => {
                this.style.opacity = '';
            }, 80);
        });
    });
}

function addTouchFeedback() {
    // Adicionar feedback tátil para dispositivos móveis
    const touchElements = document.querySelectorAll('.btn, .card, .bottom-nav-item, .table tbody tr');
    
    touchElements.forEach(element => {
        element.addEventListener('touchstart', function() {
            // Vibração sutil (se suportada)
            if ('vibrate' in navigator) {
                navigator.vibrate(5);
            }
        });
    });
}

function enableSmoothScrolling() {
    // Scroll suave para navegação
    document.documentElement.style.scrollBehavior = 'smooth';
    
    // Scroll para o topo ao navegar entre páginas (app-like)
    window.addEventListener('beforeunload', function() {
        window.scrollTo(0, 0);
    });
}

// Adicionar animação CSS para ripple
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        to {
            transform: scale(2);
            opacity: 0;
        }
    }
    
    /* Animações para transições de página */
    .page-transition {
        animation: slideInUp 0.3s ease-out;
    }
    
    @keyframes slideInUp {
        from {
            transform: translateY(20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    /* Loading states */
    .btn.loading {
        position: relative;
        color: transparent !important;
    }
    
    .btn.loading::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 20px;
        height: 20px;
        margin: -10px 0 0 -10px;
        border: 2px solid currentColor;
        border-top-color: transparent;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);

// Aplicar animação de entrada na página
document.addEventListener('DOMContentLoaded', function() {
    const mainContent = document.querySelector('.main-content');
    if (mainContent) {
        mainContent.classList.add('page-transition');
    }
});

// Função para mostrar loading nos botões
function showButtonLoading(button) {
    button.classList.add('loading');
    button.disabled = true;
}

function hideButtonLoading(button) {
    button.classList.remove('loading');
    button.disabled = false;
}

/**
 * Auto-refresh CSRF token em caso de erro 400
 */
function handleCSRFError(form) {
    fetch(window.location.href)
        .then(response => response.text())
        .then(html => {
            const parser = new DOMParser();
            const newDoc = parser.parseFromString(html, 'text/html');
            const newToken = newDoc.querySelector('meta[name="csrf-token"]')?.content;
            
            if (newToken) {
                // Atualiza token no meta tag
                const metaTag = document.querySelector('meta[name="csrf-token"]');
                if (metaTag) {
                    metaTag.content = newToken;
                }
                
                // Atualiza token no formulário
                const csrfInput = form.querySelector('input[name="csrf_token"]');
                if (csrfInput) {
                    csrfInput.value = newToken;
                }
                
                // Reenvía o formulário
                setTimeout(() => {
                    form.submit();
                }, 500);
            }
        })
        .catch(error => {
            // Se falhar, recarrega a página
            window.location.reload();
        });
}

// Adicionar aos formulários para melhor UX
document.addEventListener('DOMContentLoaded', function() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('button[type="submit"], input[type="submit"]');
            if (submitBtn) {
                showButtonLoading(submitBtn);
            }
        });
    });
});

// Tornar funções globais para uso nos templates
window.CropLinkMobile = {
    showButtonLoading,
    hideButtonLoading,
    createRippleEffect,
    enhanceAppLikeExperience
};


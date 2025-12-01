/**
 * Widget do Chatbot - Rádio Entre Rios
 *
 * Interface responsiva do chatbot que se integra com o backend
 */

class RadioChatbot {
    constructor(config = {}) {
        // Configurações
        this.config = {
            apiUrl: config.apiUrl || this.getApiUrl(),
            position: config.position || 'bottom-right', // bottom-right, bottom-left
            welcomeMessage: config.welcomeMessage || 'Oi! Eu sou o Chatinho, o assistente virtual da Rádio Entre Rios FM 105.5! Como posso ajudar você hoje?',
            botName: config.botName || 'Chatinho',
            primaryColor: config.primaryColor || '#FF6B00', // Laranja da rádio
            maxMessages: config.maxMessages || 50,
            debug: config.debug || false
        };

        // Estado
        this.isOpen = false;
        this.isTyping = false;
        this.conversationHistory = [];
        this.messageCount = 0;

        // Inicializar
        this.init();
    }

    /**
     * Detecta a URL correta da API baseado no ambiente
     */
    getApiUrl() {
        // Tenta obter a URL base do site
        const currentUrl = window.location.origin;

        // Se estiver rodando localmente, usa o caminho absoluto
        if (currentUrl.includes('localhost') || currentUrl.includes('127.0.0.1')) {
            return currentUrl + '/chatbot/chat_api.php';
        }

        // Para ambiente de produção, usa URL absoluta
        return currentUrl + '/chatbot/chat_api.php';
    }

    /**
     * Inicializa o chatbot
     */
    init() {
        // LIMPEZA AUTOMÁTICA: Remove históricos muito grandes (fix emergencial)
        this.cleanupOldHistory();

        this.createHTML();
        this.attachEventListeners();
        this.loadHistory();

        // Só adiciona mensagem de boas-vindas se não houver histórico
        // E NÃO salva ela no histórico (para evitar duplicação)
        if (this.conversationHistory.length === 0) {
            this.addBotMessage(this.config.welcomeMessage, false); // false = não salvar no histórico
        }

        this.log('Chatbot inicializado');
        this.log('API URL: ' + this.config.apiUrl);

        // FIX para WordPress: Força estilos e clicabilidade
        this.ensureClickable();
    }

    /**
     * Limpa históricos muito grandes que causam erro na API
     */
    cleanupOldHistory() {
        try {
            const saved = localStorage.getItem('radioChatbotHistory');
            if (saved) {
                const parsed = JSON.parse(saved);
                // Se histórico tem mais de 15 mensagens, limpa completamente
                if (parsed.length > 15) {
                    this.log(`⚠️ Histórico muito grande (${parsed.length} mensagens). LIMPANDO para resolver erros.`);
                    localStorage.removeItem('radioChatbotHistory');
                    return;
                }
            }

            // LIMPEZA FORÇADA: Verifica versão do chatbot
            // Se não tiver versão ou for antiga, limpa tudo
            const version = localStorage.getItem('radioChatbotVersion');
            const CURRENT_VERSION = '1.1.0'; // Incrementa a cada fix crítico

            if (version !== CURRENT_VERSION) {
                this.log(`🔄 Atualizando chatbot de ${version || 'antiga'} para ${CURRENT_VERSION}. Limpando histórico.`);
                localStorage.removeItem('radioChatbotHistory');
                localStorage.setItem('radioChatbotVersion', CURRENT_VERSION);
            }
        } catch (e) {
            // Se der erro ao ler, limpa tudo
            localStorage.clear();
        }
    }

    /**
     * Garante que o botão seja sempre clicável (proteção contra WordPress)
     */
    ensureClickable() {
        const toggle = document.getElementById('chatbot-toggle');
        const chatbot = document.getElementById('radio-chatbot');

        if (!toggle || !chatbot) return;

        // Força estilos imediatamente
        toggle.style.zIndex = '999999';
        toggle.style.pointerEvents = 'auto';
        toggle.style.cursor = 'pointer';
        chatbot.style.zIndex = '999999';

        // Re-força estilos a cada 2 segundos (proteção contra plugins que modificam)
        setInterval(() => {
            if (toggle.style.pointerEvents !== 'auto') {
                toggle.style.pointerEvents = 'auto';
                this.log('⚠️ Pointer-events corrigido automaticamente');
            }
            if (toggle.style.zIndex !== '999999') {
                toggle.style.zIndex = '999999';
                chatbot.style.zIndex = '999999';
                this.log('⚠️ Z-index corrigido automaticamente');
            }
        }, 2000);
    }

    /**
     * Cria a estrutura HTML do chatbot
     */
    createHTML() {
        const chatbotHTML = `
            <div id="radio-chatbot" class="radio-chatbot ${this.config.position}">
                <!-- Backdrop (fundo escuro mobile) -->
                <div class="chatbot-backdrop" id="chatbot-backdrop"></div>

                <!-- Botão flutuante -->
                <button id="chatbot-toggle" class="chatbot-toggle" aria-label="Abrir chat">
                    <svg class="icon-chat" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2ZM20 16H6L4 18V4H20V16Z" fill="white"/>
                        <path d="M7 9H17V11H7V9ZM7 12H14V14H7V12Z" fill="white"/>
                    </svg>
                    <svg class="icon-close" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M19 6.41L17.59 5L12 10.59L6.41 5L5 6.41L10.59 12L5 17.59L6.41 19L12 13.41L17.59 19L19 17.59L13.41 12L19 6.41Z" fill="white"/>
                    </svg>
                    <span class="notification-badge" style="display: none;">1</span>
                </button>

                <!-- Janela do chat -->
                <div id="chatbot-window" class="chatbot-window">
                    <!-- Header -->
                    <div class="chatbot-header">
                        <div class="chatbot-header-info">
                            <div class="chatbot-avatar">
                                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 5C13.66 5 15 6.34 15 8C15 9.66 13.66 11 12 11C10.34 11 9 9.66 9 8C9 6.34 10.34 5 12 5ZM12 19.2C9.5 19.2 7.29 17.92 6 15.98C6.03 13.99 10 12.9 12 12.9C13.99 12.9 17.97 13.99 18 15.98C16.71 17.92 14.5 19.2 12 19.2Z" fill="white"/>
                                </svg>
                            </div>
                            <div>
                                <div class="chatbot-title">${this.config.botName}</div>
                                <div class="chatbot-status">
                                    <span class="status-dot"></span>
                                    Online
                                </div>
                            </div>
                        </div>
                        <button class="chatbot-minimize chatbot-close-btn" aria-label="Fechar">
                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M19 6.41L17.59 5L12 10.59L6.41 5L5 6.41L10.59 12L5 17.59L6.41 19L12 13.41L17.59 19L19 17.59L13.41 12L19 6.41Z" fill="white"/>
                            </svg>
                        </button>
                    </div>

                    <!-- Mensagens -->
                    <div id="chatbot-messages" class="chatbot-messages">
                        <!-- As mensagens serão inseridas aqui -->
                    </div>

                    <!-- Input -->
                    <div class="chatbot-input-container">
                        <div class="chatbot-input-wrapper">
                            <textarea
                                id="chatbot-input"
                                class="chatbot-input"
                                placeholder="Digite sua mensagem..."
                                rows="1"
                                maxlength="500"
                            ></textarea>
                            <button id="chatbot-send" class="chatbot-send" aria-label="Enviar mensagem">
                                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M2.01 21L23 12L2.01 3L2 10L17 12L2 14L2.01 21Z" fill="currentColor"/>
                                </svg>
                            </button>
                        </div>
                        <div class="chatbot-footer">
                            Powered by DK Mídia AI
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', chatbotHTML);
    }

    /**
     * Anexa event listeners
     */
    attachEventListeners() {
        const toggle = document.getElementById('chatbot-toggle');
        const minimize = document.querySelector('.chatbot-minimize');
        const sendBtn = document.getElementById('chatbot-send');
        const input = document.getElementById('chatbot-input');
        const backdrop = document.getElementById('chatbot-backdrop');
        const chatWindow = document.getElementById('chatbot-window');

        // Toggle chatbot - VERSÃO REFORÇADA para WordPress
        // Adiciona múltiplos listeners para garantir captura
        const handleToggle = (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            this.toggleChat();
        };

        // Listener normal
        toggle.addEventListener('click', handleToggle, false);

        // Listener com capture (prioridade)
        toggle.addEventListener('click', handleToggle, true);

        // Touch events para mobile
        toggle.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.toggleChat();
        }, { passive: false });

        // BACKDROP DESABILITADO - não fecha mais ao clicar no fundo
        // (removido para evitar fechamento acidental)

        // Impedir que cliques na janela do chat a fechem
        if (chatWindow) {
            chatWindow.addEventListener('click', (e) => {
                // Impede que o clique na janela se propague e feche o chat
                e.stopPropagation();
            });
        }

        // Botão X (minimize) - ÚNICA forma de fechar o chat
        minimize.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('🔴 Botão X clicado - fechando chatbot');
            this.closeChat();
        }, { passive: false });

        // Adiciona suporte explícito para touch em mobile
        minimize.addEventListener('touchstart', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('👆 Botão fechar TOUCH START');
            this.closeChat();
        }, { passive: false });

        minimize.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('👆 Botão fechar TOUCH END');
        }, { passive: false });

        // Enviar mensagem
        sendBtn.addEventListener('click', () => this.sendMessage());
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        input.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
    }

    /**
     * Abre/fecha o chat
     */
    toggleChat() {
        this.isOpen = !this.isOpen;
        const chatbot = document.getElementById('radio-chatbot');
        const toggle = document.getElementById('chatbot-toggle');
        const badge = toggle.querySelector('.notification-badge');

        if (this.isOpen) {
            chatbot.classList.add('open');
            badge.style.display = 'none';
            document.getElementById('chatbot-input').focus();
        } else {
            chatbot.classList.remove('open');
        }

        this.log('Chat ' + (this.isOpen ? 'aberto' : 'fechado'));
    }

    /**
     * Fecha o chat explicitamente (não faz toggle)
     */
    closeChat() {
        console.log('🔥 closeChat() foi chamado!');
        this.isOpen = false;
        const chatbot = document.getElementById('radio-chatbot');

        if (chatbot) {
            chatbot.classList.remove('open');
            console.log('✅ Classe "open" removida do chatbot');
        } else {
            console.error('❌ Elemento #radio-chatbot não encontrado!');
        }

        this.log('Chat fechado');
    }

    /**
     * Envia mensagem do usuário
     */
    async sendMessage() {
        const input = document.getElementById('chatbot-input');
        const message = input.value.trim();

        if (!message || this.isTyping) return;

        // Limpar input
        input.value = '';
        input.style.height = 'auto';

        // Adicionar mensagem do usuário
        this.addUserMessage(message);

        // Adicionar ao histórico
        this.conversationHistory.push({
            role: 'user',
            text: message
        });

        // Mostrar typing indicator
        this.showTyping();

        try {
            // Enviar apenas as últimas 6 mensagens do histórico (3 trocas)
            // para evitar payload muito grande
            const recentHistory = this.conversationHistory.slice(-6);

            // Fazer requisição à API
            const response = await fetch(this.config.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    history: recentHistory
                })
            });

            const data = await response.json();

            this.hideTyping();

            if (data.success) {
                const botMessage = data.data.message;
                // addBotMessage já salva no histórico automaticamente (saveToHistory = true por padrão)
                this.addBotMessage(botMessage);
            } else {
                // Mensagem de erro também é salva no histórico
                this.addBotMessage('Desculpe, ocorreu um erro: ' + (data.error || 'Erro desconhecido'));
            }
        } catch (error) {
            this.hideTyping();
            this.addBotMessage('Desculpe, não consegui me conectar ao servidor. Tente novamente.');
            this.log('Erro na requisição: ' + error, 'error');
        }
    }

    /**
     * Adiciona mensagem do usuário
     * @param {string} message - Mensagem do usuário
     * @param {boolean} saveToHistory - Se deve salvar no histórico (padrão: true)
     */
    addUserMessage(message, saveToHistory = true) {
        const messagesContainer = document.getElementById('chatbot-messages');
        const messageHTML = `
            <div class="message user-message">
                <div class="message-content">${this.escapeHtml(message)}</div>
                <div class="message-time">${this.getCurrentTime()}</div>
            </div>
        `;
        messagesContainer.insertAdjacentHTML('beforeend', messageHTML);
        this.scrollToBottom();
        this.messageCount++;

        // Só adiciona ao histórico se saveToHistory for true
        if (saveToHistory) {
            this.conversationHistory.push({
                role: 'user',
                text: message,
                timestamp: new Date().toISOString()
            });
            this.saveHistory();
        }
    }

    /**
     * Adiciona mensagem do bot
     * @param {string} message - Mensagem do bot
     * @param {boolean} saveToHistory - Se deve salvar no histórico (padrão: true)
     */
    addBotMessage(message, saveToHistory = true) {
        const messagesContainer = document.getElementById('chatbot-messages');
        const messageHTML = `
            <div class="message bot-message">
                <div class="message-avatar">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 5C13.66 5 15 6.34 15 8C15 9.66 13.66 11 12 11C10.34 11 9 9.66 9 8C9 6.34 10.34 5 12 5ZM12 19.2C9.5 19.2 7.29 17.92 6 15.98C6.03 13.99 10 12.9 12 12.9C13.99 12.9 17.97 13.99 18 15.98C16.71 17.92 14.5 19.2 12 19.2Z" fill="currentColor"/>
                    </svg>
                </div>
                <div class="message-bubble">
                    <div class="message-content">${this.formatMessage(message)}</div>
                    <div class="message-time">${this.getCurrentTime()}</div>
                </div>
            </div>
        `;
        messagesContainer.insertAdjacentHTML('beforeend', messageHTML);
        this.scrollToBottom();

        // Mostrar notificação se chat estiver fechado
        if (!this.isOpen) {
            const badge = document.querySelector('.notification-badge');
            badge.style.display = 'flex';
        }

        // Só adiciona ao histórico se saveToHistory for true
        // (Usado para evitar salvar mensagem de boas-vindas)
        if (saveToHistory) {
            this.conversationHistory.push({
                role: 'bot',
                text: message,
                timestamp: new Date().toISOString()
            });
            this.saveHistory();
        }
    }

    /**
     * Mostra indicador de digitação
     */
    showTyping() {
        this.isTyping = true;
        const messagesContainer = document.getElementById('chatbot-messages');
        const typingHTML = `
            <div class="message bot-message typing-indicator" id="typing-indicator">
                <div class="message-avatar">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 5C13.66 5 15 6.34 15 8C15 9.66 13.66 11 12 11C10.34 11 9 9.66 9 8C9 6.34 10.34 5 12 5ZM12 19.2C9.5 19.2 7.29 17.92 6 15.98C6.03 13.99 10 12.9 12 12.9C13.99 12.9 17.97 13.99 18 15.98C16.71 17.92 14.5 19.2 12 19.2Z" fill="currentColor"/>
                    </svg>
                </div>
                <div class="message-bubble">
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        `;
        messagesContainer.insertAdjacentHTML('beforeend', typingHTML);
        this.scrollToBottom();
    }

    /**
     * Esconde indicador de digitação
     */
    hideTyping() {
        this.isTyping = false;
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    /**
     * Formata mensagem (suporta quebras de linha e links)
     */
    formatMessage(message) {
        message = this.escapeHtml(message);

        // Converter quebras de linha
        message = message.replace(/\n/g, '<br>');

        // Converter URLs em links
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        message = message.replace(urlRegex, '<a href="$1" target="_blank" rel="noopener">$1</a>');

        // Converter números de telefone em links
        const phoneRegex = /(\+?\d{2}\s?)?\(?\d{2}\)?\s?\d{4,5}-?\d{4}/g;
        message = message.replace(phoneRegex, '<a href="tel:$&">$&</a>');

        return message;
    }

    /**
     * Escapa HTML para prevenir XSS
     */
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }

    /**
     * Retorna horário atual formatado
     */
    getCurrentTime() {
        const now = new Date();
        return now.getHours().toString().padStart(2, '0') + ':' +
               now.getMinutes().toString().padStart(2, '0');
    }

    /**
     * Scroll para o final das mensagens
     */
    scrollToBottom() {
        const messagesContainer = document.getElementById('chatbot-messages');
        setTimeout(() => {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }, 100);
    }

    /**
     * Salva histórico no localStorage
     */
    saveHistory() {
        try {
            // Validação: filtra mensagens válidas antes de salvar
            const validHistory = this.conversationHistory.filter(msg => {
                return msg &&
                       msg.role &&
                       msg.text &&
                       typeof msg.text === 'string' &&
                       msg.text.trim().length > 0;
            });

            // Manter apenas as últimas N mensagens
            const history = validHistory.slice(-this.config.maxMessages);
            localStorage.setItem('radioChatbotHistory', JSON.stringify(history));

            // Atualiza o histórico em memória se limpou algo
            if (validHistory.length !== this.conversationHistory.length) {
                this.conversationHistory = validHistory;
                this.log('Mensagens inválidas removidas ao salvar histórico');
            }
        } catch (e) {
            this.log('Erro ao salvar histórico: ' + e, 'error');
        }
    }

    /**
     * Carrega histórico do localStorage
     */
    loadHistory() {
        try {
            const saved = localStorage.getItem('radioChatbotHistory');
            if (saved) {
                const parsed = JSON.parse(saved);

                // CORREÇÃO: Limita histórico a 10 mensagens (5 trocas)
                if (parsed.length > 10) {
                    this.log(`Histórico muito grande (${parsed.length} mensagens). Mantendo apenas as últimas 10.`);
                }

                // Validação: remove mensagens vazias ou inválidas
                this.conversationHistory = parsed.filter(msg => {
                    // Verifica se tem role e text válidos
                    if (!msg || !msg.role || !msg.text) {
                        this.log('Mensagem inválida removida do histórico', 'warning');
                        return false;
                    }

                    // Verifica se o texto não está vazio
                    if (typeof msg.text !== 'string' || msg.text.trim().length === 0) {
                        this.log('Mensagem vazia removida do histórico', 'warning');
                        return false;
                    }

                    return true;
                }).slice(-10); // MANTÉM APENAS AS ÚLTIMAS 10 MENSAGENS

                // Se limpou mensagens, salva o histórico limpo
                if (this.conversationHistory.length !== parsed.length) {
                    this.log(`Histórico limpo: ${parsed.length - this.conversationHistory.length} mensagens removidas`);
                    this.saveHistory();
                }

                // Renderiza mensagens do histórico na interface (apenas últimas 10)
                this.renderHistoryMessages();
            }
        } catch (e) {
            this.log('Erro ao carregar histórico: ' + e, 'error');
            // Em caso de erro, limpa o histórico corrompido
            localStorage.removeItem('radioChatbotHistory');
            this.conversationHistory = [];
        }
    }

    /**
     * Renderiza mensagens do histórico na interface
     */
    renderHistoryMessages() {
        if (this.conversationHistory.length === 0) return;

        this.log(`Renderizando ${this.conversationHistory.length} mensagens do histórico`);

        // Limpa mensagens existentes para evitar duplicação
        const messagesContainer = document.getElementById('chatbot-messages');
        if (messagesContainer) {
            messagesContainer.innerHTML = '';
        }

        this.conversationHistory.forEach(msg => {
            if (msg.role === 'user') {
                this.addUserMessage(msg.text, false); // false = não salvar (já está no histórico)
            } else if (msg.role === 'bot') {
                this.addBotMessage(msg.text, false); // false = não salvar (já está no histórico)
            }
        });
    }

    /**
     * Limpa histórico
     */
    clearHistory() {
        this.conversationHistory = [];
        localStorage.removeItem('radioChatbotHistory');
        document.getElementById('chatbot-messages').innerHTML = '';
        this.addBotMessage(this.config.welcomeMessage);
        this.log('Histórico limpo');
    }

    /**
     * Log de debug
     */
    log(message, level = 'info') {
        if (this.config.debug) {
            console.log(`[RadioChatbot ${level.toUpperCase()}]`, message);
        }
    }
}

// Inicializar chatbot quando o DOM estiver pronto
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initChatbot);
} else {
    initChatbot();
}

function initChatbot() {
    // PROTEÇÃO: Verifica se já existe uma instância
    if (window.radioChatbot) {
        console.log('[RadioChatbot] Instância já existe. Ignorando inicialização duplicada.');
        return;
    }

    // Verifica se já existe o elemento no DOM (evita duplicação)
    if (document.getElementById('radio-chatbot')) {
        console.log('[RadioChatbot] Elemento já existe no DOM. Ignorando inicialização duplicada.');
        return;
    }

    window.radioChatbot = new RadioChatbot({
        // apiUrl será detectado automaticamente se não for fornecido
        position: 'bottom-right',
        debug: true // Ativado para debug - mude para false após testar
    });

    console.log('[RadioChatbot] Inicializado com sucesso - Versão 1.1.0');
}

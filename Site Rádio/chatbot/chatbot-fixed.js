/**
 * Widget do Chatbot - Rádio Entre Rios
 *
 * Interface responsiva do chatbot que se integra com o backend
 */

class RadioChatbot {
    constructor(config = {}) {
        // Configurações
        this.config = {
            apiUrl: config.apiUrl || '/chatbot/chat_api.php',
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
     * Inicializa o chatbot
     */
    init() {
        this.createHTML();
        this.attachEventListeners();
        this.loadHistory();
        this.addBotMessage(this.config.welcomeMessage);
        this.log('Chatbot inicializado');

        // FIX para WordPress: Força estilos e clicabilidade
        this.ensureClickable();
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

        // Fechar ao clicar no backdrop (fundo escuro)
        if (backdrop) {
            backdrop.addEventListener('click', (e) => {
                // IMPORTANTE: Só fecha se o chatbot estiver realmente aberto
                if (this.isOpen) {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('🌑 Backdrop clicado - fechando chatbot');
                    this.closeChat();
                }
            });
        }

        minimize.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('🔴 Botão fechar CLICADO');
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
            // Fazer requisição à API
            const response = await fetch(this.config.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    history: this.conversationHistory
                })
            });

            const data = await response.json();

            this.hideTyping();

            if (data.success) {
                const botMessage = data.data.message;
                this.addBotMessage(botMessage);

                // Adicionar ao histórico
                this.conversationHistory.push({
                    role: 'bot',
                    text: botMessage
                });

                this.saveHistory();
            } else {
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
     */
    addUserMessage(message) {
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
    }

    /**
     * Adiciona mensagem do bot
     */
    addBotMessage(message) {
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
            // Manter apenas as últimas N mensagens
            const history = this.conversationHistory.slice(-this.config.maxMessages);
            localStorage.setItem('radioChatbotHistory', JSON.stringify(history));
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
                this.conversationHistory = JSON.parse(saved);
            }
        } catch (e) {
            this.log('Erro ao carregar histórico: ' + e, 'error');
        }
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
    window.radioChatbot = new RadioChatbot({
        apiUrl: '/chatbot/chat_api.php',
        position: 'bottom-right',
        debug: false // Mude para true durante testes
    });
}

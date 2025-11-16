/**
 * TTS Player - Solução Híbrida
 * Google Cloud TTS (municipais/regionais) + Web Speech API (nacionais)
 * Rádio Entre Rios - Sistema de Notícias
 */

class TTSPlayer {
    constructor() {
        this.currentAudio = null; // Para Google TTS MP3
        this.speechSynthesis = window.speechSynthesis; // Para Web Speech API
        this.currentUtterance = null;
        this.isPlaying = false;
        this.currentButton = null;

        // Configurações Web Speech API
        this.voiceConfig = {
            lang: 'pt-BR',
            rate: 0.95, // Um pouco mais devagar (mais natural)
            pitch: 1.0,
            volume: 1.0
        };

        // Selecionar melhor voz brasileira disponível
        this.selectBestVoice();
    }

    /**
     * Seleciona a melhor voz pt-BR disponível
     */
    selectBestVoice() {
        if (this.speechSynthesis.getVoices().length === 0) {
            // Aguardar vozes carregarem
            this.speechSynthesis.addEventListener('voiceschanged', () => {
                this.findBestBrazilianVoice();
            });
        } else {
            this.findBestBrazilianVoice();
        }
    }

    findBestBrazilianVoice() {
        const voices = this.speechSynthesis.getVoices();

        // Prioridade: pt-BR > pt > qualquer voz
        const brazilianVoice = voices.find(voice => voice.lang === 'pt-BR');
        const portugueseVoice = voices.find(voice => voice.lang.startsWith('pt'));

        this.selectedVoice = brazilianVoice || portugueseVoice || voices[0];

        console.log('🎤 Voz selecionada:', this.selectedVoice?.name || 'Padrão');
    }

    /**
     * Toca áudio (detecta automaticamente se usa MP3 ou Web Speech API)
     */
    play(noticia, button) {
        // Parar qualquer reprodução anterior
        this.stop();

        this.currentButton = button;
        this.updateButtonState(button, 'loading');

        // Verificar se tem áudio MP3 pré-gerado (Google TTS)
        if (noticia.audio_url && noticia.audio_url.trim() !== '') {
            this.playGoogleTTS(noticia.audio_url, button);
        } else {
            // Usar Web Speech API
            this.playBrowserTTS(noticia, button);
        }
    }

    /**
     * Toca áudio MP3 do Google Cloud TTS
     */
    playGoogleTTS(audioUrl, button) {
        console.log('🎵 Tocando Google TTS:', audioUrl);

        this.currentAudio = new Audio(audioUrl);

        this.currentAudio.addEventListener('canplay', () => {
            this.currentAudio.play();
            this.isPlaying = true;
            this.updateButtonState(button, 'playing');
        });

        this.currentAudio.addEventListener('ended', () => {
            this.isPlaying = false;
            this.updateButtonState(button, 'stopped');
            this.currentAudio = null;
        });

        this.currentAudio.addEventListener('error', (e) => {
            console.error('❌ Erro ao carregar áudio:', e);
            this.updateButtonState(button, 'error');
            alert('Erro ao carregar áudio. Tentando método alternativo...');

            // Fallback para Web Speech API
            this.playBrowserTTS(noticia, button);
        });
    }

    /**
     * Toca usando Web Speech API (browser nativo)
     */
    playBrowserTTS(noticia, button) {
        console.log('🔊 Tocando Web Speech API:', noticia.titulo);

        // Preparar texto
        const texto = this.prepareText(noticia.titulo, noticia.resumo || noticia.conteudo);

        if (!texto || texto.length < 10) {
            alert('Texto muito curto para reprodução de áudio');
            this.updateButtonState(button, 'error');
            return;
        }

        // Criar utterance
        this.currentUtterance = new SpeechSynthesisUtterance(texto);
        this.currentUtterance.lang = this.voiceConfig.lang;
        this.currentUtterance.rate = this.voiceConfig.rate;
        this.currentUtterance.pitch = this.voiceConfig.pitch;
        this.currentUtterance.volume = this.voiceConfig.volume;

        if (this.selectedVoice) {
            this.currentUtterance.voice = this.selectedVoice;
        }

        // Event listeners
        this.currentUtterance.onstart = () => {
            this.isPlaying = true;
            this.updateButtonState(button, 'playing');
        };

        this.currentUtterance.onend = () => {
            this.isPlaying = false;
            this.updateButtonState(button, 'stopped');
            this.currentUtterance = null;
        };

        this.currentUtterance.onerror = (e) => {
            console.error('❌ Erro Web Speech API:', e);
            this.isPlaying = false;
            this.updateButtonState(button, 'error');
            alert('Erro ao reproduzir áudio: ' + e.error);
        };

        // Iniciar reprodução
        this.speechSynthesis.speak(this.currentUtterance);
    }

    /**
     * Prepara texto para TTS (limpa e formata)
     */
    prepareText(titulo, conteudo) {
        // Combinar título e conteúdo
        let texto = titulo + '. ' + (conteudo || '');

        // Remover HTML tags
        const temp = document.createElement('div');
        temp.innerHTML = texto;
        texto = temp.textContent || temp.innerText || '';

        // Limpar caracteres especiais
        texto = texto.replace(/[\r\n]+/g, '. ');
        texto = texto.replace(/\s+/g, ' ');
        texto = texto.trim();

        // Limitar tamanho (evitar textos muito longos)
        if (texto.length > 2000) {
            texto = texto.substring(0, 2000);
            // Cortar na última frase completa
            const lastPeriod = texto.lastIndexOf('.');
            if (lastPeriod > 1000) {
                texto = texto.substring(0, lastPeriod + 1);
            }
        }

        return texto;
    }

    /**
     * Para a reprodução
     */
    stop() {
        // Parar MP3 do Google TTS
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.currentAudio = null;
        }

        // Parar Web Speech API
        if (this.speechSynthesis.speaking) {
            this.speechSynthesis.cancel();
        }

        this.isPlaying = false;

        if (this.currentButton) {
            this.updateButtonState(this.currentButton, 'stopped');
        }

        this.currentUtterance = null;
        this.currentButton = null;
    }

    /**
     * Toggle play/pause
     */
    toggle(noticia, button) {
        if (this.isPlaying && this.currentButton === button) {
            this.stop();
        } else {
            this.play(noticia, button);
        }
    }

    /**
     * Atualiza estado visual do botão
     */
    updateButtonState(button, state) {
        if (!button) return;

        const icon = button.querySelector('i');
        if (!icon) return;

        // Remover classes anteriores
        icon.classList.remove('fa-play', 'fa-pause', 'fa-spinner', 'fa-spin', 'fa-exclamation-triangle');

        switch (state) {
            case 'loading':
                icon.classList.add('fa-spinner', 'fa-spin');
                button.disabled = true;
                break;

            case 'playing':
                icon.classList.add('fa-pause');
                button.disabled = false;
                button.classList.add('tts-playing');
                break;

            case 'stopped':
                icon.classList.add('fa-play');
                button.disabled = false;
                button.classList.remove('tts-playing');
                break;

            case 'error':
                icon.classList.add('fa-exclamation-triangle');
                button.disabled = false;
                button.classList.remove('tts-playing');
                setTimeout(() => {
                    icon.classList.remove('fa-exclamation-triangle');
                    icon.classList.add('fa-play');
                }, 3000);
                break;
        }
    }

    /**
     * Detecta tipo de áudio (para badge visual)
     */
    getAudioType(noticia) {
        if (noticia.audio_url && noticia.audio_url.trim() !== '') {
            // Tem MP3 pré-gerado do Google TTS
            if (noticia.categoria === 'local') {
                return 'premium'; // WaveNet
            } else {
                return 'standard'; // Standard Voice
            }
        } else {
            return 'browser'; // Web Speech API
        }
    }
}

// Instância global
window.ttsPlayer = new TTSPlayer();

/**
 * Adiciona botões TTS aos cards de notícias
 */
function addTTSButtonsToNewsCards() {
    const newsCards = document.querySelectorAll('.news-card, .noticia-item');

    newsCards.forEach(card => {
        // Verificar se já tem botão
        if (card.querySelector('.tts-button')) return;

        // Extrair dados da notícia do card
        const noticia = extractNoticiaFromCard(card);

        if (!noticia) return;

        // Criar botão
        const button = createTTSButton(noticia);

        // Adicionar ao card (pode precisar ajustar seletor)
        const actionArea = card.querySelector('.card-actions, .noticia-footer');
        if (actionArea) {
            actionArea.prepend(button);
        } else {
            card.appendChild(button);
        }
    });
}

/**
 * Extrai dados da notícia do card HTML
 */
function extractNoticiaFromCard(card) {
    try {
        return {
            id: card.dataset.noticiaId || card.id,
            titulo: card.querySelector('.news-title, .noticia-titulo')?.textContent || '',
            resumo: card.querySelector('.news-summary, .noticia-resumo')?.textContent || '',
            conteudo: card.querySelector('.news-content, .noticia-conteudo')?.textContent || '',
            categoria: card.dataset.categoria || 'nacional',
            audio_url: card.dataset.audioUrl || ''
        };
    } catch (e) {
        console.error('Erro ao extrair notícia:', e);
        return null;
    }
}

/**
 * Cria botão TTS
 */
function createTTSButton(noticia) {
    const button = document.createElement('button');
    button.className = 'tts-button';
    button.title = 'Ouvir notícia';

    const audioType = window.ttsPlayer.getAudioType(noticia);

    // Ícone e badge
    button.innerHTML = `
        <i class="fas fa-play"></i>
        <span class="tts-badge tts-badge-${audioType}">
            ${audioType === 'premium' ? '🎵' : audioType === 'standard' ? '🔊' : '📻'}
        </span>
    `;

    // Event listener
    button.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        window.ttsPlayer.toggle(noticia, button);
    });

    return button;
}

// Auto-inicializar quando DOM carregar
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', addTTSButtonsToNewsCards);
} else {
    addTTSButtonsToNewsCards();
}

// Re-adicionar botões quando novos cards forem carregados (AJAX)
const observer = new MutationObserver(() => {
    addTTSButtonsToNewsCards();
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

// Service Worker - Rádio Entre Rios PWA
// Versão 1.0.0

const CACHE_VERSION = 'v1.0.0';
const CACHE_NAME = `radio-entre-rios-${CACHE_VERSION}`;

// Assets para cache offline
const OFFLINE_ASSETS = [
  '/',
  '/pwa/offline.html',
  '/pwa/manifest.json',
  '/pwa/icons/icon-192x192.png',
  '/pwa/icons/icon-512x512.png'
];

// URLs que NUNCA devem ser cacheadas
const NO_CACHE_PATTERNS = [
  '/wp-admin',
  '/wp-login',
  'stream.zeno.fm',
  'admin-ajax.php'
];

// Verificar se URL não deve ser cacheada
function shouldNotCache(url) {
  return NO_CACHE_PATTERNS.some(pattern => url.includes(pattern));
}

// ===== INSTALAÇÃO DO SERVICE WORKER =====
self.addEventListener('install', (event) => {
  console.log('[SW] Instalando Service Worker v' + CACHE_VERSION);

  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[SW] Cache aberto, adicionando assets offline');
        return cache.addAll(OFFLINE_ASSETS.map(url => new Request(url, { cache: 'reload' })));
      })
      .catch((error) => {
        console.error('[SW] Erro ao cachear assets:', error);
      })
  );

  // Ativar imediatamente
  self.skipWaiting();
});

// ===== ATIVAÇÃO E LIMPEZA DE CACHES ANTIGOS =====
self.addEventListener('activate', (event) => {
  console.log('[SW] Ativando Service Worker v' + CACHE_VERSION);

  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('[SW] Removendo cache antigo:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );

  // Assumir controle de todas as páginas imediatamente
  return self.clients.claim();
});

// ===== ESTRATÉGIAS DE FETCH =====
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Não cachear URLs bloqueadas
  if (shouldNotCache(url.href)) {
    return event.respondWith(fetch(request));
  }

  // Network First para APIs e conteúdo dinâmico
  if (url.pathname.includes('/wp-json/') ||
      url.pathname.includes('rds_api.php') ||
      url.pathname.includes('radio_metadata_api.php') ||
      url.pathname.includes('/wp-content/noticias/index.php')) {
    return event.respondWith(networkFirst(request));
  }

  // Cache First para assets estáticos
  if (request.destination === 'image' ||
      request.destination === 'style' ||
      request.destination === 'script' ||
      request.destination === 'font') {
    return event.respondWith(cacheFirst(request));
  }

  // Stale While Revalidate para páginas HTML
  if (request.destination === 'document') {
    return event.respondWith(staleWhileRevalidate(request));
  }

  // Padrão: tentar rede primeiro
  return event.respondWith(networkFirst(request));
});

// ===== ESTRATÉGIA: Network First =====
async function networkFirst(request) {
  try {
    const response = await fetch(request);

    // Só cachear respostas válidas
    if (response && response.status === 200) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }

    return response;
  } catch (error) {
    console.log('[SW] Network First falhou, buscando cache:', request.url);
    const cachedResponse = await caches.match(request);

    if (cachedResponse) {
      return cachedResponse;
    }

    // Retornar página offline se for documento HTML
    if (request.destination === 'document') {
      return caches.match('/pwa/offline.html');
    }

    return new Response('Offline', {
      status: 503,
      statusText: 'Service Unavailable'
    });
  }
}

// ===== ESTRATÉGIA: Cache First =====
async function cacheFirst(request) {
  const cachedResponse = await caches.match(request);

  if (cachedResponse) {
    return cachedResponse;
  }

  try {
    const response = await fetch(request);

    if (response && response.status === 200) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }

    return response;
  } catch (error) {
    console.log('[SW] Cache First falhou:', request.url);
    return new Response('Offline', { status: 503 });
  }
}

// ===== ESTRATÉGIA: Stale While Revalidate =====
async function staleWhileRevalidate(request) {
  const cachedResponse = await caches.match(request);

  const fetchPromise = fetch(request).then((response) => {
    if (response && response.status === 200) {
      const cache = caches.open(CACHE_NAME);
      cache.then((c) => c.put(request, response.clone()));
    }
    return response;
  }).catch(() => {
    // Se falhar, retornar página offline
    return caches.match('/pwa/offline.html');
  });

  return cachedResponse || fetchPromise;
}

// ===== PUSH NOTIFICATIONS =====
self.addEventListener('push', (event) => {
  console.log('[SW] Push notification recebida');

  const data = event.data ? event.data.json() : {};

  const options = {
    body: data.body || 'Nova atualização da Rádio Entre Rios',
    icon: '/pwa/icons/icon-192x192.png',
    badge: '/pwa/icons/icon-72x72.png',
    vibrate: [200, 100, 200],
    data: {
      url: data.url || '/',
      timestamp: Date.now()
    },
    actions: [
      {
        action: 'open',
        title: 'Abrir',
        icon: '/pwa/icons/icon-96x96.png'
      },
      {
        action: 'close',
        title: 'Fechar'
      }
    ],
    tag: data.tag || 'general',
    requireInteraction: false
  };

  event.waitUntil(
    self.registration.showNotification(
      data.title || 'Rádio Entre Rios 105.5 FM',
      options
    )
  );
});

// ===== CLIQUE EM NOTIFICAÇÃO =====
self.addEventListener('notificationclick', (event) => {
  console.log('[SW] Notificação clicada:', event.action);

  event.notification.close();

  if (event.action === 'close') {
    return;
  }

  // Abrir URL da notificação
  const urlToOpen = event.notification.data.url || '/';

  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then((windowClients) => {
        // Verificar se já existe uma janela aberta
        for (let client of windowClients) {
          if (client.url === urlToOpen && 'focus' in client) {
            return client.focus();
          }
        }

        // Se não existe, abrir nova janela
        if (clients.openWindow) {
          return clients.openWindow(urlToOpen);
        }
      })
  );
});

// ===== BACKGROUND SYNC =====
self.addEventListener('sync', (event) => {
  console.log('[SW] Background sync:', event.tag);

  if (event.tag === 'sync-favorites') {
    event.waitUntil(syncFavorites());
  }
});

async function syncFavorites() {
  try {
    // Buscar favoritos pendentes do IndexedDB ou localStorage
    const favorites = JSON.parse(localStorage.getItem('pending_favorites') || '[]');

    for (const newsId of favorites) {
      try {
        await fetch('/wp-json/radio/v1/favorite', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ news_id: newsId })
        });

        console.log('[SW] Favorito sincronizado:', newsId);
      } catch (error) {
        console.error('[SW] Erro ao sincronizar favorito:', newsId, error);
        throw error; // Retry automático
      }
    }

    // Limpar lista após sincronização
    localStorage.removeItem('pending_favorites');
    console.log('[SW] Todos os favoritos sincronizados');

  } catch (error) {
    console.error('[SW] Erro ao sincronizar favoritos:', error);
    throw error;
  }
}

// ===== MENSAGENS DO CLIENTE =====
self.addEventListener('message', (event) => {
  console.log('[SW] Mensagem recebida:', event.data);

  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }

  if (event.data && event.data.type === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.keys().then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => caches.delete(cacheName))
        );
      }).then(() => {
        event.ports[0].postMessage({ success: true });
      })
    );
  }
});

console.log('[SW] Service Worker carregado - Rádio Entre Rios v' + CACHE_VERSION);

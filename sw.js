const CACHE_NAME = 'signsos-v1';
const ASSETS_TO_CACHE = [
  '/devsoc_26/index.html',
  '/devsoc_26/manifest.json',
  '/devsoc_26/icons/icon-192.png',
  '/emergency_numbers.json',
  '/devsoc_26/icons/icon-512.png'
];

// Install
self.addEventListener('install', (event) => {
  console.log('SignSOS SW: Installing...');
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
  self.skipWaiting();
});

// Activate
self.addEventListener('activate', (event) => {
  console.log('SignSOS SW: Activated');
  event.waitUntil(
    caches.keys().then((names) => {
      return Promise.all(
        names.map((name) => {
          if (name !== CACHE_NAME) {
            return caches.delete(name);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch - network first, cache fallback
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Never cache API calls
  if (url.pathname.includes('/predict') ||
      url.pathname.includes('/health') ||
      url.hostname !== location.hostname) {
    event.respondWith(
      fetch(event.request).catch(() => {
        return new Response(JSON.stringify({ gesture: null, error: "offline" }), {
          headers: { 'Content-Type': 'application/json' }
        });
      })
    );
    return;
  }

  // For app assets - network first, cache fallback
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, clone);
        });
        return response;
      })
      .catch(() => {
        return caches.match(event.request);
      })
  );
});
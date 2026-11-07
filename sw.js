// This file makes the website installable on mobile devices
self.addEventListener('install', (e) => {
  console.log('[Service Worker] Install');
});

self.addEventListener('fetch', (e) => {
  // Just passes the request to the network so your API still works perfectly
  e.respondWith(fetch(e.request));
});
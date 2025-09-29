'use strict';

const chatEl = document.getElementById('chat');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const copyBtn = document.getElementById('copyBtn');
let currentTabId = null;
let lastUrlInActiveTab = null;
let panelVisible = true;

function sanitizePlain(text) {
  return String(text || '').replace(/\*\*/g, '');
}

function linkify(text) {
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  return sanitizePlain(text).replace(urlRegex, (url) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`);
}

function pushMsg(role, text, opts = {}) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');
  if (opts.external) wrap.classList.add('external');
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = role === 'user' ? 'Вы' : 'ИИ';
  const body = document.createElement('div');
  body.innerHTML = linkify(text);
  wrap.appendChild(meta);
  wrap.appendChild(body);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
}

async function getBackendBaseUrl() {
  return new Promise(resolve => {
    chrome.storage.local.get({ backendBaseUrl: 'http://localhost:8000' }, (res) => {
      resolve(res.backendBaseUrl.replace(/\/$/, ''));
    });
  });
}

async function sendChat(message) {
  const baseUrl = await getBackendBaseUrl();
  let pageUrl = null, pageTitle = null, pageText = null;

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  currentTabId = tab?.id;

  if (!currentTabId) {
    pushMsg('assistant', 'Не удалось определить активную вкладку.');
    return;
  }

  let pageData = null;
  try {
    pageData = await new Promise((resolve) => {
      chrome.tabs.sendMessage(currentTabId, { type: 'REQUEST_CONTEXT' }, (resp) => {
        if (chrome.runtime.lastError) return resolve(null);
        resolve(resp || null);
      });
    });
  } catch (_) {}

  pageUrl = pageData?.url || tab.url;
  pageTitle = pageData?.title || tab.title;
  pageText = pageData?.text || '';

  lastUrlInActiveTab = pageUrl;

  const resp = await fetch(baseUrl + '/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, page_url: pageUrl, page_title: pageTitle, page_text: pageText })
  });
  if (!resp.ok) throw new Error('Ошибка сервера: ' + resp.status);
  return resp.json();
}

// Send on button click
sendBtn?.addEventListener('click', async () => {
  const text = (chatInput.value || '').trim();
  if (!text) return;
  chatInput.value = '';
  pushMsg('user', text);
  try {
    const data = await sendChat(text);
    if (data?.answer) pushMsg('assistant', data.answer, { external: !!data?.used_external });
    if (Array.isArray(data?.sources) && data.sources.length) {
      const lines = data.sources.map(s => {
        const t = s?.title || s?.url || '';
        const u = s?.url || '';
        return `${t}: ${u}`;
      }).filter(Boolean);
      if (lines.length) pushMsg('assistant', lines.join('\n'), { external: !!data?.used_external });
    }
  } catch (e) {
    pushMsg('assistant', 'Ошибка: ' + (e?.message || String(e)));
  }
});

// Enter handling: Shift+Enter -> newline; Enter -> send
chatInput?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    if (e.shiftKey) {
      // allow newline
      return;
    }
    e.preventDefault();
    sendBtn?.click();
  }
});

// Copy chat content
copyBtn?.addEventListener('click', async () => {
  try {
    const lines = Array.from(chatEl.querySelectorAll('.msg')).map(w => {
      const who = w.querySelector('.meta')?.textContent || '';
      const text = w.querySelector('div:nth-child(2)')?.textContent || '';
      return (who ? who + ': ' : '') + text;
    });
    const blob = new Blob([lines.join('\n\n')], { type: 'text/plain' });
    await navigator.clipboard.writeText(await blob.text());
    copyBtn.textContent = 'Скопировано';
    setTimeout(() => { copyBtn.textContent = 'Копировать'; }, 1500);
  } catch (e) {
    copyBtn.textContent = 'Ошибка';
    setTimeout(() => { copyBtn.textContent = 'Копировать'; }, 1500);
  }
});

// Refresh context on tab changes while the side panel is open
chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  if (currentTabId !== tabId) {
    currentTabId = tabId;
    if (chatEl) chatEl.innerHTML = '';
    try {
      const tab = await chrome.tabs.get(tabId);
      lastUrlInActiveTab = tab.url;
    } catch (e) {
      lastUrlInActiveTab = null;
    }
  }
});

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && currentTabId === tabId) {
    const newUrl = tab.url;
    let shouldClear = true;
    if (lastUrlInActiveTab && newUrl) {
      try {
        const prev = new URL(lastUrlInActiveTab);
        const next = new URL(newUrl);
        if (prev.hostname === next.hostname) {
          shouldClear = false;
        }
      } catch (e) {
        shouldClear = true;
      }
    }

    if (shouldClear) {
      if (chatEl) chatEl.innerHTML = '';
    }
    lastUrlInActiveTab = newUrl;
  }
});

document.addEventListener('DOMContentLoaded', async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    currentTabId = tab?.id;
    lastUrlInActiveTab = tab?.url;
  } catch (_) {}
});




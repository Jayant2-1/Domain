const API_BASE = '/api';

export async function askQuestion(username, topic, question) {
  const res = await fetch(`${API_BASE}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, topic, question }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Tutor ask (non-streaming) - two-stage reasoning.
 */
export async function askTutor(username, topic, question, thinkingEnabled = true) {
  const res = await fetch(`${API_BASE}/tutor/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      username,
      topic,
      question,
      thinking_enabled: thinkingEnabled,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Tutor ask (streaming) - SSE two-stage reasoning.
 *
 * @param {string} username
 * @param {string} topic
 * @param {string} question
 * @param {function} onEvent - Callback called with each SSE event object.
 *   Events: thinking_start, thinking_done, answering_start, answering_done,
 *           done, error, validation_failed
 * @returns {Promise<void>}
 */
export async function askTutorStream(username, topic, question, onEvent, history = []) {
  const res = await fetch(`${API_BASE}/tutor/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, topic, question, history }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Parse SSE events from buffer
    const lines = buffer.split('\n');
    buffer = lines.pop() || ''; // keep incomplete line in buffer

    let currentEventType = null;
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEventType = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        const dataStr = line.slice(6);
        try {
          const data = JSON.parse(dataStr);
          onEvent(data);
        } catch {
          // ignore parse errors on partial data
        }
        currentEventType = null;
      }
      // blank lines are event separators (already handled)
    }
  }
}

export async function sendFeedback(interactionId, feedback) {
  const res = await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ interaction_id: interactionId, feedback }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function getSkills(username) {
  const res = await fetch(`${API_BASE}/skills/${encodeURIComponent(username)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut, Brain, Send, Menu } from 'lucide-react';
import ChatMessage from '../components/ChatMessage';
import Sidebar from '../components/Sidebar';
import EmptyState from '../components/EmptyState';
import { useAuth } from '../context/AuthContext';
import { askTutorStream, sendFeedback, getSkills, getHealth } from '../api';

export default function DashboardPage() {
  const { user, logout, getToken } = useAuth();
  const navigate = useNavigate();
  const username = user?.username || 'student';

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [skills, setSkills] = useState([]);
  const [serverStatus, setServerStatus] = useState('loading');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        await getHealth();
        if (!cancelled) setServerStatus('online');
      } catch {
        if (!cancelled) setServerStatus('offline');
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => { cancelled = true; clearInterval(interval); };
  }, []);

  const refreshSkills = useCallback(async () => {
    try {
      const data = await getSkills(username);
      setSkills(data.skills || []);
    } catch { /* ok on first use */ }
  }, [username]);

  useEffect(() => { refreshSkills(); }, [refreshSkills]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || loading) return;

    setInput('');
    setLoading(true);

    let assistantIdx;
    setMessages(prev => {
      assistantIdx = prev.length + 1;
      return [
        ...prev,
        { role: 'user', content: q },
        {
          role: 'assistant', content: '', analysis: null, answer: null,
          thinkingTime: null, isThinking: true, interactionId: null, feedback: null,
        },
      ];
    });

    try {
      // Send last 6 messages as conversation history
      const recentHistory = messages
        .slice(-6)
        .filter(m => m.content)
        .map(m => ({ role: m.role, content: m.content }));

      await askTutorStream(username, 'dsa', q, (event) => {
        setMessages(prev => {
          const updated = [...prev];
          const msg = { ...updated[assistantIdx] };
          switch (event.event) {
            case 'thinking_start': msg.isThinking = true; break;
            case 'thinking_done':
              msg.isThinking = false;
              msg.analysis = event.analysis || '';
              msg.thinkingTime = event.thinking_time;
              msg.intent = event.intent;
              break;
            case 'answering_start': msg.isAnswering = true; break;
            case 'answering_done':
              msg.isAnswering = false;
              msg.answer = event.answer || '';
              msg.content = event.answer || '';
              msg.answeringTime = event.answering_time;
              break;
            case 'done':
              msg.isThinking = false;
              msg.isAnswering = false;
              msg.interactionId = event.interaction_id;
              msg.totalTime = event.total_time;
              msg.feedback = null;
              break;
            case 'validation_failed':
              msg.isThinking = false;
              msg.content = event.message;
              msg.answer = event.message;
              break;
            case 'error':
              msg.isThinking = false;
              msg.isAnswering = false;
              msg.content = `Error: ${event.message}`;
              msg.answer = `Error: ${event.message}`;
              break;
            default: break;
          }
          updated[assistantIdx] = msg;
          return updated;
        });
      }, recentHistory);
      refreshSkills();
    } catch (err) {
      setMessages(prev => {
        const updated = [...prev];
        if (updated[assistantIdx]) {
          updated[assistantIdx] = {
            ...updated[assistantIdx],
            isThinking: false, isAnswering: false,
            content: `Error: ${err.message}. Is the backend running?`,
            answer: `Error: ${err.message}`,
          };
        }
        return updated;
      });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleFeedback = async (msgIndex, value) => {
    const msg = messages[msgIndex];
    if (!msg.interactionId || msg.feedback !== null) return;
    try {
      await sendFeedback(msg.interactionId, value);
      setMessages(prev => prev.map((m, i) =>
        i === msgIndex ? { ...m, feedback: value } : m
      ));
      refreshSkills();
    } catch (err) { console.error('Feedback failed:', err); }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestion = (text) => {
    setInput(text);
    inputRef.current?.focus();
  };

  const handleLogout = () => { logout(); navigate('/'); };

  return (
    <div className="dashboard">
      <Sidebar
        username={username}
        skills={skills}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />

      <div className="dash-main">
        <header className="dash-header">
          <div className="dash-header-left">
            <button className="icon-btn" onClick={() => setSidebarOpen(!sidebarOpen)}>
              <Menu size={18} />
            </button>
            <div className="header-brand">
              <Brain size={20} className="brand-icon-svg" />
              <h2>MLML</h2>
            </div>
          </div>
          <div className="dash-header-right">
            <span className={`status-badge ${serverStatus}`}>
              {serverStatus === 'online' ? 'Connected' :
               serverStatus === 'offline' ? 'Offline' : 'Checking...'}
            </span>
            <button className="icon-btn" onClick={handleLogout} title="Logout">
              <LogOut size={18} />
            </button>
          </div>
        </header>

        <div className="chat-messages">
          {messages.length === 0 && !loading ? (
            <EmptyState onSuggestion={handleSuggestion} />
          ) : (
            messages.map((msg, i) => (
              <ChatMessage
                key={i}
                message={msg}
                onFeedback={(val) => handleFeedback(i, val)}
              />
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-area">
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder="Ask me anything about data structures & algorithms..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={loading}
          />
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={loading || !input.trim()}
          >
            {loading ? (
              <div className="typing-indicator"><span/><span/><span/></div>
            ) : (
              <Send size={18} />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

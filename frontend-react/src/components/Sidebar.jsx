import { Brain, X } from 'lucide-react';

export default function Sidebar({ username, skills, isOpen, onToggle }) {
  return (
    <aside className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
      <div className="sidebar-header">
        <div className="brand">
          <Brain size={22} className="brand-icon-svg" />
          <div>
            <h1>MLML</h1>
            <span>DSA Tutor</span>
          </div>
        </div>
        <button className="icon-btn sidebar-close" onClick={onToggle}>
          <X size={18} />
        </button>
      </div>

      <div className="sidebar-user">
        <div className="user-avatar">{username[0]?.toUpperCase()}</div>
        <span className="user-name">{username}</span>
      </div>

      <div className="skills-section">
        <h3>Skill Ratings</h3>
        {skills.length === 0 ? (
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Ask questions to build your profile
          </p>
        ) : (
          skills.map((s) => (
            <div key={s.topic} className="skill-item">
              <span className="skill-name">{s.topic.replace(/_/g, ' ')}</span>
              <span className={`skill-rating ${
                s.rating < 900 ? 'low' : s.rating < 1100 ? 'mid' : 'high'
              }`}>
                {Math.round(s.rating)}
              </span>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}

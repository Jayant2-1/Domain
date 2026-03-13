import { useState } from 'react';

/**
 * ThoughtBlock - Collapsible "Thought for X seconds" panel.
 * Shows the model's structured reasoning (analysis) in a
 * DeepSeek-style expandable block.
 */
export default function ThoughtBlock({ analysis, thinkingTime, isThinking }) {
  const [expanded, setExpanded] = useState(false);

  // While still thinking, show the animated spinner
  if (isThinking) {
    return (
      <div className="thought-block thinking">
        <div className="thought-header">
          <div className="thinking-spinner" />
          <span className="thought-label">Thinking...</span>
        </div>
      </div>
    );
  }

  // No analysis to show
  if (!analysis) return null;

  const timeLabel = thinkingTime
    ? `Thought for ${thinkingTime}s`
    : 'Thought process';

  return (
    <div className={`thought-block ${expanded ? 'expanded' : 'collapsed'}`}>
      <button
        className="thought-header"
        onClick={() => setExpanded(prev => !prev)}
        aria-expanded={expanded}
      >
        <svg
          className={`thought-chevron ${expanded ? 'rotated' : ''}`}
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
        >
          <path
            d="M6 4L10 8L6 12"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span className="thought-label">{timeLabel}</span>
      </button>
      {expanded && (
        <div className="thought-content">
          {analysis}
        </div>
      )}
    </div>
  );
}

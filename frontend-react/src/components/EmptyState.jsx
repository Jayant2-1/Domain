const SUGGESTIONS = [
  'Explain how binary search works step by step',
  'What is a linked list and when should I use it?',
  'Compare BFS vs DFS with examples',
  'How does dynamic programming work?',
  'Explain the two-pointer technique',
  'What is the time complexity of merge sort?',
];

export default function EmptyState({ onSuggestion }) {
  return (
    <div className="empty-state">
      <div className="icon">🧠</div>
      <h3>MLML — Your DSA Tutor</h3>
      <p>
        Ask me anything about <strong>Data Structures & Algorithms</strong>.
        I can explain concepts, solve problems, compare approaches, and help you prepare for coding interviews.
      </p>
      <div className="suggestion-chips">
        {SUGGESTIONS.map((text, i) => (
          <button key={i} className="suggestion-chip" onClick={() => onSuggestion(text)}>
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}

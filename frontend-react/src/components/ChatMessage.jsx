import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ThoughtBlock from './ThoughtBlock';

const MarkdownContent = ({ children }) => (
  <ReactMarkdown
    remarkPlugins={[remarkGfm]}
    components={{
      code({ node, inline, className, children, ...props }) {
        const match = /language-(\w+)/.exec(className || '');
        return !inline && match ? (
          <SyntaxHighlighter
            style={oneDark}
            language={match[1]}
            PreTag="div"
            {...props}
          >
            {String(children).replace(/\n$/, '')}
          </SyntaxHighlighter>
        ) : (
          <code className={className} {...props}>
            {children}
          </code>
        );
      },
    }}
  >
    {children}
  </ReactMarkdown>
);

export default function ChatMessage({ message, onFeedback }) {
  const { role, content, feedback, newRating, analysis, thinkingTime, isThinking, answer } = message;

  // For tutor responses with thinking mode
  const hasTutor = role === 'assistant' && (analysis || isThinking || answer !== undefined);

  return (
    <div className={`message ${role}`}>
      {hasTutor ? (
        <>
          <ThoughtBlock
            analysis={analysis}
            thinkingTime={thinkingTime}
            isThinking={isThinking}
          />
          {answer !== undefined && answer !== null ? (
            <div className="final-answer"><MarkdownContent>{answer}</MarkdownContent></div>
          ) : !isThinking ? (
            <div><MarkdownContent>{content}</MarkdownContent></div>
          ) : null}
        </>
      ) : (
        <div>{role === 'assistant' ? <MarkdownContent>{content}</MarkdownContent> : content}</div>
      )}
      {role === 'assistant' && newRating != null && (
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
          Rating updated &rarr; {Math.round(newRating)}
        </div>
      )}
      {role === 'assistant' && feedback === null && !isThinking && (
        <div className="message-feedback">
          <button
            className="feedback-btn"
            onClick={() => onFeedback(1)}
            title="Helpful"
          >
            Helpful
          </button>
          <button
            className="feedback-btn"
            onClick={() => onFeedback(-1)}
            title="Not helpful"
          >
            Not helpful
          </button>
        </div>
      )}
      {role === 'assistant' && feedback !== null && (
        <div className="message-feedback">
          <span className={`feedback-btn ${feedback > 0 ? 'selected-good' : 'selected-bad'}`}>
            {feedback > 0 ? 'Marked helpful' : 'Marked not helpful'}
          </span>
        </div>
      )}
    </div>
  );
}

import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Brain, Zap, Target, BarChart3, Code, GraduationCap,
  ArrowRight, Sparkles, Shield,
} from 'lucide-react';
import Scene3D from '../components/Scene3D';
import ScrollSection from '../components/ScrollSection';
import Navbar from '../components/Navbar';

const features = [
  {
    icon: Brain,
    title: 'Adaptive AI Tutor',
    desc: 'Powered by a fine-tuned Mistral-7B model that adapts to your skill level using ELO ratings.',
  },
  {
    icon: Zap,
    title: 'Real-Time Reasoning',
    desc: 'Watch the AI think through problems step-by-step with our transparent reasoning pipeline.',
  },
  {
    icon: Target,
    title: 'RAG-Enhanced Answers',
    desc: 'Retrieval-Augmented Generation pulls the most relevant DSA concepts for every question.',
  },
  {
    icon: BarChart3,
    title: 'Skill Tracking',
    desc: 'ELO-based rating system tracks your progress across 10+ DSA topics in real time.',
  },
  {
    icon: Code,
    title: 'Code Examples',
    desc: 'Clean Python implementations with time & space complexity analysis for every explanation.',
  },
  {
    icon: Shield,
    title: 'Fully Offline',
    desc: 'Runs entirely on your hardware. No API calls, no data leaves your machine.',
  },
];

const stats = [
  { value: '17K+', label: 'Training Samples' },
  { value: '10+', label: 'DSA Topics' },
  { value: '7B', label: 'Model Parameters' },
  { value: '96%', label: 'Training Accuracy' },
];

export default function LandingPage() {
  return (
    <div className="landing">
      <Navbar />

      {/* Hero */}
      <header className="hero">
        <Scene3D className="hero-scene" />
        <div className="hero-content">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <span className="hero-badge">
              <Sparkles size={14} /> AI-Powered DSA Tutor
            </span>
            <h1>
              Master Data Structures
              <br />
              <span className="gradient-text">& Algorithms</span>
            </h1>
            <p className="hero-subtitle">
              An adaptive AI tutor that learns your level and guides you
              through DSA concepts with real-time reasoning, code examples,
              and personalized difficulty scaling.
            </p>
          </motion.div>
          <motion.div
            className="hero-actions"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            <Link to="/register" className="btn-primary">
              <GraduationCap size={18} />
              Start Learning
              <ArrowRight size={16} />
            </Link>
            <Link to="/dashboard" className="btn-secondary">
              Try It Now
            </Link>
          </motion.div>
        </div>
      </header>

      {/* Stats */}
      <ScrollSection className="stats-bar">
        <div className="stats-grid">
          {stats.map((s) => (
            <div key={s.label} className="stat-item">
              <span className="stat-value">{s.value}</span>
              <span className="stat-label">{s.label}</span>
            </div>
          ))}
        </div>
      </ScrollSection>

      {/* Features */}
      <section className="features-section">
        <ScrollSection className="section-header">
          <h2>Built for Serious Learners</h2>
          <p>Everything you need to go from beginner to advanced in DSA.</p>
        </ScrollSection>
        <div className="features-grid">
          {features.map((f, i) => (
            <ScrollSection key={f.title} className="feature-card" delay={i * 0.1}>
              <div className="feature-icon">
                <f.icon size={24} />
              </div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </ScrollSection>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="how-section">
        <ScrollSection className="section-header">
          <h2>How It Works</h2>
          <p>Three simple steps to accelerate your DSA learning.</p>
        </ScrollSection>
        <div className="steps-grid">
          {[
            { step: '01', title: 'Ask a Question', desc: 'Pick a topic and ask anything — from basics to advanced problems.' },
            { step: '02', title: 'AI Reasons & Responds', desc: 'Watch the AI think step-by-step, then delivers a tailored explanation.' },
            { step: '03', title: 'Rate & Improve', desc: 'Give feedback. The system adapts difficulty to your growing skill level.' },
          ].map((s, i) => (
            <ScrollSection key={s.step} className="step-card" delay={i * 0.15}>
              <span className="step-number">{s.step}</span>
              <h3>{s.title}</h3>
              <p>{s.desc}</p>
            </ScrollSection>
          ))}
        </div>
      </section>

      {/* CTA */}
      <ScrollSection className="cta-section">
        <h2>Ready to Level Up?</h2>
        <p>Join and start mastering DSA with an AI that adapts to you.</p>
        <Link to="/register" className="btn-primary btn-lg">
          <GraduationCap size={20} />
          Create Free Account
          <ArrowRight size={18} />
        </Link>
      </ScrollSection>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-brand">
            <Brain size={20} />
            <span>MLML</span>
          </div>
          <p>&copy; {new Date().getFullYear()} MLML — Offline Adaptive DSA Tutor</p>
        </div>
      </footer>
    </div>
  );
}

-- ============================================================
-- Offline Adaptive DSA Tutor — Database Schema (SQLite)
-- Normalised, indexed, no JSON blobs
-- ============================================================

CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);

CREATE TABLE IF NOT EXISTS topics (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS skill_ratings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic_id   INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    rating     REAL    NOT NULL DEFAULT 1000.0,
    matches    INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, topic_id)
);
CREATE INDEX IF NOT EXISTS idx_skill_ratings_user_topic
    ON skill_ratings (user_id, topic_id);

CREATE TABLE IF NOT EXISTS interactions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic_id       INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    question       TEXT    NOT NULL,
    rag_context    TEXT,
    response       TEXT    NOT NULL,
    difficulty     REAL    NOT NULL,
    rating_before  REAL    NOT NULL,
    rating_after   REAL,
    feedback       INTEGER,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_interactions_user     ON interactions (user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_feedback ON interactions (feedback);
CREATE INDEX IF NOT EXISTS idx_interactions_created  ON interactions (created_at);

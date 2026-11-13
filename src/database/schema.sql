CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS resource (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    reader_count BIGINT CHECK (reader_count IS NULL OR reader_count >= 0)
);

CREATE TABLE IF NOT EXISTS article_text (
    id UUID PRIMARY KEY,
    topic TEXT NOT NULL,
    text TEXT NOT NULL
);

ALTER TABLE article_text ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ;
COMMENT ON COLUMN article_text.published_at IS 'Дата публикации статьи в источнике (UTC)';

CREATE INDEX IF NOT EXISTS article_text_published_at_idx
    ON article_text (published_at);

CREATE TABLE IF NOT EXISTS article_sources (
    article_id UUID PRIMARY KEY REFERENCES article_text(id) ON DELETE CASCADE,
    resource_id UUID NOT NULL REFERENCES resource(id) ON DELETE RESTRICT,
    source TEXT NOT NULL CHECK (source <> '')
);

CREATE INDEX IF NOT EXISTS article_sources_resource_idx
    ON article_sources (resource_id);

CREATE TABLE IF NOT EXISTS article_metrics (
    article_id UUID PRIMARY KEY REFERENCES article_text(id) ON DELETE CASCADE,
    views BIGINT CHECK (views IS NULL OR views >= 0),
    reactions BIGINT CHECK (reactions IS NULL OR reactions >= 0),
    comments BIGINT CHECK (comments IS NULL OR comments >= 0)
);

ALTER TABLE article_metrics
    ADD COLUMN IF NOT EXISTS popularity_pattern_score DOUBLE PRECISION
    CHECK (popularity_pattern_score BETWEEN -10 AND 10);

ALTER TABLE article_metrics ADD COLUMN IF NOT EXISTS audience BIGINT
    CHECK (audience IS NULL OR audience >= 0);
ALTER TABLE article_metrics ADD COLUMN IF NOT EXISTS collected_at TIMESTAMPTZ;
COMMENT ON COLUMN article_metrics.audience
    IS 'Аудитория источника на момент сбора этих метрик';
COMMENT ON COLUMN article_metrics.collected_at IS 'Время сбора социальных метрик (UTC)';

COMMENT ON COLUMN article_metrics.popularity_pattern_score
    IS 'оценка соответствия паттернам популярности';

CREATE TABLE IF NOT EXISTS article_vectors (
    article_id UUID PRIMARY KEY REFERENCES article_text(id) ON DELETE CASCADE,
    embedding vector NOT NULL,
    embedding_model TEXT
);

-- Legacy vectors have unknown provenance until they are recomputed.
ALTER TABLE article_vectors ADD COLUMN IF NOT EXISTS embedding_model TEXT;
-- Dimensions are validated against the configured embedding profile in Python.
-- This cast preserves legacy coordinates and allows explicit model migrations.
ALTER TABLE article_vectors ALTER COLUMN embedding TYPE vector;
COMMENT ON COLUMN article_vectors.embedding_model
    IS 'Модель, которой получен вектор; NULL означает неизвестное происхождение';

-- Removed extension: metrics keep a single current row per article.
DROP TABLE IF EXISTS article_metric_observations;

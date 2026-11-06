-- Full overview of the five application tables, without a row limit.
-- Unlinked articles and resources are also included.
SELECT
    r.id AS resource_id,
    r.name AS resource_name,
    r.reader_count,
    t.id AS article_id,
    t.topic,
    t.text,
    t.published_at,
    s.source,
    m.views,
    m.reactions,
    m.comments,
    m.audience AS audience_at_metric_collection,
    m.collected_at AS metrics_collected_at,
    m.popularity_pattern_score,
    v.embedding::text AS text_vector,
    vector_dims(v.embedding) AS vector_dimensions
FROM public.resource AS r
FULL OUTER JOIN public.article_sources AS s ON s.resource_id = r.id
FULL OUTER JOIN public.article_text AS t ON t.id = s.article_id
LEFT JOIN public.article_metrics AS m ON m.article_id = t.id
LEFT JOIN public.article_vectors AS v ON v.article_id = t.id
ORDER BY r.name NULLS LAST, t.id;

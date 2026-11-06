"""Rank popular vector-search candidates using text first, then topic and score."""

import asyncio
import math
import unicodedata
from dataclasses import asdict, dataclass, field

from src.database.model_embedding import TextEmbedder
from src.database.table_classes import ArticleVectorTable
from src.database.topic_detector import ArticleTopicDetector
from src.database.vector_utils import normalize_vector


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True)
class SimilarArticle:
    article_id: str
    text: str
    topic: str
    popularity_score: float
    text_similarity: float | None
    topic_similarity: float
    rank_score: float
    resource_name: str | None = None
    source: str | None = None
    selection_method: str = "content_similarity"

    def __post_init__(self):
        if not self.article_id or not self.text.strip() or not self.topic.strip():
            raise ValueError("Reference article requires an identity, text and topic")
        if not math.isfinite(self.popularity_score) or not -10 <= self.popularity_score <= 10:
            raise ValueError("Reference popularity score must be in [-10, 10]")
        if self.selection_method not in ("content_similarity", "topic_popularity"):
            raise ValueError("Unknown example selection method")
        if self.text_similarity is None and self.selection_method != "topic_popularity":
            raise ValueError("Content search requires a known text similarity")
        values = (self.topic_similarity, self.rank_score)
        if self.text_similarity is not None:
            values += (self.text_similarity,)
        for value in values:
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError("Reference similarities and rank must be finite and in [-1, 1]")
        object.__setattr__(self, "article_id", str(self.article_id))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    topic: str
    articles: list[SimilarArticle]
    candidates_found: int
    min_score: float
    selection_method: str = "content_similarity"
    fallback_candidates_found: int = 0
    warnings: list[str] = field(default_factory=list)

    def retrieval_metadata(self):
        return dict(candidates_found=self.candidates_found, min_popularity_score=self.min_score,
                    selection_method=self.selection_method, fallback_candidates_found=self.fallback_candidates_found,
                    warnings=self.warnings)

    def to_dict(self):
        return dict(topic=self.topic, candidates_found=self.candidates_found, min_score=self.min_score,
                    selection_method=self.selection_method, fallback_candidates_found=self.fallback_candidates_found,
                    warnings=self.warnings, articles=[a.to_dict() for a in self.articles])


class SearchSimillar:
    """Use existing E5 vectors and observed module-2 labels in PostgreSQL."""

    def __init__(self, vectors: ArticleVectorTable, embedder: TextEmbedder,
                 topic_detector=None, *, min_score=5.0, min_similarity=0.6,
                 candidate_limit=100, text_weight=.8, topic_weight=.15, popularity_weight=.05):
        weights = (text_weight, topic_weight, popularity_weight)
        if (not all(math.isfinite(w) and w >= 0 for w in weights)
                or not math.isclose(sum(weights), 1) or text_weight <= topic_weight + popularity_weight):
            raise ValueError("Weights must sum to 1 and give text more weight than the other signals combined")
        if (not math.isfinite(min_score) or not -10 <= min_score <= 10
                or not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1):
            raise ValueError("Invalid minimum score or text similarity")
        if not isinstance(candidate_limit, int) or isinstance(candidate_limit, bool) or candidate_limit < 1:
            raise ValueError("Candidate limit must be a positive integer")
        self.vectors, self.embedder = vectors, embedder
        self.topic_detector = topic_detector or ArticleTopicDetector(embedder)
        self.min_score, self.min_similarity, self.candidate_limit = min_score, min_similarity, candidate_limit
        self.weights = weights
        self._topics = {}

    def _topic_vector(self, topic):
        if topic not in self._topics:
            self._topics[topic] = normalize_vector(
                self.embedder.embed_query(f"Тема статьи: {topic}"))
        return self._topics[topic]

    def _rank(self, rows, text, topic):
        query_topic = self._topic_vector(topic)
        ranked = []
        for row in rows:
            if normalized_text(row["text"]) == normalized_text(text):
                continue
            other_topic = self._topic_vector(row["topic"])
            topic_similarity = max(-1.0, min(1.0, math.fsum(a*b for a, b in zip(query_topic, other_topic))))
            similarity = max(-1.0, min(1.0, float(row["text_similarity"])))
            score = float(row["score"])
            rank = (self.weights[0]*similarity + self.weights[1]*topic_similarity
                    + self.weights[2]*(score+10)/20)
            ranked.append(SimilarArticle(str(row["article_id"]), row["text"], row["topic"],
                score, similarity, topic_similarity, rank, row.get("resource_name"), row.get("source")))
        ranked.sort(key=lambda a: (-a.rank_score, -a.popularity_score, -a.text_similarity, a.article_id))
        unique, seen = [], set()
        for article in ranked:
            key = normalized_text(article.text)
            if key not in seen:
                unique.append(article)
                seen.add(key)
        return unique

    @staticmethod
    def _best_in_topic(rows, text, topic):
        # Popularity alone determines this order; low content similarity must
        # not be confused with a content match or hide a better-scoring example.
        ordered = sorted(rows, key=lambda row: (-float(row["score"]), str(row["article_id"])))
        articles, seen = [], {normalized_text(text)}
        for row in ordered:
            key = normalized_text(row["text"])
            if row["topic"] != topic or key in seen:
                continue
            similarity = row.get("text_similarity")
            if similarity is not None:
                similarity = max(-1.0, min(1.0, float(similarity)))
            score = float(row["score"])
            articles.append(SimilarArticle(str(row["article_id"]), row["text"], topic,
                score, similarity, 1.0, (score+10)/20, row.get("resource_name"), row.get("source"),
                selection_method="topic_popularity"))
            seen.add(key)
        return articles

    async def search(self, text: str, topic: str | None = None, *, limit: int | None = None,
                     exclude_article_id=None) -> SearchResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be nonempty")
        if topic is not None and (not isinstance(topic, str) or not topic.strip()):
            raise ValueError("Topic must be nonempty when provided")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise ValueError("Limit must be a positive integer")
        embedding = await asyncio.to_thread(self.embedder.embed_document, text)
        if topic is None:
            topic = await asyncio.to_thread(self.topic_detector.detect_from_embedding, embedding)
        topic = topic.strip()
        rows = await self.vectors.search_popular(embedding, min_score=self.min_score,
            min_similarity=self.min_similarity, limit=self.candidate_limit, exclude_article_id=exclude_article_id)
        articles = await asyncio.to_thread(self._rank, rows, text, topic)
        method, fallback_count, warnings = "content_similarity", 0, []
        if not articles:
            fallback = await self.vectors.best_by_topic(embedding, topic=topic,
                limit=self.candidate_limit, exclude_article_id=exclude_article_id)
            articles = await asyncio.to_thread(self._best_in_topic, fallback, text, topic)
            fallback_count = len(fallback)
            method = "topic_popularity" if articles else "general"
            warning = "Похожих по содержанию статей не найдено. "
            warning += (f"Для примеров подобраны лучшие тексты по теме «{topic}»."
                        if articles else f"В БД также нет доступных примеров по теме «{topic}». Рекомендации будут общими редакторскими.")
            warnings.append(warning)
        return SearchResult(topic, articles if limit is None else articles[:limit], len(rows), self.min_score,
                            method, fallback_count, warnings)

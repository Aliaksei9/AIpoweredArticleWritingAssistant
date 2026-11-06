"""Orchestrate retrieval and LLM editing without writing articles back to DB."""


class TextCorrectionModule:
    def __init__(self, search, analysis):
        self.search, self.analysis = search, analysis

    async def prepare(self, text, topic=None, *, exclude_article_id=None):
        found = await self.search.search(text, topic, exclude_article_id=exclude_article_id)
        plan = await self.analysis.prepare(text, found.articles, topic=found.topic)
        plan.context["warnings"] = list(dict.fromkeys([*found.warnings, *plan.context["warnings"]]))
        plan.context["retrieval_method"] = found.selection_method
        return found, plan

    async def correct(self, text, topic=None, *, exclude_article_id=None):
        found, plan = await self.prepare(text, topic, exclude_article_id=exclude_article_id)
        result = await self.analysis.execute(plan)
        result["retrieval"] = found.retrieval_metadata()
        return result

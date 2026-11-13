"""A bounded, audience-normalized popularity assessment formula."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AssesmentConformPPmetricsBased:
    """Assess a post relative to its channel's audience.

    Combine weighted log1p rates into activity P, then return
    10*tanh(log(P/log(2))/temperature). Views and audience must be
    positive; zero reactions/comments add no activity. Reference
    rates give zero when all signals reach their reference.
    """

    FORMULA_VERSION = "log-activity-v3"

    view_reference_rate: float = 0.10
    reaction_reference_rate: float = 0.0008
    comment_reference_rate: float = 0.00005
    temperature: float = 1.5

    def __post_init__(self) -> None:
        for value in (
            self.view_reference_rate,
            self.reaction_reference_rate,
            self.comment_reference_rate,
            self.temperature,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Reference rates must be positive finite numbers")

    @staticmethod
    def _metric(value: int | float | None) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Metrics must be nonnegative finite numbers") from exc
        if not math.isfinite(number) or number < 0:
            raise ValueError("Metrics must be nonnegative finite numbers")
        return number

    def assess(
        self,
        views: int | float | None,
        reactions: int | float | None,
        comments: int | float | None,
        audience: int | float | None,
    ) -> float | None:
        v, r, c, a = map(self._metric, (views, reactions, comments, audience))
        if v is None or r is None or c is None or a is None or v == 0 or a == 0:
            return None

        def log_activity(count: float, reference: float) -> float:
            if count == 0:
                return -math.inf  # log(0): no activity, not a fixed score penalty.
            log_ratio = math.log(count) - math.log(a) - math.log(reference)
            # log(log1p(exp(log_ratio))) without forming count/(reference*A).
            if log_ratio < -36:
                return log_ratio  # Relative approximation error is below 1.2e-16.
            return math.log(max(log_ratio, 0.0) + math.log1p(math.exp(-abs(log_ratio))))

        terms = [
            math.log(weight) + log_activity(count, reference)
            for weight, count, reference in (
                (0.2, v, self.view_reference_rate),
                (0.4, r, self.reaction_reference_rate),
                (0.4, c, self.comment_reference_rate),
            )
        ]
        # Log-sum-exp preserves even extremely small positive activity.
        largest = max(terms)
        log_total = largest + math.log(sum(math.exp(term - largest) for term in terms))
        score = 10.0 * math.tanh((log_total - math.log(math.log(2.0))) / self.temperature)
        return round(score, 6)

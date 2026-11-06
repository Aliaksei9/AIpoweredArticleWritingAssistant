"""Predict social counts from topic, predicted popularity and resource metadata."""

from .predictor import SocialMetricsPrediction, SocialMetricsPredictor
from .resources import ResourceInfo

__all__ = ["SocialMetricsPrediction", "SocialMetricsPredictor", "ResourceInfo"]

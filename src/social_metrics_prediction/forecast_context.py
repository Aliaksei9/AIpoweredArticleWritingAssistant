"""Observed metric ages for training; a fixed seven-day production forecast."""
from datetime import datetime,timedelta,timezone
import math
from src.database.time_utils import utc_datetime

FORECAST_HORIZON_DAYS=7.
MIN_HORIZON_DAYS=2.
MAX_HORIZON_DAYS=14.
PUBLICATION_TIMEZONE=timezone(timedelta(hours=3))


def validate_horizon(value):
    if isinstance(value,bool):raise ValueError('Horizon must be a number of days')
    try:value=float(value)
    except (TypeError,ValueError,OverflowError) as error:raise ValueError('Horizon must be a number of days') from error
    if not math.isfinite(value) or not MIN_HORIZON_DAYS<=value<=MAX_HORIZON_DAYS:
        raise ValueError('Forecast horizon must be between 2 and 14 days')
    return value


def observation_horizon(published_at,collected_at):
    return (utc_datetime(collected_at,field='collected_at')-utc_datetime(published_at)).total_seconds()/86400.


def forecast_context(published_at=None):
    publication=utc_datetime(published_at or datetime.now(timezone.utc))
    return dict(published_at=publication.isoformat(),horizon_days=FORECAST_HORIZON_DAYS,
                fixed_horizon=True,
                forecast_at=(publication+timedelta(days=FORECAST_HORIZON_DAYS)).isoformat())

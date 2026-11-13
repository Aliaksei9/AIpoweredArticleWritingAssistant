"""Timezone-aware timestamps shared by collection and forecasting."""
from datetime import datetime,timezone


def utc_datetime(value, *, field='published_at'):
    if isinstance(value,str):
        try:value=datetime.fromisoformat(value.replace('Z','+00:00'))
        except ValueError as error:raise ValueError(f'{field} must be an ISO datetime with a timezone') from error
    if not isinstance(value,datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{field} must include a timezone')
    return value.astimezone(timezone.utc)

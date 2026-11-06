"""Telegram-specific credentials, addressing and API lifecycle."""

from contextlib import asynccontextmanager
import os

from .base import CollectionContext, SourceAdapter, require_env


class TelegramSource(SourceAdapter):
    code = "TG"

    @staticmethod
    def _client_arguments():
        try:
            api_id = int(require_env("TG_API_ID"))
        except ValueError:
            raise ValueError("TG_API_ID must be a positive integer; set it in .env") from None
        if api_id <= 0:
            raise ValueError("TG_API_ID must be a positive integer; set it in .env")
        return os.getenv("TG_SESSION", ".telegram_session"), api_id, require_env("TG_API_HASH")

    def validate_configuration(self):
        self._client_arguments()

    def normalize_resource(self, resource):
        resource = super().normalize_resource(resource)
        return resource if resource.startswith(("@", "http://", "https://")) else "@"+resource

    @asynccontextmanager
    async def open(self, context: CollectionContext):
        from telethon import TelegramClient
        from ..telegram import TGdataCollection
        options = context.options
        async with TelegramClient(*self._client_arguments()) as client:
            yield TGdataCollection(context.connection, client, context.embedder,
                max_age_days=options.max_age_days, as_of=options.as_of)

    async def login(self):
        from telethon import TelegramClient
        async with TelegramClient(*self._client_arguments()):
            print("Telegram session is authorized", flush=True)


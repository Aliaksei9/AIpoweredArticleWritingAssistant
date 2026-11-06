"""Connection setup releases its resource on errors and cancellation."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.database.connection import connect_database


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_vector_registration_closes_connection_and_preserves_error(self):
        for failure in (RuntimeError("registration failed"), asyncio.CancelledError()):
            connection = SimpleNamespace(close=AsyncMock())
            with self.subTest(error=type(failure).__name__), patch(
                "src.database.connection.psycopg.AsyncConnection.connect",
                AsyncMock(return_value=connection),
            ), patch(
                "src.database.connection.register_vector_async", AsyncMock(side_effect=failure)
            ):
                with self.assertRaises(type(failure)) as error:
                    await connect_database("fake-dsn")
                self.assertIs(error.exception, failure)
            connection.close.assert_awaited_once()

    async def test_successful_connection_remains_open_for_its_caller(self):
        connection = SimpleNamespace(close=AsyncMock())
        with patch(
            "src.database.connection.psycopg.AsyncConnection.connect", AsyncMock(return_value=connection)
        ), patch("src.database.connection.register_vector_async", AsyncMock()) as register:
            self.assertIs(await connect_database("fake-dsn"), connection)
        register.assert_awaited_once_with(connection)
        connection.close.assert_not_awaited()

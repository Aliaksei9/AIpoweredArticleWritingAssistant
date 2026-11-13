from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from model_tests.collect_training_target import OpenDiscussionClient
from model_tests.discover_channels import CachedEntityClient


class TrainingExpansionTests(unittest.IsolatedAsyncioTestCase):
    async def test_closed_messages_are_excluded_but_known_zero_engagement_is_kept(self):
        oldest = datetime.now(timezone.utc)-timedelta(days=14)
        def message(number, *, comments=True, reactions=True, expired=False):
            return SimpleNamespace(id=number, date=oldest if expired else oldest+timedelta(days=1),
                replies=SimpleNamespace(comments=comments,replies=0),
                reactions=SimpleNamespace(results=[]) if reactions else None)
        messages=[message(1),message(2,comments=False),message(3,reactions=False),
                  message(4,comments=False,expired=True),message(5)]
        class Client:
            async def iter_messages(self,*args,**kwargs):
                for value in messages:
                    yield value
        wrapped=OpenDiscussionClient(Client(),oldest)
        actual=[m.id async for m in wrapped.iter_messages('channel')]
        self.assertEqual(actual,[1,4])
        self.assertEqual(wrapped.skipped_closed,2)

    async def test_entity_cache_is_consulted_before_fetching_channel(self):
        client=SimpleNamespace(get_input_entity=AsyncMock(return_value='cached-input'),
                               get_entity=AsyncMock(return_value='channel'))
        self.assertEqual(await CachedEntityClient(client).get_entity('@public'),'channel')
        client.get_input_entity.assert_awaited_once_with('@public')
        client.get_entity.assert_awaited_once_with('cached-input')


if __name__=='__main__':
    unittest.main()

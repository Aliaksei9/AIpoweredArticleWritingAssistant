"""Real PostgreSQL: repeated collection updates one row and creates no history."""
import asyncio
from datetime import datetime,timedelta,timezone
import json
import os
from uuid import uuid4
from dotenv import load_dotenv
from src.database.connection import connect_database
from src.database.table_classes import ResourceTable,ArticleTextTable,ArticleTableOfSources,ArticleMetricTable


class RollbackFixture(Exception):pass


async def main():
    load_dotenv('.env');now=datetime.now(timezone.utc);article=uuid4();resource=uuid4()
    async with await connect_database(os.environ['DATABASE_URL']) as conn:
        missing=await (await conn.execute("SELECT to_regclass('public.article_metric_observations') IS NULL AS removed")).fetchone()
        assert missing['removed']
        before=await (await conn.execute('SELECT count(*) AS n FROM article_metrics')).fetchone()
        try:
            async with conn.transaction():
                await ResourceTable(conn).upsert(resource,'Single row integration fixture',1000)
                await ArticleTextTable(conn).upsert(article,'Игры','Текст для проверки.',now-timedelta(days=7))
                await ArticleTableOfSources(conn).link(article,resource,'TG')
                metrics=ArticleMetricTable(conn);first=now-timedelta(days=1)
                await metrics.upsert(article,100,5,1,1000,collected_at=first)
                await metrics.upsert(article,150,8,2,1100,collected_at=now)
                await metrics.upsert(article,50,1,0,900,collected_at=first)
                row=await metrics.get(article)
                assert row['views']==150 and row['reactions']==8 and row['audience']==1100 and row['collected_at']==now
                count=await (await conn.execute('SELECT count(*) AS n FROM article_metrics WHERE article_id=%s',(article,))).fetchone()
                assert count['n']==1
                raise RollbackFixture()
        except RollbackFixture:pass
        after=await (await conn.execute('SELECT count(*) AS n FROM article_metrics')).fetchone();assert before==after
        print(json.dumps(dict(status='passed',one_current_row=True,latest_update_preserved=True,no_history_table=True,test_rolled_back=True)))


if __name__=='__main__':asyncio.run(main(),loop_factory=asyncio.SelectorEventLoop)

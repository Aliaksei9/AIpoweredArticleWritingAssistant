"""Common social inputs, one current label per article, and API context."""
import unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import httpx
import numpy as np
from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased
from src.database.embedding_config import EMBEDDING_SIGNATURE,EMBEDDING_DIMENSIONS
from src.social_metrics_prediction import SocialMetricsPredictor,ResourceInfo
from src.social_metrics_prediction.dataset import metric_dataset_from_rows
from src.social_metrics_prediction.forecast_context import forecast_context,observation_horizon,validate_horizon
from src.social_metrics_prediction.unified_models import UnifiedRateModel
from src.social_metrics_prediction.unified_features import INPUT_FIELDS
from src.controller import EditorController
from src.ui.app import create_app


class ContextTests(unittest.TestCase):
    def test_timezone_and_horizon_validation(self):
        value=forecast_context('2026-11-12T15:00:00+03:00')
        self.assertEqual(value['published_at'],'2026-11-12T12:00:00+00:00')
        self.assertEqual(observation_horizon(value['published_at'],value['forecast_at']),7)
        self.assertEqual(value['horizon_days'],7.)
        self.assertTrue(value['fixed_horizon'])
        with self.assertRaises(ValueError):forecast_context('2026-11-12T12:00:00')
        for age in (0,15,float('nan'),True):
            with self.subTest(age=age),self.assertRaises(ValueError):validate_horizon(age)

    def test_dataset_requires_one_current_row_per_article(self):
        assessor=AssesmentConformPPmetricsBased();date=datetime(2026,11,1,tzinfo=timezone.utc)
        row=dict(article_id='a',text='Текст',topic='Игры',views=100,reactions=2,comments=1,audience=1000,
                 score=assessor.assess(100,2,1,1000),embedding=None,embedding_model=None,
                 resource_id='r',source='TG',published_at=date,collected_at=date+timedelta(days=6))
        data=metric_dataset_from_rows([row],[])
        self.assertEqual(data.horizons,[6.])
        values=data.inputs([0],[1.25])[0]
        self.assertEqual(values['score'],1.25)
        self.assertNotIn('views',values)
        with self.assertRaisesRegex(ValueError,'one current metric row'):
            metric_dataset_from_rows([row,dict(row,collected_at=date+timedelta(days=7))],[])

    def model(self):
        rows=[dict(topic='Игры' if i%2 else 'Кино',resource_id='r',source='TG',audience=1000+i*100,score=float(i-3),
                   published_at=f'2026-11-0{i+1}T12:00:00+00:00',horizon_days=3.+i) for i in range(6)]
        vectors=np.zeros((6,EMBEDDING_DIMENSIONS),np.float32)
        for i in range(6):vectors[i,i]=1
        counts=np.asarray([[100+i*20,3+i,i%3] for i in range(6)])
        config=dict(views=dict(family='ridge',alpha=10.,strength=.5),
                    reactions=dict(family='neighbours',neighbours=3,temperature=1.,quantile=.95,strength=1.),
                    comments=dict(family='ridge',alpha=1000.,strength=.5))
        model=SocialMetricsPredictor().fit(rows,counts,document_embeddings=vectors,configuration=config)
        model.set_resources([ResourceInfo('r','Канал',1000)])
        return model,rows,vectors

    def test_all_three_models_receive_identical_complete_matrix(self):
        model,rows,vectors=self.model();seen=[];original=UnifiedRateModel.raw
        def capture(owner,features,inputs,*args,**kwargs):
            seen.append((id(features),features.shape));return original(owner,features,inputs,*args,**kwargs)
        with patch.object(UnifiedRateModel,'raw',capture):result=model.predict_rows(rows,document_embeddings=vectors)
        self.assertEqual(result.shape,(6,3));self.assertEqual(len(seen),3)
        self.assertEqual(len(set(seen)),1)
        self.assertEqual(tuple(model.shared_input_fields),INPUT_FIELDS)
        original=model.shared_encoder_.transform(rows[:1],vectors[:1])
        for change in [dict(topic='Другая тема'),dict(score=5.),dict(audience=5000),dict(published_at='2026-11-05T23:00:00+00:00'),dict(horizon_days=10.),dict(resource_id='new')]:
            with self.subTest(change=change):
                changed=model.shared_encoder_.transform([dict(rows[0],**change)],vectors[:1])
                self.assertFalse(np.array_equal(original,changed))
        self.assertFalse(np.array_equal(original,model.shared_encoder_.transform(rows[:1],vectors[1:2])))

    def test_shared_head_cache_does_not_leak_between_requests(self):
        model,rows,vectors=self.model()
        model.estimators_[2].model=model.estimators_[1].model
        calls=[];original=UnifiedRateModel.raw
        def capture(owner,features,inputs,*args,**kwargs):
            calls.append(id(owner));return original(owner,features,inputs,*args,**kwargs)
        with patch.object(UnifiedRateModel,'raw',capture):
            first=model.predict_rows(rows[:1],document_embeddings=vectors[:1])
            second=model.predict_rows(rows[1:2],document_embeddings=vectors[1:2])
        self.assertEqual(len(calls),4)
        self.assertEqual(calls.count(id(model.estimators_[1].model)),2)
        matrix=model.shared_encoder_.transform(rows[:1],vectors[:1])
        expected=np.column_stack([head.predict(matrix,rows[:1]) for head in model.estimators_])
        np.testing.assert_array_equal(first,expected)
        self.assertEqual(second.shape,(1,3))

    def test_shared_model_round_trip_and_invalid_document(self):
        model,rows,vectors=self.model()
        with TemporaryDirectory() as folder:
            path=Path(folder)/'model.joblib';model.save(path);loaded=SocialMetricsPredictor.load(path)
            np.testing.assert_array_equal(model.predict_rows(rows,document_embeddings=vectors),loaded.predict_rows(rows,document_embeddings=vectors))
            self.assertEqual(loaded.horizon_metrics_for('r'),['views','reactions','comments'])
            self.assertEqual(loaded.metadata['embedding_signature'],EMBEDDING_SIGNATURE)
            with self.assertRaises(ValueError):loaded.predict_rows(rows)
            with self.assertRaises(ValueError):loaded.predict_rows(rows,document_embeddings=np.zeros_like(vectors))

    def test_reaction_model_does_not_depend_on_other_target_labels(self):
        model,rows,vectors=self.model()
        counts=np.asarray([[100+i*20,3+i,i%3] for i in range(len(rows))])
        configuration=model.metadata['selection']['configuration']
        before=model.predict_rows(rows,document_embeddings=vectors)
        changed=counts.copy();changed[:,0]*=1000;changed[:,2]+=10000
        other=SocialMetricsPredictor().fit(rows,changed,document_embeddings=vectors,configuration=configuration)
        np.testing.assert_array_equal(before[:,1],other.predict_rows(rows,document_embeddings=vectors)[:,1])
        changed=counts.copy();changed[:,1]*=1000
        other=SocialMetricsPredictor().fit(rows,changed,document_embeddings=vectors,configuration=configuration)
        np.testing.assert_array_equal(before[:,[0,2]],other.predict_rows(rows,document_embeddings=vectors)[:,[0,2]])

    def test_finer_reaction_configuration_keeps_shared_inputs_and_distinct_model(self):
        model,rows,vectors=self.model()
        configuration=model.metadata['selection']['configuration']
        configuration['reactions']=dict(family='neighbours',same_resource=True,context_weight=.003,
            quantile=.85,neighbours=5,temperature=.06,strength=1.)
        counts=np.asarray([[100+i*20,3+i,i%3] for i in range(len(rows))])
        refined=SocialMetricsPredictor().fit(rows,counts,document_embeddings=vectors,configuration=configuration)
        self.assertEqual(len({id(head.model) for head in refined.estimators_}),3)
        self.assertEqual(tuple(refined.shared_input_fields),INPUT_FIELDS)
        expected=model.predict_rows(rows,document_embeddings=vectors)
        actual=refined.predict_rows(rows,document_embeddings=vectors)
        np.testing.assert_array_equal(actual[:,[0,2]],expected[:,[0,2]])

    def test_public_prediction_always_uses_seven_days(self):
        model,rows,vectors=self.model()
        row=dict(rows[0],horizon_days=7.)
        expected=model.predict_rows([row],document_embeddings=vectors[:1])[0]
        actual=model.predict(row['topic'],row['score'],'r',document_embedding=vectors[0],published_at=row['published_at'])
        np.testing.assert_array_equal(list(actual.to_dict().values()),expected)
        with self.assertRaises(TypeError):
            model.predict(row['topic'],row['score'],'r',document_embedding=vectors[0],horizon_days=14.)


class ForecastAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.resource=ResourceInfo('r','Канал',1000)
        owner=self
        class Engine:
            social=SimpleNamespace(uses_extended_features=True,horizon_sensitive_metrics=['views','reactions','comments'],resources=[owner.resource])
            detector=SimpleNamespace(topics=[SimpleNamespace(name='Игры')])
            def evaluate(self,text,topic,resource,published_at):
                owner.received=published_at
                return dict(topic=topic,score=2.,metrics=dict(views=100,reactions=3,comments=1))
        async def resources():return [self.resource]
        self.controller=EditorController(engine=Engine(),resource_loader=resources)
        self.app=create_app(self.controller)
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),base_url='http://testserver')

    async def asyncTearDown(self):
        await self.client.aclose();await self.app.state.jobs.aclose();await self.app.state.language_jobs.aclose();await self.controller.aclose()

    async def test_endpoint_passes_normalized_plan_and_explains_horizon_scope(self):
        response=await self.client.post('/api/predictions',json=dict(text='Новый текст.',topic='Игры',resource_id='r',published_at='2026-11-12T15:00:00+03:00'))
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.received,'2026-11-12T12:00:00+00:00')
        self.assertEqual(response.json()['forecast']['horizon_sensitive_metrics'],['views','reactions','comments'])
        self.assertEqual(response.json()['forecast']['horizon_days'],7.)
        self.assertEqual(response.json()['forecast']['forecast_at'],'2026-11-19T12:00:00+00:00')
        self.assertTrue(response.json()['forecast']['fixed_horizon'])

    async def test_endpoint_rejects_naive_dates_and_invalid_horizons(self):
        for extra in [dict(published_at='2026-11-12T12:00:00'),dict(horizon_days=1),dict(horizon_days=7),dict(horizon_days=15)]:
            response=await self.client.post('/api/predictions',json=dict(text='Текст.',resource_id='r',**extra))
            self.assertEqual(response.status_code,422)

    async def test_workspace_and_page_explain_fixed_seven_day_forecast(self):
        workspace=(await self.client.get('/api/workspace')).json()
        self.assertEqual(workspace['forecast']['horizon_days'],7.)
        self.assertTrue(workspace['forecast']['fixed_horizon'])
        self.assertNotIn('min_horizon_days',workspace['forecast'])
        page=(await self.client.get('/')).text
        self.assertNotIn('id="horizon-days"',page)
        self.assertIn('Прогноз социальных метрик',page)
        self.assertIn('через семь дней',page)
        self.assertIn('Планируемое время публикации',page)

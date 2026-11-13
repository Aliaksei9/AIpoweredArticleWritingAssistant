"""Alternative inputs do not leak labels or fit preprocessors at prediction time."""
import pickle
import unittest

import numpy as np

from model_tests.alternative_metric_models import AlternativeFeatures,AlternativeRegressor,SemanticNeighbour


class AlternativeMetricTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(42);vectors=rng.normal(size=(40,16))
        vectors[:,:8]/=np.linalg.norm(vectors[:,:8],axis=1)[:,None]
        rows=[dict(topic='Игры',resource_id=str(i%2),source='TG',audience=10000,score=float(i%7-3)) for i in range(40)]
        counts=np.asarray([[1000+i*10,20+i,2+i%3] for i in range(40)])
        texts=['Новая игра выходит завтра. Новость номер '+str(i) for i in range(40)]
        return rows,counts,vectors,texts

    def test_target_independence(self):
        rows,counts,vectors,texts=self.fixture()
        for family in ('ridge','neighbour'):
            config=dict(family='ridge',features='dense',target='counts',alpha=100.)
            cls=AlternativeRegressor if family=='ridge' else SemanticNeighbour
            if family=='neighbour':config=dict(neighbours=10,temperature=.05)
            first=cls(config).fit(rows,counts,vectors,texts).predict(rows,vectors,texts)
            changed=counts.copy();changed[:,1]*=10000
            other=cls(config).fit(rows,changed,vectors,texts).predict(rows,vectors,texts)
            np.testing.assert_array_equal(first[:,[0,2]],other[:,[0,2]])

    def test_text_vocab_and_scaler_remain_training_only(self):
        rows,counts,vectors,texts=self.fixture()
        for mode in ('dense','pca','word','char','native','native_pca'):
            pre=AlternativeFeatures(mode).fit(rows,vectors,texts)
            first=pre.transform(rows[:2],vectors[:2],texts[:2])
            pre.transform(rows[:2],vectors[:2]+10,['никогда не встречавшийся текст']*2)
            other=pre.transform(rows[:2],vectors[:2],texts[:2])
            if hasattr(first,'toarray'):first=first.toarray();other=other.toarray()
            np.testing.assert_array_equal(first,other)

    def test_serialization_and_nonnegative_outputs(self):
        rows,counts,vectors,texts=self.fixture()
        model=AlternativeRegressor(dict(family='ridge',features='word',target='rates',alpha=30.)).fit(rows,counts,vectors,texts)
        loaded=pickle.loads(pickle.dumps(model))
        first=model.predict(rows,vectors,texts,strength=2.)
        np.testing.assert_array_equal(first,loaded.predict(rows,vectors,texts,strength=2.))
        self.assertTrue(np.isfinite(first).all() and np.all(first>=0))

    def test_poisson_counts_are_positive_and_do_not_mutate_labels(self):
        rows,counts,vectors,texts=self.fixture();saved=counts.copy()
        config=dict(family='cat',features='native',target='counts',loss='Poisson',depth=4)
        model=AlternativeRegressor(config).fit(rows,counts,vectors,texts)
        p=model.predict(rows[:2],vectors[:2],texts[:2])
        self.assertTrue(np.isfinite(p).all() and np.all(p>=0))
        np.testing.assert_array_equal(counts,saved)

    def test_direct_neural_heads_remain_independent(self):
        from model_tests.direct_metric_neural import DirectMetricNeural
        rows,counts,vectors,texts=self.fixture();c=dict(features='dense',target='counts')
        first=DirectMetricNeural(c,epochs=2,threads=1).fit(rows,counts,vectors,texts).predict(rows,vectors,texts)
        changed=counts.copy();changed[:,1]*=100
        other=DirectMetricNeural(c,epochs=2,threads=1).fit(rows,changed,vectors,texts).predict(rows,vectors,texts)
        np.testing.assert_array_equal(first[:,[0,2]],other[:,[0,2]])

    def test_cross_resource_transfer_is_finite_and_independent(self):
        from model_tests.cross_resource_transfer import CrossResourceTransfer
        rows,counts,vectors,texts=self.fixture()
        config=dict(neighbours=5,temperature=.02,ratio_cap=10.,exclude_resource=True)
        first=CrossResourceTransfer(config).fit(rows,counts,vectors,texts).predict(rows,vectors,texts)
        changed=counts.copy();changed[:,1]*=1000
        model=CrossResourceTransfer(config).fit(rows,changed,vectors,texts)
        other=model.predict(rows,vectors,texts)
        np.testing.assert_array_equal(first[:,[0,2]],other[:,[0,2]])
        self.assertTrue(np.isfinite(other).all() and np.all(other>=0))

    def test_neural_poisson_predicts_the_fitted_mean_without_subtraction(self):
        import torch
        from model_tests.direct_metric_neural import DirectMetricNeural
        rows,counts,vectors,texts=self.fixture()
        model=DirectMetricNeural(dict(features='dense',target='poisson'),epochs=1,threads=1).fit(rows,counts,vectors,texts)
        with torch.no_grad():
            for network in model.networks:
                network[-1].weight.zero_();network[-1].bias.zero_()
        np.testing.assert_allclose(model.raw(rows,vectors,texts),model._prior(rows)+1,rtol=1e-6)

    def test_shared_prediction_cache_preserves_independent_heads(self):
        from model_tests.alternative_metric_models import AlternativeMetricHead,predict_heads
        rows,counts,vectors,texts=self.fixture()
        model=AlternativeRegressor(dict(family='ridge',features='dense',target='counts',alpha=100.)).fit(rows,counts,vectors,texts)
        heads={name:AlternativeMetricHead(model,i,.5+i*.25) for i,name in enumerate(('views','reactions','comments'))}
        expected=np.column_stack([h.predict(rows,vectors,texts) for h in heads.values()])
        np.testing.assert_array_equal(expected,predict_heads(heads,rows,vectors,texts))

    def test_calendar_models_keep_targets_independent(self):
        from model_tests.calendar_count_models import CalendarCountRegressor
        rows,counts,_,_=self.fixture()
        rows=[dict(r,published_at=f'2026-11-{1+i%8:02d}T12:00:00+00:00',horizon_days=float(3+i%10)) for i,r in enumerate(rows)]
        first=CalendarCountRegressor(100.).fit(rows,counts).predict(rows)
        changed=counts.copy();changed[:,1]*=100
        other=CalendarCountRegressor(100.).fit(rows,changed).predict(rows)
        np.testing.assert_array_equal(first[:,[0,2]],other[:,[0,2]])

    def test_stacked_pipeline_roundtrip_and_cached_predictions(self):
        from model_tests.alternative_metric_models import AlternativeMetricHead,predict_heads
        from model_tests.residual_count_stacking import CountCorrection,StackedCountPipeline,StackedMetricHead
        rows,counts,vectors,texts=self.fixture()
        model=AlternativeRegressor(dict(family='ridge',features='dense',target='counts',alpha=100.)).fit(rows,counts,vectors,texts)
        base={m:AlternativeMetricHead(model,i,1.) for i,m in enumerate(('views','reactions','comments'))}
        predictions=predict_heads(base,rows,vectors,texts);priors=model._prior(rows)
        correction=CountCorrection(1000.).fit(rows,counts,predictions,priors)
        pipeline=StackedCountPipeline(base,{0:(correction,.25),1:(correction,.5)})
        heads={m:StackedMetricHead(pipeline,i) for i,m in enumerate(base)}
        expected=pipeline.predict(rows,vectors,texts)
        loaded=pickle.loads(pickle.dumps(heads))
        np.testing.assert_array_equal(expected,predict_heads(loaded,rows,vectors,texts))

    def test_oof_calibration_recovers_a_known_scale(self):
        from model_tests.calibrated_reactions import calibration_coefficient
        prior=np.asarray([10.,20.,30.]);prediction=prior+np.asarray([2.,-4.,8.])
        target=prior+1.5*(prediction-prior)
        self.assertAlmostEqual(calibration_coefficient(target,prior,prediction),1.5)
        self.assertEqual(calibration_coefficient(prior,prior,prior),1.)


if __name__=='__main__':unittest.main()

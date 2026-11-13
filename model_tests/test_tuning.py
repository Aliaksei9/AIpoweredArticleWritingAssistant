"""Behavior of baseline corrections and training-only statistics."""
import unittest

import numpy as np
from sklearn.linear_model import Ridge

from model_tests.social_metric_candidates import ResourceMeanBaseline
from model_tests.tuning_candidates import ResidualRatePredictor
from src.social_metrics_prediction.baseline_correction import ChannelMeanCorrection
from src.social_metrics_prediction.features import MetricFeatureEncoder


class TuningTests(unittest.TestCase):
    @staticmethod
    def data():
        rows=[];counts=[]
        for channel in range(3):
            for score in range(-8,9):
                rows.append(dict(source='TG',resource_id=str(channel),audience=100000,
                                 topic='Игры',score=score))
                counts.append([10000*(channel+1)+score*400,200+score*10,40+score*3])
        return rows,np.asarray(counts)

    def test_learns_within_channel_effect_on_unseen_scores(self):
        rows,counts=self.data()
        fit=np.asarray([i for i,r in enumerate(rows) if r['score']%2==0])
        holdout=np.asarray([i for i,r in enumerate(rows) if r['score']%2!=0])
        train=[rows[i] for i in fit];validation=[rows[i] for i in holdout]
        model=ResidualRatePredictor(alpha=.01).fit(train,counts[fit])
        baseline=ResourceMeanBaseline().fit(train,counts[fit])
        error=np.mean((model.predict_rows(validation)-counts[holdout])**2)
        original=np.mean((baseline.predict_rows(validation)-counts[holdout])**2)
        self.assertLess(error,original*.05)

    def test_zero_correction_matches_training_mean_and_audience_scaling(self):
        rows,counts=self.data()
        model=ResidualRatePredictor(blend=0).fit(rows,counts)
        baseline=ResourceMeanBaseline().fit(rows,counts)
        np.testing.assert_array_equal(model.predict_rows(rows),baseline.predict_rows(rows))
        doubled=[dict(r,audience=200000) for r in rows]
        np.testing.assert_array_equal(model.predict_rows(doubled),baseline.predict_rows(doubled))
        unknown=[dict(rows[0],resource_id='new')]
        np.testing.assert_array_equal(model.predict_rows(unknown),baseline.predict_rows(unknown))
        with self.assertRaises(ValueError):model.predict_rows([dict(rows[0],source='VK')])

    def test_application_wrapper_matches_experimental_rate_predictions(self):
        rows,counts=self.data()
        experiment=ResidualRatePredictor(alpha=10,blend=.5).fit(rows,counts)
        encoder=MetricFeatureEncoder().fit(rows)
        features=encoder.transform(rows)
        offset=len(encoder.categories_.categories_[0])
        columns=tuple(range(offset,offset+len(encoder.categories_.categories_[1])))
        rates=counts/100000.*1000.
        models=[ChannelMeanCorrection(Ridge(alpha=10),columns,blend=.5).fit(features,rates[:,i]) for i in range(3)]
        validation=rows+[dict(rows[0],resource_id='new',topic='Новая тема',score=4.)]
        predicted=np.column_stack([m.predict(encoder.transform(validation)) for m in models])
        np.testing.assert_allclose(np.maximum(0,predicted),experiment.predict_rates(validation),atol=1e-8)


if __name__=='__main__':
    unittest.main()

"""History models preserve chronology and generalize topic/score effects."""
from datetime import datetime,timedelta,timezone
import unittest

import numpy as np

from model_tests.history_candidates import ConditionalHistoryPredictor
from src.social_metrics_prediction.features import MetricFeatureEncoder
from src.social_metrics_prediction.history_regression import ChannelHistoryRegressor


class HistoryModelTests(unittest.TestCase):
    @staticmethod
    def data():
        rows=[];counts=[]
        for channel in range(3):
            for i in range(60):
                rows.append(dict(source='TG',resource_id=str(channel),topic='Игры' if i%2 else 'Кино',
                    audience=100000,score=float(i%9-4),
                    published_at=(datetime(2026,10,28,tzinfo=timezone.utc)+timedelta(hours=i)).isoformat()))
                counts.append([10000*(channel+1)+i*50,100+i*2,10+i])
        return rows,np.asarray(counts)

    def test_history_is_invariant_to_input_row_order(self):
        rows,counts=self.data()
        model=ConditionalHistoryPredictor(half_life=7,topic_strength=.5,score_strength=.5).fit(rows,counts)
        order=np.random.default_rng(3).permutation(len(rows))
        other=ConditionalHistoryPredictor(half_life=7,topic_strength=.5,score_strength=.5).fit([rows[i] for i in order],counts[order])
        np.testing.assert_array_equal(model.predict_rows(rows),other.predict_rows(rows))

    def test_application_rates_match_experiment_for_all_outputs(self):
        rows,counts=self.data()
        config=dict(half_life=30,upper_quantile=.95,topic_strength=1.,topic_shrink=20.,score_strength=.5,slope_ridge=100.)
        experiment=ConditionalHistoryPredictor(**config).fit(rows,counts)
        encoder=MetricFeatureEncoder().fit(rows)
        features=encoder.transform(rows)
        offset=len(encoder.categories_.categories_[0])
        columns=tuple(range(offset,offset+len(encoder.categories_.categories_[1])))
        rates=counts/100.
        models=[ChannelHistoryRegressor(resource_columns=columns,**config).fit_with_dates(
            features,rates[:,i],published_at=[r['published_at'] for r in rows]) for i in range(3)]
        # An odd audience avoids exact half-integer ties after equivalent
        # scalar/vector floating-point reductions for an unknown channel.
        validation=rows+[dict(rows[0],resource_id='new',topic='Новая тема',score=8.,audience=100001)]
        predicted=np.column_stack([m.predict(encoder.transform(validation)) for m in models])
        audience=np.asarray([r['audience'] for r in validation])[:,None]
        np.testing.assert_array_equal(np.rint(np.maximum(0,predicted)*audience/1000.),experiment.predict_rows(validation))

    def test_missing_dates_do_not_turn_arbitrary_row_order_into_recency(self):
        rows,counts=self.data();encoder=MetricFeatureEncoder().fit(rows)
        features=encoder.transform(rows);offset=len(encoder.categories_.categories_[0])
        columns=tuple(range(offset,offset+len(encoder.categories_.categories_[1])))
        a=ChannelHistoryRegressor(columns).fit(features,counts[:,0]/100.)
        b=ChannelHistoryRegressor(columns).fit(features[::-1],counts[::-1,0]/100.)
        np.testing.assert_allclose(a.predict(features),b.predict(features),atol=1e-10)
        self.assertEqual(a.channels_without_dates_,3)


if __name__=='__main__':unittest.main()

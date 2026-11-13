"""Kernel heads cannot use validation labels or other metric targets."""
import unittest
import pickle

import numpy as np

from model_tests.conditional_kernel_metrics import KernelChannelPredictor,SingleMetricHead


class KernelMetricTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(topic='Игры' if i%3 else 'Кино',resource_id=str(i%2),source='TG',
                   audience=10000,score=float(i%9-4)) for i in range(60)]
        counts=np.asarray([[1000+i*10,20+i,2+i%3] for i in range(60)])
        return rows,counts

    def test_targets_are_separable(self):
        rows,counts=self.fixture();first=KernelChannelPredictor(upper_quantile=.95).fit(rows,counts).predict_rows(rows)
        modified=counts.copy();modified[:,1]*=10000
        second=KernelChannelPredictor(upper_quantile=.95).fit(rows,modified).predict_rows(rows)
        np.testing.assert_array_equal(first[:,[0,2]],second[:,[0,2]])

    def test_single_and_batch_predictions_agree(self):
        rows,counts=self.fixture();model=KernelChannelPredictor().fit(rows,counts)
        batch=model.predict_rows(rows)
        single=np.vstack([model.predict_rows([r]) for r in rows])
        np.testing.assert_array_equal(single,batch)

    def test_unseen_channel_and_extreme_scores_are_finite(self):
        rows,counts=self.fixture();model=KernelChannelPredictor().fit(rows,counts)
        p=model.predict_rows([dict(rows[0],resource_id='unseen'),dict(rows[0],score=10.)])
        self.assertTrue(np.isfinite(p).all() and np.all(p>=0))
        self.assertTrue(np.all(p==np.rint(p)))

    def test_head_has_stable_serialization_path(self):
        from model_tests.advanced_pipeline_study import SingleMetricHead as RunnerHead
        self.assertIs(RunnerHead,SingleMetricHead)
        self.assertEqual(SingleMetricHead.__module__,'model_tests.conditional_kernel_metrics')
        rows,counts=self.fixture()
        head=SingleMetricHead(1,dict(kind='kernel',parameters={})).fit(rows,counts)
        loaded=pickle.loads(pickle.dumps(head))
        np.testing.assert_array_equal(head.predict(rows),loaded.predict(rows))


if __name__=='__main__':unittest.main()

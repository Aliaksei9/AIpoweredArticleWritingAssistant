"""Each target head is independent of the labels of the other metrics."""
import unittest
import numpy as np
from model_tests.independent_metric_candidates import ViewsModel,ReactionsModel,CommentsModel


class IndependentMetricTests(unittest.TestCase):
    def test_changing_reactions_does_not_change_view_predictions(self):
        rows=[dict(topic='Игры',resource_id=str(i%2),source='TG',audience=10000,score=float(i%9-4)) for i in range(40)]
        counts=np.asarray([[1000+i*10,20+i,2+i%3] for i in range(40)])
        original=ViewsModel(dict(upper_quantile=.95)).fit(rows,counts).predict(rows)
        changed=counts.copy();changed[:,1]*=10000
        other=ViewsModel(dict(upper_quantile=.95)).fit(rows,changed).predict(rows)
        np.testing.assert_array_equal(original,other)

    def test_heads_keep_separate_target_identity(self):
        heads=[ViewsModel({}),ReactionsModel({}),CommentsModel({})]
        self.assertEqual([h.metric for h in heads],[0,1,2])
        self.assertEqual(len({id(h.configuration) for h in heads}),3)


if __name__=='__main__':unittest.main()

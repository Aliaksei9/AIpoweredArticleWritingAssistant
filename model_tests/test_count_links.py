"""Check training-only calibration and independent target heads."""
import unittest

import numpy as np

from model_tests.count_link_candidates import CalibratedCountLink


class CountLinkTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(topic='Игры',resource_id=str(i%2),source='TG',audience=10000,
                   score=float(i%9-4)) for i in range(60)]
        counts=np.asarray([[1000+i*10,20+i,2+i%3] for i in range(60)])
        return rows,counts

    def test_unseen_resource_and_extreme_scores_remain_finite(self):
        rows,counts=self.fixture();model=CalibratedCountLink(0).fit(rows,counts)
        unseen=[dict(rows[0],resource_id='new',score=-10.),dict(rows[0],score=10.)]
        predicted=model.predict(unseen)
        self.assertTrue(np.isfinite(predicted).all())
        self.assertTrue((predicted>=0).all())

    def test_reaction_labels_do_not_influence_views(self):
        rows,counts=self.fixture()
        first=CalibratedCountLink(0).fit(rows,counts).predict(rows)
        changed=counts.copy();changed[:,1]*=1000
        second=CalibratedCountLink(0).fit(rows,changed).predict(rows)
        np.testing.assert_array_equal(first,second)

    def test_prediction_does_not_fit_on_input_rows(self):
        rows,counts=self.fixture();model=CalibratedCountLink(1).fit(rows,counts)
        original=model.predict(rows[:2])
        model.predict([dict(rows[0],resource_id='unknown')])
        np.testing.assert_array_equal(original,model.predict(rows[:2]))


if __name__=='__main__':unittest.main()

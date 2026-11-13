"""Research neural feature transforms remain training-only and reproducible."""
import pickle
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.assesment_conform_of_PP_text_based.features import FEATURE_DIMENSIONS
from model_tests.advanced_text_networks import AdvancedTextRegressor,NeuralFeatures,TextPredictorBundle


class AdvancedTextTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(32,FEATURE_DIMENSIONS)).astype(np.float32)
        return x,np.linspace(-8,8,len(x),dtype=np.float32)

    def test_all_architectures_are_bounded_and_serializable(self):
        x,y=self.fixture()
        for kind in ('dense','additive','residual','ordinal'):
            with self.subTest(kind=kind):
                c=dict(kind=kind,widths=(12,8),dropout=.2,epochs=2,lr=.001,decay=.1,pca=8)
                model=AdvancedTextRegressor(c,threads=1).fit(x,y)
                result=model.predict(x[:4]);loaded=pickle.loads(pickle.dumps(model))
                np.testing.assert_array_equal(result,loaded.predict(x[:4]))
                self.assertTrue(np.isfinite(result).all() and np.all(np.abs(result)<=10))

    def test_transform_does_not_fit_on_prediction_inputs(self):
        x,_=self.fixture();pre=NeuralFeatures(pca=8).fit(x)
        original=pre.transform(x[:3]);pre.transform(x[:3]+1000)
        np.testing.assert_array_equal(original,pre.transform(x[:3]))

    def test_mixup_reproducibility_and_caller_rng(self):
        x,y=self.fixture();c=dict(widths=(8,),dropout=.2,epochs=3,lr=.001,decay=.1,mixup=.15)
        state=torch.random.get_rng_state().clone()
        first=AdvancedTextRegressor(c,threads=1).fit(x,y)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        other=AdvancedTextRegressor(c,threads=1).fit(x,y)
        np.testing.assert_array_equal(first.predict(x),other.predict(x))

    def test_bundle_uses_declared_weights(self):
        x,y=self.fixture();c=dict(widths=(8,),dropout=0.,epochs=1,lr=.001,decay=0.)
        a=AdvancedTextRegressor(c,seed=1,threads=1).fit(x,y)
        b=AdvancedTextRegressor(c,seed=2,threads=1).fit(x,y)
        bundle=TextPredictorBundle([a,b],[.25,.75])
        np.testing.assert_allclose(bundle.predict(x),.25*a.predict(x)+.75*b.predict(x))

    def test_invalid_labels_rejected(self):
        x,y=self.fixture();y[0]=11
        with self.assertRaises(ValueError):AdvancedTextRegressor({}).fit(x,y)

    def test_cached_model_from_other_training_rows_is_rejected(self):
        from model_tests.advanced_neural_study import fingerprint,fitted_model
        x,y=self.fixture();indices=np.arange(16);configuration=dict(widths=(8,),epochs=1)
        saved=dict(fingerprint=fingerprint('corpus',indices,configuration,42),model='cached')
        with patch.object(Path,'exists',return_value=True),patch('model_tests.advanced_neural_study.joblib.load',return_value=saved):
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                fitted_model(Path('unused.joblib'),x,y,indices+16,configuration,'corpus')


if __name__=='__main__':unittest.main()

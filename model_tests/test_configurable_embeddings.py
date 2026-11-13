"""Embedding configuration changes require a coherent vector/model profile."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

from src.database.embedding_config import EmbeddingSettings


class ConfigurableEmbeddingTests(unittest.TestCase):
    def test_dimension_and_prefix_changes_have_distinct_provenance(self):
        default=EmbeddingSettings()
        self.assertEqual(default.signature,default.model)
        profiles=[default,EmbeddingSettings(dimensions=768),EmbeddingSettings(document_prefix=''),
                  EmbeddingSettings(query_prefix=''),EmbeddingSettings(model='local/other-model')]
        self.assertEqual(len({p.signature for p in profiles}),len(profiles))
        with self.assertRaises(ValueError):EmbeddingSettings(dimensions=0)
        with self.assertRaises(ValueError):EmbeddingSettings.from_environment({'EMBEDDING_DIMENSIONS':'bad'})

    def test_nondefault_model_dimension_and_prefixes_work_in_new_process(self):
        code='''
import tempfile
from pathlib import Path
from unittest.mock import Mock,patch
from types import SimpleNamespace
import numpy as np
from src.database.embedding_config import MODEL_NAME,EMBEDDING_DIMENSIONS,EMBEDDING_SIGNATURE
from src.database.model_embedding import TextEmbedder
from src.database.table_classes import ArticleVectorTable
from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased,TextTopicEncoder
from src.assesment_conform_of_PP_text_based.features import FEATURE_DIMENSIONS
assert MODEL_NAME=='test/another-model' and EMBEDDING_DIMENSIONS==768 and FEATURE_DIMENSIONS==1536
factory=Mock(return_value=SimpleNamespace(get_embedding_dimension=lambda:768))
with patch.dict('sys.modules',{'sentence_transformers':SimpleNamespace(SentenceTransformer=factory)}):
    TextEmbedder()
factory.assert_called_once_with('test/another-model')
ArticleVectorTable._vector([1.]+[0.]*767)
class Fake:
    def embed_document(self,text):return [1.]+[0.]*767
    def embed_query(self,text):return [1.]+[0.]*767
encoder=TextTopicEncoder(embedder=Fake())
features=encoder.transform(['a','b','c','d'],['t']*4)
model=AssesmentConformPPtextBased(encoder=encoder,epochs=1,num_threads=1).train_features(features,[-1,0,1,2])
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'model.joblib';model.save(path)
    restored=AssesmentConformPPtextBased.load(path)
    assert restored.encoder.embedding_signature==EMBEDDING_SIGNATURE
    assert len(restored.model.predict(features))==4
    import joblib
    payload=joblib.load(path);payload['embedding_signature']='incompatible-prefix-profile'
    joblib.dump(payload,path)
    try:AssesmentConformPPtextBased.load(path)
    except ValueError:pass
    else:raise AssertionError('An incompatible embedding profile was accepted')
from src.database.table_classes import MODEL_NAME as STORED_PROFILE
assert STORED_PROFILE==EMBEDDING_SIGNATURE and STORED_PROFILE!=MODEL_NAME
print('configured dimension and artifact round trip passed')
'''
        env=dict(os.environ,EMBEDDING_MODEL='test/another-model',EMBEDDING_DIMENSIONS='768',
                 EMBEDDING_DOCUMENT_PREFIX='',EMBEDDING_QUERY_PREFIX='')
        result=subprocess.run([sys.executable,'-c',code],env=env,cwd=Path(__file__).resolve().parents[1],
                              capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()

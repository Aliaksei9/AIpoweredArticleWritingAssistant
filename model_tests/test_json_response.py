"""Strict shared JSON decoding must reject ambiguity before domain validation."""
import os
import unittest
from unittest.mock import patch

from src.text_correction.errors import InvalidLLMResponse
from src.text_correction.json_response import load_json_response
from src.text_correction.openrouter_client import OpenRouterConfig, DEFAULT_FALLBACK_MODELS


class JSONResponseTests(unittest.TestCase):
    def test_nested_duplicate_keys_and_nonfinite_constants_are_rejected(self):
        values=['{"items":[{"id":1,"id":2}]}', '{"items":[NaN]}',
                '{"value":Infinity}', '{"value":-Infinity}', 'not JSON', None]
        for value in values:
            with self.subTest(value=value), self.assertRaisesRegex(InvalidLLMResponse,'Invalid response'):
                load_json_response(value,'Invalid response')

    def test_separate_objects_may_use_the_same_keys_and_preserve_unicode(self):
        value=load_json_response('{"items":[{"text":"Привет 🎮"},{"text":"Другой текст"}]}','Invalid response')
        self.assertEqual(value['items'],[{'text':'Привет 🎮'},{'text':'Другой текст'}])

    def test_environment_defaults_match_the_configuration_and_allow_overrides(self):
        with patch.dict(os.environ,{},clear=True):
            defaults=OpenRouterConfig();value=OpenRouterConfig.from_env()
            for name in ('model','max_input_tokens','max_output_tokens','timeout_seconds','retries','free_only','reasoning_enabled'):
                self.assertEqual(getattr(value,name),getattr(defaults,name))
            self.assertEqual(value.fallback_models,DEFAULT_FALLBACK_MODELS)
        with patch.dict(os.environ,dict(OPENROUTER_MAX_INPUT_TOKENS='12000',
            OPENROUTER_MAX_OUTPUT_TOKENS='4000',OPENROUTER_TIMEOUT_SECONDS='25.5',
            OPENROUTER_RETRIES='1',OPENROUTER_FREE_ONLY='false',OPENROUTER_REASONING_ENABLED='yes'),clear=True):
            value=OpenRouterConfig.from_env()
            self.assertEqual((value.max_input_tokens,value.max_output_tokens,value.timeout_seconds,value.retries),
                             (12000,4000,25.5,1))
            self.assertFalse(value.free_only);self.assertTrue(value.reasoning_enabled)

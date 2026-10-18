import json
from pathlib import Path
import tempfile
import unittest
from airavoid.provenance import cache_identity,validate_cache_entry


class ProvenanceTests(unittest.TestCase):
    def test_wrong_source_digest_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'detections.jsonl'
            p.with_name('benchmark.json').write_text(json.dumps({'input_sha256':'abc'}))
            with self.assertRaises(ValueError):cache_identity(p,'video.mp4','xyz',1280,720,25)
            self.assertEqual(cache_identity(p,'relocated.mp4','abc',1280,720,25)['status'],'sha256_verified')

    def test_legacy_without_digest_is_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'detections.jsonl'
            self.assertEqual(cache_identity(p,'v.mp4','abc',1280,720,25)['status'],'unverified_legacy_cache')
            p.with_name('benchmark.json').write_text(json.dumps({'resolution':[1920,1080]}))
            with self.assertRaises(ValueError):cache_identity(p,'v.mp4','abc',1280,720,25)

    def test_cache_timestamp_and_invalid_observations_rejected(self):
        validate_cache_entry({'time':.08,'observations':[[1,2,4,8,.5,0,'sot']]},2,.08)
        with self.assertRaises(ValueError):validate_cache_entry({'time':.08},2,.2)
        with self.assertRaises(ValueError):validate_cache_entry({'detections':[[1,2,4,8,float('nan'),0]]},2,.08)
        with self.assertRaises(ValueError):validate_cache_entry({'observations':[[1,2,4,8,.5,7]]},2,.08)

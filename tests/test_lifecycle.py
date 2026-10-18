import json
from pathlib import Path
import tempfile
import unittest
from airavoid.lifecycle import RunSession


class LifecycleTests(unittest.TestCase):
    def test_failed_rerun_cannot_reuse_old_success_and_cleans_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'benchmark.json').write_text('{}');closed=[]
            with self.assertRaises(ValueError):
                with RunSession(root) as run:
                    run.register(lambda:closed.append('first'));run.register(lambda:closed.append('second'))
                    raise ValueError('intentional failure')
            self.assertFalse((root/'benchmark.json').exists())
            self.assertEqual(closed,['second','first'])
            self.assertEqual(json.loads((root/'run_status.json').read_text())['state'],'failed')

    def test_complete_requires_final_benchmark(self):
        with tempfile.TemporaryDirectory() as folder:
            with RunSession(folder):Path(folder,'benchmark.json').write_text('{}')
            self.assertEqual(json.loads(Path(folder,'run_status.json').read_text())['state'],'complete')
            with self.assertRaises(RuntimeError):
                with RunSession(folder):pass

"""Durable completion status and cleanup of resources owned by one CLI run."""
from datetime import datetime,timezone
import json
from pathlib import Path


class RunSession:
    def __init__(self,output,metadata=None,benchmark_name='benchmark.json'):
        self.output=Path(output);self.metadata=metadata or {};self.callbacks=[]
        self.benchmark_name=benchmark_name

    def status(self,state,**extra):
        (self.output/'run_status.json').write_text(json.dumps({**self.metadata,'state':state,
            'started_utc':self.started,'updated_utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2),encoding='utf-8')

    def __enter__(self):
        self.output.mkdir(parents=True,exist_ok=True);self.started=datetime.now(timezone.utc).isoformat()
        # A failed rerun must never leave an old benchmark claiming completion.
        self.output.joinpath(self.benchmark_name).unlink(missing_ok=True)
        self.status('running');return self

    def register(self,callback):self.callbacks.append(callback)

    def __exit__(self,kind,error,traceback):
        cleanup_errors=[]
        for callback in reversed(self.callbacks):
            try:callback()
            except Exception as e:cleanup_errors.append(type(e).__name__+': '+str(e))
        if error is not None:
            self.status('failed',error_type=kind.__name__,error=str(error),cleanup_errors=cleanup_errors)
            return False
        if cleanup_errors or not self.output.joinpath(self.benchmark_name).exists():
            self.status('failed',error='Cleanup failed or final benchmark absent',cleanup_errors=cleanup_errors)
            raise RuntimeError('Run did not complete cleanly; see run_status.json')
        self.status('complete',cleanup_errors=[])
        return False


def stop_encoder(process):
    """Terminate only the encoder spawned by this run if normal closing failed."""
    if process.poll() is not None:return
    process.terminate()
    try:process.wait(timeout=5)
    except Exception:
        process.kill();process.wait(timeout=5)

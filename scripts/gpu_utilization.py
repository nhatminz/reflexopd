"""Optional benchmark-only nvidia-smi sampling, with no new Python dependency."""
import os
import subprocess
import threading


class GPUUtilization:
    def __init__(self, enabled=False):
        self.enabled=enabled;self.samples=[];self.error=None
        self.stop_event=threading.Event();self.thread=None

    def _sample(self):
        # Standalone benchmark uses logical cuda:0. Honor numeric/UUID visibility.
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
        while not self.stop_event.is_set():
            try:
                result=subprocess.run(['nvidia-smi','-i',gpu,'--query-gpu=utilization.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True)
                self.samples.append(float(result.stdout.strip()))
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                self.error=str(exc);return
            self.stop_event.wait(.5)

    def start(self):
        if self.enabled:
            self.thread=threading.Thread(target=self._sample,daemon=True);self.thread.start()
        return self

    def finish(self):
        self.stop_event.set()
        if self.thread is not None:self.thread.join(timeout=4)
        return dict(gpu_utilization_mean=sum(self.samples)/len(self.samples) if self.samples else None,
                    gpu_utilization_samples=len(self.samples),gpu_utilization_error=self.error)

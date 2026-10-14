"""Real timing and optional measured power integrations; no fabricated energy values."""
import csv,time,statistics,platform
from pathlib import Path
import torch
import numpy as np

def integrate_power(samples,idle_w,start,end):
    s=sorted((float(t),max(0.,float(p)-idle_w)) for t,p in samples if start<=float(t)<=end)
    if len(s)<2:return None
    return float(np.trapezoid([p for _,p in s],[t for t,_ in s]))

def read_power_csv(path):
    with open(path,newline='') as f:
        rows=csv.DictReader(f)
        return [(float(r['timestamp_s']),float(r['power_w'])) for r in rows]

def measure(model,input_shape=(1,3,32,32),device='cpu',warmup=50,runs=1000,power_csv=None,idle_w=0.,expected_sampling_hz=100):
    model=model.to(device).eval(); x=torch.randn(*input_shape,device=device)
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(warmup):model(x)
        sync(); start=time.perf_counter();times=[];windows=[]
        for _ in range(runs):
            sync();a=time.perf_counter();model(x);sync();b=time.perf_counter();times.append((b-a)*1000);windows.append((a,b))
        end=time.perf_counter()
    energy=None;source='unavailable'
    if power_csv:
        samples=read_power_csv(power_csv)
        energy_total=integrate_power(samples,idle_w,start,end)
        if energy_total is not None: energy=energy_total/runs;source='measured_csv'
    return {'latency_ms':float(statistics.mean(times)),'p50_latency_ms':float(np.percentile(times,50)),
            'p95_latency_ms':float(np.percentile(times,95)),'throughput_sps':runs/(end-start),
            'energy_j':energy,'energy_source':source,'memory_mb':sum(p.numel()*p.element_size() for p in model.parameters())/1e6,
            'platform':platform.platform(),'device':str(device),'warmup':warmup,'runs':runs,'power_sampling_expected_hz':expected_sampling_hz}

class CalibratedEnergy:
    """Opt-in surrogate, explicitly tagged estimated; never interchangeable with physical power samples."""
    def __init__(self,average_dynamic_power_w):
        if average_dynamic_power_w<=0:raise ValueError('Positive calibrated power required')
        self.power_w=float(average_dynamic_power_w)
    def attach(self,report):
        report=dict(report);report['energy_j']=report['latency_ms']*.001*self.power_w
        report['energy_source']='estimated_calibrated_power';return report

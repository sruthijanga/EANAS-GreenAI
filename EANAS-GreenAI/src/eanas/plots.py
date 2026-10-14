from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def search_plots(result_dir):
    root=Path(result_dir); history=json.loads((root/'history.json').read_text())
    fig,ax=plt.subplots(figsize=(7,4.5));ax.plot([x['iteration'] for x in history],[x['best_score'] for x in history],marker='o')
    ax.set(xlabel='Search Iteration',ylabel='Best Normalized Objective',title='EANAS Search Convergence');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'convergence.jpg',dpi=400);plt.close(fig)
    results=json.loads((root/'pareto_front.json').read_text())
    if results:
        fig,ax=plt.subplots(figsize=(7,4.5));ax.scatter([x['latency_ms'] for x in results],[100*x['accuracy'] for x in results],s=[max(30,200*x['energy_j']) for x in results])
        ax.set(xlabel='Latency (ms)',ylabel='Classification Accuracy (%)',title='Feasible Pareto Frontier');ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(root/'pareto.jpg',dpi=400);plt.close(fig)

#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eanas.plots import search_plots
p=argparse.ArgumentParser();p.add_argument('--result-dir',default='results/search');a=p.parse_args();search_plots(a.result_dir);print('Generated figures in',a.result_dir)

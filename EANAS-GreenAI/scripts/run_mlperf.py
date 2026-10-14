#!/usr/bin/env python3
"""Official MLPerf Tiny integration entrypoint. Does not misrepresent image classification as MLPerf Tiny."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--benchmark-repo',required=True,help='Local clone of https://github.com/mlcommons/tiny');p.add_argument('--command',nargs=argparse.REMAINDER,help='Official benchmark task command after --command');a=p.parse_args()
repo=Path(a.benchmark_repo).resolve()
if not repo.is_dir():p.error('Benchmark repository does not exist')
if not a.command:p.error('Specify task-specific official MLPerf Tiny command after --command')
print('Executing official MLPerf Tiny command in',repo,flush=True)
subprocess.run(a.command,cwd=repo,check=True)

import json
from pathlib import Path
from dataclasses import fields
import yaml

def load_yaml(path):return yaml.safe_load(Path(path).read_text())
def write_json(path,obj):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(obj,indent=2))

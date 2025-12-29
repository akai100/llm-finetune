import yaml
import argparse
import copy
from pathlib import Path

class Config:
    def __init__(self, paths):
        self.cfg = {}
        for p in paths:
            self._merge(self._load_yaml(p))
        self._validate()
        self._post_process()

    def _load_yaml(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _merge(self, new_cfg):
        def merge_dict(a, b):
            for k, v in b.items():
                if k in a and isinstance(a[k], dict) and isinstance(v, dict):
                    merge_dict(a[k], v)
                else:
                    a[k] = v
        merge_dict(self.cfg, new_cfg)

    def override_from_cli(self):
        parser = argparse.ArgumentParser()
        for k, v in self.cfg.items():
            if isinstance(v, (int, float, str, bool)):
                parser.add_argument(f"--{k}", type=type(v))
        args = vars(parser.parse_args())
        for k, v in args.items():
            if v is not None:
                self.cfg[k] = v

    def _validate(self):
        required = ["model_name", "output_dir"]
        for r in required:
            if r not in self.cfg:
                raise ValueError(f"Missing config field: {r}")

    def _post_process(self):
        self.cfg["output_dir"] = Path(self.cfg["output_dir"])
        self.cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    def __getattr__(self, item):
        return self.cfg.get(item)

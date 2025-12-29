class Config:
    def __init__(self, cfg_path):
        self.cfg = self._load(cfg_path)
        self._validate()
        self._post_process()

    def _load(self, path):
        ...
    def merge(self, other_cfg):
        ...
    def override_from_cli(self, args):
        ...

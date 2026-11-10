"""Allow mmcv 2.2.0 in the import-time version checks of mmdet 3.2 and mmdet3d 1.4."""

import importlib.util
from pathlib import Path

for package in ("mmdet", "mmdet3d"):
    path = Path(importlib.util.find_spec(package).origin)
    source = path.read_text()
    patched = source.replace("mmcv_maximum_version = '2.2.0'", "mmcv_maximum_version = '2.3.0'")
    if patched != source:
        path.write_text(patched)
        print(f"{package}: mmcv_maximum_version -> 2.3.0")

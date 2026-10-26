# Third-party components

The project itself is proprietary (see LICENSE). The components below are
not owned by the project and keep their own licences.

| Component | Where | Licence |
|---|---|---|
| MegaDetector V6 (Microsoft AI for Good, via PytorchWildlife) | `models/MDV6-yolov9-c.pt` | MIT |
| PytorchWildlife | Python package (`requirements-lock.txt`) | MIT |
| Ultralytics YOLO11 | `yolo11n.pt`; `models/melia_wildlife.pt` is fine-tuned from it; Python package `ultralytics` | AGPL-3.0 |
| SpeciesNet (Google) | downloaded at runtime from Kaggle (not in this repository) | Apache-2.0 |
| Snapshot Serengeti (LILA BC, University of Minnesota Lion Center) | used for training; images not in this repository | CDLA-Permissive-1.0 |

Other Python packages are installed from PyPI with the versions in
`requirements-lock.txt`, each under its own licence.

Camera pictures from this site and the system's logs are kept in a private
repository and are not published here.

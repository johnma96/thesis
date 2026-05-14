# Legacy dependency files

These files have been superseded by `pyproject.toml` + `uv.lock` (root directory).

| File | Replaced by |
|---|---|
| `requirements.txt` | `[project.dependencies]` in `pyproject.toml` |
| `requirements-pytorch-cpu.txt` | `[project.optional-dependencies.pytorch-cpu]` + `uv sync --extra pytorch-cpu` |
| `requirements-pytorch-cu126.txt` | `[project.optional-dependencies.pytorch-cu126]` + `uv sync --extra pytorch-cu126` |
| `environment.yml` | `pyproject.toml` + `uv sync` |
| `setup.py` | `[build-system]` in `pyproject.toml` (hatchling) |
| `main.ipynb` | Template stub — not used in this project |

For pip-compatible exports (no uv required) use:
- `requirements-export-cpu.txt` — CPU environment
- `requirements-export-cu126.txt` — GPU/CUDA 12.6 environment

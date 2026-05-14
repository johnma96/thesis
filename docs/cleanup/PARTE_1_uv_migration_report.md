# Part 1 Report — Migration to uv

**Status:** ✅ Complete  
**Branch:** `chore/repo-cleanup-uv`

---

## Summary

| Item | Result |
|---|---|
| `pyproject.toml` created | ✅ 30 direct deps + 5 extras (pytorch-cpu, pytorch-cu126, notebooks, api, dev) |
| `uv.lock` generated | ✅ 340 packages resolved in ~7s |
| `spectralcrop` importable via `uv run` | ✅ |
| `torch 2.9.1+cu126` available | ✅ |
| CPU requirements exported | ✅ `requirements-export-cpu.txt` |
| GPU requirements exported | ✅ `requirements-export-cu126.txt` |
| Legacy files archived | ✅ `archive/legacy_dependencies/` |
| `install.md` rewritten | ✅ uv-based instructions |

---

## Key design decisions

### PyTorch CPU vs GPU handling

PyTorch's index-based distribution (`download.pytorch.org/whl/{cpu,cu126}`) creates a
conflict when uv tries to resolve both variants simultaneously.

**Solution:** `[tool.uv] conflicts` declaration marks `pytorch-cpu` and `pytorch-cu126`
as mutually exclusive extras. uv resolves each independently and stores both resolution
paths in `uv.lock`. The lock file correctly selects the right wheels at install time.

```bash
uv sync --extra pytorch-cpu    # PC A (no GPU)
uv sync --extra pytorch-cu126  # PC B (RTX 3050)
```

### pytorch-requirements exports: 2 files (not 1)

A single `requirements.txt` cannot express both CPU and GPU torch wheels because they come
from different indexes. Two separate exports are maintained:
- `requirements-export-cpu.txt` — for pip users on CPU-only machines
- `requirements-export-cu126.txt` — for pip users on CUDA 12.6 machines

### `lazypredict==0.2.12` placement

`lazypredict` was used only in `notebooks/301-jmmz-classification-exploration.ipynb`
(initial 12-algorithm sweep). It is NOT required for the production CNN-2D pipeline.
Placed in the `notebooks` extra, not in `[project.dependencies]`.

### `typer` added as new dependency

`typer` was not in `requirements.txt` but is required by the new `main.py` CLI
(Part 2). It is a direct dependency of the orchestrator.

---

## Files created / modified / archived

| Action | File |
|---|---|
| Created | `pyproject.toml` |
| Created | `uv.lock` |
| Created | `requirements-export-cpu.txt` |
| Created | `requirements-export-cu126.txt` |
| Updated | `install.md` |
| Archived | `archive/legacy_dependencies/requirements.txt` |
| Archived | `archive/legacy_dependencies/requirements-pytorch-cpu.txt` |
| Archived | `archive/legacy_dependencies/requirements-pytorch-cu126.txt` |
| Archived | `archive/legacy_dependencies/environment.yml` |
| Archived | `archive/legacy_dependencies/setup.py` |
| Archived | `archive/legacy_dependencies/main.ipynb` |

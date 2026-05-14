# Legacy pipeline files

These three files were generated in October 2025 during the exploratory phase of the project (commit `1d94703`: *"Pipelines by AI to process data: Problem with PCAs"*).

They were **never used** in the actual project. The real preprocessing pipeline is implemented in:

- `spectralcrop/data/hypercube_processor.py` — `HypercubeProcessor` class
- `notebooks/101-jmmz-preprocess-data.ipynb` — produces `data/interim/masked_reflectance.zarr`

## Why archived and not deleted

Kept for historical reference in case any algorithmic ideas (e.g., COG export, HTML report generation) are useful in future work.

| File | Notes |
|---|---|
| `pipeline_envi_cpu.py` | CPU-only ENVI→COG pipeline (550 lines, never executed) |
| `pipeline_envi_dask_zarr.py` | Dask/Zarr pipeline with **invalid Python syntax** (literal newlines in strings) — never ran |
| `pipeline_original.md` | Design spec / planning doc for the two pipeline scripts above |

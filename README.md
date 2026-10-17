# Kuo Model

Biophysical cone-to-RGC network model based on Kuo et al. Simulates the retinal circuit from cone photoreceptors through bipolar cells to retinal ganglion cells (RGCs).

## Components

- **`cone_model.py`** — Biophysical cone photocurrent model
- **`bipolar_network_model_2d.py`** — 2D bipolar-RGC network model (BipolarNetworkModel2D)
- **`barcode_stim.py`** — Barcode stimulus generation
- **`cone_kuo_pipeline.py`** — Full cone-to-Kuo pipeline
- **`kuo_bipolar_onoff_barcode_2d.py`** — Top-level runner script

## Usage

```python
from cone_kuo_pipeline import ConeKuoPipeline

pipeline = ConeKuoPipeline()
pipeline.run()
```

See `kuo_bipolar_onoff_barcode_2d.py` for a complete example.

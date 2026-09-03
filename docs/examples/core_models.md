# Core models

[`examples/midas.py`](https://github.com/bank-of-england/nowcast-midas/blob/main/examples/midas.py)
fits a single-indicator [`MIDAS`][nowcast_midas.midas.MIDAS], a multi-regressor
[`MultiMIDAS`][nowcast_midas.multi_midas.MultiMIDAS], and the full
[`MidasCombo`][nowcast_midas.midas_combo.MidasCombo] pipeline on sampled
mixed-frequency data. Run it from the repository root:

```bash
python examples/midas.py
```

```python
--8 < --"../examples/midas.py"
```

# Real-time combination pipeline

[`examples/realtime_midas.py`](https://github.com/bank-of-england/nowcast-midas/blob/main/examples/realtime_midas.py)
drives the SC-MIDAS combination through `forecast-realtime`, building synthetic
vintage metadata from sampled data (no real-time database required). It needs the
optional real-time extras:

```bash
pip install -e ".[realtime]"
python examples/realtime_midas.py
```

```python
--8 < --"../examples/realtime_midas.py"
```

# Interpreting the decomposition dataset

Every `nowcast_midas` model can decompose a forecast into additive components.
`MIDAS`, `OLS`, `MultiMIDAS`, and `MidasCombo` each expose a `forecast_decomp()` method that
returns a **long-format** table where **one row = one additive component** of a
single horizon's forecast.

The examples below use the sampled data returned by
`nowcast_midas.utils.sample_combo_data()` and the
`MidasCombo.forecast_decomp()` method. A real-time integration can enrich the
same rows with vintage and revision metadata.

---

## 1. The golden rule: contributions add up

The only invariant you ever need is that **`contribution` sums to the forecast**:

- **Level** decomposition — the components of one forecast:
  $$\hat y_h \;=\; \sum_{\text{component}} \text{contribution}$$
- **Revision** decomposition — the change between two vintages:
  $$\hat y_{v_1} - \hat y_{v_0} \;=\; \sum_{\text{component}} \text{contribution}$$

`contribution` is always the single source of truth. `weight` and `news` are
*optional* explanatory factors (see [§5](#5-weight-and-news)).

---

## 2. Columns

`forecast_decomp()` returns the model columns. A real-time integration can add
the remaining columns when it stores one decomposition for each vintage.

| Column | From | Meaning |
|---|---|---|
| `horizon` / `forecast_horizon` | model | Steps ahead (0 = nowcast) |
| `date` | model | Target period being forecast |
| `component` | model | **Contributing component** (see [§3](#3-component-namespace)) |
| `contribution` | model | **Signed additive contribution** (target units) |
| `weight` | model | Scalar multiplier, or `NaN` (see [§5](#5-weight-and-news)) |
| `variable` | realtime | The target being forecast |
| `source` | realtime | Model label (distinguishes multiple models) |
| `vintage_date` | realtime | The vintage this decomposition was computed at |
| `base_vintage_date` | realtime | Previous vintage (revision rows); `NaT` for level |
| `decomposition` | realtime | `"level"` or `"revision"` |
| `revision_source` | realtime | `"news"` / `"reestimation"` / `"interaction"`; blank for level |
| `news` | realtime | Surprise factor = `contribution / weight`; often `NaN` |
| `forecast_metric` | realtime | Transform the forecast is expressed in (`levels`, `pop`, …) |

---

## 3. Component namespace

All three models share the same **row types**, distinguished by the `component`
label. The `weight` convention is what matters:

| Row type | `component` | `contribution` | `weight` |
|---|---|---|---|
| Intercept | `intercept` | $\alpha$ | `1.0` |
| MIDAS data block | `<variable>` | $\sum_j x_j\,(\beta w_j)$ | **`NaN`** |
| Quarterly data block | `<variable>` | $\sum_j z_j\,\delta_j$ | **`NaN`** |
| Dummy | `dummy_{i}` | $d_i\,\gamma_i$ | $\gamma_i$ |
| AR lag | `ar_lag{k}` | $y_{t-k}\,\phi_k$ | $\phi_k$ |

**Why the data blocks have `weight = NaN`:** a MIDAS block's contribution is a
*sum over lags* $\sum_j x_j\,\beta w_j$ — many observations, each with its own
coefficient. There is no single scalar that multiplies a single observable, so
`weight` (and therefore `news`) is left blank. The intercept, dummy and AR rows
*are* single coefficient × single observable, so they carry a scalar weight.

---

## 4. Per-model specifics

### MIDAS

One monthly regressor. Standalone, the data block is named `X` by default
(`forecast_decomp(..., regressor_name="X")`); inside a combo it takes the
indicator's name.

```
intercept          weight = 1.0
X (or <variable>)  weight = NaN     # the weighted-lag block
dummy_0, dummy_1…  weight = gamma_i # only if dummies were fitted
ar_lag1, ar_lag2…  weight = phi_k   # only if n_ar_lags > 0
```

### MultiMIDAS

Same layout, but **one data-block row per regressor** — monthly (MIDAS) *and*
quarterly (linear). Both kinds of data block use `weight = NaN`.

```
intercept                  weight = 1.0
<monthly_var_1>            weight = NaN   # MIDAS weighted-lag block
<monthly_var_2>            weight = NaN
<quarterly_var_1>          weight = NaN   # quarterly linear block (also a lag sum)
dummy_0, dummy_1…          weight = gamma_i
ar_lag1, ar_lag2…          weight = phi_k
```

### MidasCombo

A combination flattens down to its leaf indicator models. Every component is
namespaced **`"{model}::{component}"`**, and each leaf's contribution is scaled
by its *effective* combination weight $w_\text{eff}$:

$$\hat y_h = \sum_{\text{model}} w_{\text{eff, model}} \sum_{\text{component}} \text{contribution}_{\text{model, component}}$$

- `gdp_growth_rate::intercept` → `weight` = $w_\text{eff} \times 1 = w_\text{eff}$
  (so the **combo weight lives on the intercept row**).
- `gdp_growth_rate::gdp_growth_rate` → `weight` = `NaN` (the MIDAS block).
- `gdp_growth_rate::dummy_0` → `weight` = $w_\text{eff}\,\gamma$.

The effective weights across leaf models sum to 1 (for a weighted combo), so the
namespaced contributions still add up to the combined forecast.

---

## 5. `weight` and `news`

For **linear** rows the contribution factorises:

$$\text{contribution} = \underbrace{\text{weight}}_{w_i} \times \underbrace{\text{news}}_{x_i - E[x_i\mid\Omega_{v_0}]}$$

- `weight` — the sensitivity of the forecast to that input.
- `news` — the **surprise** (how much the input differed from expectation). In
  `rt_model.decompositions` it is backed out as `contribution / weight`.

Both are **optional** and are `NaN` for the MIDAS / quarterly data blocks,
because those have no single scalar weight.

> ⚠️ **Two things are called "news".** The `revision_source == "news"` *value*
> is the slice of a revision caused by new data. The `news` *column* is the
> surprise magnitude. They are unrelated — the column is often `NaN` even on
> `revision_source == "news"` rows.

---

## 6. Level vs revision

| `decomposition` | `base_vintage_date` | `revision_source` | Adds up to |
|---|---|---|---|
| `level` | `NaT` | blank | the forecast $\hat y$ |
| `revision` | old vintage | `news` / `reestimation` / `interaction` | the change $\Delta\hat y$ |

For revision rows, the `revision_source` values split the revision into three
economic pieces:

- **`news`** — new data arrived (parameters held fixed). *This is the only slice
  that represents a data release.*
- **`reestimation`** — parameters changed (data held fixed), e.g. because a new
  target outturn grew the sample.
- **`interaction`** — the data × parameter cross-term.

---

## 7. Worked reading

From `model.forecast_decomp(aggregate=True)`:

| component | contribution | weight | decomposition | revision_source |
|---|---|---|---|---|
| `gdp_growth_rate::intercept` | 0.0575 | 0.5229 | level | — |
| `gdp_growth_rate::gdp_growth_rate` | 0.0410 | NaN | level | — |

Read as: within the combo, the `gdp_growth_rate` MIDAS model has an **effective
weight of 0.52** (on its intercept row); its intercept adds **+0.0575** and its
monthly-data block adds **+0.0410** to the nowcast. The data block has no scalar
weight, so `weight`/`news` are blank — but its *contribution* is fully defined.

---

## 8. Recipes

**Which indicator's new data moved the nowcast, and by how much**

```python
dec = model.forecast_decomp()
moved = (
    dec[(dec["decomposition"] == "revision") & (dec["revision_source"] == "news")]
    .assign(indicator=lambda d: d["component"].str.split("::").str[0])
    .groupby(["source", "vintage_date", "base_vintage_date", "indicator"])[
        "contribution"
    ]
    .sum()
    .reset_index()
    .query("contribution != 0")
)
```

`contribution` is the amount; `component` (→ `indicator`) is which model;
`source` separates multiple models. To name the *actual monthly release*
(date + value), join your own source-data table — that detail is not in the
decomposition.

**Reconstruct a leaf model's raw (pre-combo) forecast**

```python
lvl = dec[dec["decomposition"] == "level"].assign(
    indicator=lambda d: d["component"].str.split("::").str[0]
)
w = (
    lvl[lvl["component"].str.contains("intercept")]
    .groupby(["source", "vintage_date", "indicator"])["weight"]
    .first()
)
tot = lvl.groupby(["source", "vintage_date", "indicator"])["contribution"].sum()
raw = tot / w  # divide the combined contribution by the effective combo weight
```

**Sanity-check additivity**

```python
# Level rows for one vintage/source should sum to the forecast value.
lvl.groupby(["source", "vintage_date", "date"])["contribution"].sum()
```

---

## TL;DR

- One row per additive component; **`contribution` always sums to the forecast**.
- `intercept` / `dummy_i` / `ar_lagk` carry a scalar `weight`; **MIDAS and
  quarterly data blocks carry `weight = NaN`** (a lag sum has no single weight).
- `MidasCombo` namespaces every component as `model::component` and puts the
  **effective combo weight on the intercept row**. Pass `aggregate=True` to
  collapse each leaf model to a single row instead.
- Filter `decomposition == "revision"` + `revision_source == "news"` and read
  `contribution` to see which indicator's new data moved the forecast.

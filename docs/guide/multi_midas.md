# Multi-regressor MIDAS

The [`MultiMIDAS`](../api.md#nowcast_midas.multi_midas.MultiMIDAS) class
extends the single-indicator MIDAS regression to **several
high-frequency (monthly) regressors plus, optionally, quarterly
regressors** that enter the model linearly.  Every regressor can keep
its own weighting scheme, lag length and start lag.

## Model

For each forecast horizon $h$ the model is

$$
y_{t+h} \;=\; \alpha
   \;+\; \sum_{k=1}^{K} \beta_k \sum_{j=0}^{L_k-1} w_k(j;\theta_k)\, X_{k,t,j}
   \;+\; \sum_{\ell=1}^{Q} \sum_{m=0}^{M_\ell-1} \delta_{\ell,m}\, Z_{\ell,t-m}
   \;+\; \mathbf{d}_{t+h}^{\!\top}\gamma
   \;+\; \sum_{p=1}^{P} \phi_p\, y_{t+h-p}
   \;+\; \varepsilon_{t+h}
$$

where

* $X_{k,t}$ is the $k$-th **monthly** indicator with its own MIDAS weight
  function $w_k(\cdot;\theta_k)$ and slope $\beta_k$,
* $Z_{\ell,t}$ is the $\ell$-th **quarterly** regressor entering
  linearly with one $\delta_{\ell,m}$ per quarterly lag,
* $\mathbf{d}_{t+h}$ are optional outlier dummies,
* $\phi_p$ are optional AR coefficients of the target.

## Estimator routing

Weighting methods fall into two groups:

* **Linear-in-parameters** (`almon`, `unrestricted`): the lag weights are
  a linear map of their parameters, so the corresponding term can be
  written as fixed design columns. For `almon`, $w(j)=\sum_i c_i\,j^i$
  gives $X_k\,w_k = (X_k V_k)\,c_k$ with $V_k$ the polynomial (Vandermonde)
  basis; `unrestricted` uses $X_k$ directly. These are estimated by OLS
  with $\beta_k$ fixed to $1.0$.
* **Nonlinear** (`exp_almon`, `beta`): the weights depend nonlinearly on
  $\theta_k$ (exponential/Beta density with normalisation), so the model
  estimates the weight function and slope $\beta_k$ together.

| Monthly methods                     | Estimator |
|-------------------------------------|-----------|
| All `almon` or `unrestricted`       | joint **OLS** (closed-form) |
| At least one `exp_almon` or `beta`  | joint **NLS** (Levenberg-Marquardt with JAX Jacobian) |

The routing rule selects the solver, but it does not change each variable's
role in the model. `almon`/`unrestricted` terms and quarterly regressors
always enter as fixed design columns with $\beta_k = 1.0$. The model
estimates only the `exp_almon`/`beta` weight functions and their slopes on
the NLS path.

!!! note "Why linear, not normalised, inside NLS"
    Keeping a normalised weight scheme such as `almon`'s $w/\sum w$
    alongside a separately estimated slope $\beta_k$ would leave $\theta_k$ identified
    only up to scale (the scale cancels in the normalisation), so the
    optimiser can drift to degenerate parameters. Estimating these terms
    linearly removes the redundant degree of freedom and makes an
    `almon` variable yield **identical** estimates whether the model is
    solved by OLS or by NLS. The trade-off is interpretive: the recovered
    `weights` are unnormalised ($V_k c_k$, with $\beta_k = 1.0$) rather
    than summing to one.

Quarterly regressors are always linear and are folded into both paths
without changing the routing rule.

## Input data

`fit` takes a target frame with `date`/`value` and a long-format
regressor frame with `date`/`variable`/`value`:

```python
target = pd.DataFrame({"date": q_dates, "value": y})
regressors = pd.DataFrame({"date": ..., "variable": ..., "value": ...})
```

The examples below use the built-in simulator so every block runs as-is;
later blocks continue from these names:

```python
from nowcast_midas import MultiMIDAS
from nowcast_midas.utils import sample_combo_data

# long-format frames with monthly_1..3 (ME) + quarterly_1 (QE) regressors
target, regressors, info = sample_combo_data(n_quarters=60, seed=42)
```

## Specifying the model

The simplest call shares the same weight scheme across all monthly
indicators:

```python
from nowcast_midas import MultiMIDAS

model = MultiMIDAS(
    variables=["monthly_1", "monthly_2"],
    method="almon",
    n_lags=6,
    horizons=[0, 1],
)
model.fit(target, regressors)
```

For per-variable overrides — including mixing monthly and quarterly
regressors — pass [`VariableSpec`](../api.md#nowcast_midas.specs.VariableSpec)
objects:

```python
from nowcast_midas import VariableSpec

model = MultiMIDAS(
    variables=[
        VariableSpec("monthly_1", method="exp_almon", n_lags=6),
        VariableSpec("monthly_2", method="almon", n_lags=3),
        VariableSpec("quarterly_1", frequency="QE", n_lags=2),
    ],
    horizons=[0, 1],
)
model.fit(target, regressors)
```

`VariableSpec(frequency="QE", n_lags=M)` requests a quarterly regressor
with $M$ quarterly lags; `method`, `n_pars_weights` and `estimator` are
ignored for quarterly variables.

## Outputs

After `fit`, results for each horizon are stored in `model.fits_[h]`
([`FittedMultiMidas`](../api.md#nowcast_midas.multi_midas.FittedMultiMidas)):

* `alpha`, `gamma` (dummy coefficients), `phi` (AR coefficients).
* `variable_fits[name]` →
  [`VariableFit`](../api.md#nowcast_midas.multi_midas.VariableFit) with
    * `beta` — slope; estimated separately only for nonlinear methods
      (`exp_almon`, `beta`). It is fixed to $1.0$ for the
      linearly-estimated methods (`almon`, `unrestricted`) and for
      quarterly regressors, regardless of whether the model was solved by
      OLS or NLS,
    * `theta` — weight-shape parameters (or quarterly deltas),
    * `weights` — evaluated lag weights (length `n_lags`).

For the nonlinear methods, `weights` are the normalised MIDAS weights
$w_k(\theta_k)$. For `almon`/`unrestricted`, `weights` are the
unnormalised lag coefficients ($V_k c_k$ or $c_k$). For quarterly
regressors, `weights` is the vector of linear coefficients
$(\delta_{\ell,0}, \dots, \delta_{\ell,M-1})$.

## Forecasting

```python
fc = model.forecast(regressors)
```

`forecast` returns a **long-format** DataFrame with one row per fitted
horizon and the columns `date`, `horizon`, `spec`, `value`.  `spec` is
the `"+"`-joined regressor variable names (a MultiMIDAS forecast is a
single joint prediction).  Monthly and quarterly lag rows are built
independently, and a missing lag in any regressor anchors that horizon's
forecast to NaN.

`fit()` additionally stores the long-format in-sample fitted values as
`model.fits_df_` (`date`, `horizon`, `value`), and `forecast()` stores its
result as `model.forecasts_df_`.

```python
dec = model.forecast_decomp(regressors)
```

`forecast_decomp` splits each horizon's forecast into additive components
— one row per regressor block plus `intercept`, `dummy_i` and `ar_lagk`
— that sum back to the forecast. See
[Interpreting decompositions](decomposition.md).

## Pipeline integration

Use [`MultiMidasSpec`](../api.md#nowcast_midas.specs.MultiMidasSpec) to embed
a `MultiMIDAS` block as a single named source inside
[`MidasCombo`](../api.md#nowcast_midas.midas_combo.MidasCombo):

```python
from nowcast_midas import ComboSpec, MidasCombo, MultiMidasSpec

multi_spec = MultiMidasSpec(
    name="multi_block",
    variables=[
        VariableSpec("monthly_1", method="almon", n_lags=6),
        VariableSpec("monthly_2", method="exp_almon", n_lags=6),
        VariableSpec("quarterly_1", frequency="QE", n_lags=1),
    ],
)

combo = ComboSpec(name="my_combo", sources=[multi_spec], method="average")
pipe = MidasCombo(combo_specs=combo, horizons=4)
pipe.fit(target, regressors)
forecasts = pipe.forecast()
```

The fitted MultiMIDAS model is exposed via
`pipe.multi_midas_instances_[name]` and its per-horizon `FittedMultiMidas`
under `pipe.multi_midas_models_[name][h]`.

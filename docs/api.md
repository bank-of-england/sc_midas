# API Reference

This script writes the API manifest for the public nowcast_midas API. Zensical renders the API content from the current source.

## High-level Pipeline

::: nowcast_midas.midas_combo.MidasCombo
    options:
      show_source: false
      show_root_heading: true

## MIDAS Regression (single indicator)

::: nowcast_midas.midas.MIDAS
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.midas.FittedMidas
    options:
      show_source: false
      show_root_heading: true

## Quarterly OLS (single regressor)

::: nowcast_midas.ols.OLS
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.ols.FittedOLS
    options:
      show_source: false
      show_root_heading: true

## Multi-Regressor MIDAS

::: nowcast_midas.multi_midas.FittedMultiMidas
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.multi_midas.MultiMIDAS
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.multi_midas.VariableFit
    options:
      show_source: false
      show_root_heading: true

## Specifications

::: nowcast_midas.specs.ComboSpec
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.specs.MidasSpec
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.specs.MultiMidasSpec
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.specs.OLSSpec
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.specs.VariableSpec
    options:
      show_source: false
      show_root_heading: true

## Weighting Schemes

::: nowcast_midas.temporal_weights.almon
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.temporal_weights.beta
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.temporal_weights.exp_almon
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.temporal_weights.get_weights
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.temporal_weights.unrestricted
    options:
      show_source: false
      show_root_heading: true

## Combination Weights

::: nowcast_midas.combo_weights.clipped_ols
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.combo_weights.constrained_least_squares
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.combo_weights.fit_average
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.combo_weights.fit_weights
    options:
      show_source: false
      show_root_heading: true

## Example Data

::: nowcast_midas.utils.sample_combo_data
    options:
      show_source: false
      show_root_heading: true

::: nowcast_midas.utils.sample_data
    options:
      show_source: false
      show_root_heading: true

from importlib.metadata import version

from .midas import MIDAS, FittedMidas
from .midas_combo import MidasCombo
from .multi_midas import FittedMultiMidas, MultiMIDAS, VariableFit
from .ols import OLS, FittedOLS
from .specs import ComboSpec, MidasSpec, MultiMidasSpec, OLSSpec, VariableSpec

__version__ = version("nowcast-midas")


__all__ = [
    "MIDAS",
    "OLS",
    "ComboSpec",
    "FittedMidas",
    "FittedMultiMidas",
    "FittedOLS",
    "MidasCombo",
    "MidasSpec",
    "MultiMIDAS",
    "MultiMidasSpec",
    "OLSSpec",
    "VariableFit",
    "VariableSpec",
]

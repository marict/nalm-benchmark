import torch

from stable_nalu.layer.inalu import INALULayer

from ..abstract import ExtendedTorchModule
from .basic import BasicCell, BasicLayer
from .dag import DAGLayer
from .gradient_bandit_nac import GradientBanditNACCell, GradientBanditNACLayer
from .gradient_bandit_nalu import (GradientBanditNALUCell,
                                   GradientBanditNALULayer)
from .gumbel_mnac import GumbelMNACCell, GumbelMNACLayer
from .gumbel_nac import GumbelNACCell, GumbelNACLayer
from .gumbel_nalu import GumbelNALUCell, GumbelNALULayer
from .hard_softmax_nac import HardSoftmaxNACCell, HardSoftmaxNACLayer
from .hard_softmax_nalu import HardSoftmaxNALUCell, HardSoftmaxNALULayer
from .independent_nac import IndependentNACCell, IndependentNACLayer
from .independent_nalu import IndependentNALUCell, IndependentNALULayer
from .linear_nac import LinearNACCell, LinearNACLayer
from .linear_nalu import LinearNALUCell, LinearNALULayer
from .mcfc import (MCFullyConnected, MulMCFC, MulMCFCSignINALU,
                   MulMCFCSignRealNPU)
from .mnac import MNACCell, MNACLayer
from .nac import NACCell, NACLayer
from .nalu import NALUCell, NALULayer
from .npu import NPULayer
from .npu_real import RealNPULayer
from .pos_nac import PosNACCell, PosNACLayer
from .pos_nalu import PosNALUCell, PosNALULayer
from .re_regualized_linear_mnac import (ReRegualizedLinearMNACCell,
                                        ReRegualizedLinearMNACLayer)
from .re_regualized_linear_nac import (ReRegualizedLinearNACCell,
                                       ReRegualizedLinearNACLayer)
from .re_regualized_linear_nalu import (ReRegualizedLinearNALUCell,
                                        ReRegualizedLinearNALULayer)
from .re_regualized_linear_pos_nac import (ReRegualizedLinearPosNACCell,
                                           ReRegualizedLinearPosNACLayer)
from .regualized_linear_mnac import (RegualizedLinearMNACCell,
                                     RegualizedLinearMNACLayer)
from .regualized_linear_nac import (RegualizedLinearNACCell,
                                    RegualizedLinearNACLayer)
from .regualized_linear_nalu import (RegualizedLinearNALUCell,
                                     RegualizedLinearNALULayer)
from .silly_re_regualized_linear_mnac import (SillyReRegualizedLinearMNACCell,
                                              SillyReRegualizedLinearMNACLayer)
from .softmax_nac import SoftmaxNACCell, SoftmaxNACLayer
from .softmax_nalu import SoftmaxNALUCell, SoftmaxNALULayer

unit_name_to_layer_class = {
    "NAC": NACLayer,
    "MNAC": MNACLayer,
    "NALU": NALULayer,
    "PosNAC": PosNACLayer,
    "PosNALU": PosNALULayer,
    "GumbelNAC": GumbelNACLayer,
    "GumbelMNAC": GumbelMNACLayer,
    "GumbelNALU": GumbelNALULayer,
    "LinearNAC": LinearNACLayer,
    "LinearNALU": LinearNALULayer,
    "SoftmaxNAC": SoftmaxNACLayer,
    "SoftmaxNALU": SoftmaxNALULayer,
    "IndependentNAC": IndependentNACLayer,
    "IndependentNALU": IndependentNALULayer,
    "HardSoftmaxNAC": HardSoftmaxNACLayer,
    "HardSoftmaxNALU": HardSoftmaxNALULayer,
    "GradientBanditNAC": GradientBanditNACLayer,
    "GradientBanditNALU": GradientBanditNALULayer,
    "RegualizedLinearNAC": RegualizedLinearNACLayer,
    "RegualizedLinearMNAC": RegualizedLinearMNACLayer,
    "RegualizedLinearNALU": RegualizedLinearNALULayer,
    "ReRegualizedLinearNAC": ReRegualizedLinearNACLayer,
    "ReRegualizedLinearMNAC": ReRegualizedLinearMNACLayer,
    "ReRegualizedLinearNALU": ReRegualizedLinearNALULayer,
    "ReRegualizedLinearPosNAC": ReRegualizedLinearPosNACLayer,
    "SillyReRegualizedLinearNAC": None,
    "SillyReRegualizedLinearMNAC": SillyReRegualizedLinearMNACLayer,
    "SillyReRegualizedLinearNALU": None,
    "NPU": NPULayer,
    "RealNPU": RealNPULayer,
    "iNALU": INALULayer,
    "MCFC": MCFullyConnected,
    "MulMCFC": MulMCFC,
    "MulMCFCSignINALU": MulMCFCSignINALU,
    "MulMCFCSignRealNPU": MulMCFCSignRealNPU,
    "DAG": DAGLayer,
}

unit_name_to_cell_class = {
    "NAC": NACCell,
    "MNAC": MNACCell,
    "NALU": NALUCell,
    "PosNAC": PosNACCell,
    "PosNALU": PosNALUCell,
    "GumbelNAC": GumbelNACCell,
    "GumbelMNAC": GumbelMNACCell,
    "GumbelNALU": GumbelNALUCell,
    "SoftmaxNAC": SoftmaxNACCell,
    "SoftmaxNALU": SoftmaxNALUCell,
    "IndependentNAC": IndependentNACCell,
    "IndependentNALU": IndependentNALUCell,
    "HardSoftmaxNAC": HardSoftmaxNACCell,
    "HardSoftmaxNALU": HardSoftmaxNALUCell,
    "GradientBanditNAC": GradientBanditNACCell,
    "GradientBanditNALU": GradientBanditNALUCell,
    "RegualizedLinearNAC": RegualizedLinearNACCell,
    "RegualizedLinearNALU": RegualizedLinearNALUCell,
    "ReRegualizedLinearNAC": ReRegualizedLinearNACCell,
    "ReRegualizedLinearMNAC": ReRegualizedLinearMNACCell,
    "ReRegualizedLinearNALU": ReRegualizedLinearNALUCell,
    "ReRegualizedLinearPosNAC": ReRegualizedLinearPosNACCell,
}


class GeneralizedLayer(ExtendedTorchModule):
    """Abstracts all layers, both basic, NAC and NALU

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
        unit_name: name of the unit (e.g. NAC, Sigmoid, Tanh)
    """

    UNIT_NAMES = set(unit_name_to_layer_class.keys()) | BasicLayer.ACTIVATIONS

    def __init__(
        self, in_features, out_features, unit_name, writer=None, name=None, **kwags
    ):
        super().__init__("layer", name=name, writer=writer, **kwags)
        self.in_features = in_features
        self.out_features = out_features
        self.unit_name = unit_name

        if unit_name in unit_name_to_layer_class:
            Layer = unit_name_to_layer_class[unit_name]
            self.layer = Layer(in_features, out_features, writer=self.writer, **kwags)
        else:
            self.layer = BasicLayer(
                in_features,
                out_features,
                activation=unit_name,
                writer=self.writer,
                **kwags,
            )

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, input):
        return self.layer(input)

    def extra_repr(self):
        return "in_features={}, out_features={}, unit_name={}".format(
            self.in_features, self.out_features, self.unit_name
        )


class GeneralizedCell(ExtendedTorchModule):
    """Abstracts all cell, RNN-tanh, RNN-ReLU, GRU, LSTM, NAC and NALU

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
        unit_name: name of the unit (e.g. RNN-tanh, LSTM, NAC)
    """

    UNIT_NAMES = set(unit_name_to_cell_class.keys()) | {
        "GRU",
        "LSTM",
        "RNN-tanh",
        "RNN-ReLU",
        "RNN-linear",
    }

    def __init__(self, input_size, hidden_size, unit_name, writer=None, **kwags):
        super().__init__("cell", writer=writer, **kwags)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.unit_name = unit_name

        if unit_name in unit_name_to_cell_class:
            Cell = unit_name_to_cell_class[unit_name]
            self.cell = Cell(input_size, hidden_size, writer=self.writer, **kwags)
        elif unit_name == "none":
            self.cell = PassThoughCell(input_size, hidden_size, **kwags)
        elif unit_name == "GRU":
            self.cell = torch.nn.GRUCell(input_size, hidden_size)
        elif unit_name == "LSTM":
            self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        elif unit_name == "RNN-tanh":
            self.cell = torch.nn.RNNCell(input_size, hidden_size, nonlinearity="tanh")
        elif unit_name == "RNN-ReLU":
            self.cell = torch.nn.RNNCell(input_size, hidden_size, nonlinearity="relu")
        elif unit_name == "RNN-linear":
            self.cell = BasicCell(
                input_size,
                hidden_size,
                activation="linear",
                writer=self.writer,
                **kwags,
            )
        else:
            raise NotImplementedError(f"{unit_name} is not an implemented cell type")

    def reset_parameters(self):
        self.cell.reset_parameters()

    def forward(self, x_t, h_tm1):
        return self.cell(x_t, h_tm1)

    def extra_repr(self):
        return "input_size={}, hidden_size={}, unit_name={}".format(
            self.input_size, self.hidden_size, self.unit_name
        )

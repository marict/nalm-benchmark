import argparse
import ast
import datetime
import math
import os
import random
from decimal import Decimal

import numpy as np
import runpod_service.wandb_setup as wandb
import torch

# Enable anomaly detection for debugging gradient issues
torch.autograd.set_detect_anomaly(True, check_nan=True)

import misc.utils as utils
import stable_nalu
import stable_nalu.functional.regualizer as Regualizer
from debug_utils import tap_context
from stable_nalu.layer import DAGLayer
from stable_nalu.layer.dag import DAGLayer


def print_dag_internal_state(
    x_tensor,
    y_tensor,
    t_tensor,
    g_tensor,
    o_tensor,
    out_logits_tensor,
    value_vec_inter_tensor,
    state_label,
    o_sign_tensor=None,
    o_mag_tensor=None,
    dag_layer=None,
):
    """Print DAG internal state. Extracts [0] from all tensors and formats output."""

    # Extract and convert all tensors
    x_vals = x_tensor[0].detach().cpu().view(-1).tolist()
    y_val = float(y_tensor[0].detach().cpu().view(-1)[0].item())
    t_val = float(t_tensor[0].detach().cpu().view(-1)[0].item())
    g_state = g_tensor[0].detach().cpu().tolist()
    o_state = o_tensor[0].detach().cpu().tolist()
    out_logits_state = out_logits_tensor[0].detach().cpu()
    value_vec_inter_state = value_vec_inter_tensor[0].detach().cpu().tolist()

    # Handle optional sign/mag tensors
    o_sign_state = None
    o_mag_state = None
    if o_sign_tensor is not None:
        o_sign_state = o_sign_tensor[0].detach().cpu().tolist()
    if o_mag_tensor is not None:
        o_mag_state = o_mag_tensor[0].detach().cpu().tolist()

    print(f"Sample statistics ({state_label} state):")
    print(f"input={[round(x, 5) for x in x_vals]}")
    print(f"output={round(y_val, 5)}, target={round(t_val, 5)}")
    print(f"G ({state_label.lower()}): {[round(g, 5) for g in g_state]}")

    # Get intermediate values at each step if DAG layer is available
    working_values = None
    if (
        dag_layer is not None
        and hasattr(dag_layer, "_debug_working_mag")
        and hasattr(dag_layer, "_debug_working_sign")
    ):
        if (
            len(dag_layer._debug_working_mag) > 0
            and len(dag_layer._debug_working_sign) > 0
        ):
            # Combine mag and sign to get actual values
            working_values = []
            for i in range(len(dag_layer._debug_working_mag)):
                mag = (
                    dag_layer._debug_working_mag[i][0].detach().cpu()
                )  # First batch element
                sign = (
                    dag_layer._debug_working_sign[i][0].detach().cpu()
                )  # First batch element
                values = (mag * sign).tolist()
                working_values.append(values)

    for step_idx, step_values in enumerate(o_state):
        rounded_selectors = [round(v, 5) for v in step_values]
        g_value = round(g_state[step_idx], 1) if step_idx < len(g_state) else "N/A"

        # Get intermediate values available at this step
        if working_values is not None and step_idx < len(working_values):
            intermediate_vals = [round(v, 5) for v in working_values[step_idx]]

            # Compute the result of this step (stored in next working_values if available)
            computed_value = "N/A"
            if step_idx + 1 < len(working_values):
                # The new value is stored at position num_initial_nodes + step_idx
                new_idx = dag_layer.num_initial_nodes + step_idx
                if new_idx < len(working_values[step_idx + 1]):
                    computed_value = round(working_values[step_idx + 1][new_idx], 5)

            print(f"    Step {step_idx}:")
            print(f"        selectors: {rounded_selectors}")
            print(f"        inputs:    {intermediate_vals}")
            print(f"        G: {g_value} → computed_value: {computed_value}")
        else:
            # Fallback to old format if debug data not available
            print(f"    Step {step_idx}: selectors {rounded_selectors} G: {g_value}")

        # Only print separate sign/mag if they exist (not in unified selector mode)
        if o_sign_state is not None and o_mag_state is not None:
            rounded_sign = [round(o, 5) for o in o_sign_state[step_idx]]
            rounded_mag = [round(o, 5) for o in o_mag_state[step_idx]]
            print(f"        sign ({state_label.lower()}): {rounded_sign}")
            print(f"        mag ({state_label.lower()}): {rounded_mag}")

    # Add output selector information
    if out_logits_state is not None:
        selected_idx = torch.argmax(out_logits_state).item()
        one_hot = torch.zeros_like(out_logits_state)
        one_hot[selected_idx] = 1.0

        print(f"Output Selector ({state_label.lower()}):")
        print(
            f"\tlogits ({state_label.lower()}): {[round(v, 5) for v in out_logits_state.tolist()]}"
        )
        if state_label.upper() == "SOFT":
            print(f"\tselected (one-hot): {[int(v) for v in one_hot.tolist()]}")
        print(f"\tselected_node: {selected_idx}")

        if value_vec_inter_state is not None:
            print(
                f"\tintermediate_values ({state_label.lower()}): {[round(v, 5) for v in value_vec_inter_state]}"
            )
            print(
                f"\tselected_value ({state_label.lower()}): {round(value_vec_inter_state[selected_idx], 5)}"
            )


class NoOpScheduler:
    """A scheduler that does nothing - always returns the initial learning rate."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        """Do nothing when stepping."""
        pass

    def get_last_lr(self):
        """Return the current learning rates from the optimizer."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        """Return empty state dict."""
        return {}

    def load_state_dict(self, state_dict):
        """Do nothing when loading state dict."""
        pass


def get_default_note():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Parse arguments
parser = argparse.ArgumentParser(description="Runs the simple function static task")
parser.add_argument(
    "--id",
    action="store",
    default=-1,
    type=int,
    help="Unique id to identify experiment",
)
parser.add_argument(
    "--layer-type",
    action="store",
    default="NALU",
    choices=list(stable_nalu.network.SimpleFunctionStaticNetwork.UNIT_NAMES),
    type=str,
    help="Specify the layer type, e.g. Tanh, ReLU, NAC, NALU",
)
parser.add_argument(
    "--operation",
    action="store",
    default="add",
    choices=["add", "sub", "mul", "div"],
    type=str,
    help="Specify the operation to use, e.g. add, mul, squared",
)
parser.add_argument(
    "--num-subsets",
    action="store",
    default=2,
    type=int,
    help="Specify the number of subsets to use",
)
parser.add_argument(
    "--regualizer",
    action="store",
    default=10,
    type=float,
    help="Specify the regualization lambda to be used",
)
parser.add_argument(
    "--regualizer-z",
    action="store",
    default=0,
    type=float,
    help="Specify the z-regualization lambda to be used",
)
parser.add_argument(
    "--regualizer-oob",
    action="store",
    default=1,
    type=float,
    help="Specify the oob-regualization lambda to be used",
)
parser.add_argument(
    "--first-layer",
    action="store",
    default=None,
    help="Set the first layer to be a different type",
)
parser.add_argument(
    "--max-iterations",
    action="store",
    default=100000,
    type=int,
    help="Specify the max number of iterations to use",
)
parser.add_argument(
    "--batch-size",
    action="store",
    default=128,
    type=int,
    help="Specify the batch-size to be used for training",
)
parser.add_argument(
    "--seed", action="store", default=0, type=int, help="Specify the seed to use"
)
parser.add_argument(
    "--interpolation-range",
    action="store",
    default=[1, 2],
    type=ast.literal_eval,
    help="Specify the interpolation range that is sampled uniformly from",
)
parser.add_argument(
    "--extrapolation-range",
    action="store",
    default=[2, 4],
    type=ast.literal_eval,
    help="Specify the extrapolation range that is sampled uniformly from",
)
parser.add_argument(
    "--input-size", action="store", default=2, type=int, help="Specify the input size"
)
parser.add_argument(
    "--output-size", action="store", default=1, type=int, help="Specify the output size"
)
parser.add_argument(
    "--subset-ratio",
    action="store",
    default=0.5,
    type=float,
    help="Specify the subset-size as a fraction of the input-size",
)
parser.add_argument(
    "--overlap-ratio",
    action="store",
    default=0.0,
    type=float,
    help="Specify the overlap-size as a fraction of the input-size",
)
parser.add_argument(
    "--simple",
    action="store_true",
    default=False,
    help="Use a very simple dataset with t = sum(v[0:2]) + sum(v[4:6])",
)
parser.add_argument(
    "--hidden-size",
    action="store",
    default=2,
    type=int,
    help="Specify the vector size of the hidden layer.",
)
parser.add_argument(
    "--nac-mul",
    action="store",
    default="none",
    choices=["none", "normal", "safe", "max-safe", "mnac", "npu", "real-npu"],
    type=str,
    help="Make the second NAC a multiplicative NAC, used in case of a just NAC network.",
)
parser.add_argument(
    "--oob-mode",
    action="store",
    default="clip",
    choices=["regualized", "clip"],
    type=str,
    help="Choose of out-of-bound should be handled by clipping or regualization.",
)
parser.add_argument(
    "--regualizer-scaling",
    action="store",
    default="linear",
    choices=["exp", "linear"],
    type=str,
    help="Use an expoentational scaling from 0 to 1, or a linear scaling.",
)
parser.add_argument(
    "--regualizer-scaling-start",
    action="store",
    default=1000000,
    type=int,
    help="Start linear scaling at this global step.",
)
parser.add_argument(
    "--regualizer-scaling-end",
    action="store",
    default=2000000,
    type=int,
    help="Stop linear scaling at this global step.",
)
parser.add_argument(
    "--regualizer-shape",
    action="store",
    default="linear",
    choices=["squared", "linear", "none"],
    type=str,
    help="Use either a squared or linear shape for the bias and oob regualizer. Use none so W reg in tensorboard is logged at 0",
)
parser.add_argument(
    "--mnac-epsilon",
    action="store",
    default=0,
    type=float,
    help="Set the idendity epsilon for MNAC.",
)
parser.add_argument(
    "--nalu-bias",
    action="store_true",
    default=False,
    help="Enables bias in the NALU gate",
)
parser.add_argument(
    "--nalu-two-nac",
    action="store_true",
    default=False,
    help="Uses two independent NACs in the NALU Layer",
)
parser.add_argument(
    "--nalu-two-gate",
    action="store_true",
    default=False,
    help="Uses two independent gates in the NALU Layer",
)
parser.add_argument(
    "--nalu-mul",
    action="store",
    default="normal",
    choices=["normal", "safe", "trig", "max-safe", "mnac", "golden-ratio"],
    help="Multplication unit, can be normal, safe, trig",
)
parser.add_argument(
    "--nalu-gate",
    action="store",
    default="normal",
    choices=["normal", "regualized", "obs-gumbel", "gumbel", "golden-ratio"],
    type=str,
    help="Can be normal, regualized, obs-gumbel, or gumbel",
)
parser.add_argument(
    "--nac-weight",
    action="store",
    default="normal",
    choices=["normal", "golden-ratio"],
    type=str,
    help="Way to calculate the NAC+.",
)

parser.add_argument(
    "--optimizer",
    action="store",
    default="adam",
    choices=["adam", "sgd"],
    type=str,
    help="The optimization algorithm to use, Adam or SGD",
)
parser.add_argument(
    "--learning-rate",
    action="store",
    default=1e-3,
    type=float,
    help="Specify the learning-rate",
)
parser.add_argument(
    "--lr-cosine",
    action="store_true",
    default=False,
    help="Use cosine LR decay from --learning-rate to --lr-min over --max-iterations",
)
parser.add_argument(
    "--lr-min",
    action="store",
    default=1e-6,
    type=float,
    help="Minimum learning rate for cosine decay",
)
parser.add_argument(
    "--momentum",
    action="store",
    default=0.0,
    type=float,
    help="Specify the nestrov momentum, only used with SGD",
)

parser.add_argument(
    "--no-cuda",
    action="store_true",
    default=False,
    help=f"Force no CUDA (cuda usage is detected automatically as {torch.cuda.is_available()})",
)
parser.add_argument(
    "--name-prefix",
    action="store",
    default="simple_function_static",
    type=str,
    help="Where the data should be stored",
)
parser.add_argument(
    "--remove-existing-data",
    action="store_true",
    default=False,
    help="Should old results be removed",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    default=False,
    help="Should network measures (e.g. gates) and gradients be shown",
)
parser.add_argument(
    "--reg-scale-type",
    action="store",
    default="heim",
    choices=["heim", "madsen"],
    type=str,
    help="Type of npu regularisation scaling to use. Matches respective author's papers",
)
parser.add_argument(
    "--regualizer-beta-start",
    action="store",
    default=1e-5,
    type=float,
    help="Starting value of the beta scale factor.",
)
parser.add_argument(
    "--regualizer-beta-end",
    action="store",
    default=1e-4,
    type=float,
    help="Final value of the beta scale factor.",
)
parser.add_argument(
    "--regualizer-beta-step",
    action="store",
    default=10000,
    type=int,
    help="Update the regualizer-beta-start value every x steps.",
)
parser.add_argument(
    "--regualizer-beta-growth",
    action="store",
    default=10,
    type=int,
    help="Scale factor to grow the regualizer-beta-start value by.",
)
parser.add_argument(
    "--regualizer-l1",
    action="store_true",
    default=False,
    help="Add L1 regularization loss term. Be sure the regualizer-scaling is set",
)
parser.add_argument(
    "--regualizer-npu-w",
    action="store",
    default=0,
    type=int,
    help="Use sparisty reg on npu weights. Int represents the amount to scale reg by. 0 means off",
)
parser.add_argument(
    "--regualizer-gate",
    type=int,
    default=0,
    help="Use sparisty reg on npu gate. Int represents the amount to scale reg by. 0 means off",
)
parser.add_argument(
    "--npu-clip",
    action="store",
    default="none",
    choices=["none", "w", "g", "wg", "wig"],
    help="Type of parameters (if any) to clip in a NPU/RealNPU module",
)
parser.add_argument(
    "--npu-Wr-init",
    action="store",
    default="xavier-uniform",
    choices=["xavier-uniform", "xavier-uniform-constrained"],
    help="Init method to use for the W_real of the NPU. xavier-uniform= NPU paper init method,"
    "xavier-uniform-constrained= NAU init method",
)

parser.add_argument(
    "--pytorch-precision", type=int, default=32, help="Precision for pytorch to work in"
)

parser.add_argument(
    "--nmu-noise",
    action="store_true",
    default=False,
    help="Applies/ unapplies multiplicative noise from a ~U[1,5] during training. Aids with failure ranges on a vinilla NMU.",
)
parser.add_argument(
    "--nau-noise",
    action="store_true",
    default=False,
    help="Applies/ unapplies additive noise from a ~U[1,5] during training.",
)

parser.add_argument(
    "--no-save",
    action="store_true",
    default=False,
    help="Do not save model at the end of training",
)
parser.add_argument(
    "--load-checkpoint",
    action="store_true",
    default=False,
    help="Loads a saved checkpoint and resumes training",
)
parser.add_argument(
    "--log-interval",
    action="store",
    default=1000,
    type=int,
    help="Log to tensorboard every X epochs.",
)

parser.add_argument(
    "--clip-grad-norm",
    action="store",
    default=None,
    type=float,
    help="Norm clip value for gradients.",
)

parser.add_argument(
    "--nru-div-mode",
    action="store",
    default="div",
    choices=["div", "div-sepSign"],
    help="Division type for NRU. div calcs mag and sign in one go. div-sepSign calcs sign separately",
)
parser.add_argument(
    "--realnpu-reg-type",
    action="store",
    default="W",
    choices=["W", "bias"],
    help="W penalises {-1,1}. bias penalises {-1,0,1}.",
)

parser.add_argument(
    "--clip-grad-value",
    action="store",
    default=None,
    type=float,
    help="Clip value for gradients i.e. [-value, value].",
)
parser.add_argument(
    "--reinit",
    action="store_true",
    default=False,
    help="Enables iNALU's reinitialization scheme",
)
parser.add_argument(
    "--reinit-epoch-interval",
    action="store",
    default=10,
    type=int,
    help="Check after this many epochs if reinitialization can occur.",
)
parser.add_argument(
    "--reinit-max-stored-losses",
    action="store",
    default=5000,
    type=int,
    help="Number of losses that need to be collected before reinitialization can occur.",
)
parser.add_argument(
    "--reinit-loss-thr",
    action="store",
    default=1.0,
    type=float,
    help="Reinitialization only occurs if the avg accumulated loss is greater than this threshold.",
)

parser.add_argument(
    "--num-bins",
    action="store",
    default=5,
    type=int,
    help="Number of bins for |x| binned MSE analysis (default: 5)",
)

parser.add_argument(
    "--note",
    action="store",
    default=None,
    type=str,
    help="Note to add to wandb run name (default: human readable datetime)",
)

parser.add_argument(
    "--no-open-browser",
    action="store_true",
    default=False,
    help="Don't open browser for wandb (useful for automated tests)",
)

parser.add_argument(
    "--max-target-magnitude",
    action="store",
    default=None,
    type=float,
    help="Maximum allowed magnitude for target values (filters out extreme division results)",
)

parser.add_argument(
    "--div-regularizer",
    action="store",
    default=None,
    type=float,
    help="Division regularizer epsilon for x/(x^2 + eps^2) term. If None, no division regularizer is applied.",
)

args = parser.parse_args()

# Initialize wandb with note
note = args.note if args.note else get_default_note()
operation = args.operation
placeholder_name = f"local - {operation}"
if note:
    placeholder_name = f"{placeholder_name} - {note}"
run = wandb.init_wandb(
    local_project="nalm-benchmark",
    placeholder_name=placeholder_name,
    open_browser=not args.no_open_browser,
)

utils.set_pytorch_precision(args.pytorch_precision)
setattr(args, "cuda", torch.cuda.is_available() and not args.no_cuda)

# Print configuration
print(f"running")
print(f"  - layer_type: {args.id}")
print(f"  - layer_type: {args.layer_type}")
print(f"  - first_layer: {args.first_layer}")
print(f"  - operation: {args.operation}")
print(f"  - num_subsets: {args.num_subsets}")
print(f"  - regualizer: {args.regualizer}")
print(f"  - regualizer_z: {args.regualizer_z}")
print(f"  - regualizer_oob: {args.regualizer_oob}")
print(f"  -")
print(f"  - max_iterations: {args.max_iterations}")
print(f"  - batch_size: {args.batch_size}")
print(f"  - seed: {args.seed}")
print(f"  -")
print(f"  - interpolation_range: {args.interpolation_range}")
print(f"  - extrapolation_range: {args.extrapolation_range}")
print(f"  - input_size: {args.input_size}")
print(f"  - output_size: {args.output_size}")
print(f"  - subset_ratio: {args.subset_ratio}")
print(f"  - overlap_ratio: {args.overlap_ratio}")
print(f"  - simple: {args.simple}")
print(f"  -")
print(f"  - hidden_size: {args.hidden_size}")
print(f"  - nac_mul: {args.nac_mul}")
print(f"  - oob_mode: {args.oob_mode}")
print(f"  - regualizer_scaling: {args.regualizer_scaling}")
print(f"  - regualizer_scaling_start: {args.regualizer_scaling_start}")
print(f"  - regualizer_scaling_end: {args.regualizer_scaling_end}")
print(f"  - regualizer_shape: {args.regualizer_shape}")
print(f"  - mnac_epsilon: {args.mnac_epsilon}")
print(f"  - nalu_bias: {args.nalu_bias}")
print(f"  - nalu_two_nac: {args.nalu_two_nac}")
print(f"  - nalu_two_gate: {args.nalu_two_gate}")
print(f"  - nalu_mul: {args.nalu_mul}")
print(f"  - nalu_gate: {args.nalu_gate}")
print(f"  - nac_weight: {args.nac_weight}")
print(f"  -")
print(f"  - optimizer: {args.optimizer}")
print(f"  - learning_rate: {args.learning_rate}")
print(f"  - momentum: {args.momentum}")
print(f"  -")
print(f"  - cuda: {args.cuda}")
print(f"  - name_prefix: {args.name_prefix}")
print(f"  - remove_existing_data: {args.remove_existing_data}")
print(f"  - verbose: {args.verbose}")
print(f"  -")
print(f"  - reg_scale_type: {args.reg_scale_type}")
print(f"  - regualizer_beta_start: {args.regualizer_beta_start}")
print(f"  - regualizer_beta_end: {args.regualizer_beta_end}")
print(f"  - regualizer_beta_step: {args.regualizer_beta_step}")
print(f"  - regualizer_beta_growth: {args.regualizer_beta_growth}")
print(f"  - regualizer_l1: {args.regualizer_l1}")
print(f"  - regualizer-npu-w: {args.regualizer_npu_w}")
print(f"  - regualizer-gate: {args.regualizer_gate}")
print(f"  - npu-clip: {args.npu_clip}")
print(f"  - npu-Wr-init: {args.npu_Wr_init}")
print(f"  -")
print(f"  - pytorch-precision: {torch.get_default_dtype()}")
print(f"  -")
print(f"  - no-save: {args.no_save}")
print(f"  - load-checkpoint: {args.load_checkpoint}")
print(f"  - log-interval: {args.log_interval}")
print(f"  -")
print(f"  - clip-grad-norm: {args.clip_grad_norm}")
print(f"  - nru_div_mode: {args.nru_div_mode}")
print(f"  - realnpu_reg_type: {args.realnpu_reg_type}")
print(f"  -")
print(f"  - reinit: {args.reinit}")
print(f"  - reinit_epoch_interval: {args.reinit_epoch_interval}")
print(f"  - reinit_max_stored_losses: {args.reinit_max_stored_losses}")
print(f"  - reinit_loss_thr: {args.reinit_loss_thr}")
print(f"  - num_bins: {args.num_bins}")
print(f"  -")


def get_npu_Wr_init_writer_value():
    if args.npu_Wr_init == "xavier-uniform":
        return "xu"
    elif args.npu_Wr_init == "xavier-uniform-constrained":
        return "xuc"
    else:
        raise ValueError(f"Invalid arg ({args.npu_Wr_init}) given for npu_Wr_init")


summary_writer = stable_nalu.writer.DummySummaryWriter()

# Set threads
if "LSB_DJOB_NUMPROC" in os.environ:
    torch.set_num_threads(int(os.environ["LSB_DJOB_NUMPROC"]))


# Set seed
def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

# set epsilon for numerical stability
eps = torch.finfo().eps

# Setup datasets
dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
    operation=args.operation,
    input_size=args.input_size,
    subset_ratio=args.subset_ratio,
    overlap_ratio=args.overlap_ratio,
    num_subsets=args.num_subsets,
    simple=args.simple,
    use_cuda=args.cuda,
    seed=args.seed,
    max_result_magnitude=args.max_target_magnitude,
)
print(f"  -")
print(f"  - dataset: {dataset.print_operation()}")
# Interpolation and extrapolation seeds are from random.org
dataset_train = iter(
    dataset.fork(sample_range=args.interpolation_range).dataloader(
        batch_size=args.batch_size
    )
)
dataset_valid_interpolation_data = next(
    iter(
        dataset.fork(sample_range=args.interpolation_range, seed=43953907).dataloader(
            batch_size=10000
        )
    )
)
dataset_test_extrapolation_data = next(
    iter(
        dataset.fork(sample_range=args.extrapolation_range, seed=8689336).dataloader(
            batch_size=10000
        )
    )
)

# setup model
model = stable_nalu.network.SingleLayerNetwork(
    args.layer_type,
    input_size=dataset.get_input_size(),
    output_size=args.output_size,
    writer=summary_writer.every(args.log_interval).verbose(args.verbose),
    first_layer=args.first_layer,
    hidden_size=args.hidden_size,
    nac_oob=args.oob_mode,
    regualizer_shape=args.regualizer_shape,
    regualizer_z=args.regualizer_z,
    mnac_epsilon=args.mnac_epsilon,
    nac_mul=args.nac_mul,
    nalu_bias=args.nalu_bias,
    nalu_two_nac=args.nalu_two_nac,
    nalu_two_gate=args.nalu_two_gate,
    nalu_mul=args.nalu_mul,
    nalu_gate=args.nalu_gate,
    nac_weight=args.nac_weight,
    regualizer_gate=args.regualizer_gate,
    regualizer_npu_w=args.regualizer_npu_w,
    npu_clip=args.npu_clip,
    npu_Wr_init=args.npu_Wr_init,
    nru_div_mode=args.nru_div_mode,
    realnpu_reg_type=args.realnpu_reg_type,
    dag_depth=args.num_subsets + 1,
    # dag_depth=1,
)
model.reset_parameters()
if args.cuda:
    model.cuda()
criterion = torch.nn.MSELoss()

# Build optimizer
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )
else:
    raise ValueError(f"{args.optimizer} is not a valid optimizer algorithm")

# Optional cosine LR schedule over training duration
if args.lr_cosine:
    eta_min = max(0.0, float(args.lr_min))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.max_iterations), eta_min=eta_min
    )
else:
    scheduler = NoOpScheduler(optimizer)


# Compute bins for |x|
def compute_bins(x, num_bins: int = 5):
    x_abs = x.abs().min(dim=1).values
    # Use quantiles to ensure roughly equal sample counts per bin
    quantiles = torch.quantile(x_abs, torch.linspace(0, 1, num_bins + 1))
    bin_edges = quantiles

    # Assign samples to bins based on quantiles
    bin_indices = torch.zeros_like(x_abs, dtype=torch.long)
    for i in range(num_bins):
        if i == 0:
            mask = x_abs <= bin_edges[i + 1]
        elif i == num_bins - 1:
            mask = x_abs > bin_edges[i]
        else:
            mask = (x_abs > bin_edges[i]) & (x_abs <= bin_edges[i + 1])
        bin_indices[mask] = i

    print("=" * 60)
    print("📊 BINNED MSE ANALYSIS SETUP")
    print("=" * 60)
    print(f"📈 Input tensor shape: {x.shape}")
    print(f"🔢 Number of bins: {num_bins}")
    print(f"📏 min|x| range: [{x_abs.min().item():.6f}, {x_abs.max().item():.6f}]")
    print()
    print("📋 Bin boundaries (quantile-based):")
    for i in range(num_bins):
        if i == 0:
            print(f"   Bin {i}: min|x| ≤ {bin_edges[i + 1]:.6f}")
        elif i == num_bins - 1:
            print(f"   Bin {i}: min|x| > {bin_edges[i]:.6f}")
        else:
            print(f"   Bin {i}: {bin_edges[i]:.6f} < min|x| ≤ {bin_edges[i + 1]:.6f}")
    print()
    print("🎯 Sample distribution per bin:")
    for i in range(num_bins):
        count = (bin_indices == i).sum()
        print(f"   Bin {i}: {count} samples")
    print("=" * 60)

    return bin_indices


def compute_binned_mse(data, bin_indices, name: str) -> dict:
    """
    Compute MSE for different log-spaced bins of |x| values.

    Args:
        data: Tuple of (x, t) where x is input tensor and t is target tensor
        num_bins: Number of log-spaced bins to create

    Returns:
        Dictionary with bin counts and MSE per bin
    """
    x, t = data

    # Compute MSE per bin
    total_bins = bin_indices.max() + 1
    result = {}
    total_mse = 0
    total_samples = 0
    for bin_idx in range(total_bins):
        bin_samples = bin_indices == bin_idx
        count = bin_samples.sum()
        if count > 0:
            bin_data = (x[bin_samples], t[bin_samples])
            bin_mse = test_model(bin_data)
            bin_mse_scalar = float(bin_mse.detach().cpu().item())
            result[f"mse/{name}/bin_{bin_idx}"] = bin_mse_scalar
            total_mse += bin_mse_scalar * count
            total_samples += count
        else:
            result[f"mse/{name}/bin_{bin_idx}"] = float("nan")

    # Compute weighted average MSE across all bins
    interpolation_error = (
        total_mse / total_samples if total_samples > 0 else float("nan")
    )

    return result, interpolation_error


def test_model(data):
    with torch.no_grad(), model.no_internal_logging(), model.no_random():
        model.eval()
        x, t = data
        err = criterion(model(x), t)
        model.train()
        return err


# Train model
print(model)
print()
# only print inits of small models
# utils.print_model_params(model) if args.input_size <= 10 else None
print()

use_npu_scaling = (
    args.regualizer_l1
    or (args.regualizer_npu_w and args.reg_scale_type == "heim")
    or (args.regualizer_gate and args.reg_scale_type == "heim")
)
if use_npu_scaling:
    # Decimal type required to avoid accumulation of fp precision errors when multiplying by growth factor
    args.regualizer_beta_start = Decimal(str(args.regualizer_beta_start))
    # Decimal and fp arithmetic don't mix so beta end must also be a decimal
    args.regualizer_beta_end = Decimal(str(args.regualizer_beta_end))
r_l1_scale = args.regualizer_beta_start

"""Resuming previous training"""
resume_epoch = 0
if args.load_checkpoint:
    resume_epoch = stable_nalu.writer.load_model("no-tb-writer", model, optimizer)
    if resume_epoch > args.max_iterations:
        raise ValueError(
            f"{args.max_iterations} must be larger than or equal to the loaded models resume epoch {resume_epoch}"
        )
    if resume_epoch != 0:
        for i, j in zip(range(resume_epoch), dataset_train):
            (x_train, t_train) = j
    print("Checkpoint loaded")
    print(
        "train %d: %.5f, inter: %.5f, extra: %.5f"
        % (
            resume_epoch,
            test_model((x_train, t_train)),
            test_model(dataset_valid_interpolation_data),
            test_model(dataset_test_extrapolation_data),
        )
    )
"""------------------"""
if args.reinit:
    epoch_losses = []
    reinit_counter = 0

patience_counter = 0
early_stop = False

bin_indices = compute_bins(dataset_valid_interpolation_data[0])
for epoch_i, (x_train, t_train) in zip(
    range(resume_epoch, args.max_iterations + 1), dataset_train
):
    tap_context.set_epoch_i(epoch_i)
    summary_writer.set_iteration(epoch_i)

    # Prepear model
    # model.set_parameter("tau", max(0.5, math.exp(-1e-5 * epoch_i)))
    optimizer.zero_grad()
    log_dict = {}

    # Compute binned MSE for interpolation data (also returns total interpolation error)
    binned_mse, interpolation_error = compute_binned_mse(
        dataset_valid_interpolation_data, bin_indices, "inter"
    )
    log_dict.update(binned_mse)

    _es_thr = 1e-10
    PATIENCE = 3
    if interpolation_error < _es_thr:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            extrapolation_error = test_model(dataset_test_extrapolation_data)
            print(
                f"Early stopping at step {epoch_i}: inter={interpolation_error:.10f}, extra={extrapolation_error.detach().cpu().item():.10f}"
            )
            log_dict["early_stopped"] = 1
            early_stop = True
    else:
        patience_counter = 0

    # forward
    y_train = model(x_train)

    regualizers = model.regualizer()
    if args.regualizer_scaling == "linear":
        r_w_scale = max(
            0,
            min(
                1,
                (
                    (epoch_i - args.regualizer_scaling_start)
                    / (args.regualizer_scaling_end - args.regualizer_scaling_start)
                ),
            ),
        )
    elif args.regualizer_scaling == "exp":
        r_w_scale = 1 - math.exp(-1e-5 * epoch_i)

    l1_loss = 0
    if args.regualizer_l1:
        l1_loss = Regualizer.l1(model.parameters())

    if use_npu_scaling:
        # the beta_start value will be updated accordingly to be the correct beta value for the epoch.
        # It is done this way to avoid having initialise another variable outside the epoch loop
        if args.regualizer_beta_start <= args.regualizer_beta_end:
            if epoch_i % args.regualizer_beta_step == 0 and epoch_i != 0:
                if args.regualizer_beta_start < args.regualizer_beta_end:
                    args.regualizer_beta_start *= args.regualizer_beta_growth
        else:
            if epoch_i % args.regualizer_beta_step == 0 and epoch_i != 0:
                if args.regualizer_beta_start > args.regualizer_beta_end:
                    args.regualizer_beta_start /= args.regualizer_beta_growth

        r_l1_scale = float(
            args.regualizer_beta_start
        )  # Decimal doesn't work for tensorboard or mixed fp arithmetic

    # mse loss
    loss_train_criterion = criterion(y_train, t_train)
    loss_train_regualizer = (
        args.regualizer * r_w_scale * regualizers["W"]
        + regualizers["g"]
        + args.regualizer_z * regualizers["z"]
        + args.regualizer_oob * regualizers["W-OOB"]
        + args.regualizer_l1 * r_l1_scale * l1_loss
        + args.regualizer_npu_w
        * (r_l1_scale if args.reg_scale_type == "heim" else r_w_scale)
        * regualizers["W-NPU"]
        + args.regualizer_gate
        * (r_l1_scale if args.reg_scale_type == "heim" else r_w_scale)
        * regualizers["g-NPU"]
        + (
            (0.05 * regualizers["inalu"])
            if (interpolation_error < 1 and epoch_i > 10000)
            else 0
        )
    )

    loss_train = loss_train_criterion + loss_train_regualizer
    dag = next((m for m in model.modules() if isinstance(m, DAGLayer)), None)

    log_dict["loss/train"] = float(loss_train_criterion.detach().cpu().item())
    log_dict["mse/inter"] = float(interpolation_error)

    # Log training batch target statistics
    log_dict["targets/train_mean"] = float(t_train.mean().detach().cpu().item())
    log_dict["targets/train_max"] = float(t_train.abs().max().detach().cpu().item())

    log_dict["lr"] = float(scheduler.get_last_lr()[0])
    if dag is not None and hasattr(dag, "_last_train_G"):
        log_dict["mean/G"] = float(dag._last_train_G.cpu().mean().item())
        o_l1_mean = float(dag._last_train_O.cpu().abs().sum(dim=-1).mean().item())
        log_dict["mean/O"] = o_l1_mean

    commit = False
    if epoch_i % args.log_interval == 0 or early_stop:
        commit = True
        extrapolation_error = test_model(dataset_test_extrapolation_data)
        log_dict["mse/extra"] = float(extrapolation_error.detach().cpu().item())

        print(
            "train %d: %.10f, inter: %.10f, extra: %.10f"
            % (
                epoch_i,
                loss_train_criterion.detach().cpu().item(),
                interpolation_error,
                extrapolation_error.detach().cpu().item(),
            )
        )

        # Print extrapolation example after running through model to get internal state
        if dag is not None:
            # Get a random extrapolation example
            x_extra, t_extra = dataset_test_extrapolation_data
            random_idx = epoch_i % x_extra.size(
                0
            )  # Use epoch for reproducible "randomness"
            x_extra_sample = x_extra[random_idx : random_idx + 1]
            t_extra_sample = t_extra[random_idx : random_idx + 1]

            # Run model on extrapolation sample to get internal state
            with torch.no_grad(), model.no_internal_logging(), model.no_random():
                model.eval()
                y_extra_sample = model(x_extra_sample)
                model.train()

            # Print extrapolation example statistics
            print_dag_internal_state(
                x_tensor=x_extra_sample,
                y_tensor=y_extra_sample,
                t_tensor=t_extra_sample,
                g_tensor=dag._last_eval_G,
                o_tensor=dag._last_eval_O,
                out_logits_tensor=dag._last_eval_out_logits,
                value_vec_inter_tensor=dag._last_eval_value_vec_inter,
                state_label="HARDENED eval",
                dag_layer=dag,
            )
            # Inspect gate/selection on the first sample (concise)
            print_dag_internal_state(
                x_tensor=x_train,
                y_tensor=y_train,
                t_tensor=t_train,
                g_tensor=dag._last_train_G,
                o_tensor=dag._last_train_O,
                out_logits_tensor=dag._last_train_out_logits,
                value_vec_inter_tensor=dag._last_train_value_vec_inter,
                state_label="SOFT training",
                dag_layer=dag,
            )

    if early_stop:
        print(f"Early stopped at step {epoch_i}")
        wandb.wrapper.log(log_dict, step=epoch_i)
        break

    # Optimize model
    if loss_train.requires_grad:
        loss_train.backward()

        clip_thresh = (
            float("inf") if args.clip_grad_norm == None else args.clip_grad_norm
        )
        try:
            # Log gradient norms before clipping
            norm_before_clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_thresh, error_if_nonfinite=True
            ).item()
            log_dict["gradients/norm_before_clip"] = float(norm_before_clip)

            # Apply gradient clipping
            if args.clip_grad_norm != None:
                # Verify clipping worked: compute actual norm after clipping
                norm_after_clip = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf"), error_if_nonfinite=True
                ).item()

                log_dict["gradients/norm_after_clip"] = float(norm_after_clip)

                log_dict["gradients/clipping_ratio"] = float(
                    norm_after_clip / norm_before_clip
                )
            else:
                log_dict["gradients/clipping_ratio"] = 1.0
        except:
            import pdb

            pdb.set_trace()
            raise

        if args.clip_grad_value != None:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)

        optimizer.step()
        scheduler.step()
    model.optimize(loss_train_criterion)

    # Log gradients if in verbose mode
    if args.verbose and epoch_i % args.log_interval == 0:
        model.log_gradients()
        # model.log_gradient_elems()

    """
    inalu reinit conditions:
    - every 10th epoch (and not the first epoch) where the number of stored errors is over 5,000.
    - if the average err value of the first half of the errors is less than the 2nd half + sdev and the avg loss of the
    latter half is larger than 1
    """
    if args.reinit:
        epoch_losses.append(interpolation_error)

        if (
            epoch_i % args.reinit_epoch_interval == 0
            and epoch_i > 0
            and len(epoch_losses) > args.reinit_max_stored_losses
        ):
            losses_last_half = epoch_losses[len(epoch_losses) // 2 :]
            if np.mean(epoch_losses[0 : len(epoch_losses) // 2]) <= (
                np.mean(losses_last_half) + np.std(losses_last_half)
            ) and (np.mean(losses_last_half) > args.reinit_loss_thr):
                model.reset_parameters()
                print(f"reinit number {reinit_counter}")
                epoch_losses = []
                reinit_counter += 1

    # Log to W&B with step number
    wandb.wrapper.log(log_dict, step=epoch_i, commit=commit)

# Compute validation loss
loss_valid_inter = test_model(dataset_valid_interpolation_data)
loss_valid_extra = test_model(dataset_test_extrapolation_data)
loss_train_capped = test_model([x_train, t_train])

# Write results for this training
print(f"finished:")
if args.reinit:
    print(f"Reinitialized {reinit_counter} times")

print(f"  - loss_train_capped: {loss_train_capped}")
print(f"  - loss_train (+reg loss): {loss_train}")
print(f"  - loss_train_criterion: {loss_train_criterion}")
print(f"  - loss_valid_inter: {loss_valid_inter}")
print(f"  - loss_valid_extra: {loss_valid_extra}")
print()
# Skip printing model params for now
# utils.print_model_params(model)

# Play completion sound on macOS
import os
import platform

if platform.system() == "Darwin":  # macOS
    try:
        if early_stop:
            os.system("afplay /System/Library/Sounds/Funk.aiff")
        else:
            os.system("afplay /System/Library/Sounds/Submarine.aiff")
    except:
        pass  # Silently fail if sound doesn't work

if not args.no_save:
    model.writer._root.close()  # fix - close summary writer before saving model to avoid thread locking issues
    # Use saved weights to visualize the intermediate values.
    stable_nalu.writer.save_model_checkpoint(
        summary_writer.name,
        epoch_i + 1,
        model,
        optimizer,
        {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
    )

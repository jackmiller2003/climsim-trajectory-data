"""
Script for processing ClimSim data, converting into chunks which are easily used by a PyTorch Dataset.
"""

import numpy as np
import xarray as xr
import argparse
from pathlib import Path
import torch
import json
from tqdm import tqdm


if __name__ == "__main__":
    # Create an argparse to get location of data
    parser = argparse.ArgumentParser(description="Process ClimSim data")

    # /g/data/x77/jm0124/ClimSim_data/ClimSim_low_res/cache/datasets--LEAP--ClimSim_low-res/snapshots/bab82a2ebdc750a0134ddcd0d5813867b92eed2a
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Path to the normalisation meta data folder (provided with this repository).",
        default="meta_data/normalisations",
    )

    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path specifying where to save the normalisation array.",
    )

    parser.add_argument(
        "--save_as_torch_tensor",
        action="store_true",
        help="Save as torch tensor (otherwise npy)",
        default=False,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    parser.add_argument(
        "--variable_selection",
        type=str,
        default="variable_selections/default.json",
        help="Which variables to complete predictions with",
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)

    mean_dataset = xr.open_dataset(data_path / "inputs" / "input_mean.nc")
    std_dataset = xr.open_dataset(data_path / "inputs" / "input_std.nc")

    # Load the variable selection
    with open(args.variable_selection, "r") as f:
        variable_selection = json.load(f)

    list_of_vars_with_mean_and_std = []

    for name, levels in variable_selection.items():
        if name not in mean_dataset:
            raise ValueError(f"Variable {name} not in mean dataset")
        if name not in std_dataset:
            raise ValueError(f"Variable {name} not in std dataset")

        mean = mean_dataset[name]
        std = std_dataset[name]

        for level in levels:
            list_of_vars_with_mean_and_std.append(
                (mean[level].values, std[level].values)
            )

    numpy_array = np.array(list_of_vars_with_mean_and_std)

    if args.save_as_torch_tensor:
        torch.save(
            torch.tensor(numpy_array, dtype=torch.float32),
            f"{output_path.absolute()}.pt",
        )
    else:
        np.save(f"{output_path.absolute()}.npy", numpy_array)

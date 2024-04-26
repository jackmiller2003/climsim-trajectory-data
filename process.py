"""
Script for processing ClimSim data, converting into chunks which are easily used by a PyTorch Dataset.
"""

import numpy as np
import netCDF4 as nc
import xarray as xr
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import gc
import json

from src.common import repo_root

DICT_OF_DAYS_IN_MONTH = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}

FLATTENED_SIZE = 617
N_LEVELS = 60
N_COLS = 384

# These are fixed training/val/test splits.
START_OF_TRAIN_SPLIT = "0001-02"
END_OF_TRAIN_SPLIT = "0006-12"
START_OF_VAL_SPLIT = "0007-02"
END_OF_VAL_SPLIT = "0007-12"
START_OF_TEST_SPLIT = "0008-02"
END_OF_TEST_SPLIT = "0008-12"

# Typically 1 file is 1.9 MB (we will round to 2 MB)
MB_SIZE_OF_FILE = 2
TOTAL_NUM_OF_INDICES_WITHOUT_SUB_SELECTION = 556


def get_ordered_file_list_from_date_range(
    date_range: tuple[str, str], folder_path: Path, verbose: bool = True
) -> list[Path]:
    """
    Returns oredered file list of netcdf snapshots from date range.
    """

    start_date, end_date = date_range
    start_year = int(start_date[:4])
    start_month = int(start_date[5:7])
    end_year = int(end_date[:4])
    end_month = int(end_date[5:7])

    file_list = []

    if verbose:
        print(f"Start year: {start_year}, Start month: {start_month}")
        print(f"End year: {end_year}, End month: {end_month}")

    for year in range(start_year, end_year + 1):
        if year == start_year:
            start_month = start_month
        else:
            start_month = 1

        if year == end_year:
            end_month = end_month
        else:
            end_month = 12

        for month in range(start_month, end_month + 1):
            folder_str = f"{year:04}-{month:02}"

            # Get all relevant files in folder
            folder = folder_path / folder_str

            if folder.exists():
                for file in folder.iterdir():

                    # We only take input files for the predictions
                    if file.name.startswith("E3SM-MMF.mli."):
                        path_to_file = folder / file
                        file_list.append(path_to_file)
            else:
                print(f"Folder {folder} does not exist")

    # Order the list by E3SM-MMF.mli.(0001-02-23-32400).nc

    file_list = list(sorted(file_list, key=lambda x: file_name_to_time(x.name)[1]))

    # Assert each file is exactly 20 minutes apart
    for i, file in enumerate(file_list):
        if i != 0:
            current_interp_dict, current_time = file_name_to_time(file.name)
            prev_interp_dict, prev_time = file_name_to_time(file_list[i - 1].name)

            time_diff = current_time - prev_time

            assert (
                time_diff == 20 * 60
            ), f"Time difference between {file_list[i].name} and {file_list[i-1].name} {time_diff}.\nPrevious interp dict:{prev_interp_dict}.\nCurrent interp dict: {current_interp_dict}"

    return file_list


def file_name_to_time(file_name: str) -> tuple[dict[str, int], int]:
    """
    Get the time from the file name.
    """
    date_of_file = file_name.split(".")[-2]
    date_split = date_of_file.split("-")
    year_of_file = int(date_split[0])
    month_of_file = int(date_split[1])
    day_of_file = int(date_split[2])
    seconds_of_day = int(date_split[3])

    dict_of_interpreted_times = {
        "year": year_of_file,
        "month": month_of_file,
        "day": day_of_file,
        "seconds": seconds_of_day,
    }

    SECONDS_IN_A_DAY = 24 * 60 * 60

    days_in_this_month = DICT_OF_DAYS_IN_MONTH[int(month_of_file)]

    assert int(day_of_file) <= days_in_this_month, f"Day {day_of_file} is too large"

    # Number of seconds in the years up to the point
    total_time = (int(year_of_file) - 1) * 365 * SECONDS_IN_A_DAY

    # Number of seconds in the months of the current year
    for month in range(1, int(month_of_file)):
        total_time += DICT_OF_DAYS_IN_MONTH[month] * SECONDS_IN_A_DAY

    # Number of seconds in the days of the current month
    total_time += (int(day_of_file) - 1) * SECONDS_IN_A_DAY

    # Number of seconds in the current day
    total_time += int(seconds_of_day)

    return dict_of_interpreted_times, total_time


def convert_json_to_indices(
    variable_selection_file: Path = Path("variable_selections/default.json"),
    all_variables: Path = Path("variable_selections/all_variables.json"),
    all_variables_with_num_of_indices: Path = Path(
        "variable_selections/all_variables_with_num_of_indices.json"
    ),
) -> list[int]:
    """
    Convert a json file to a list of indices.
    """

    with open(variable_selection_file, "r") as f:
        variable_selection_dict = json.load(f)

    with open(all_variables, "r") as f:
        all_variables_list = json.load(f)

    with open(all_variables_with_num_of_indices, "r") as f:
        all_variables_with_num_of_indices_list = json.load(f)

    # Assert variable selection is subset of all variables
    assert set(variable_selection_dict.keys()).issubset(set(all_variables_list))

    # Assert that the value of the maximal indices is less than the number of indices
    for variable, indices in variable_selection_dict.items():
        assert max(indices) < all_variables_with_num_of_indices_list[variable]

    return variable_selection_dict


def save_arrays_from_file_list(
    list_of_files: list[Path],
    dict_of_variables_and_indices_to_save: dict[str, list[int]],
    total_num_of_indices: int,
    path_to_save: Path,
    n_cdfs_per_chunk: int,
    save_as_torch: bool = False,
) -> None:

    for file_num, file in enumerate(list_of_files):
        netcdf_data = xr.open_dataset(file)

        array_shape = (1, total_num_of_indices, N_COLS)
        individual_array = np.zeros(array_shape)

        starting_index = 0

        # Loop through each variable in the dataset
        for name, variable in sorted(
            list(netcdf_data.variables.items()), key=lambda x: x[0]
        ):

            if name not in dict_of_variables_and_indices_to_save:
                continue

            indices_to_extract = dict_of_variables_and_indices_to_save[name]

            variable_values = variable[indices_to_extract].values

            individual_array[
                starting_index : starting_index + len(indices_to_extract)
            ] = variable_values

        index_of_chunked_array = file_num % n_cdfs_per_chunk

        if index_of_chunked_array == 0:
            current_arrays = []
            starting_time_of_chunk = file_name_to_time(file.name)[1]

        current_arrays.append(individual_array)

        if index_of_chunked_array == n_cdfs_per_chunk - 1:

            # Concatenate along first dimension
            concatenated_array = np.concatenate(current_arrays, axis=0)
            ending_time_of_chunk = file_name_to_time(file.name)[1]

            name_of_array = f"{starting_time_of_chunk}_{ending_time_of_chunk}"

            if save_as_torch:
                torch_array = torch.tensor(concatenated_array, dtype=torch.float32)

                torch.save(
                    torch_array,
                    path_to_save / f"{name_of_array}.pt",
                )

                del torch_array
            else:
                np.save(
                    path_to_save / f"{name_of_array}.npy",
                    concatenated_array,
                )

            del concatenated_array

        # Just in case the arrays stay on the heap
        if file_num % 1000 == 0:
            gc.collect()


if __name__ == "__main__":
    # Create an argparse to get location of data
    parser = argparse.ArgumentParser(description="Process ClimSim data")

    # /g/data/x77/jm0124/ClimSim_data/ClimSim_low_res/cache/datasets--LEAP--ClimSim_low-res/snapshots/bab82a2ebdc750a0134ddcd0d5813867b92eed2a
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Path to the data, this should be the custom cache directory specified in download.py i.e. snapshots/[stuff] with the folders of months and years in it.",
    )

    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path specifying where to save the processed torch arrays.",
    )

    parser.add_argument(
        "--mb_per_chunk", type=int, help="Size of chunks in MBs", default=5000
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

    output_path = Path(args.output_path)

    train_file_list = get_ordered_file_list_from_date_range(
        (START_OF_TRAIN_SPLIT, END_OF_TRAIN_SPLIT),
        args.data_path / "train",
        args.verbose,
    )

    val_file_list = get_ordered_file_list_from_date_range(
        (START_OF_VAL_SPLIT, END_OF_VAL_SPLIT), args.data_path / "train", args.verbose
    )

    test_file_list = get_ordered_file_list_from_date_range(
        (START_OF_TEST_SPLIT, END_OF_TEST_SPLIT), args.data_path / "train", args.verbose
    )

    length_of_list = len(train_file_list)

    if args.verbose:
        print(f"Train file list length: {len(train_file_list)}")
        print(f"Val file list length: {len(val_file_list)}")
        print(f"Test file list length: {len(test_file_list)}")

    data_split_file_list_to_proc = [train_file_list, val_file_list, test_file_list]

    dict_of_variables_and_indices_to_save = convert_json_to_indices(
        args.variable_selection
    )

    # Get total number of indices
    total_num_of_indices = sum(
        [len(indices) for indices in dict_of_variables_and_indices_to_save.values()]
    )

    n_cdfs_per_chunk = args.mb_per_chunk / (
        MB_SIZE_OF_FILE
        * (total_num_of_indices / TOTAL_NUM_OF_INDICES_WITHOUT_SUB_SELECTION)
    )

    for file_list in data_split_file_list_to_proc:
        name_of_partition = (
            "train"
            if file_list == train_file_list
            else "val" if file_list == val_file_list else "test"
        )

        path_to_save = output_path / name_of_partition

        if not path_to_save.exists():
            path_to_save.mkdir(parents=True)

        save_arrays_from_file_list(
            list_of_files=file_list,
            dict_of_variables_and_indices_to_save=dict_of_variables_and_indices_to_save,
            total_num_of_indices=total_num_of_indices,
            path_to_save=path_to_save,
            n_cdfs_per_chunk=n_cdfs_per_chunk,
            save_as_torch_tensor=args.save_as_torch_tensor,
        )

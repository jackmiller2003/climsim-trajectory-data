"""
Script for processing ClimSim data, converting into chunks which are easily used by a PyTorch Dataset.
"""

import numpy as np
import netCDF4 as nc
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import gc

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


def get_ordered_file_list_from_date_range(
    date_range: tuple[str, str], folder_path: Path
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

    print(f"Start year: {start_year}, Start month: {start_month}")
    print(f"End year: {end_year}, End month: {end_month}")

    for year in range(start_year, end_year + 1):
        if year == start_year:
            start_month = start_month
        else:
            start_month = 1
        for month in range(start_month, end_month + 1):
            folder_str = f"{year:04}-{month:02}"

            # Get all relevant files in folder
            folder = folder_path / folder_str

            if folder.exists():
                for file in folder.iterdir():
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

    total_time = 0

    total_time += (int(year_of_file) - 1) * 365 * SECONDS_IN_A_DAY

    for month in range(1, int(month_of_file)):
        total_time += DICT_OF_DAYS_IN_MONTH[month] * SECONDS_IN_A_DAY

    total_time += (int(day_of_file) - 1) * SECONDS_IN_A_DAY
    total_time += int(seconds_of_day)

    return dict_of_interpreted_times, total_time


if __name__ == "__main__":

    # Some parameters

    START_OF_TRAIN_SPLIT = "0001-02"
    END_OF_TRAIN_SPLIT = "0006-12"
    START_OF_VAL_SPLIT = "0007-02"
    END_OF_VAL_SPLIT = "0007-12"
    START_OF_TEST_SPLIT = "0008-02"
    END_OF_TEST_SPLIT = "0008-12"

    CHUNK_SIZE = 1000

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

    raise NotImplementedError("This script is not yet to be modified...")

    args = parser.parse_args()

    train_file_list = get_ordered_file_list_from_date_range(
        (START_OF_TRAIN_SPLIT, END_OF_TRAIN_SPLIT), args.data_path / "train"
    )

    val_file_list = get_ordered_file_list_from_date_range(
        (START_OF_VAL_SPLIT, END_OF_VAL_SPLIT), args.data_path / "train"
    )

    test_file_list = get_ordered_file_list_from_date_range(
        (START_OF_TEST_SPLIT, END_OF_TEST_SPLIT), args.data_path / "train"
    )

    length_of_list = len(train_file_list)

    print(f"Train file list length: {len(train_file_list)}")
    print(f"Val file list length: {len(val_file_list)}")
    print(f"Test file list length: {len(test_file_list)}")

    data_split_file_list_to_proc = []

    if args.data_split == "train":
        data_split_file_list_to_proc = [train_file_list]
    elif args.data_split == "val":
        data_split_file_list_to_proc = [val_file_list]
    elif args.data_split == "test":
        data_split_file_list_to_proc = [test_file_list]
    else:
        data_split_file_list_to_proc = [train_file_list, val_file_list, test_file_list]

    if args.indices_to_save is not None:
        indices_to_save = [int(index) for index in args.indices_to_save]
        print(f"Saving indices: {indices_to_save}")
    else:
        print("Saving all indices")

    for file_list in data_split_file_list_to_proc:
        variables_dim_size = (
            len(indices_to_save) if args.indices_to_save is not None else FLATTENED_SIZE
        )

        name_of_partition = (
            "train"
            if file_list == train_file_list
            else "val" if file_list == val_file_list else "test"
        )

        start_index = args.index_range[0]
        end_index = min(args.index_range[1], len(file_list))

        first_dim_of_chunked_array = min(args.chunk_size, end_index - start_index)

        chunked_array = np.zeros(
            (first_dim_of_chunked_array, variables_dim_size, N_COLS)
        )

        print(
            f"Processing {args.data_split} files. Setting chunked array to size: {chunked_array.shape}"
        )

        latest_ending_i = start_index

        for i in tqdm(
            range(args.index_range[0], end_index),
            desc="Processing train files",
        ):

            train_file = file_list[i]

            snapshot_data = nc.Dataset(train_file, "r")

            tensors = []

            # Loop through each variable in the dataset
            for name, variable in snapshot_data.variables.items():

                if i == 0:
                    print(f"Variable name: {name}, shape: {variable.shape}")

                if variable.shape == (N_COLS,):
                    # Reshape variable to be (1, N_COLS)
                    reshaped_variable = np.reshape(variable, (1, N_COLS))
                    tensors.append(reshaped_variable)
                elif variable.shape == (N_LEVELS, N_COLS):
                    tensors.append(variable)

            # Concatenate tensors along first dimension
            concatenated_tensor = np.concatenate(tensors, axis=0)

            index_of_chunked_array = i % args.chunk_size

            if args.indices_to_save is not None:
                indices_to_save = [int(index) for index in args.indices_to_save]

                concatenated_tensor = concatenated_tensor[indices_to_save, :]

            chunked_array[index_of_chunked_array] = concatenated_tensor

            if index_of_chunked_array == args.chunk_size - 1 or i == end_index - 1:

                if args.save_as_torch_tensor:
                    torch.save(
                        torch.tensor(chunked_array, dtype=torch.float32),
                        args.output_path
                        / f"{name_of_partition}_chunk_{latest_ending_i}_{i}.pt",
                    )
                else:
                    np.save(
                        args.output_path
                        / f"{name_of_partition}_chunk_{latest_ending_i}_{i}.npy",
                        chunked_array,
                    )

                chunked_array = np.zeros((args.chunk_size, variables_dim_size, N_COLS))
                latest_ending_i = i + 1

            # delete the snapshot and concatenated tensor to free up memory
            snapshot_data.close()
            del concatenated_tensor

            if i % 1000 == 0:
                gc.collect()

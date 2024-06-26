import datasets
from pathlib import Path
from huggingface_hub import snapshot_download
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_year_and_month(
    year: str, month: str, data_split: str, custom_cache_dir: Path, verbose: bool = True
) -> Path | None:
    datasets.config.DOWNLOADED_DATASETS_PATH = custom_cache_dir

    if verbose:
        print(f"Downloading data for {year}-{month} to {custom_cache_dir}")

    # Download the dataset from the Hugging Face Hub
    download_path = snapshot_download(
        repo_id="LEAP/ClimSim_low-res",
        allow_patterns=f"{data_split}/{year}-{month}/*",
        cache_dir=custom_cache_dir,
        repo_type="dataset",
        etag_timeout=120,
        resume_download=True,
    )

    # If fails return None
    if download_path is None:
        return None

    return Path(download_path)


def download_in_parallel(
    start_year: int,
    end_year: int,
    start_month: int = 1,
    end_month: int = 12,
    num_workers: int = 12,
    data_split: str = "train",
    custom_cache_dir: Path = Path("data"),
    verbose: bool = True,
) -> None:
    months = list(range(1, 13))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create a list to store future results
        futures = []
        for year in range(start_year, end_year + 1):

            if year == start_year:
                months = months[start_month - 1 :]

            if year == end_year:
                months = months[: end_month - 1]

            print(f"Downloading data for months {months}")

            year = f"{year:04d}"

            for month in months:
                month = f"{month:02d}"

                # Submit download tasks to the executor
                futures.append(
                    executor.submit(
                        download_year_and_month,
                        year,
                        month,
                        data_split,
                        custom_cache_dir,
                        verbose,
                    )
                )

        # Process as completed
        for future in as_completed(futures):
            result = future.result()
            if verbose and result is not None:
                print(f"Download completed for {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, help="Year to download")
    parser.add_argument("--end_year", type=int, help="Year to end download")
    parser.add_argument("--start_month", type=int, help="Month to start download")
    parser.add_argument("--end_month", type=int, help="Month to end download")
    parser.add_argument("--data_split", type=str, help="Data split to download")
    parser.add_argument(
        "--custom_cache_dir", type=str, help="Custom cache directory to download data"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    if args.verbose:
        print(f"Using parameters:\n{args}\n")

    download_in_parallel(
        start_year=args.start_year,
        end_year=args.end_year,
        data_split=args.data_split,
        start_month=args.start_month,
        end_month=args.end_month,
        custom_cache_dir=Path(args.custom_cache_dir),
        verbose=args.verbose,
    )

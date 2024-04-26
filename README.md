# climsim-trajectory-data

Small repository for converting ClimSim low resoluton data to trajectories usable by PyTorch.

## Setup

To setup the environment to complete the download, we will use a python virutal environment. To do so, use the following commands:

```python
python3 -m venv climsim-trajectory-data-venv
source climsim-trajectory-data-venv/bin/activate
pip install -r requirements.txt --no-cache-dir
```

## Completing the download

To download the entire dataset (or a subset by changing the start and end years and the start and end months) into a directory (say the directory data in this repository), one can use the script `download.py` in the following way:

```python
source climsim-trajectory-data-venv/bin/activate
python download.py --start_year 0001 --end_year 0009 --data_split train --custom_cache_dir data --verbose
```

## Processing the data

Once we have downloaded it, we now need to process the data.

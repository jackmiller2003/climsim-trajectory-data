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

```python
source climsim-trajectory-data-venv/bin/activate
python process.py --output_path torch_data --mb_per_chunk 5000 --save_as_torch_tensor --verbose --variable_selection variable_selections/default.json
```

The main two arguments to change here is the `output_path` which controls where the chunked arrays are stored and the other is `mb_per_chunk` which determines the length of chunks. One should probably choose this based on the size of the GPUs upon which models are being trained.

If one wanted to use PBS to schedule jobs in parallel, you can use the start/end year/month flags. For example:

```bash
#!/bin/bash

#PBS -N [name]
#PBS -q [queue]
#PBS -P [project]
#PBS -l walltime=4:00:00
#PBS -l ncpus=16
#PBS -l mem=64GB
#PBS -l jobfs=100GB
#PBS -l storage=[storage]

cd [directory]
source climsim-trajectory-data-venv/bin/activate

python process.py --output_path [output_path] --data_path [data_path] --mb_per_chunk 12000 --save_as_torch_tensor --verbose --variable_selection variable_selections/default.json --start_year 1 --end_year 7 --start_month 2 --end_month 12
```
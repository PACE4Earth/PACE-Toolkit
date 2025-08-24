import os
import ast
import json
import xarray as xr

base_path = '/p/scratch/hclimrep/pavel1/CorrDiff/CorrDiff_Output_Code4Earth'

files = [
    os.path.join(base_path, f)
    for f in os.listdir(base_path)
    if '.nc' in f
]

files.sort()

this_path = files[-1]

GROUPS = ['prediction', 'truth']

for group in GROUPS:
    with xr.open_dataset(this_path, group=group, engine='netcdf4') as ds:
        print(group, '\n', ds, '\n')

        
    
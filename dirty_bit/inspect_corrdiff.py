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
# this_path = '/p/scratch/hclimrep/pavel1/CorrDiff/crea2_6h/Output/Output_allVars/corrdiff_output_ensemble_6h_8.nc'

with xr.open_dataset(this_path, group='prediction', engine='netcdf4') as ds:
    print(ds)
    # cfg = ast.literal_eval(ds.attrs['cfg'])
    # for k, v in cfg.items():
    #     print(k, v)
    #     print()
    # da = ds.isel(time=0).values
    # print(da)    
#     output_filename = cfg['generation']['io']['output_filename']
    
# try:
#     with xr.open_dataset(output_filename, engine='netcdf4') as ds:
#         print(ds)
# except Exception as e:
#     print(e)    

        
    
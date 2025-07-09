try:
    import cdsapi
except:
    raise ImportError("Install cdsapi.")

import os

output_file = os.path.join(os.getcwd(), 'data', 'era5_sample.nc')
print('Will save file to: ', output_file)

# Initialize CDS API
c = cdsapi.Client()

# Download ERA5 data
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'geopotential', 'u_component_of_wind', 'v_component_of_wind',
        ],
        'pressure_level': [
            '500', '850',
        ],
        'year': '2023',
        'month': '06',
        'day': '01',
        'time': [
            '00:00', '06:00', '12:00', '18:00',
        ],
        # 'area': [
        #     50, -130, 20, -60,
        # ],
    },
    str(output_file)
)


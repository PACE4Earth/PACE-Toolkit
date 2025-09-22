import os

import numpy as np
import torch
import zarr
from torch import nn
from torch.nn import functional as F
        
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from .operators import standardize

VARIABLES = [
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure',
    'vmax_10m',
    'total_precipitation',
]

class CorrelationMap(nn.Module):
    """
    Computes point-wise correlation statistics for a given list of variables
    in a time series.
    """
    def __init__(self, grid, device=None):
        """
        Initializes the module.

        Args:
            grid (dict): A dictionary containing grid information like 'lat' and 'lon'.
            variables (list): A list of strings with the names of the variables
                              to be found in the input sample dictionary.
            device (str, optional): The device to run the computations on. 
                                    Defaults to the 'DEVICE' env var or 'cpu'.
        """
        super().__init__()
        
        if device is None:
            self.device = os.getenv('DEVICE', 'cpu')
        else:
            self.device = device
            
        self.variables = VARIABLES
        # c = len(self.variables)  # Number of channels is now dynamic
        self.c = 3
        self.h = grid['lat'].shape[0]
        self.w = grid['lon'].shape[0]
        
        # Buffers to store running sums for correlation calculation
        # We use register_buffer so they are moved to the correct device
        # with .to(device) and included in the state_dict, but are not
        # considered model parameters by the optimizer.
        self.register_buffer('sum_c', torch.zeros(1, self.c, self.h, self.w, dtype=torch.float32))
        self.register_buffer('sum_c_sq', torch.zeros(1, self.c, self.h, self.w, dtype=torch.float32))
        self.register_buffer('sum_cc', torch.zeros(self.c, self.c, self.h, self.w, dtype=torch.float32))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))

    def forward(self, sample: dict):
        """
        Updates the running sums with a new data sample.
        """
        processed_tensors = []
        variable_names = []
        for name, tensor in sample.items():
            if isinstance(tensor, torch.Tensor) and name in self.variables:
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)
                elif tensor.ndim == 4:
                ############################################# this, the [0] index chooses ensemble member
                    tensor = tensor[0, [0]]
                
                processed_tensors.append(standardize(tensor))
                variable_names.append(name)
                    
        data = torch.stack(processed_tensors, dim=1).contiguous()
        c, n, h, w = data.shape
        
        # for di in data:
        #     print(di.min(), di.max())

        with torch.no_grad():
            self.count += 1 # data.shape[1]
            self.sum_c += torch.sum(data, dim=0, keepdim=True)
            self.sum_c_sq += torch.sum(data**2, dim=0, keepdim=True)
            self.sum_cc += torch.einsum('bchw,bdhw->cdhw', data, data)
        
        return None
        # return torch.zeros(1, 1, device=data.device)

    def compute_correlation(self, epsilon=1e-8):
        """
        Computes the Pearson correlation matrix from the accumulated sums.
        """
        if self.count == 0:
            print("Cannot compute correlation with zero samples.")
            return None
        
        N = self.count
        
        # E[X] = sum(X) / N
        mean_c = self.sum_c / N
        
        # Cov(X, Y) = E[XY] - E[X]E[Y]
        # The covariance matrix is E[X*X^T] - E[X]*E[X]^T
        cov_matrix = (self.sum_cc / N) - torch.einsum('ichw,jchw->ijchw', mean_c, mean_c).squeeze(2)

        # Var(X) = E[X^2] - E[X]^2
        var_c = (self.sum_c_sq / N) - mean_c**2
        std_dev_c = torch.sqrt(var_c) # Shape: [1, C, H, W]
        
        # Denominator for the correlation formula: std(X) * std(Y)
        # This is the outer product of the standard deviation vector.
        denominator = torch.einsum('ichw,jchw->ijchw', std_dev_c, std_dev_c).squeeze(2)
        
        # Correlation(X, Y) = Cov(X, Y) / (std(X) * std(Y))
        correlation_matrix = cov_matrix / (denominator + epsilon)
        
        return correlation_matrix

    def reset(self):
        """Resets the internal statistics."""
        self.sum_c.zero_()
        self.sum_c_sq.zero_()
        self.sum_cc.zero_()
        self.count.zero_()
    
    # def evaluate(self):
        
    #     sum_c_prod = self.sum_c * self.sum_c.transpose(1, 0)
    #     numerator = self.count * self.sum_cc - sum_c_prod
        
    #     var_term = self.count * self.sum_c_sq - self.sum_c_sq
    #     denominator_sq = var_term*var_term.transpose(1,0)
    #     denominator = torch.sqrt(denominator_sq + 1e-6)
        
    #     print(numerator.shape, denominator.shape)
        
    #     correlation_map = numerator / denominator
        
    #     return correlation_map
    
    def evaluate(self, logger, comm):
        """
        Performs a collective reduction and computes the correlation map.

        1.  All ranks write their local data to a shared Zarr archive.
        2.  All ranks wait at a barrier for writing to complete.
        3.  Rank 0 reads and aggregates all data from the archive.
        4.  Rank 0 computes the final correlation map and returns it.
        5.  Other ranks return None.
        """
    
        # correlation_map = self.compute_correlation().squeeze(0)
        
        # print(correlation_map.shape)
        
        # output_zarr_path = logger.path
    
        
        # # === Part 1: Reduction Logic (executed by ALL ranks) ===
        # zarr_path = logger.path
        # synchronizer = zarr.ProcessSynchronizer(f'{zarr_path}.sync')
        # store = zarr.DirectoryStore(zarr_path)
        # root = zarr.group(store=store, synchronizer=synchronizer, overwrite=False)

        # # 3. Write the data variables and link them to coordinates via attributes
        # corr_arr = root.create_dataset(
        #     'correlation_map',
        #     data=correlation_map.numpy(),
        #     dtype='f4'
        # )
        # corr_arr.attrs['_ARRAY_DIMENSIONS'] = ['var_1', 'var_2', 'lat', 'lon']

        # print('Correlation map computed and saved.')
        
        # return correlation_map
        
        rank = comm.Get_rank()
        size = comm.Get_size()
        zarr_path = logger.path

        # === Part 1: Reduction Logic (executed by ALL ranks) ===
        synchronizer = zarr.ProcessSynchronizer(f'{zarr_path}.sync')
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store, synchronizer=synchronizer, overwrite=False)

        # Each rank saves its local data to a unique group
        rank_group = root.create_group(f'corr_map_rank_{rank}', overwrite=True)
        rank_group.array('count', np.array(self.count))
        rank_group.array('sum_c', self.sum_c.numpy())
        rank_group.array('sum_c_sq', self.sum_c_sq.numpy())
        rank_group.array('sum_cc', self.sum_cc.numpy())
        
        # Wait for all ranks to finish writing
        comm.Barrier()

        # === Part 2: Aggregation & Calculation (executed ONLY by rank 0) ===
        print(rank)
        if rank == 0:
            print("Rank 0 is aggregating results from all ranks...")
            # Initialize with its own data
            global_count = self.count
            global_sum_c = self.sum_c.clone()
            global_sum_c_sq = self.sum_c_sq.clone()
            global_sum_cc = self.sum_cc.clone()

            # Loop through other ranks and add their contributions
            for i in range(1, size):
                other_rank_group = root[f'corr_map_rank_{i}']
                global_count += other_rank_group['count'][()]
                global_sum_c += torch.from_numpy(other_rank_group['sum_c'][:])
                global_sum_c_sq += torch.from_numpy(other_rank_group['sum_c_sq'][:])
                global_sum_cc += torch.from_numpy(other_rank_group['sum_cc'][:])

            print(f"Aggregation complete. Total count: {global_count}")

            # ... (aggregation as before) ...
            
            N = global_count
            
            # --- Correlation Calculation (using a more stable formula) ---
            
            # Numerator: N * sum(XY) - sum(X) * sum(Y)
            sum_x = global_sum_c.squeeze(0)
            sum_y = global_sum_c.squeeze(0) # for the correlation, this is the same tensor
            
            # The numerator's cross-product term is a bit tricky
            sum_x_sum_y = torch.einsum('chw,dhw->cdhw', sum_x, sum_y)
            numerator = N * global_sum_cc - sum_x_sum_y

            # Denominator: sqrt((N * sum(X^2) - sum(X)^2) * (N * sum(Y^2) - sum(Y)^2))
            sum_x_sq = global_sum_c_sq.squeeze(0)
            sum_y_sq = global_sum_c_sq.squeeze(0) # same tensor

            # The variance terms
            var_x_term = N * sum_x_sq - sum_x**2
            var_y_term = N * sum_y_sq - sum_y**2
            
            # Clamp for numerical stability
            var_x_term = torch.clamp(var_x_term, min=1e-4)
            var_y_term = torch.clamp(var_y_term, min=1e-4)

            # The full denominator
            denominator_sq = torch.einsum('chw,dhw->cdhw', var_x_term, var_y_term)
            denominator = torch.sqrt(denominator_sq + 1e-4)

            # Final correlation map
            correlation_map = numerator / (denominator + 1e-4)

            
            print(correlation_map.shape, correlation_map.min(), correlation_map.max())
            
            output_zarr_path = logger.path
        
            # 3. Write the data variables and link them to coordinates via attributes
            corr_arr = root.create_dataset(
                'correlation_map',
                data=correlation_map.numpy(),
                dtype='f4'
            )
            corr_arr.attrs['_ARRAY_DIMENSIONS'] = ['var_1', 'var_2', 'lat', 'lon']

            print('Correlation map computed and saved.')
         
            self.sum_c = torch.zeros(1, self.c, self.h, self.w, dtype=torch.float32)
            self.sum_c_sq = torch.zeros(1, self.c, self.h, self.w, dtype=torch.float32)
            self.sum_cc = torch.zeros(self.c, self.c, self.h, self.w, dtype=torch.float32)
            self.count = torch.tensor(0, dtype=torch.long)

            return correlation_map
        else:
            
            self.sum_c = torch.zeros(1, self.c, self.h, self.w, dtype=torch.float32)
            self.sum_c_sq = torch.zeros(1, self.c, self.h, self.w, dtype=torch.float32)
            self.sum_cc = torch.zeros(self.c, self.c, self.h, self.w, dtype=torch.float32)
            self.count = torch.tensor(0, dtype=torch.long)
            
            return None
    
    def visualize(self):
        
        correlation_map = self.evaluate()
        
        print(correlation_map.shape)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        
        im = ax.pcolormesh(correlation_map[1,2], vmin=-1, vmax=1, cmap='seismic')
        
        fig.colorbar(im, ax=ax)
        
        plt.savefig('/p/project1/hclimrep/vozar2/PACE-Toolkit/pace/plots/corrs_map.png')
        plt.close("all")
        
        return None
    
if __name__=="__main__":
    # 1. Define your grid and the list of variables you care about
    my_variables = [
        '2m_temperature',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'mean_sea_level_pressure'
    ]
    grid_info = {'lat': torch.rand(721), 'lon': torch.rand(1440)}

    # 2. Instantiate the module with your specific variables
    corr_calculator = CorrelationMap(grid=grid_info, variables=my_variables, device='cpu')

    # 3. In your training/data loop, create a sample and pass it
    for i in range(100): # Simulate iterating through 100 time steps
        # Create a dummy sample dictionary
        sample = {
            '2m_temperature': torch.randn(1, 1, 721, 1440),
            '10m_u_component_of_wind': torch.randn(1, 1, 721, 1440),
            '10m_v_component_of_wind': torch.randn(1, 1, 721, 1440),
            'mean_sea_level_pressure': torch.randn(1, 1, 721, 1440),
            # ... other variables the module will ignore
            'geopotential': torch.randn(1, 1, 721, 1440) 
        }
        
        # Update the running statistics
        corr_calculator(sample)

    # 4. After processing all data, compute the final correlation matrix
    correlation_matrix = corr_calculator.compute_correlation()

    # The result is a tensor of shape [4, 4, 721, 1440]
    # where correlation_matrix[i, j, h, w] is the correlation between
    # variable i and variable j at the grid point (h, w).
    print("Shape of correlation matrix:", correlation_matrix.shape)
    print("Number of samples processed:", corr_calculator.count.item())
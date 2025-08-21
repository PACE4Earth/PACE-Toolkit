import os

import torch
from torch import nn
from torch.nn import functional as F
        
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# class CorrelationMap(nn.Module):
#     def __init__(self, grid):
#         super().__init__()
        
#         self.device = os.getenv('DEVICE')  # Get device from environment variable or default to CPU
#         c = 4
#         h = grid['lat'].shape[0]
#         w = grid['lon'].shape[0]
        
#         self.sum_c = torch.zeros(1, c, h, w, device=self.device, dtype=torch.float32)
#         self.sum_c_sq = torch.zeros(1, c, h, w, device=self.device, dtype=torch.float32)
#         self.sum_cc = torch.zeros(c, c, h, w, device=self.device, dtype=torch.float32)
#         self.count = 0
        
#     def forward(self, sample):
        
#         data = torch.cat(
#             [
#                 sample['2m_temperature'],
#                 sample['10m_u_component_of_wind'],
#                 sample['10m_v_component_of_wind'],
#                 sample['mean_sea_level_pressure'],
#             ],
#             dim=1,
#         )
        
#         self.count = self.count + 1
        
#         self.sum_c = self.sum_c + data
#         self.sum_cc = self.sum_cc + data*data.transpose(1, 0)
#         self.sum_c_sq = self.sum_c_sq + data**2
        
#         if self.count == 6480:
#             self.visualize()
        
#         return None

class CorrelationMap(nn.Module):
    """
    Computes point-wise correlation statistics for a given list of variables
    in a time series.
    """
    def __init__(self, grid, variables: list, device=None):
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
            
        self.variables = variables
        c = len(self.variables)  # Number of channels is now dynamic
        h = grid['lat'].shape[0]
        w = grid['lon'].shape[0]
        
        # Buffers to store running sums for correlation calculation
        # We use register_buffer so they are moved to the correct device
        # with .to(device) and included in the state_dict, but are not
        # considered model parameters by the optimizer.
        self.register_buffer('sum_c', torch.zeros(1, c, h, w, dtype=torch.float32))
        self.register_buffer('sum_c_sq', torch.zeros(1, c, h, w, dtype=torch.float32))
        self.register_buffer('sum_cc', torch.zeros(c, c, h, w, dtype=torch.float32))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))

    def forward(self, sample: dict):
        """
        Updates the running sums with a new data sample.
        """
        # Dynamically build the tensor from the list of variables
        tensors_to_cat = [sample[var] for var in self.variables]
        data = torch.cat(tensors_to_cat, dim=1).to(self.device)
        
        # Ensure data has a batch dimension of 1 for broadcasting
        if data.dim() == 3:
            data = data.unsqueeze(0) # Shape: [1, C, H, W]

        # No gradients needed for these calculations
        with torch.no_grad():
            self.count += data.shape[0]
            self.sum_c += torch.sum(data, dim=0, keepdim=True)
            self.sum_c_sq += torch.sum(data**2, dim=0, keepdim=True)
            
            # einsum is a clear way to compute the outer product batch-wise
            # 'bchw,bdhw->cdhw' means for each item in the batch, compute the
            # outer product of the channel vectors at each (h,w) location.
            self.sum_cc += torch.einsum('bchw,bdhw->cdhw', data, data)
        
        return None

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
    
    def evaluate(self):
        
        sum_c_prod = self.sum_c * self.sum_c.transpose(1, 0)
        numerator = self.count * self.sum_cc - sum_c_prod
        
        var_term = self.count * self.sum_c_sq - self.sum_c_sq
        denominator_sq = var_term*var_term.transpose(1,0)
        denominator = torch.sqrt(denominator_sq + 1e-6)
        
        print(numerator.shape, denominator.shape)
        
        correlation_map = numerator / denominator
        
        return correlation_map
    
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
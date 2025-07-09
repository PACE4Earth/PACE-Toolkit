import torch
import matplotlib.pyplot as plt

from utils.dataset import UnifiedDataset
from metrics.geostrophic_new import GeostrophicWind


def load_sample(level_idx=15):
    dataset = UnifiedDataset()
    sample = dataset[0]  # Shape: [time, channel, level, y, x]

    u = sample['u_component_of_wind'][0, 0, level_idx]
    v = sample['v_component_of_wind'][0, 0, level_idx]
    phi = sample['geopotential'][0, 0, level_idx]

    dx = dataset.grid['dx']
    dy = dataset.grid['dy']
    f = dataset.grid['f']
    lat = dataset.grid['lat']
    lon = dataset.grid['lon']

    return phi, u, v, dx, dy, f, lat, lon


def compute_geostrophic_metrics(phi, u, v, dx, dy, f, lat):
    gw = GeostrophicWind(dx, dy, f, lat)
    return gw.compute_with_actual_wind(phi, u, v)


def plot_quiver(u, v, u_g, v_g, lat, lon, stride=10, level_idx=15, filename="uv_vs_geostrophic.png"):
    u_plot = u[::stride, ::stride]
    v_plot = v[::stride, ::stride]
    ug_plot = u_g[::stride, ::stride]
    vg_plot = v_g[::stride, ::stride]

    lat_plot = lat[::stride]
    lon_plot = lon[::stride]
    Lon, Lat = torch.meshgrid(lon_plot, lat_plot, indexing='xy')

    plt.figure(figsize=(14, 6))
    plt.quiver(Lon, Lat, u_plot, v_plot, scale=2000, headwidth=2, color='black', label='Wind (u,v)')
    plt.quiver(Lon, Lat, ug_plot, vg_plot, scale=2000, headwidth=2, color='blue', label='Geostrophic Wind (ug, vg)')
    plt.title(f'Wind vs Geostrophic Wind (Level {level_idx})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    # plt.show()


def plot_ratio(ratio, lat, lon, level_idx=15, filename="ageo_to_geo_ratio.png"):
    Lon, Lat = torch.meshgrid(lon, lat, indexing='xy')

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(Lon, Lat, ratio, cmap='viridis', shading='auto', vmin=0, vmax=2)
    plt.colorbar(label='Ageostrophic / Geostrophic Wind Magnitude', )
    plt.title(f'Ratio of Ageostrophic to Geostrophic Wind (Level {level_idx})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    # plt.show()


def main():
    level_idx = 15
    phi, u, v, dx, dy, f, lat, lon = load_sample(level_idx)
    u_g, v_g, u_ag, v_ag, ratio = compute_geostrophic_metrics(phi, u, v, dx, dy, f, lat)

    plot_quiver(u, v, u_g, v_g, lat, lon, level_idx=level_idx)
    plot_ratio(ratio, lat, lon, level_idx=level_idx)


if __name__ == "__main__":
    main()

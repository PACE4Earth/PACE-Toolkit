import os
from pathlib import Path
from typing import List

def get_env_path(var_name: str) -> Path:
    val = os.environ.get(var_name)
    if not val:
        raise EnvironmentError(f"Environment variable '{var_name}' is not set.")
    return Path(val)

# Base directories (from environment variables)
ERA5_BASE = get_env_path("ERA5_DATA_PATH")
CORRDIFF_BASE = get_env_path("CORRDIFF_DATA_PATH")
GRAPHCAST_BASE = get_env_path("GRAPHCAST_FORECAST_PATH")

# === GraphCast ===
def get_graphcast_file(timestamp: str) -> Path:
    """e.g., '20210101_00' → /.../20210101_00.nc"""
    return GRAPHCAST_BASE / f"{timestamp}.nc"

def list_graphcast_files() -> List[Path]:
    """List all GraphCast forecast files."""
    return sorted(GRAPHCAST_BASE.glob("*.nc"))
    
# === ERA5 ===
def get_era5_output_path(timestamp: str) -> Path:
    """e.g., '20210101_00' → /.../20210101_00/output.nc"""
    return ERA5_BASE / timestamp / "output.nc"

def list_era5_files() -> List[Path]:
    """List all ERA5 output.nc files (one per timestamp)."""
    return sorted(ERA5_BASE.glob("*/output.nc"))

# === CorrDiff ===
def get_corrdiff_file(timestamp: str) -> Path:
    """e.g., '20210101_00' → /.../forecasts/20210101_00.nc"""
    return CORRDIFF_BASE / "forecasts" / f"{timestamp}.nc"

def list_corrdiff_files() -> List[Path]:
    """List all CorrDiff forecast files."""
    return sorted((CORRDIFF_BASE / "forecasts").glob("*.nc"))

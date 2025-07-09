import os

def get_env_path(env_var, default=None):
    """
    Returns the absolute path from environment variable env_var.
    If env_var not set, returns the default relative path.
    """
    path = os.getenv(env_var)
    if path is not None:
        return path
    elif default is not None:
        return default
    else:
        raise RuntimeError(f"Environment variable {env_var} not set and no default provided.")

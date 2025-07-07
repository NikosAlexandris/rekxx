import shlex
import subprocess
from pathlib import Path
import netCDF4 as nc
import typer
import xarray as xr
from rich import print
from rekx.backend import RechunkingBackend
from rekx.constants import VERBOSE_LEVEL_DEFAULT
from rekx.models import (
    XarrayVariableSet,
    select_xarray_variable_set_from_dataset,
    validate_variable_set,
)
from pathlib import Path
import typer
from rekx.constants import DRY_RUN_DEFAULT
from rekx.nccopy.constants import (
    FIX_UNLIMITED_DIMENSIONS_DEFAULT,
    CACHE_SIZE_DEFAULT,
    CACHE_ELEMENTS_DEFAULT,
    CACHE_PREEMPTION_DEFAULT,
    COMPRESSION_FILTER_DEFAULT,
    COMPRESSION_LEVEL_DEFAULT,
    SHUFFLING_DEFAULT,
    RECHUNK_IN_MEMORY_DEFAULT,
)


def rechunk_multiple_netcdf_files(
    input_filepath: Path,
    output_filepath: Path | None,
    time: int,
    latitude: int,
    longitude: int,
    min_longitude: float | None = None,
    min_x: int | None = None,
    max_longitude: float | None = None,
    max_x: int | None = None,
    min_latitude: float | None = None,
    min_y: int | None = None,
    max_latitude: float | None = None,
    max_y: int | None = None,
    fix_unlimited_dimensions: bool = FIX_UNLIMITED_DIMENSIONS_DEFAULT,
    variable_set: list[XarrayVariableSet] = list[XarrayVariableSet.all],
    every_nth_timestamp: int | None = None,
    mask_and_scale: bool = False,
    drop_other_variables: bool = True,
    cache_size: int | None = CACHE_SIZE_DEFAULT,
    cache_elements: int | None = CACHE_ELEMENTS_DEFAULT,
    cache_preemption: float | None = CACHE_PREEMPTION_DEFAULT,
    compression: str = COMPRESSION_FILTER_DEFAULT,
    compression_level: int = COMPRESSION_LEVEL_DEFAULT,
    shuffling: str | None = SHUFFLING_DEFAULT,
    memory: bool = RECHUNK_IN_MEMORY_DEFAULT,
    mode: str = 'w-',
    overwrite_output: bool = False,
    dry_run: bool = DRY_RUN_DEFAULT,
    backend: RechunkingBackend = RechunkingBackend.xarray,
    # dask_scheduler: str | None = None,
    verbose: int = VERBOSE_LEVEL_DEFAULT,
):
    """
    Rechunk a NetCDF4 dataset with options to fine tune the output

    """
    try:
        with xr.open_dataset(input_filepath, engine="h5netcdf") as dataset:
            variable_set = validate_variable_set(variable_set)
            selected_variables = select_xarray_variable_set_from_dataset(
                XarrayVariableSet, variable_set, dataset
            )
            backend_name = backend.name
            backend = backend.get_backend()
            command = backend.rechunk(
                input_filepath=input_filepath,
                variables=list(selected_variables),
                output_filepath=output_filepath,
                time=time,
                latitude=latitude,
                longitude=longitude,
                min_longitude=min_longitude,
                min_x=min_x,
                max_longitude=max_longitude,
                max_x=max_x,
                min_latitude=min_latitude,
                min_y=min_y,
                max_latitude=max_latitude,
                max_y=max_y,
                every_nth_timestamp=every_nth_timestamp,
                mask_and_scale=mask_and_scale,
                drop_other_variables=drop_other_variables,
                fix_unlimited_dimensions=fix_unlimited_dimensions,
                cache_size=cache_size,
                cache_elements=cache_elements,
                cache_preemption=cache_preemption,
                compression=compression,
                compression_level=compression_level,
                shuffling=shuffling,
                memory=memory,
                mode=mode,
                overwrite_output=overwrite_output,
                dry_run=dry_run,
                verbose=verbose,
            )

            # -------------------------------------------- Re-Design Me ------
            # Only nccopy backend returns executable commands
            if backend_name == RechunkingBackend.nccopy.name:
                subprocess.run(shlex.split(command), check=True)
                command_arguments = shlex.split(command)
                try:
                    subprocess.run(command_arguments, check=True)
                    print(f"Command {command} executed successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred while executing the command: {e}")

            # if backend_name == RechunkingBackend.xarray.name:
            else:
                return command
                # logger.info(f"Rechunking completed: {command}")
            # -------------------------------------------- Re-Design Me ------

    except Exception as e:
        typer.echo(f"Error processing {input_filepath.name}: {str(e)}")

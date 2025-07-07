import shlex
from distributed import LocalCluster
from rekx.dask_configuration import auto_configure_for_large_dataset
import subprocess
from pathlib import Path
from typing import Optional
from rekx.api.rechunk import rechunk_multiple_netcdf_files
import typer
from rekx.typer.parameters import (
    typer_option_variable_set,
    typer_argument_source_path,
    typer_argument_source_path_with_pattern,
    typer_argument_time_chunk_size,
    typer_argument_latitude_chunk_size,
    typer_argument_longitude_chunk_size,
    typer_option_output_directory,
    typer_option_filename_pattern,
    typer_option_every_nth_timestamp,
    typer_option_mask_and_scale,
    typer_option_overwrite_output,
    typer_option_dry_run,
    typer_option_verbose,
    typer_option_latitude_in_degrees,
    typer_option_y_index,
    typer_option_longitude_in_degrees,
    typer_option_x_index,
    typer_option_drop_other_variables,
    typer_option_rechunking_backend,
    typer_option_compresion,
    typer_option_compresion_level,
    typer_option_shuffling,
    typer_option_fix_unlimited_dimensions,
    typer_option_nccopy_cache_size,
    typer_option_nccopy_cache_elements,
    typer_option_nccopy_cache_preemption,
    typer_option_nccopy_memory,
    typer_option_workers,
    typer_option_threads_per_worker,
    typer_option_memory_limit,
    typer_option_auto_memory_limit,
    typer_option_open_file_mode,
    typer_option_overwrite_output,
)
import xarray as xr
from rich import print
from typing_extensions import Annotated
from rekx.backend import RechunkingBackend
from rekx.constants import VERBOSE_LEVEL_DEFAULT
from rekx.log import logger
from rekx.models import (
    XarrayVariableSet,
    select_xarray_variable_set_from_dataset,
    validate_variable_set,
)
from dask.distributed import Client
from functools import partial
from pathlib import Path
from typing import Optional
import typer
import dask
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
from rekx.constants import OVERWRITE_OUTPUT_DEFAULT


def modify_chunk_size(
    netcdf_file,
    variable,
    chunk_size,
):
    """
    Modify the chunk size of a variable in a NetCDF file.

    Parameters:
    - nc_file: path to the NetCDF file
    - variable_name: name of the variable to modify
    - new_chunk_size: tuple specifying the new chunk size, e.g., (2600, 2600)
    """
    with nc.Dataset(netcdf_file, "r+") as dataset:
        variable = dataset.variables[variable]

        if variable.chunking() != [None]:
            variable.set_auto_chunking(chunk_size)
            print(
                f"Modified chunk size for variable '{variable}' in file '{netcdf_file}' to {chunk_size}."
            )

        else:
            print(
                f"Variable '{variable}' in file '{netcdf_file}' is not chunked. Skipping."
            )


def rechunk(
    input_filepath: Annotated[Path, typer_argument_source_path],
    output_filepath: Annotated[
        Optional[Path], typer.Argument(help="Path to the output NetCDF file.")
    ],
    time: Annotated[int, typer.Option(help="New chunk size for the `time` dimension.")],
    latitude: Annotated[
        int, typer.Option(help="New chunk size for the `lat` dimension.")
    ],
    longitude: Annotated[
        int, typer.Option(help="New chunk size for the `lon` dimension.")
    ],
    fix_unlimited_dimensions: Annotated[
        bool, typer.Option(help="Convert unlimited size input dimensions to fixed size dimensions in output.")
    ] = FIX_UNLIMITED_DIMENSIONS_DEFAULT,
    variable_set: Annotated[
        list[XarrayVariableSet], typer.Option(help="Set of Xarray variables to rechunk [bold red]Not Fully Functional Yet![/bold red]")] = list[XarrayVariableSet.all],
    cache_size: Optional[int] = CACHE_SIZE_DEFAULT,
    cache_elements: Optional[int] = CACHE_ELEMENTS_DEFAULT,
    cache_preemption: Optional[float] = CACHE_PREEMPTION_DEFAULT,
    compression: str = COMPRESSION_FILTER_DEFAULT,
    compression_level: int = COMPRESSION_LEVEL_DEFAULT,
    shuffling: str = SHUFFLING_DEFAULT,
    memory: bool = RECHUNK_IN_MEMORY_DEFAULT,
    mode: Annotated[ str, typer.Option(help="Writing file mode")] = 'w-',
    overwrite_output: Annotated[bool, typer.Option(help="Overwrite existing output file")] = False,
    dry_run: Annotated[bool, typer_option_dry_run] = DRY_RUN_DEFAULT,
    backend: Annotated[
        RechunkingBackend,
        typer.Option(
            help="Backend to use for rechunking. [code]nccopy[/code] [red]Not Implemented Yet![/red]"
        ),
    ] = RechunkingBackend.xarray,
    dask_scheduler: Annotated[
        Optional[str], typer.Option(help="The port:ip of the dask scheduler")
    ] = None,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
):
    """
    Rechunk a NetCDF4 dataset with options to fine tune the output

    """
    if verbose:
        import time as timer

        rechunking_timer_start = timer.time()

    # if dask_scheduler:
    #     from dask.distributed import Client
    #     client = Client(dask_scheduler)
    #     typer.echo(f"Using Dask scheduler at {dask_scheduler}")

    try:
        with xr.open_dataset(input_filepath, engine="h5netcdf") as dataset:

            # def validate_variable_set(variable_set_input: list[str]) -> list[XarrayVariableSet]:
            #     if not variable_set_input:
            #         # Use a sensible default or raise
            #         return [XarrayVariableSet.all]
            #     validated = []
            #     for v in variable_set_input:
            #         if v in XarrayVariableSet.__members__:
            #             validated.append(XarrayVariableSet[v])
            #         else:
            #             raise ValueError(f"Invalid variable set: {v}")
            #     return validated

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
                dry_run=dry_run,  # = True : just return the command!
                verbose=verbose,
            )

            if dry_run:
                if verbose:
                    print(
                        f"[bold]Dry run[/bold] the [bold]following command that would be executed[/bold] :",
                        f"    {command}",
                    )

                return  # Exit for a dry run

            else:
                # Only nccopy backend returns executable commands
                
                if backend_name == RechunkingBackend.nccopy.name:
                    subprocess.run(shlex.split(command), check=True)
                    command_arguments = shlex.split(command)
                    try:
                        subprocess.run(command_arguments, check=True)
                        print(f"Command {command} executed successfully.")
                    except subprocess.CalledProcessError as e:
                        print(f"An error occurred while executing the command: {e}")

                else:
                    logger.info(f"Rechunking completed: {command}")

            if verbose:
                rechunking_timer_end = timer.time()
                elapsed_time = rechunking_timer_end - rechunking_timer_start
                logger.info(f"Rechunking via {backend} took {elapsed_time:.2f} seconds")
                print(f"Rechunking took {elapsed_time:.2f} seconds.")

    except Exception as e:
        typer.echo(f"Error processing {input_filepath.name}: {str(e)}")


def rechunk_netcdf_files(
    # Input
    source_path: Annotated[Path, typer_argument_source_path_with_pattern],
    # Chunks
    time: Annotated[int, typer_argument_time_chunk_size],
    latitude: Annotated[int, typer_argument_latitude_chunk_size],
    longitude: Annotated[int, typer_argument_longitude_chunk_size],
    # Slicing
    min_longitude: Annotated[Optional[float], typer_option_longitude_in_degrees] = None,
    min_x: Annotated[Optional[int], typer_option_x_index] = None,
    max_longitude: Annotated[Optional[float], typer_option_longitude_in_degrees] = None,
    max_x: Annotated[Optional[int], typer_option_x_index] = None,
    min_latitude: Annotated[Optional[float], typer_option_latitude_in_degrees] = None,
    min_y: Annotated[Optional[int], typer_option_y_index] = None,
    max_latitude: Annotated[Optional[float], typer_option_latitude_in_degrees] = None,
    max_y: Annotated[Optional[int], typer_option_y_index] = None,
    every_nth_timestamp: Annotated[Optional[int], typer_option_every_nth_timestamp] = None,
    # Input
    pattern: Annotated[str, typer_option_filename_pattern] = "*.nc",
    # Options for reading / opening data
    mask_and_scale: Annotated[bool, typer_option_mask_and_scale] = False,
    # Output & options
    output_directory: Annotated[Path, typer_option_output_directory] = Path('.'),
    mode: Annotated[str, typer_option_open_file_mode] = 'w',
    overwrite_output: Annotated[bool, typer_option_overwrite_output] = OVERWRITE_OUTPUT_DEFAULT,
    # Data processing
    variable_set: Annotated[
        list[XarrayVariableSet], typer_option_variable_set] = list[XarrayVariableSet.all],
    drop_other_variables: Annotated[bool, typer_option_drop_other_variables] = True,
    # Rechunking
    backend: Annotated[RechunkingBackend, typer_option_rechunking_backend] = RechunkingBackend.xarray,
    memory: Annotated[bool, typer_option_nccopy_memory] = RECHUNK_IN_MEMORY_DEFAULT,
    # Output data structure
    fix_unlimited_dimensions: Annotated[bool, typer_option_fix_unlimited_dimensions] = FIX_UNLIMITED_DIMENSIONS_DEFAULT,
    compression: Annotated[str, typer_option_compresion] = COMPRESSION_FILTER_DEFAULT,
    compression_level: Annotated[int, typer_option_compresion_level] = COMPRESSION_LEVEL_DEFAULT,
    shuffling: Annotated[Optional[str], typer_option_shuffling] = SHUFFLING_DEFAULT,
    cache_size: Annotated[Optional[int], typer_option_nccopy_cache_size] = CACHE_SIZE_DEFAULT,
    cache_elements: Annotated[Optional[int], typer_option_nccopy_cache_elements] = CACHE_ELEMENTS_DEFAULT,
    cache_preemption: Annotated[Optional[float], typer_option_nccopy_cache_preemption] = CACHE_PREEMPTION_DEFAULT,
    # Dask configuration
    workers: Annotated[int, typer_option_workers] = 4,
    threads_per_worker: Annotated[int, typer_option_threads_per_worker] = 1,
    memory_limit: Annotated[Optional[int], typer_option_memory_limit] = None,
    auto_memory_limit: Annotated[bool, typer_option_auto_memory_limit] = True,
    dry_run: Annotated[bool, typer_option_dry_run] = False,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
) -> None:
    """Rechunk multiple NetCDF files in parallel"""
    if verbose:
        import time as timer

        rechunking_timer_start = timer.time()

    if not memory_limit:
        memory_limit = auto_memory_limit

   # Auto-configure Dask for single large file processing
    dask_configuration = auto_configure_for_large_dataset(
        memory_limit=memory_limit,
        workers=workers,
        threads_per_worker=threads_per_worker,
        verbose=verbose,
    )

    # Resolve input files
    if source_path.is_file():
        input_file_paths = [source_path]

    elif source_path.is_dir():
        input_file_paths = list(source_path.glob(pattern))

    else:
        raise ValueError(f"Invalid input path: {source_path}")
    
    if not input_file_paths:
        typer.echo("No files found matching pattern")
        return

    # Dry run logic
    if dry_run:
        dry_run_message = (
            f"Dry running operations that [bold]would be performed[/bold] :"
            f"\n> Configure Dask : {dask_configuration}"
            f"\n> Match filename pattern [code]{pattern}[/code] in [code]{source_path}[/code]"
            f"\n> Number of files matched : {len(list(input_file_paths))}"
            f"\n> Write rechunked data in [code]{output_directory}[/code]"
        )
        print(dry_run_message)
        return

    else:
        # "Make" the output directory
        if input_file_paths and not output_directory.exists():
            output_directory.mkdir(parents=True, exist_ok=True)
            if verbose > 0:
                print(f"[yellow]Creating the requested output directory[/yellow] [code]{output_directory}[/code].")

    output_filename_base = f"{time}_{latitude}_{longitude}_{compression}_{compression_level}"
    if shuffling and compression_level > 0:
        output_filename_base += "_shuffled"
    output_files = [
        output_directory / f"{f.stem}_{output_filename_base}{f.suffix}"
        for f in input_file_paths
    ]

    # Launch a local Dask cluster
    if dask_configuration:
        # with LocalCluster(**dask_configuration) as cluster, Client(cluster) as client:
        cluster = LocalCluster(**dask_configuration)
        client = Client(cluster)
    else:
        client = Client()

        print(f"Dashboard: {client.dashboard_link}")

    if verbose:
        print(f"Processing {len(input_file_paths)} files with {workers} workers")

    # Create processing function with fixed parameters
    # with multiprocessing.Pool(processes=workers) as pool:
    partial_rechunk_command = partial(
        rechunk_multiple_netcdf_files,
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
        fix_unlimited_dimensions=fix_unlimited_dimensions,
        variable_set=variable_set,
        every_nth_timestamp=every_nth_timestamp,
        mask_and_scale=mask_and_scale,
        drop_other_variables=drop_other_variables,
        cache_size=cache_size,
        cache_elements=cache_elements,
        cache_preemption=cache_preemption,
        compression=compression,
        compression_level=compression_level,
        shuffling=shuffling,
        memory=memory,
        mode=mode,
        overwrite_output=overwrite_output,
        dry_run=dry_run,  # just return the command!
        backend=backend,
        verbose=verbose,
    )
        # pool.map(partial_rechunk_command, input_file_paths)

    # Build the task graph
    tasks = []
    for in_file, out_file in zip(input_file_paths, output_files):
        task = dask.delayed(partial_rechunk_command, name="Rechunk data in NetCDf file")(
            in_file,
            out_file,
        )
        tasks.append(task)

        # Print command immediately if verbose
        if verbose:
            # Get command without executing (dry-run=True)
            command = rechunk_multiple_netcdf_files(
                in_file,
                out_file, 
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
                fix_unlimited_dimensions=fix_unlimited_dimensions,
                variable_set=variable_set,
                every_nth_timestamp=every_nth_timestamp,
                mask_and_scale=mask_and_scale,
                drop_other_variables=drop_other_variables,
                cache_size=cache_size,
                cache_elements=cache_elements,
                cache_preemption=cache_preemption,
                compression=compression,
                compression_level=compression_level,
                shuffling=shuffling,
                memory=memory,
                mode=mode,
                overwrite_output=overwrite_output,
                dry_run=True,  # just return the command !
                backend=backend,
                verbose=verbose,
            )
            print(f"[green]>[/green] {command}")
            # print(f"  [green]>[/green] [code dim]{cmd}[/code dim]")


    if not dry_run:
        dask.compute(*tasks)
        # client.close()
        # cluster.close()

        if verbose:
            print(f"[bold green]Rechunking operations [code]{backend.name}[/code] complete.[/bold green]")
            rechunking_timer_end = timer.time()
            elapsed_time = rechunking_timer_end - rechunking_timer_start
            logger.info(f"Rechunking via {backend} took {elapsed_time:.2f} seconds")
            print(f"Elapsed time {elapsed_time:.2f} seconds.")

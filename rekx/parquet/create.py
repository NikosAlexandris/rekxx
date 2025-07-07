import multiprocessing
import traceback
from functools import partial
from pathlib import Path
import fsspec
import xarray as xr
from fsspec.implementations.reference import LazyReferenceMapper
from kerchunk.hdf import SingleHdf5ToZarr
from rich import print
from rekx.constants import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_RECORD_SIZE,
    OVERWRITE_OUTPUT_DEFAULT,
    DRY_RUN_DEFAULT,
    VERBOSE_LEVEL_DEFAULT,
)
from rekx.log import logger


def create_parquet_store(
    input_file: Path,
    output_parquet_store: Path,
    cache_size: int = DEFAULT_CACHE_SIZE,
    record_size: int = DEFAULT_RECORD_SIZE,
    overwrite_output: bool = OVERWRITE_OUTPUT_DEFAULT,
    dry_run: bool = DRY_RUN_DEFAULT,
):
    """ """
    log_messages = []
    log_messages.append("Logging execution of create_parquet_store()")

    if not dry_run:
        # Write chunked data as a NetCDF file
        if output_parquet_store.exists():
            if not overwrite_output:
                print(f"A store named '{output_parquet_store}' exists already. Process abord. You may wish to retry by adding ``--overwrite_output` ?")
                return  # Exit the function without writing the file
        else:
            output_parquet_store.mkdir(parents=True, exist_ok=True)

        try:
            log_messages.append(f"Creating a filesystem mapper for {output_parquet_store}")
            filesystem = fsspec.filesystem("file")

            # Create LazyReferenceMapper to pass to SingleHdf5ToZarr
            lazy_output = LazyReferenceMapper.create(
                root=str(output_parquet_store),
                fs=filesystem,
                cache_size=cache_size,
                record_size=record_size,
                engine='fastparquet',
            )
            log_messages.append(f"Created the filesystem mapper {lazy_output}")

            log_messages.append(f"Kerchunking the file {input_file}")
            # zarr_like_object = SingleHdf5ToZarr(str(input_file), out=lazy_output)
            zarr_like_object = SingleHdf5ToZarr(str(input_file))
            parquet_store = zarr_like_object.translate()
            log_messages.append(f"Kerchunked the file {input_file}")

        except Exception:
            print(f"Failed processing file [code]{input_file}[/code]")
            log_messages.append(f"Exception occurred")
            log_messages.append("Traceback (most recent call last):")
            tb_lines = traceback.format_exc().splitlines()
            for line in tb_lines:
                log_messages.append(line)
            raise

        finally:
            logger.info("\n".join(log_messages))

        # logger.info(f"Returning a Parquet store : {output_parquet_store}")
        return parquet_store


def create_single_parquet_store(
    input_file_path,
    output_directory,
    cache_size: int = DEFAULT_CACHE_SIZE,
    record_size: int = DEFAULT_RECORD_SIZE,
    overwrite_output: bool = OVERWRITE_OUTPUT_DEFAULT,
    dry_run: bool = DRY_RUN_DEFAULT,
    verbose: int = VERBOSE_LEVEL_DEFAULT,
):
    """Helper function for create_multiple_parquet_stores()"""

    # Build a name for the output Parquet store (which is a directory)
    filename = input_file_path.stem
    single_parquet_store = output_directory / f"{filename}.parquet"

    # Generate Parquet store
    parquet_store = create_parquet_store(
            input_file=input_file_path,
        output_parquet_store=single_parquet_store,
        cache_size=cache_size,
        record_size=record_size,
        overwrite_output=overwrite_output,
        dry_run=dry_run,
    )
    if verbose > 0:
        print(f"  [code]{single_parquet_store}[/code]")

    if verbose > 1:
        dataset = xr.open_dataset(
            str(parquet_store),
            engine="kerchunk",
            storage_options=dict(remote_protocol="file"),
        )
        print(dataset)

    return parquet_store


def create_multiple_parquet_stores(
    source_directory: Path,
    output_directory: Path,
    pattern: str = "*.nc",
    cache_size: int = DEFAULT_CACHE_SIZE,
    record_size: int = DEFAULT_RECORD_SIZE,
    workers: int = 4,
    overwrite_output: bool = OVERWRITE_OUTPUT_DEFAULT,
    dry_run: bool = DRY_RUN_DEFAULT,
    verbose: int = VERBOSE_LEVEL_DEFAULT,
):
    """ """
    input_file_paths = list(source_directory.glob(pattern))
    # if verbose:
    #     print(f'Input file paths : {input_file_paths}')
    if not input_file_paths:
        print(
            "No files found in [code]{source_directory}[/code] matching the pattern [code]{pattern}[/code]!"
        )
        return

    if output_directory.exists():
        if not overwrite_output:
            print(f"A directory named '{output_directory}' exists already. Process abord. You may wish to retry by adding ``--overwrite_output` ?")
            return  # Exit the function without writing the file
    try:
        output_directory = output_directory.parent / output_directory.name
        output_directory.mkdir(parents=True, exist_ok=True)

        with multiprocessing.Pool(processes=workers) as pool:
            print(
                f"Creating the following Parquet stores in [code]{output_directory}[/code] : "
            )
            partial_create_parquet_references = partial(
                create_single_parquet_store,
                output_directory=output_directory,
                cache_size=cache_size,
                record_size=record_size,
                overwrite_output=overwrite_output,
                dry_run=dry_run,
                verbose=verbose,
            )
            pool.map(partial_create_parquet_references, input_file_paths)
        if verbose:
            print(f"Done!")

    except Exception as e:
        print(f"Failed creating multiple Parquet stores !")
        raise

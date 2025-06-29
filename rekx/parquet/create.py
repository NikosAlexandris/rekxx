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
    DEFAULT_RECORD_SIZE,
)
from rekx.log import logger


def create_parquet_store(
    input_file: Path,
    output_parquet_store: Path,
    record_size: int = DEFAULT_RECORD_SIZE,
):
    """ """
    log_messages = []
    log_messages.append("Logging execution of create_parquet_store()")
    output_parquet_store.mkdir(parents=True, exist_ok=True)

    try:
        log_messages.append(f"Creating a filesystem mapper for {output_parquet_store}")
        filesystem = fsspec.filesystem("file")
        output = LazyReferenceMapper.create(
            root=str(output_parquet_store),  # does not handle Path
            fs=filesystem,
            record_size=record_size,
        )
        log_messages.append(f"Created the filesystem mapper {output}")

        log_messages.append(f"Kerchunking the file {input_file}")
        single_zarr = SingleHdf5ToZarr(str(input_file), out=output)
        single_zarr.translate()
        log_messages.append(f"Kerchunked the file {input_file}")

    except Exception as e:
        print(f"Failed processing file [code]{input_file}[/code] : {e}")
        log_messages.append(f"Exception occurred: {e}")
        log_messages.append("Traceback (most recent call last):")

        tb_lines = traceback.format_exc().splitlines()
        for line in tb_lines:
            log_messages.append(line)

        raise

    finally:
        logger.info("\n".join(log_messages))

    logger.info(f"Returning a Parquet store : {output_parquet_store}")
    return output_parquet_store


def create_single_parquet_store(
    input_file_path,
    output_directory,
    record_size: int = DEFAULT_RECORD_SIZE,
    verbose: int = 0,
):
    """Helper function for create_multiple_parquet_stores()"""
    filename = input_file_path.stem
    single_parquet_store = output_directory / f"{filename}.parquet"
    create_parquet_store(
        input_file_path,
        output_parquet_store=single_parquet_store,
        record_size=record_size,
    )
    if verbose > 0:
        print(f"  [code]{single_parquet_store}[/code]")

    if verbose > 1:
        dataset = xr.open_dataset(
            str(single_parquet_store),
            engine="kerchunk",
            storage_options=dict(remote_protocol="file"),
        )
        print(dataset)


def create_multiple_parquet_stores(
    source_directory: Path,
    output_directory: Path,
    pattern: str = "*.nc",
    record_size: int = DEFAULT_RECORD_SIZE,
    workers: int = 4,
    verbose: int = 0,
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
    output_directory.mkdir(parents=True, exist_ok=True)
    with multiprocessing.Pool(processes=workers) as pool:
        print(
            f"Creating the following Parquet stores in [code]{output_directory}[/code] : "
        )
        partial_create_parquet_references = partial(
            create_single_parquet_store,
            output_directory=output_directory,
            record_size=record_size,
            verbose=verbose,
        )
        pool.map(partial_create_parquet_references, input_file_paths)
    if verbose:
        print(f"Done!")

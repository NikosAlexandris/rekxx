import shutil
import xarray as xr
import dask.bag
from pathlib import Path
import fsspec
from fsspec.implementations.reference import LazyReferenceMapper
from kerchunk.combine import MultiZarrToZarr
from rich import print
from typing_extensions import Annotated
from rekx.parquet.create import (
    create_multiple_parquet_stores,
    create_single_parquet_store,
)
from rekx.constants import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_RECORD_SIZE,
    DRY_RUN_DEFAULT,
    OVERWRITE_OUTPUT_DEFAULT,
    VERBOSE_LEVEL_DEFAULT,
)
from rekx.typer.parameters import (
    typer_argument_kerchunk_combined_reference,
    typer_argument_source_directory,
    typer_option_filename_pattern,
    typer_option_cache_size,
    typer_option_record_size,
    typer_option_number_of_workers,
    typer_option_overwrite_output,
    typer_option_dry_run,
    typer_option_verbose,
)


def parquet_reference(
    input_file: Path,
    output_directory: Path = Path("."),
    record_size: int = DEFAULT_RECORD_SIZE,
    overwrite_output: bool = OVERWRITE_OUTPUT_DEFAULT,
    dry_run: Annotated[bool, typer_option_dry_run] = DRY_RUN_DEFAULT,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
):
    """Create Parquet references from an HDF5/NetCDF file"""
    filename = input_file.stem
    output_parquet_store = output_directory / f"{filename}.parquet"

    if dry_run:
        print(f"[bold]Dry running operations that would be performed[/bold]:")
        print(
            f"> Creating Parquet references to [code]{input_file}[/code] in [code]{output_parquet_store}[/code]"
        )
        return  output_parquet_store

    parquet_store = create_single_parquet_store(
        input_file_path=input_file,
        output_directory=output_directory,
        record_size=record_size,
        overwrite_output=overwrite_output,
        dry_run=dry_run,
        verbose=verbose,
    )

    # return output_parquet_store
    return parquet_store


def parquet_multi_reference(
    source_directory: Annotated[Path, typer_argument_source_directory],
    output_directory: Path = Path("."),
    pattern: Annotated[str, typer_option_filename_pattern] = "*.nc",
    cache_size: Annotated[int, typer_option_cache_size] = DEFAULT_CACHE_SIZE,
    record_size: Annotated[int, typer_option_record_size] = DEFAULT_RECORD_SIZE,
    workers: Annotated[int, typer_option_number_of_workers] = 4,
    overwrite_output: Annotated[bool, typer_option_overwrite_output] = OVERWRITE_OUTPUT_DEFAULT,
    dry_run: Annotated[bool, typer_option_dry_run] = DRY_RUN_DEFAULT,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
):
    """Create Parquet references from an HDF5/NetCDF file"""
    input_file_paths = list(source_directory.glob(pattern))

    if not input_file_paths:
        print("No files found in the source directory matching the pattern.")
        return

    if dry_run:
        print(f"[bold]Dry running operations that would be performed[/bold]:")
        print(
            f"> Reading files in [code]{source_directory}[/code] matching the pattern [code]{pattern}[/code]"
        )
        print(f"> Number of files matched : {len(input_file_paths)}")
        print(f"> Creating Parquet stores in [code]{output_directory}[/code]")
        return  # Exit for a dry run

    create_multiple_parquet_stores(
        source_directory=source_directory,
        output_directory=output_directory,
        pattern=pattern,
        cache_size=cache_size,
        record_size=record_size,
        workers=workers,
        overwrite_output=overwrite_output,
        dry_run=dry_run,
        verbose=verbose,
    )


def generate_parquet_reference(
    source_directory: Annotated[Path, typer_argument_source_directory],
    output_parquet_store: Annotated[
        Path, typer_argument_kerchunk_combined_reference
    ] = "combined_kerchunk.parquet",
    pattern: Annotated[str, typer_option_filename_pattern] = "*.nc",
    cache_size: Annotated[int, typer_option_cache_size] = DEFAULT_CACHE_SIZE,
    record_size: Annotated[int, typer_option_record_size] = DEFAULT_RECORD_SIZE,
    workers: Annotated[int, typer_option_number_of_workers] = 4,
    overwrite_output: Annotated[bool, typer_option_overwrite_output] = OVERWRITE_OUTPUT_DEFAULT,
    dry_run: Annotated[bool, typer_option_dry_run] = DRY_RUN_DEFAULT,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
):
    """
    Generate Kerchunk references for CESM output.
    """
    if dry_run:
        print(
            f"[bold]Dry run[/bold] of [bold]operations that would be performed[/bold]:"
        )

    # Get list of input data files
    input_file_paths = list(source_directory.glob(pattern))
    # input_file_paths = list(map(str, input_file_paths))
    input_file_paths.sort()

    if output_parquet_store.exists():
        if not overwrite_output:
            print(
                f"A directory named '{output_parquet_store}' exists already. Process abord. You may wish to retry by adding ``--overwrite_output` ?"
            )
            return  # Exit the function without writing the file

    # Create temporary directory for intermediate files
    temporary_path = output_parquet_store.parent / "intermediates.tmp"
    temporary_path.mkdir(exist_ok=True)

    try:
        # Parallelize generating references using dask.bag

         # Create bag and map with additional parameters
        bag = dask.bag.from_sequence(
            seq=input_file_paths,
            npartitions=len(input_file_paths),
        ).map(
            parquet_reference,
            output_directory=temporary_path,  # Pass as keyword argument
            record_size=record_size,           # Pass as keyword argument
            overwrite_output=overwrite_output, # Pass as keyword argument
            dry_run=dry_run,                   # Pass as keyword argument
            verbose=verbose                    # Pass as keyword argument
        )
        input_references = bag.compute()

        # create the output directory if needed
        output_parquet_store = output_parquet_store.parent / output_parquet_store.name
        output_parquet_store.mkdir(parents=True, exist_ok=True)
        filesystem = fsspec.filesystem("file")

        # Create LazyReferenceMapper to pass to MultiZarrToZarr
        lazy_output = LazyReferenceMapper.create(
            root=str(output_parquet_store),
            fs=filesystem,
            cache_size=cache_size,
            record_size=record_size,
            engine="fastparquet",
        )

        # Combine multiple Zarr references (one per file) to
        # a single aggregate reference file
        multi_zarr = MultiZarrToZarr(
            input_references,
            # inline_threshold=5000,
            remote_protocol="file",
            concat_dims=["time"],
            identical_dims=["lat", "lon"],
            coo_map={"time": "cf:time"},
            out=lazy_output,
        )
        multi_zarr.translate()

        if verbose > 1:
            dataset = xr.open_dataset(
                str(output_parquet_store),  # does not handle Path
                engine="kerchunk",
                storage_options=dict(remote_protocol="file", chunks={}),
                # storage_options=dict(skip_instance_cache=True, remote_protocol="file"),
            )
            print(dataset)

    except Exception:
        print(f"Failed creating the [code]{output_parquet_store}[/code] !")
        import traceback

        traceback.print_exc()

    # finally:
    #     # Cleanup intermediates
    #     for f in temporary_path.glob("temporary_*.parquet"):
    #         print(f"file : {f}")
    #         shutil.rmtree(f)
    #     shutil.rmtree(temporary_path)

import shutil
from pathlib import Path
import typer
import xarray as xr
import fsspec
from fsspec.implementations.reference import LazyReferenceMapper
from kerchunk.combine import MultiZarrToZarr
from rich import print
from typing_extensions import Annotated
from rekx.constants import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_RECORD_SIZE,
    OVERWRITE_OUTPUT_DEFAULT,
    DRY_RUN_DEFAULT,
    VERBOSE_LEVEL_DEFAULT,
)
from rekx.progress import DisplayMode, display_context
from rekx.typer.parameters import (
    typer_argument_kerchunk_combined_reference,
    typer_argument_source_directory,
    typer_option_filename_pattern,
    typer_option_cache_size,
    typer_option_record_size,
    typer_option_overwrite_output,
    typer_option_dry_run,
    typer_option_verbose,
)
from pathlib import Path


def combine_multiple_parquet_stores(
    source_directory: Path,
    output_parquet_store: Path,
    pattern: str = "*.parquet",
    cache_size: int = DEFAULT_CACHE_SIZE,
    record_size: int = DEFAULT_RECORD_SIZE,
    overwrite_output: bool = OVERWRITE_OUTPUT_DEFAULT,
    dry_run: bool = False,
    verbose: int = VERBOSE_LEVEL_DEFAULT,
):
    """

    Parameters
    ----------
    - record_size : int
        * for disk/storage efficiency
    - cache_size : int
        * for memory/access efficiency

    """
    if dry_run:
        print(
            f"[bold]Dry run[/bold] of [bold]operations that would be performed[/bold]:"
        )
    input_references = list(source_directory.glob(pattern))
    input_references = list(map(str, input_references))
    input_references.sort()

    if verbose:
        print(
            f"> Reading references in [code]{source_directory}[/code] matching the pattern [code]{pattern}[/code]"
        )
        print(f"> Number of references matched: {len(input_references)}")

        print(
            f"> Writing combined reference to [code]{output_parquet_store}[/code]"
        )

    if not dry_run:
        # Write chunked data as a NetCDF file
        if output_parquet_store.exists():
            if not overwrite_output:
                print(f"A store named '{output_parquet_store}' exists already. Process abord. You may wish to retry by adding ``--overwrite_output` ?")
                return  # Exit the function without writing the file
        try:
            output_parquet_store = output_parquet_store.parent / output_parquet_store.name
            output_parquet_store.mkdir(parents=True, exist_ok=True)
            filesystem = fsspec.filesystem("file")

            # Create LazyReferenceMapper to pass to MultiZarrToZarr
            lazy_output = LazyReferenceMapper.create(
                root=str(output_parquet_store),
                fs=filesystem,
                cache_size=cache_size,
                record_size=record_size,
                engine='fastparquet',
            )
            # Combine to single references
            multi_zarr = MultiZarrToZarr(
                input_references,
                remote_protocol="file",
                concat_dims=["time"],
                identical_dims=["lat", "lon"],
                coo_map={"time": "cf:time"},
                out=lazy_output,
            )
            multi_zarr.translate()
            lazy_output.flush()

            if verbose:
                print(f"\n[bold]> Combined Parquet store name :[/bold] {output_parquet_store}")
            return output_parquet_store

        except Exception as e:
            print(f"Failed creating the [code]{output_parquet_store}[/code] : {e}!")
            import traceback

            traceback.print_exc()
            # return


def combine_parquet_stores_to_parquet(
    source_directory: Annotated[Path, typer_argument_source_directory],
    combined_reference: Annotated[
        Path, typer_argument_kerchunk_combined_reference
    ] = "combined_kerchunk.parquet",
    pattern: Annotated[str, typer_option_filename_pattern] = "*.parquet",
    cache_size: Annotated[int, typer_option_cache_size] = DEFAULT_CACHE_SIZE,
    record_size: Annotated[int, typer_option_record_size] = DEFAULT_RECORD_SIZE,
    overwrite_output: Annotated[bool, typer_option_overwrite_output] = OVERWRITE_OUTPUT_DEFAULT,
    remove_source_directory: Annotated[bool, typer.Option(help="Remove source directory of multiple single Parquet reference stores. [yellow]Convenience option[/yellow]")] = False,
    dry_run: Annotated[bool, typer_option_dry_run] = DRY_RUN_DEFAULT,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
):
    """Combine multiple Parquet stores into a single aggregate dataset using Kerchunk's `MultiZarrToZarr` function"""

    display_mode = DisplayMode(verbose)
    with display_context[display_mode]:
        try:
            output_parquet_store = combine_multiple_parquet_stores(
                    source_directory=source_directory,
                    output_parquet_store=combined_reference,
                    pattern=pattern,
                    cache_size=cache_size,
                    record_size=record_size,
                    overwrite_output=overwrite_output,
                    dry_run=dry_run,
                    verbose=verbose,
            )
            if verbose > 1:
                dataset = xr.open_dataset(
                    str(output_parquet_store),  # does not handle Path
                    engine="kerchunk",
                    storage_options=dict(remote_protocol="file", chunks = {})
                    # storage_options=dict(skip_instance_cache=True, remote_protocol="file"),
                )
                print(dataset)

        # return output_parquet_store

        except Exception as e:
            print(f"Failed creating the [code]{combined_reference}[/code] : {e}!")
            import traceback
            traceback.print_exc()

        finally:
            if remove_source_directory:
                shutil.rmtree(source_directory)

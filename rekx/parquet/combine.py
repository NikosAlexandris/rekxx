from rekx.log import logger
import shutil
from pathlib import Path
import fsspec
import typer
import xarray as xr
from fsspec.implementations.reference import LazyReferenceMapper
from kerchunk.combine import MultiZarrToZarr
from rich import print
from typing_extensions import Annotated
from rekx.constants import DRY_RUN_DEFAULT
from rekx.constants import (
    DEFAULT_RECORD_SIZE,
    VERBOSE_LEVEL_DEFAULT,
)
from rekx.progress import DisplayMode, display_context
from rekx.typer_parameters import (
    typer_argument_kerchunk_combined_reference,
    typer_argument_source_directory,
    typer_option_filename_pattern,
    typer_option_verbose,
)
import dask
from dask.delayed import delayed
from pathlib import Path


@delayed
def copy_path(source_path: Path, target_path: Path) -> Path:
    # Create a copy in temp directory instead of using original
    temporary_name = f"copy_{hash(str(source_path))}.parquet"
    temporary_path = target_path / temporary_name
    logger.debug(f" Copying [code]{source_path}[/code] to [code]{temporary_path}[/code]")
    shutil.copytree(source_path, temporary_path)
    return temporary_path
        

def _combine_pair_of_parquet_stores(
    pair_of_parquet_stores: list[Path],
    output_parquet_store: Path,
    pattern: str = "*.parquet",
    record_size: int | None = DEFAULT_RECORD_SIZE,
    mode: str = 'w-',
    overwrite_output: bool = False,
    dry_run: bool = False,
    verbose: int = VERBOSE_LEVEL_DEFAULT,
):
    """
    """
    if dry_run:
        print(
            f"[bold]Dry run[/bold] of [bold]operations that would be performed[/bold]:"
        )

    if verbose:
        print(
            f"> Reading pair of references : [code]{pair_of_parquet_stores}[/code]"
        )
        print(
            f"> Writing combined reference to [code]{output_parquet_store}[/code]"
        )

    if not dry_run:

        try:
            # if overwrite_output and output_parquet_store.exists():
            #     import shutil
            #     shutil.rmtree(output_parquet_store)
            output_parquet_store = output_parquet_store.parent / output_parquet_store.name
            output_parquet_store.mkdir(parents=True, exist_ok=True)
            filesystem = fsspec.filesystem("file")

            # Create LazyReferenceMapper to pass to MultiZarrToZarr
            lazy_output = LazyReferenceMapper.create(
                root=str(output_parquet_store),
                fs=filesystem,
                record_size=record_size,
            )
            # Combine to single references
            input_references = list(map(str, pair_of_parquet_stores))
            input_references.sort()
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


def combine_multiple_parquet_stores(
    source_directory: Path,
    output_parquet_store: Path,
    pattern: str = "*.parquet",
    record_size: int | None = DEFAULT_RECORD_SIZE,
    dry_run: bool = False,
    verbose: int = VERBOSE_LEVEL_DEFAULT,
):
    """
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

        try:
            output_parquet_store = output_parquet_store.parent / output_parquet_store.name
            output_parquet_store.mkdir(parents=True, exist_ok=True)
            filesystem = fsspec.filesystem("file")

            # Create LazyReferenceMapper to pass to MultiZarrToZarr
            lazy_output = LazyReferenceMapper.create(
                root=str(output_parquet_store),
                fs=filesystem,
                record_size=record_size,
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


@delayed
def combine_pair(
        pair_of_parquet_stores: list[Path],
        output_parquet_store: Path,
        record_size: int | None = DEFAULT_RECORD_SIZE,
        mode: str = 'w-',
        overwrite_output: bool = False,
        dry_run: bool = DRY_RUN_DEFAULT,
        verbose: int = VERBOSE_LEVEL_DEFAULT,
) -> Path:
    """
    Combine two Parquet files into a single intermediate file
    """
    intermediate_path = output_parquet_store / f"temp_{hash(tuple(pair_of_parquet_stores))}.parquet"
    if not dry_run:
        _combine_pair_of_parquet_stores(
            pair_of_parquet_stores=pair_of_parquet_stores,
            output_parquet_store=intermediate_path,
            record_size=record_size,
            mode=mode,
            overwrite_output=overwrite_output,
            dry_run=dry_run,
            verbose=verbose
        )
        return intermediate_path


def tree_reduction(
    source_directory: Path,
    output_parquet_store: Path,
    pattern: str = "*.parquet",
    record_size: int | None = DEFAULT_RECORD_SIZE,
    dry_run: bool = False,
    verbose: int = VERBOSE_LEVEL_DEFAULT,
):
    """
    Recursive tree reduction to combine files in parallel
    """

    reference_file_paths = sorted(source_directory.glob(pattern))
    logger.debug(f" {len(reference_file_paths)} files matching pattern '{pattern}'")

    for i, f in enumerate(reference_file_paths):
        logger.debug(f" File {i}: {f}")
    
    # Special case: single file
    if len(reference_file_paths) == 1:
        print("DEBUG: Single file case - creating copy instead of moving original")
        single_input_reference = reference_file_paths[0]
        
        return copy_path(single_input_reference, output_parquet_store)
    
    logger.debug(f" Multiple files case - starting tree reduction")

    while len(reference_file_paths) > 1:
        logger.debug(f" Tree reduction level with {len(reference_file_paths)} files")
        new_level = []
        
        for i in range(0, len(reference_file_paths), 2):
            if i + 1 < len(reference_file_paths):
                pair = reference_file_paths[i:i+2]
                logger.debug(f" Combining pair {i//2}: {[str(p) for p in pair]}")
                combined = combine_pair(
                    pair_of_parquet_stores=pair,
                    output_parquet_store=output_parquet_store,
                    record_size=record_size,
                    dry_run=dry_run,
                    verbose=verbose
                )
                new_level.append(combined)

            else:
                # FIXED: Copy odd file to temp instead of pass-through
                logger.debug(f" Odd file out: {reference_file_paths[i]} → creating temp copy")
                single_copy = copy_path(
                    source_path=reference_file_paths[i],
                    target_path=output_parquet_store
                )
                new_level.append(single_copy)

        
        reference_file_paths = new_level
        logger.debug(f" Next level will have {len(reference_file_paths)} items")
    
    logger.debug(f"Tree reduction complete, returning final delayed object {reference_file_paths}")
    return reference_file_paths[0]


def combine_parquet_stores_to_parquet(
    source_directory: Annotated[Path, typer_argument_source_directory],
    pattern: Annotated[str, typer_option_filename_pattern] = "*.parquet",
    combined_reference: Annotated[
        Path, typer_argument_kerchunk_combined_reference
    ] = "combined_kerchunk.parquet",
    record_size: int = DEFAULT_RECORD_SIZE,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Run the command without making any changes."),
    ] = False,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
):
    """Combine multiple Parquet stores into a single aggregate dataset using Kerchunk's `MultiZarrToZarr` function"""

    mode = DisplayMode(verbose)
    with display_context[mode]:
        try:
            output_parquet_store = combine_multiple_parquet_stores(
                    source_directory=source_directory,
                    output_parquet_store=combined_reference,
                    pattern=pattern,
                    record_size=record_size,
                    dry_run=dry_run,
                    verbose=verbose,
            )

        # return output_parquet_store

        except Exception as e:
            print(f"Failed creating the [code]{combined_reference}[/code] : {e}!")
            import traceback

            traceback.print_exc()

        if verbose > 1:
            dataset = xr.open_dataset(
                str(output_parquet_store),  # does not handle Path
                engine="kerchunk",
                storage_options=dict(remote_protocol="file", chunks = {})
                # storage_options=dict(skip_instance_cache=True, remote_protocol="file"),
            )
            print(dataset)


def combine_pair_wise_parquet_stores_to_parquet(
    source_directory: Annotated[Path, typer_argument_source_directory],
    pattern: Annotated[str, typer_option_filename_pattern] = "*.parquet",
    combined_reference: Annotated[
        Path, typer_argument_kerchunk_combined_reference
    ] = "combined_kerchunk.parquet",
    record_size: int = DEFAULT_RECORD_SIZE,
    mode: Annotated[ str, typer.Option(help="Writing file mode")] = 'w-',
    overwrite_output: Annotated[bool, typer.Option(help="Overwrite existing output file")] = False,
    workers: Annotated[int, typer.Option(help="Number of worker processes.")] = 4,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Run the command without making any changes."),
    ] = False,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
):
    """
    Tree-based parallel combination of Parquet stores

    """
    reference_file_paths = sorted(source_directory.glob(pattern))
    
    if dry_run:
        print(f"Would combine {len(reference_file_paths)} files using tree reduction")
        return

    # Create temporary directory for intermediate files
    temporary_path = combined_reference.parent / "intermediates.tmp"
    temporary_path.mkdir(exist_ok=True)

    # Configure Dask
    with dask.config.set(scheduler='threads', num_workers=workers):
        # Build computation graph
        final_file = tree_reduction(
            source_directory=source_directory,
            output_parquet_store=temporary_path,
            record_size=record_size,
            dry_run=dry_run,
            verbose=verbose
        )
        
        # Execute computation
        result_path = final_file.compute()
    
    # Move final result to desired location
    if combined_reference.exists() and overwrite_output:
        logger.warning(f"Overwriting exising reference {combined_reference}!")
        shutil.rmtree(combined_reference)
    result_path.rename(combined_reference)
    
    # Cleanup intermediates
    for f in temporary_path.glob("temporary_*.parquet"):
        print(f"file : {f}")
        shutil.rmtree(f)
    shutil.rmtree(temporary_path)

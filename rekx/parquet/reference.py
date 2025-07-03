from pathlib import Path
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
        return  # Exit for a dry run

    create_single_parquet_store(
        input_file_path=input_file,
        output_directory=output_directory,
        record_size=record_size,
        verbose=verbose,
    )


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

"""
Rekx is a command line interface to Kerchunk
"""

from typing import Annotated

import typer
from rich.panel import Panel

from rekx.clip import clip_netcdf_file_cli
from rekx.messages import NOT_IMPLEMENTED_CLI

from rekx.combine import combine_kerchunk_references, combine_kerchunk_references_to_parquet
from rekx.consistency import check_chunk_consistency_json
from rekx.inspect import inspect_netcdf_data
from rekx.log import initialize_logger, logger
from rekx.parquet.combine import (
    combine_parquet_stores_to_parquet,
)
from rekx.parquet.combine_pairs import (
    combine_pair_wise_parquet_stores_to_parquet_2,
    combine_pair_wise_parquet_stores_to_parquet,
)
from rekx.parquet.combine_auto import combine_multiple_parquet_stores_auto_dask
from rekx.parquet.reference import (
    generate_parquet_reference,
    parquet_reference,
    parquet_multi_reference,
)
from rekx.parquet.select import select_from_parquet
from rekx.nccopy.rechunk import (
    generate_rechunk_commands_for_multiple_netcdf,
)
from rekx.cli.rechunk import (
    modify_chunk_size,
    rechunk,
    rechunk_netcdf_files,
)
from rekx.reference import create_kerchunk_reference
from rekx.typer.rich_help_panel_names import (
    rich_help_panel_combine,
    rich_help_panel_diagnose,
    rich_help_panel_read_performance,
    rich_help_panel_rechunking,
    rich_help_panel_reference,
    rich_help_panel_select,
    rich_help_panel_select_references,
    rich_help_panel_suggest,
    rich_help_panel_zarr,
)
from rekx.convert import convert_parquet_to_zarr_store
from rekx.netcdf.convert import convert_netcdf_to_zarr_store
from rekx.read import read_performance_cli, read_performance_area_cli
from rekx.select import (
    select_fast,
    select_time_series,
    select_time_series_from_json,
)
from rekx.shapes import diagnose_chunking_shapes
from rekx.suggest import (
    suggest_chunking_shape,
    suggest_chunking_shape_alternative,
    suggest_chunking_shape_alternative_symmetrical,
)
from rekx.typer.parameters import OrderCommands, typer_option_log, typer_option_version

typer.rich_utils.Panel = Panel.fit
app = typer.Typer(
    cls=OrderCommands,
    add_completion=True,
    add_help_option=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help=f"🙾  🦖 Rekx command line interface [bold][magenta]prototype",
)


# callback


@app.callback()
def main(
    version: Annotated[bool, typer_option_version] = False,
    log: Annotated[bool, typer_option_log] = False,
):
    if log:
        initialize_logger()
        logger.info("Logging initialized")
    pass


# diagnose data structure

app.command(
    name="inspect",
    help="Inspect Xarray-supported data",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_diagnose,
)(inspect_netcdf_data)
app.command(
    name="shapes",
    help="Diagnose chunking shapes in multiple Xarray-supported data",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_diagnose,
)(diagnose_chunking_shapes)

# validate chunking in series of data

app.command(
    name="validate-json",
    help="Validate chunk size consistency along multiple Kerchunk reference files [reverse]How to get available variables?",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_diagnose,
)(check_chunk_consistency_json)

# suggest

app.command(
    name="suggest",
    no_args_is_help=True,
    help=f"Suggest a good chunking shape, [yellow]ex.[/yellow] [code]'8784,2600,2600'[/code] [reverse]Needs a review!",
    rich_help_panel=rich_help_panel_suggest,
)(suggest_chunking_shape)
app.command(
    name="suggest-alternative",
    no_args_is_help=True,
    help="Suggest a good chunking shape [red]Merge to [code]suggest",
    rich_help_panel=rich_help_panel_suggest,
)(suggest_chunking_shape_alternative)
app.command(
    name="suggest-symmetrical",
    no_args_is_help=True,
    help="Suggest a good chunking shape [red]Merge to [code]suggest",
    rich_help_panel=rich_help_panel_suggest,
)(suggest_chunking_shape_alternative_symmetrical)

# rechunk

app.command(
    name="modify-chunks",
    help=f"Modify in-place the chunk size metadata in NetCDF files {NOT_IMPLEMENTED_CLI}",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_rechunking,
)(modify_chunk_size)
app.command(
    name="rechunk",
    help=f"Rechunk NetCDF file",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_rechunking,
)(rechunk)
app.command(
    name="rechunk-multiple",
    help=f"Rechunk NetCDF file [red]Merge to [code]rechunk",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_rechunking,
)(rechunk_netcdf_files)
app.command(
    name="rechunk-generator",
    help=f"Generate variations of rechunking commands for multiple files",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_rechunking,
)(generate_rechunk_commands_for_multiple_netcdf)

# create Kerchunk reference sets

app.command(
    name="reference",
    help="Create Kerchunk JSON reference files",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_reference,
)(create_kerchunk_reference)
app.command(
    "reference-parquet",
    no_args_is_help=True,
    help=f"Create Parquet references to an HDF5/NetCDF file [red]Merge to [code]reference",
    rich_help_panel=rich_help_panel_reference,
)(parquet_reference)
app.command(
    "reference-multi-parquet",
    help=f"Create Parquet references to multiple HDF5/NetCDF files [red]Merge to [code]reference-parquet",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_reference,
)(parquet_multi_reference)

# combine Kerchunk reference sets

app.command(
    name="combine",
    help="Combine Kerchunk reference sets (JSONs to JSON)",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_combine,
)(combine_kerchunk_references)
app.command(
    name="combine-to-parquet",
    help="Combine Kerchunk reference sets into a single Parquet store (JSONs to Parquet)",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_combine,
)(combine_kerchunk_references_to_parquet)
app.command(
    "combine-parquet-stores",
    help=f"Combine multiple Parquet stores (Parquets to Parquet)",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_combine,
)(combine_parquet_stores_to_parquet)
app.command(
    "combine-pair-wise-parquet-stores",
    help=f"Combine multiple Parquet stores (Parquets to Parquet) following a pair-wise tree reduction",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_combine,
)(combine_pair_wise_parquet_stores_to_parquet)
app.command(
    "combine-pair-wise-parquet-stores-2",
    help=f"Combine multiple Parquet stores (Parquets to Parquet) following a pair-wise tree reduction",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_combine,
)(combine_pair_wise_parquet_stores_to_parquet_2)
app.command(
    "combine-parquet-stores-auto",
    help=f"Combine multiple Parquet stores (Parquets to Parquet) following a pair-wise tree reduction",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_combine,
)(combine_multiple_parquet_stores_auto_dask)

app.command(
    "parquet-to-zarr",
    help=f"Convert Parquet to local Zarr store [bold yellow reverse] Experimental ",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_zarr,
)(convert_parquet_to_zarr_store)

app.command(
    "generate-parquet",
    help=f"Generate a Parquet store with references to multiple HDF5/NetCDF files",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_combine,
)(generate_parquet_reference)

# select

app.command(
    name="clip",
    help="%< Clip time series data",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_select,
)(clip_netcdf_file_cli)

app.command(
    name="select",
    help="  Select time series over a location",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_select,
)(select_time_series)
app.command(
    name="select-fast",
    help="  Bare read time series from Xarray-supported data and optionally write to CSV [bold magenta reverse] Performance Test [/bold magenta reverse]",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_select,
)(select_fast)

app.command(
    name="select-json",
    help="  Select time series over a location from a JSON Kerchunk reference set",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_select_references,
)(select_time_series_from_json)
app.command(
    name="select-parquet",
    help=f" Select data from a Parquet references store",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_select_references,
)(select_from_parquet)
app.command(
    "netcdf-to-zarr",
    help=f"Convert multi-NetCDF to Zarr store",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_zarr,
)(convert_netcdf_to_zarr_store)

# read and load in memory for performance assessment

app.command(
    name="read-performance",
    help="  Measure read & load in-memory operations of a point time series from Xarray-supported data",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_read_performance,
)(read_performance_cli)
app.command(
    name="read-performance-area",
    help="  Measure read & load in-memory operations of an area time series from Xarray-supported data",
    no_args_is_help=True,
    rich_help_panel=rich_help_panel_read_performance,
)(read_performance_area_cli)


if __name__ == "__main__":
    app()

from rich import print
import typer
from typer.core import TyperGroup

from rekx.constants import (
    LATITUDE_MAXIMUM,
    LATITUDE_MINIMUM,
    LONGITUDE_MAXIMUM,
    LONGITUDE_MINIMUM,
)
from rekx.typer.rich_help_panel_names import (
    rich_help_panel_input_options,
    rich_help_panel_output_options,
    rich_help_panel_dask_configuration,
    rich_help_panel_nccopy,
    rich_help_panel_chunk_sizes,
    rich_help_panel_chunking_parameters,
    rich_help_panel_advanced_options,
    rich_help_panel_output_data_structure,
    rich_help_panel_select,
    rich_help_panel_time_series,
    rich_help_panel_data_variables,
    rich_help_panel_selection_options,
)
from rekx.timestamp import callback_generate_datetime_series, parse_timestamp_series


class OrderCommands(TyperGroup):
    def list_commands(self, ctx: typer.Context):
        """Return list of commands in the order they appear.

        See also
        --------
        - https://github.com/tiangolo/typer/issues/428#issuecomment-1238866548

        """
        return list(self.commands)


# Generic


def version_callback(flag: bool) -> None:
    if flag:
        from ._version import __version__

        print(f"rekx version {__version__}")
        raise typer.Exit(code=0)


typer_option_version = typer.Option(
    "--version",
    help="Show the version and exit.",
    callback=version_callback,
    is_flag=True,
    is_eager=True,
)


typer_option_log = typer.Option(
    "--log",
    help="Enable logging.",
    is_flag=True,
    # default=False,  # Default value is False, so logging is off by default
)


typer_option_dry_run = typer.Option(
    # "--dry-run",
    help=f"[bold]Run the command without making any changes ![/bold] [yellow bold reverse] Try Me! [/yellow bold reverse]",
    # default_factory = False,
    rich_help_panel=rich_help_panel_output_options,
)

humanize_typer_help = "Convert byte sizes into human-readable formats"
# humanize_typer_help = "Flag to humanize file size"
humanize_help = humanize_typer_help
typer_option_humanize = typer.Option(
    "--humanize",
    "-h",
    help=humanize_typer_help,
    # default = False,
)

# Input & Output Paths

from pathlib import Path


def callback_source_path(input_path: Path):
    """ """
    from rich import print

    if not input_path.exists():
        print(f"[red]The path [code]{input_path}[/code] does not exist[/red].")
        raise typer.Exit()

    if not input_path.is_file() and not input_path.is_dir():
        print(f"[red]The path [code]{input_path}[/code] is not valid[/red].")
        raise typer.Exit()

    return input_path


def callback_source_directory(directory: Path):
    """ """
    if not directory.exists() or not any(directory.iterdir()):
        print(
            f"[red]The directory [code]{directory}[/code] does not exist or is empty[/red]."
        )
    return directory


def callback_source_path_with_pattern(path: Path, ctx: typer.Context):
    """ """
    if not path.exists():
        raise typer.BadParameter("Path does not exist.")

    if not path.is_file() and not path.is_dir():
        raise typer.BadParameter("Path is not a file.")

    # if not path.is_readable():
    #     raise typer.BadParameter("File is not readable.")

    return path.resolve()


typer_argument_source_path = typer.Argument(
    # show_default=True,
    help="Input filename or directory path",
    rich_help_panel=rich_help_panel_time_series,
    callback=callback_source_path,
    # default_factory = None
)
typer_argument_source_directory = typer.Argument(
    show_default=True,
    help="Source directory path",
    rich_help_panel=rich_help_panel_time_series,
    callback=callback_source_directory,
    # default_factory = None
)
typer_argument_source_path_with_pattern = typer.Argument(
    show_default=True,
    help="List source file paths",
    rich_help_panel=rich_help_panel_time_series,
    callback=callback_source_path_with_pattern,
    # default_factory = None
)
typer_argument_output_directory = typer.Argument(
    show_default=True,
    help="Output directory path for reference files. Will create if inexistent.",
    rich_help_panel=rich_help_panel_time_series,
    # default_factory = None
)
typer_option_output_directory = typer.Option(
    show_default=True,
    help="Output directory path for reference files. Will create if inexistent.",
    rich_help_panel=rich_help_panel_output_options,
    # default_factory = None
)
typer_argument_kerchunk_combined_reference = typer.Argument(
    show_default=True,
    help="Combined kerchunk reference file name",
    rich_help_panel=rich_help_panel_time_series,
    # default_factory = None
)
typer_option_filename_pattern = typer.Option(
    help="Filename pattern to match",
    rich_help_panel=rich_help_panel_input_options,
)
typer_option_overwrite_output = typer.Option(
    help="Overwrite existing output",
    rich_help_panel=rich_help_panel_output_options,
)
typer_option_number_of_workers = typer.Option(
    help="Number of workers for parallel processing using `concurrent.futures`",
    rich_help_panel=rich_help_panel_advanced_options,
)


# # Time series

def parse_input_time_series(time_series: str):
    """
    """
    return Path(time_series).as_posix()


time_series_typer_help = "A time series dataset (any format supported by Xarray)"
typer_argument_time_series = typer.Argument(
    # show_default=True,
    help=time_series_typer_help,
    parser=parse_input_time_series,
    # rich_help_panel=rich_help_panel_time_series,
)
typer_option_time_series = typer.Option(
    # show_default=True,
    help=time_series_typer_help,
    rich_help_panel=rich_help_panel_time_series,
)

# Where?

longitude_typer_help = f"Longitude in decimal degrees ranging in [-180, 360]. [yellow]If ranging in [0, 360], consider the `--convert-longitude-360` option.[/yellow]"
longitude_in_degrees_help = longitude_typer_help
typer_argument_longitude_in_degrees = typer.Argument(
    help=longitude_typer_help,
    min=LONGITUDE_MINIMUM,
    max=LONGITUDE_MAXIMUM,
)
typer_option_longitude_in_degrees = typer.Option(
    help=longitude_typer_help,
    min=LONGITUDE_MINIMUM,
    max=LONGITUDE_MAXIMUM,
    rich_help_panel=rich_help_panel_selection_options,
)
typer_option_x_index = typer.Option(
    help="X coordinate index",
)

latitude_typer_help = "Latitude in decimal degrees ranging in [-90, 90]"
latitude_in_degrees_help = latitude_typer_help
typer_argument_latitude_in_degrees = typer.Argument(
    help=latitude_typer_help,
    min=LATITUDE_MINIMUM,
    max=LATITUDE_MAXIMUM,
)
typer_option_latitude_in_degrees = typer.Option(
    help=latitude_typer_help,
    min=LATITUDE_MINIMUM,
    max=LATITUDE_MAXIMUM,
    rich_help_panel=rich_help_panel_selection_options,
)
typer_option_y_index = typer.Option(
    help="Y coordinate index",
)

# When?

# timestamp_typer_help = "Quoted date-time string of data to extract from series, example: [yellow]'2112-12-21 21:12:12'[/yellow]'"
# typer_argument_timestamp = typer.Argument(
#     help=timestamp_typer_help,
#     callback=ctx_attach_requested_timezone,
#     # rich_help_panel=rich_help_panel_time_series,
#     default_factory=now_utc_datetimezone,
# )
timestamps_typer_help = "Quoted date-time strings of data to extract from series, example: [yellow]'2112-12-21, 2112-12-21 12:21:21, 2112-12-21 21:12:12'[/yellow]'"
typer_argument_timestamps = typer.Argument(
    help=timestamps_typer_help,
    parser=parse_timestamp_series,
    callback=callback_generate_datetime_series,
    #     default_factory=now_utc_datetimezone_series,
)
# typer_option_timestamps = typer.Option(
#     help='Timestamps',
#     parser=parse_timestamp_series,
#     callback=callback_generate_datetime_series,
# #     default_factory=now_utc_datetimezone_series,
# )
typer_option_start_time = typer.Option(
    help=f"Start timestamp of the period. [yellow]Overrides the `timestamps` paramter![/yellow]",
    rich_help_panel=rich_help_panel_selection_options,
    # default_factory=None,
)
# typer_option_frequency = typer.Option(
#     help=f'Frequency for the timestamp generation function',
#     rich_help_panel=rich_help_panel_time_series,
#     # default_factory='h'
# )
typer_option_end_time = typer.Option(
    help="End timestamp of the period. [yellow]Overrides the `timestamps` paramter![/yellow]",
    rich_help_panel=rich_help_panel_selection_options,
    # default_factory=None,
)
typer_option_every_nth_timestamp = typer.Option(
    help="Select every n-th timestamps, i.e. if input is every 30', setting to 2 will derive an hourly dataset ",
    rich_help_panel=rich_help_panel_selection_options,
)

# Data variables

typer_argument_variable = typer.Argument(
    show_default=True,
    help=f"Variables to select data from",
    rich_help_panel=rich_help_panel_data_variables,
    # default_factory=False,
)
typer_option_variable_set = typer.Option(
    show_default=True,
    help="Set of Xarray variables to diagnose",
    rich_help_panel=rich_help_panel_data_variables,
    # default_factory=False,
)
typer_option_list_variables = typer.Option(
    show_default=True,
    help=f"List variables in the Kerchunk reference set",
    rich_help_panel=rich_help_panel_data_variables,
    # default_factory=False,
)
typer_option_drop_other_variables = typer.Option(
    help="Drop variables other than the main one. [yellow bold]Attention, presets are the author's best guess![/yellow bold]",
    rich_help_panel=rich_help_panel_data_variables,
)

# Input options

typer_option_mask_and_scale = typer.Option(
    help="Mask and scale the series",
    rich_help_panel=rich_help_panel_input_options,
    # default_factory=False,
)
typer_option_neighbor_lookup = typer.Option(
    help="Enable nearest neighbor (inexact) lookups. Read Xarray manual on [underline]nearest-neighbor-lookups[/underline]",
    show_default=True,
    show_choices=True,
    case_sensitive=False,
    rich_help_panel=rich_help_panel_select,
    # default_factory=None, # default_factory=MethodsForInexactMatches.nearest,
)
typer_option_tolerance = typer.Option(
    help=f"Maximum distance between original & new labels for inexact matches. Read Xarray manual on [underline]nearest-neighbor-lookups[/underline]",
    #  https://docs.xarray.dev/en/stable/user-guide/indexing.html#nearest-neighbor-lookups',
    rich_help_panel=rich_help_panel_select,
    # default_factory=0.1,
)
repetitions_typer_help = (
    "Times to repeat the read operation to calculate the median read time."
)
# repetitions_typer_help = "Number of repetitions for read operation"
repetitions_help = repetitions_typer_help
typer_option_repetitions = typer.Option(
    help=repetitions_help,
    rich_help_panel=rich_help_panel_select,
    # default = 10,
)

# Arrays & Chunks

typer_argument_time_chunk_size = typer.Argument(
    help="New chunk size for the `time` dimension",
    rich_help_panel=rich_help_panel_chunk_sizes,
)
typer_argument_latitude_chunk_size = typer.Argument(
    help="New chunk size for the `lat`itude dimension",
    rich_help_panel=rich_help_panel_chunk_sizes,
)
typer_argument_longitude_chunk_size = typer.Argument(
    help="New chunk size for the `lon`gitude dimension",
    rich_help_panel=rich_help_panel_chunk_sizes,
)


def parse_variable_shape(variable_shape: str):
    if isinstance(variable_shape, str):
        return [int(dimension) for dimension in variable_shape.split(",")]


typer_option_fix_unlimited_dimensions = typer.Option(
    help="Convert unlimited size input dimensions to fixed size dimensions in output.",
    rich_help_panel=rich_help_panel_chunking_parameters,
)

typer_argument_variable_shape = typer.Argument(
    help="Variable shape",
    parser=parse_variable_shape,
    rich_help_panel=rich_help_panel_chunking_parameters,
    # default=None,
)
typer_option_record_size = typer.Option(
    help="Number of records in a reference file",
    rich_help_panel=rich_help_panel_chunking_parameters,
)
typer_option_cache_size = typer.Option(
    help="Numer of reference sets (files, like in a Parquet format) cached in memory",
    rich_help_panel=rich_help_panel_chunking_parameters,
)

# Backend

typer_option_rechunking_backend = typer.Option(
    help="Backend to use for rechunking. [code]nccopy[/code] [red]Not Implemented Yet![/red]",
    rich_help_panel=rich_help_panel_chunking_parameters,
)

# NetCDF / Generic data structure options

typer_option_compresion = typer.Option(
    help="Compression filter, applicable to all backends",
    rich_help_panel=rich_help_panel_output_data_structure,
)
typer_option_compresion_level = typer.Option(
    help="Compression level, applicable to all backends",
    rich_help_panel=rich_help_panel_output_data_structure,
)
typer_option_shuffling = typer.Option(
    help="Shuffling, applicable to all backends",
    rich_help_panel=rich_help_panel_output_data_structure,
)

# nccopy paramters and options

typer_option_nccopy_cache_size = typer.Option(
    help="[yellow bold]Applicable to `nccopy`[/yellow bold]",
    rich_help_panel=rich_help_panel_nccopy,
)
typer_option_nccopy_cache_elements = typer.Option(
    help="[yellow bold]Applicable to `nccopy`[/yellow bold]",
    rich_help_panel=rich_help_panel_nccopy,
)
typer_option_nccopy_cache_preemption = typer.Option(
    help="[yellow bold]Applicable to `nccopy`[/yellow bold]",
    rich_help_panel=rich_help_panel_nccopy,
)
typer_option_nccopy_memory = typer.Option(
    help="[yellow bold]Applicable to `nccopy`[/yellow bold]",
    rich_help_panel=rich_help_panel_nccopy,
)

# Dask configuration

typer_option_workers = typer.Option(
    help="Number of workers",
    rich_help_panel=rich_help_panel_dask_configuration,
)
typer_option_threads_per_worker = typer.Option(
    help="Threads per worker",
    rich_help_panel=rich_help_panel_dask_configuration,
)
typer_option_memory_limit = typer.Option(
    help="Memory limit for the Dask cluster in GB. [yellow bold]Will override [code]auto-memory[/code][/yellow bold]",
    rich_help_panel=rich_help_panel_dask_configuration,
)
typer_option_auto_memory_limit = typer.Option(
    help="Dask's auto memory limit per worker",
    rich_help_panel=rich_help_panel_dask_configuration,
)

# Output options

typer_option_open_file_mode = typer.Option(
    help="Writing file mode",
    rich_help_panel=rich_help_panel_output_options,
)
typer_option_verbose = typer.Option(
    "--verbose",
    "-v",
    count=True,
    is_flag=False,
    help="Show details while executing commands",
    rich_help_panel=rich_help_panel_output_options,
    # default_factory=0,
)
typer_option_rounding_places = typer.Option(
    "--rounding-places",
    "-r",
    help="Number of digits to round results to",
    show_default=True,
    rich_help_panel=rich_help_panel_output_data_structure,
    # default_factory=5,
)

typer_option_statistics = typer.Option(
    help="Calculate and display summary statistics",
    rich_help_panel=rich_help_panel_output_data_structure,
    # default=False
)
typer_option_variable_name_as_suffix = typer.Option(
    help="Suffix the output filename with the variable name",
    rich_help_panel=rich_help_panel_output_data_structure,
    # default=False
)
typer_option_output_filename = typer.Option(
    help="Output filename [underline]with [bold]appropriate[/bold] extension[/underline] for selected data. [code].nc[/code] for NetCDF, [code].csv[/code] for CSV, [code]png[/code] for PNG",
    rich_help_panel=rich_help_panel_output_data_structure,
)
typer_option_csv = typer.Option(
    help="CSV output filename",
    rich_help_panel=rich_help_panel_output_data_structure,
    # default_factory='series_in',
)

# Helpers

typer_option_convert_longitude_360 = typer.Option(
    help="Convert range of longitude values to [0, 360]",
    rich_help_panel="Helpers",
    # default_factory=False
)
typer_option_in_memory = typer.Option(
    help="Use in-memory processing",  # You may need to customize the help text
    # default_factory=False
)

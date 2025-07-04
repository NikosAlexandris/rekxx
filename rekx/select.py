import time as timer
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import fsspec
import typer
import xarray as xr
from rich import print
from typing_extensions import Annotated

from rekx.constants import (
    VERBOSE_LEVEL_DEFAULT,
)
from rekx.models import MethodForInexactMatches
from rekx.constants import VERBOSE_LEVEL_DEFAULT
from rekx.csv import to_csv
from rekx.hardcodings import exclamation_mark
from rekx.log import logger
from rekx.messages import ERROR_IN_SELECTING_DATA
from rekx.statistics import print_series_statistics
from rekx.write import write_to_netcdf
from rekx.typer.parameters import (
    typer_argument_latitude_in_degrees,
    typer_argument_longitude_in_degrees,
    typer_argument_time_series,
    typer_argument_timestamps,
    typer_option_output_filename,
    typer_option_csv,
    typer_option_end_time,
    typer_option_in_memory,
    typer_option_list_variables,
    typer_option_mask_and_scale,
    typer_option_neighbor_lookup,
    typer_option_start_time,
    typer_option_statistics,
    typer_option_time_series,
    typer_option_tolerance,
    typer_option_verbose,
)
from .utilities import set_location_indexers


def select_fast(
    time_series: Annotated[Path, typer_argument_time_series],
    variable: Annotated[str, typer.Argument(help="Variable to select data from")],
    longitude: Annotated[float, typer_argument_longitude_in_degrees],
    latitude: Annotated[float, typer_argument_latitude_in_degrees],
    time_series_2: Annotated[Path, typer_option_time_series] = None,
    tolerance: Annotated[
        Optional[float], typer_option_tolerance
    ] = 0.1,  # Customize default if needed
    # in_memory: Annotated[bool, typer_option_in_memory] = False,
    csv: Annotated[Path, typer_option_csv] = None,
    tocsv: Annotated[Path, typer_option_csv] = None,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
):
    """Bare timing to read data over a location and optionally write
    comma-separated values.

    Parameters
    ----------
    time_series:
        Path to Xarray-supported input file
    variable: str
        Name of the variable to query
    longitude: float
        The longitude of the location to read data
    latitude: float
        The latitude of the location to read data
    time_series:
        Path to second Xarray-supported input file
    tolerance: float
        Maximum distance between original and new labels for inexact matches.
        Read Xarray manual on nearest-neighbor-lookups
    csv:
        CSV output filename
    to_csv:
        CSV output filename (fast implementation from xarray-extras)

    Returns
    -------
    data_retrieval_time : float
        An estimation of the time it took to retrieve data over the requested
        location if no verbosity is asked.
    Notes
    -----
    ``mask_and_scale`` is always set to ``False`` to avoid errors related with
    decoding timestamps.

    """
    try:
        data_retrieval_start_time = timer.perf_counter()  # time()
        series = xr.open_dataset(time_series, mask_and_scale=False)[variable].sel(
            lon=longitude, lat=latitude, method="nearest"
        )
        if time_series_2:
            series_2 = xr.open_dataset(time_series_2, mask_and_scale=False)[
                variable
            ].sel(lon=longitude, lat=latitude, method="nearest")
        if csv:
            series.to_pandas().to_csv(csv)
            if time_series_2:
                series_2.to_pandas().to_csv(csv.name + "2")
        elif tocsv:
            to_csv(
                x=series,
                path=str(tocsv),
            )
            if time_series_2:
                to_csv(x=series_2, path=str(tocsv) + "2")

        data_retrieval_time = f"{timer.perf_counter() - data_retrieval_start_time:.3f}"
        if not verbose:
            return data_retrieval_time
        else:
            print(
                f"[bold green]It worked[/bold green] and took : {data_retrieval_time}"
            )

    except Exception as e:
        print(f"An error occurred: {e}")


def select_time_series(
    time_series: Path,
    variable: Annotated[str, typer.Argument(..., help="Variable name to select from")],
    longitude: Annotated[float, typer_argument_longitude_in_degrees],
    latitude: Annotated[float, typer_argument_latitude_in_degrees],
    list_variables: Annotated[bool, typer_option_list_variables] = False,
    timestamps: Annotated[Optional[Any], typer_argument_timestamps] = None,
    start_time: Annotated[Optional[datetime], typer_option_start_time] = None,
    end_time: Annotated[Optional[datetime], typer_option_end_time] = None,
    time: Annotated[
        Optional[int], typer.Option(help="New chunk size for the 'time' dimension")
    ] = None,
    lat: Annotated[
        Optional[int], typer.Option(help="New chunk size for the 'lat' dimension")
    ] = None,
    lon: Annotated[
        Optional[int], typer.Option(help="New chunk size for the 'lon' dimension")
    ] = None,
    # convert_longitude_360: Annotated[bool, typer_option_convert_longitude_360] = False,
    mask_and_scale: Annotated[bool, typer_option_mask_and_scale] = False,
    neighbor_lookup: Annotated[
        MethodForInexactMatches, typer_option_neighbor_lookup
    ] = MethodForInexactMatches.nearest,
    tolerance: Annotated[
        Optional[float], typer_option_tolerance
    ] = 0.1,  # Customize default if needed
    in_memory: Annotated[bool, typer_option_in_memory] = False,
    statistics: Annotated[bool, typer_option_statistics] = False,
    output_filename: Annotated[
        Path|None, typer_option_output_filename
    ] = None,
    # output_filename: Annotated[Path, typer_option_output_filename] = 'series_in',  #Path(),
    # variable_name_as_suffix: Annotated[bool, typer_option_variable_name_as_suffix] = True,
    # rounding_places: Annotated[Optional[int], typer_option_rounding_places] = ROUNDING_PLACES_DEFAULT,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
) -> None:
    """
    Select data using a Kerchunk reference file

    Parameters
    ----------
    time_series:
        Path to Xarray-supported input file
    variable: str
        Name of the variable to query
    longitude: float
        The longitude of the location to read data
    latitude: float
        The latitude of the location to read data
    list_variables: bool
         Optional flag to list data variables and exit without doing anything
         else.
    timestamps: str
        A string of properly formatted timestamps to be parsed and use for
        temporal selection.
    start_time: str
        A start time to generate a temporal selection period
    end_time: str
        An end time for the generation of a temporal selection period
    time: int
        New chunk size for the 'time' dimension
    lat: int
        New chunk size for the 'lat' dimension
    lon: int
        New chunk size for the 'lon' dimension
    mask_and_scale: bool
        Flag to apply masking and scaling based on the input metadata
    neighbor_lookup: str
        Method to use for inexact matches.
    tolerance: float
        Maximum distance between original and new labels for inexact matches.
        Read Xarray manual on nearest-neighbor-lookups
    statistics: bool
        Optional flag to calculate and display summary statistics
    verbose: int
        Verbosity level

    Returns
    -------

    """
    # if convert_longitude_360:
    #     longitude = longitude % 360
    # warn_for_negative_longitude(longitude)

    logger.debug(f"Command context : {typer.Context}")

    data_retrieval_start_time = timer.time()
    logger.debug(f"Starting data retrieval... {data_retrieval_start_time}")

    timer_start = timer.time()
    dataset = xr.open_dataset(
        time_series,
        mask_and_scale=mask_and_scale,
    )  # is a dataset
    timer_end = timer.time()
    logger.debug(
        f"Dataset opening via Xarray took {timer_end - timer_start:.2f} seconds"
    )

    available_variables = list(dataset.data_vars)  # Is there a faster way ?
    if list_variables:
        logger.info(
            f"The dataset contains the following variables : `{available_variables}`."
        )
        print(
            f"The dataset contains the following variables : `{available_variables}`."
        )
        return

    if not variable in available_variables:
        logger.debug(
            f"The requested variable `{variable}` does not exist! Plese select one among the available variables : {available_variables}."
        )
        print(
            f"The requested variable `{variable}` does not exist! Plese select one among the available variables : {available_variables}."
        )
        raise typer.Exit(code=0)
    else:
        timer_start = timer.time()
        time_series = dataset[variable]
        timer_end = timer.time()
        logger.debug(
            f"Data array variable selection took {timer_end - timer_start:.2f} seconds"
        )

        timer_start = timer.time()
        chunks = {"time": time, "lat": lat, "lon": lon}
        time_series.chunk(chunks=chunks)
        timer_end = timer.time()
        logger.debug(
            f"Data array rechunking took {timer_end - timer_start:.2f} seconds"
        )

    timer_start = timer.time()
    indexers = set_location_indexers(
        data_array=time_series,
        longitude=longitude,
        latitude=latitude,
        verbose=verbose,
    )
    timer_end = timer.time()
    logger.debug(f"Data array indexers : {indexers}")
    logger.debug(
        f"Data array indexers setting took {timer_end - timer_start:.2f} seconds"
    )

    try:
        timer_start = timer.time()
        location_time_series = time_series.sel(
            **indexers,
            method=neighbor_lookup,
            tolerance=tolerance,
        )
        timer_end = timer.time()
        indentation = " " * 4 * 9
        indented_location_time_series = "\n".join(
            indentation + line for line in str(location_time_series).split("\n")
        )
        logger.debug(
            f"Location time series selection :\n{indented_location_time_series}"
        )
        logger.debug(f"Location selection took {timer_end - timer_start:.2f} seconds")

        if in_memory:
            timer_start = timer.time()
            location_time_series.load()  # load into memory for faster ... ?
            timer_end = timer.time()
            logger.debug(
                f"Location selection loading in memory took {timer_end - timer_start:.2f} seconds"
            )

    except Exception as exception:
        logger.error(f"{ERROR_IN_SELECTING_DATA} : {exception}")
        print(f"{ERROR_IN_SELECTING_DATA} : {exception}")
        raise SystemExit(33)
    # ------------------------------------------------------------------------

    if start_time or end_time:
        timestamps = None  # we don't need a timestamp anymore!

        if start_time and not end_time:  # set `end_time` to end of series
            end_time = location_time_series.time.values[-1]

        elif end_time and not start_time:  # set `start_time` to beginning of series
            start_time = location_time_series.time.values[0]

        else:  # Convert `start_time` & `end_time` to the correct string format
            start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")

        timer_start = timer.time()
        location_time_series = location_time_series.sel(
            time=slice(start_time, end_time)
        )
        timer_end = timer.time()
        logger.debug(
            f"Time slicing with `start_time` and `end_time` took {timer_end - timer_start:.2f} seconds"
        )

    if timestamps is not None and not start_time and not end_time:
        if len(timestamps) == 1:
            start_time = end_time = timestamps[0]

        try:
            timer_start = timer.time()
            location_time_series = location_time_series.sel(
                time=timestamps, method=neighbor_lookup
            )
            timer_end = timer.time()
            logger.debug(
                f"Time selection with `timestamps` took {timer_end - timer_start:.2f} seconds"
            )

        except KeyError:
            print(f"No data found for one or more of the given {timestamps}.")

    if location_time_series.size == 1:
        timer_start = timer.time()
        single_value = float(location_time_series.values)
        warning = (
            f"{exclamation_mark} The selected timestamp "
            + f"{location_time_series.time.values}"
            + f" matches the single value "
            + f"{single_value}"
        )
        timer_end = timer.time()
        logger.debug(
            f"Single value conversion to float took {timer_end - timer_start:.2f} seconds"
        )
        logger.warning(warning)
        if verbose > 0:
            print(warning)

    data_retrieval_end_time = timer.time()
    logger.debug(
        f"Data retrieval took {data_retrieval_end_time - data_retrieval_start_time:.2f} seconds"
    )

    if not verbose:
        print(location_time_series.values)
    else:
        print(location_time_series)

    if statistics:  # after echoing series which might be Long!
        print_series_statistics(
            data_array=location_time_series,
            timestamps=timestamps,
            title="Selected series",
        )
    output_handlers = {
        ".nc": lambda location_time_series, path: write_to_netcdf(
            location_time_series=location_time_series,
            path=path,
            longitude=longitude,
            latitude=latitude
        ),
        ".csv": lambda location_time_series, path: to_csv(
            x=location_time_series, path=path
        ),
    }
    if output_filename:
        extension = output_filename.suffix.lower()
        if extension in output_handlers:
            output_handlers[extension](location_time_series, output_filename)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")


def select_time_series_from_json(
    reference_file: Annotated[
        Path, typer.Argument(..., help="Path to the kerchunk reference file")
    ],
    variable: Annotated[str, typer.Argument(..., help="Variable name to select from")],
    longitude: Annotated[float, typer_argument_longitude_in_degrees],
    latitude: Annotated[float, typer_argument_latitude_in_degrees],
    list_variables: Annotated[bool, typer_option_list_variables] = False,
    timestamps: Annotated[Optional[Any], typer_argument_timestamps] = None,
    start_time: Annotated[Optional[datetime], typer_option_start_time] = None,
    end_time: Annotated[Optional[datetime], typer_option_end_time] = None,
    time: Annotated[
        Optional[int], typer.Option(help="New chunk size for the 'time' dimension")
    ] = None,
    lat: Annotated[
        Optional[int], typer.Option(help="New chunk size for the 'lat' dimension")
    ] = None,
    lon: Annotated[
        Optional[int], typer.Option(help="New chunk size for the 'lon' dimension")
    ] = None,
    # convert_longitude_360: Annotated[bool, typer_option_convert_longitude_360] = False,
    mask_and_scale: Annotated[bool, typer_option_mask_and_scale] = False,
    neighbor_lookup: Annotated[
        MethodForInexactMatches, typer_option_neighbor_lookup
    ] = None,
    tolerance: Annotated[
        Optional[float], typer_option_tolerance
    ] = 0.1,  # Customize default if needed
    in_memory: Annotated[bool, typer_option_in_memory] = False,
    statistics: Annotated[bool, typer_option_statistics] = False,
    csv: Annotated[Path, typer_option_csv] = None,
    # output_filename: Annotated[Path, typer_option_output_filename] = 'series_in',  #Path(),
    # variable_name_as_suffix: Annotated[bool, typer_option_variable_name_as_suffix] = True,
    # rounding_places: Annotated[Optional[int], typer_option_rounding_places] = ROUNDING_PLACES_DEFAULT,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
) -> None:
    """
    Select data using a Kerchunk reference file

    Parameters
    ----------
    reference_file:
        Path to an input JSON Kerchunk reference file
    variable: str
        Name of the variable to query
    longitude: float
        The longitude of the location to read data
    latitude: float
        The latitude of the location to read data
    list_variables: bool
         Optional flag to list data variables and exit without doing anything
         else.
    timestamps: str
        A string of properly formatted timestamps to be parsed and use for
        temporal selection.
    start_time: str
        A start time to generate a temporal selection period
    end_time: str
        An end time for the generation of a temporal selection period
    time: int
        New chunk size for the 'time' dimension
    lat: int
        New chunk size for the 'lat' dimension
    lon: int
        New chunk size for the 'lon' dimension
    mask_and_scale: bool
        Flag to apply masking and scaling based on the input metadata
    neighbor_lookup: str
        Method to use for inexact matches.
    tolerance: float
        Maximum distance between original and new labels for inexact matches.
        Read Xarray manual on nearest-neighbor-lookups
    in_memory: bool
        ?
    statistics: bool
        Optional flag to calculate and display summary statistics
    csv:
        CSV output filename
    verbose: int
        Verbosity level
    """
    # if convert_longitude_360:
    #     longitude = longitude % 360
    # warn_for_negative_longitude(longitude)

    # logger.debug(f'Command context : {print(typer.Context)}')

    data_retrieval_start_time = timer.time()
    logger.debug(f"Starting data retrieval... {data_retrieval_start_time}")

    timer_start = timer.time()
    mapper = fsspec.get_mapper(
        "reference://",
        fo=str(reference_file),
        remote_protocol="file",
        remote_options={"skip_instance_cache": True},
    )
    timer_end = timer.time()
    logger.debug(f"Mapper creation took {timer_end - timer_start:.2f} seconds")
    timer_start = timer.time()
    dataset = xr.open_dataset(
        mapper,
        engine="zarr",
        backend_kwargs={"consolidated": False},
        chunks=None,
        mask_and_scale=mask_and_scale,
    )  # is a dataset
    timer_end = timer.time()
    logger.debug(
        f"Dataset opening via Xarray took {timer_end - timer_start:.2f} seconds"
    )

    available_variables = list(dataset.data_vars)  # Is there a faster way ?
    if list_variables:
        print(
            f"The dataset contains the following variables : `{available_variables}`."
        )
        return

    if not variable in available_variables:
        logger.error(
            f"The requested variable `{variable}` does not exist! Plese select one among the available variables : {available_variables}."
        )
        print(
            f"The requested variable `{variable}` does not exist! Plese select one among the available variables : {available_variables}."
        )
        raise typer.Exit(code=0)
    else:
        # variable
        timer_start = timer.time()
        time_series = dataset[variable]
        timer_end = timer.time()
        logger.debug(
            f"Data array variable selection took {timer_end - timer_start:.2f} seconds"
        )

        # chunking
        timer_start = timer.time()
        chunks = {"time": time, "lat": lat, "lon": lon}
        time_series.chunk(chunks=chunks)
        timer_end = timer.time()
        logger.debug(
            f"Data array rechunking took {timer_end - timer_start:.2f} seconds"
        )

        # ReviewMe --------------------------------------------------------- ?
        # in-memory
        if in_memory:
            timer_start = timer.time()
            location_time_series.load()  # load into memory for faster ... ?
            timer_end = timer.time()
            logger.debug(
                f"Location selection loading in memory took {timer_end - timer_start:.2f} seconds"
            )
        # --------------------------------------------------------------------

    timer_start = timer.time()
    indexers = set_location_indexers(
        data_array=time_series,
        longitude=longitude,
        latitude=latitude,
        verbose=verbose,
    )
    timer_end = timer.time()
    logger.debug(
        f"Data array indexers setting took {timer_end - timer_start:.2f} seconds"
    )

    try:
        timer_start = timer.time()
        location_time_series = time_series.sel(
            **indexers,
            method=neighbor_lookup,
            tolerance=tolerance,
        )
        timer_end = timer.time()
        logger.debug(f"Location selection took {timer_end - timer_start:.2f} seconds")

        # in-memory
        if in_memory:
            timer_start = timer.time()
            location_time_series.load()  # load into memory for faster ... ?
            timer_end = timer.time()
            logger.debug(
                f"Location selection loading in memory took {timer_end - timer_start:.2f} seconds"
            )

    except Exception as exception:
        logger.error(f"{ERROR_IN_SELECTING_DATA} : {exception}")
        print(f"{ERROR_IN_SELECTING_DATA} : {exception}")
        raise SystemExit(33)
    # ------------------------------------------------------------------------

    if start_time or end_time:
        timestamps = None  # we don't need a timestamp anymore!

        if start_time and not end_time:  # set `end_time` to end of series
            end_time = location_time_series.time.values[-1]

        elif end_time and not start_time:  # set `start_time` to beginning of series
            start_time = location_time_series.time.values[0]

        else:  # Convert `start_time` & `end_time` to the correct string format
            start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")

        timer_start = timer.time()
        location_time_series = location_time_series.sel(
            time=slice(start_time, end_time)
        )
        timer_end = timer.time()
        logger.debug(
            f"Time slicing with `start_time` and `end_time` took {timer_end - timer_start:.2f} seconds"
        )

    if timestamps is not None and not start_time and not end_time:
        if len(timestamps) == 1:
            start_time = end_time = timestamps[0]

        try:
            timer_start = timer.time()
            location_time_series = location_time_series.sel(
                time=timestamps, method=neighbor_lookup
            )
            timer_end = timer.time()
            logger.debug(
                f"Time selection with `timestamps` took {timer_end - timer_start:.2f} seconds"
            )

        except KeyError:
            logger.error(f"No data found for one or more of the given {timestamps}.")
            print(f"No data found for one or more of the given {timestamps}.")

    if location_time_series.size == 1:
        timer_start = timer.time()
        single_value = float(location_time_series.values)
        warning = (
            f"{exclamation_mark} The selected timestamp "
            + f"{location_time_series.time.values}"
            + f" matches the single value "
            + f"{single_value}"
        )
        timer_end = timer.time()
        logger.debug(
            f"Single value conversion to float took {timer_end - timer_start:.2f} seconds"
        )
        logger.warning(warning)
        if verbose > 0:
            print(warning)

    data_retrieval_end_time = timer.time()
    logger.debug(
        f"Data retrieval took {data_retrieval_end_time - data_retrieval_start_time:.2f} seconds"
    )

    if not verbose:
        print(location_time_series.values)
    else:
        print(location_time_series)

    # special case!
    if location_time_series is not None and timestamps is None:
        timestamps = location_time_series.time.to_numpy()

    if statistics:  # after echoing series which might be Long!
        print_series_statistics(
            data_array=location_time_series,
            timestamps=timestamps,
            title="Selected series",
        )
    if csv:
        to_csv(
            x=location_time_series,
            path=csv,
        )

    # return location_time_series

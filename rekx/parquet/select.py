import time as timer
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import typer
import xarray as xr
from rich import print
from typing_extensions import Annotated
from rekx.hardcodings import exclamation_mark
from rekx.statistics import print_series_statistics

from rekx.constants import (
    ROUNDING_PLACES_DEFAULT,
    VERBOSE_LEVEL_DEFAULT,
)
from rekx.csv import to_csv
from rekx.log import logger
from rekx.messages import ERROR_IN_SELECTING_DATA
from rekx.models import MethodForInexactMatches
from rekx.typer_parameters import (
    typer_argument_latitude_in_degrees,
    typer_argument_longitude_in_degrees,
    typer_argument_timestamps,
    typer_option_csv,
    typer_option_end_time,
    typer_option_in_memory,
    typer_option_mask_and_scale,
    typer_option_neighbor_lookup,
    typer_option_rounding_places,
    typer_option_start_time,
    typer_option_statistics,
    typer_option_tolerance,
    typer_option_variable_name_as_suffix,
    typer_option_verbose,
)
from rekx.utilities import set_location_indexers


def select_from_parquet(
    parquet_store: Annotated[Path, typer.Argument(..., help="Path to Parquet store")],
    variable: Annotated[str, typer.Argument(..., help="Variable name to select from")],
    longitude: Annotated[float, typer_argument_longitude_in_degrees],
    latitude: Annotated[float, typer_argument_latitude_in_degrees],
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
    variable_name_as_suffix: Annotated[
        bool, typer_option_variable_name_as_suffix
    ] = True,
    rounding_places: Annotated[
        Optional[int], typer_option_rounding_places
    ] = ROUNDING_PLACES_DEFAULT,
    verbose: Annotated[int, typer_option_verbose] = VERBOSE_LEVEL_DEFAULT,
) -> None:
    """Select data from a Parquet store"""

    # if convert_longitude_360:
    #     longitude = longitude % 360
    # warn_for_negative_longitude(longitude)

    logger.debug(f"Command context : {typer.Context}")

    data_retrieval_start_time = timer.time()
    logger.debug(f"Starting data retrieval... {data_retrieval_start_time}")

    # timer_start = timer.time()
    # mapper = fsspec.get_mapper(
    #     "reference://",
    #     fo=str(reference_file),
    #     remote_protocol="file",
    #     remote_options={"skip_instance_cache": True},
    # )
    # timer_end = timer.time()
    # logger.debug(f"Mapper creation took {timer_end - timer_start:.2f} seconds")
    timer_start = timer.perf_counter()
    dataset = xr.open_dataset(
        str(parquet_store),  # does not handle Path
        engine="kerchunk",
        # storage_options=dict(skip_instance_cache=True, remote_protocol="file"),
        storage_options=dict(remote_protocol="file"),
        # backend_kwargs={"consolidated": False},
        # chunks=None,
        # mask_and_scale=mask_and_scale,
    )
    timer_end = timer.perf_counter()
    logger.debug(
        f"Dataset opening via Xarray took {timer_end - timer_start:.2f} seconds"
    )

    available_variables = list(dataset.data_vars)
    if not variable in available_variables:
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

        if in_memory:
            timer_start = timer.time()
            location_time_series.load()  # load into memory for faster ... ?
            timer_end = timer.time()
            logger.debug(
                f"Location selection loading in memory took {timer_end - timer_start:.2f} seconds"
            )

    except Exception as exception:
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

    timer_start = timer.time()
    results = {
        location_time_series.name: location_time_series.to_numpy(),
    }
    timer_end = timer.time()
    logger.debug(
        f"Data series conversion to NumPy took {timer_end - timer_start:.2f} seconds"
    )

    title = "Location time series"

    # special case!
    if location_time_series is not None and timestamps is None:
        timer_start = timer.time()
        timestamps = location_time_series.time.to_numpy()
        timer_end = timer.time()
        logger.debug(
            f"Timestamps conversion to NumPy from Xarray's _time_ coordinate took {timer_end - timer_start:.2f} seconds"
        )

    if not verbose and not (statistics or csv):
        flat_array = location_time_series.values.flatten()
        print(*flat_array, sep=", ")
    if verbose > 0:
        print(location_time_series)

    if statistics:  # after echoing series which might be Long!
        timer_start = timer.time()
        print_series_statistics(
            data_array=location_time_series,
            timestamps=timestamps,
            title="Selected series",
        )
        timer_end = timer.time()
        logger.debug(
            f"Printing statistics in the console took {timer_end - timer_start:.2f} seconds"
        )

    if csv:
        timer_start = timer.time()
        to_csv(
            x=location_time_series,
            path=csv,
        )
        timer_end = timer.time()
        logger.debug(f"Exporting to CSV took {timer_end - timer_start:.2f} seconds")

    # return location_time_series

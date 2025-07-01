from pathlib import Path
from .utilities import set_location_indexers
from rekx.drop import drop_other_data_variables
from rekx.nccopy.constants import (
    COMPRESSION_FILTER_DEFAULT,
    COMPRESSION_LEVEL_DEFAULT,
    SHUFFLING_DEFAULT,
)


def rechunk_netcdf_via_xarray(
    input_filepath: Path,
    output_filepath: Path,
    time: int,
    latitude: int,
    longitude: int,
    min_longitude: float,
    max_longitude: float,
    min_latitude: float,
    max_latitude: float,
    mask_and_scale: bool = False,
    drop_other_variables: bool = True,
    fix_unlimited_dimensions: bool = False,
    # cache_size: int = CACHE_SIZE_DEFAULT,
    # cache_elements: int = CACHE_ELEMENTS_DEFAULT,
    # cache_preemption: float = CACHE_PREEMPTION_DEFAULT,
    compression: str = COMPRESSION_FILTER_DEFAULT,
    compression_level: int = COMPRESSION_LEVEL_DEFAULT,
    shuffling: bool | None = SHUFFLING_DEFAULT,
    # memory: bool = RECHUNK_IN_MEMORY_DEFAULT,
    mode: str = 'w-',
    overwrite_output: bool = False,
    encoding: dict = {},
    # unlimited_dimensions: list = ['time'],
    compute: bool = True,
    engine: str = "netcdf4",
) -> None:
    """
    Rechunk a NetCDF dataset and save it to a new file.

    Parameters
    ----------
    input_filepath : Path
        The path to the input NetCDF file.
    output_filepath : Path
        The path to the output NetCDF file where the rechunked dataset will be saved.
    chunks : Dict[str, Union[int, None]]
        A dictionary specifying the new chunk sizes for each dimension.
        Use `None` for dimensions that should not be chunked.

    Returns
    -------
    None
        The function saves the rechunked dataset to `output_filepath`.

    Examples
    --------
    # >>> rechunk_netcdf(Path("input.nc"), Path("output.nc"), {'time': 365, 'lat': 25, 'lon': 25})

    """

    # Open the dataset
    import xarray as xr

    dataset = xr.open_dataset(
        filename_or_obj=input_filepath,
        mask_and_scale=mask_and_scale,  # False == do not cast the data type. Please !
    )

    # Drop "other" data variables
    if drop_other_variables:
        dataset = drop_other_data_variables(dataset)
            
    # Clip
    indexers = set_location_indexers(
        data_array=dataset,
        longitude=slice(min_longitude, max_longitude),
        latitude=slice(min_latitude, max_latitude),
        # verbose=verbose,
    )
    # from devtools import debug
    # debug(locals())
    dataset = dataset.sel(
        **indexers,
        # tolerance=tolerance,
    )
    # from devtools import debug
    # debug(locals())
    # output_handlers = {
    #     ".nc": lambda area_time_series, output_filename: 
    #         area_time_series.to_netcdf(
    #             output_filename,
    #             # engine="h5netcdf",
    #         ),
    # }

    # Reset legacy encoding
    dataset.drop_encoding()

    # Initialise encoding
    encoding = {}

    for variable in dataset.data_vars:
        dimensions = dataset[variable].dims
        dtype = dataset[variable].dtype
        chunk_sizes = []
        if mask_and_scale:
            fill_value = dataset[variable].encoding['_FillValue']
        else:
            fill_value = None

        for dimension in dimensions:
            if dimension == "time":
                if not time:
                    time = dataset.sizes['time']
                chunk_sizes.append(time)
            elif dimension == "lat":
                chunk_sizes.append(latitude)
            elif dimension == "lon":
                chunk_sizes.append(longitude)
            else:
                chunk_sizes.append(dataset.sizes[dimension])

        # Define (Zarr v3) encoding
        encoding[variable] = {
            # "endian",
            # "szip_coding",
            "contiguous": False,
            # "blosc_shuffle",
            "shuffle": shuffling,
            # "least_significant_digit",
            # "quantize_mode",
            # "zlib",
            "dtype": dtype,
            # "significant_digits",
            '_FillValue': fill_value,
            # "szip_pixels_per_block",
            # "fletcher32",
            "chunksizes": tuple(chunk_sizes),
            "compression": compression,
            "complevel": compression_level,
        }

    # Write chunked data as a NetCDF file
    if output_filepath.exists():
        if not overwrite_output:
            print(f"The output file '{output_filepath}' already exists. It will not be overwritten.")
            # warnings.warn(f"The output file '{output_filepath}' already exists. It will not be overwritten.")
            # logger.warning(f"The output file '{output_filepath}' already exists. It will not be overwritten.")
            return  # Exit the function without writing the file
        else:
            mode = "w"
            # print(f"Overwriting the existing file '{output_filepath}'.")
            # logger.info(f"Overwriting the existing file '{output_filepath}'.")

    # extension = output_filepath.suffix.lower()
    # if extension in output_handlers:
    #     output_handlers[extension](area_time_series, output_filepath)
    # else:
    #     raise ValueError(f"Unsupported file extension: {extension}")


    if fix_unlimited_dimensions:
        unlimited_dimensions = []
    else:
        unlimited_dimensions = ['time']

    dataset.to_netcdf(
        path=output_filepath,
        mode=mode,
        engine=engine,
        encoding=encoding,
        unlimited_dims=unlimited_dimensions,
        compute=compute,
    )

    # return output_filepath

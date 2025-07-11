import enum
from pathlib import Path

# from enum import Enum
from typing import List, Type, Set

import netCDF4
import xarray as xr


class MethodForInexactMatches(str, enum.Enum):
    none = None  # only exact matches
    pad = "pad"  # ffill: propagate last valid index value forward
    backfill = "backfill"  # bfill: propagate next valid index value backward
    nearest = "nearest"  # use nearest valid index value


class XarrayVariableSet(str, enum.Enum):
    all = "all"
    dimensions = "dimensions"
    dimensions_without_coordinates = "dimensions-without-coordinates"
    coordinates = "coordinates"
    time = "time"
    latitude = "latitude"
    latitude_boundaries = "latitude-boundaries"
    longitude = "longitude"
    longitude_boundaries = "longitude-boundaries"
    location = "location"
    location_boundaries= "location-boundaries"
    data_variables = "data-variables"
    data = "data"
    metadata = "metadata"


def select_xarray_variable_set_from_dataset(
    xarray_variable_set: Type[enum.Enum],
    variable_set: list[enum.Enum],
    dataset: xr.Dataset,
) -> Set[str]:
    """
    Select user-requested set of variables from an Xarray dataset.

    Parameters
    ----------
    xarray_variable_set: enum.Enum
        The Enum model to use for selection

    variable_set: List[enum.Enum]
        The user-requested sets of variables to select based on the Enum model

    dataset: xr.Dataset
        The input Xarray dataset from which to extract the user-requested
        variables

    Returns
    -------
    Set[str]
        Selected variable names

    Examples
    --------

    Notes
    -----
    Is quasi-identical to the function
    select_netcdf_variable_set_from_dataset() with differences in terms of the
    names of attributes. See also docstring of other function.
    """
    try:
        # Core sets, _should_ always exist
        variables = set(dataset.variables)
        dimensions = set(dataset.dims)
        coordinates = set(dataset.coords)
        data_variables = set(dataset.data_vars)

        dimensions_without_coordinates = dimensions.difference(coordinates) 
        time_coordinate = {dataset.time.name}
        latitude_coordinate = {dataset.lat.name}
        longitude_coordinate = {dataset.lon.name}
        location_coordinates = latitude_coordinate.union(longitude_coordinate)

        # Boundary variables (may not exist)
        lat_bnds = getattr(dataset, 'lat_bnds', None)
        lon_bnds = getattr(dataset, 'lon_bnds', None)
        data_variables_latitude_boundaries = {lat_bnds.name} if lat_bnds is not None else set()
        data_variables_longitude_boundaries = {lon_bnds.name} if lon_bnds is not None else set()
        data_variables_location_boundaries = data_variables_latitude_boundaries.union(data_variables_longitude_boundaries)

        # Metadata handling
        data_variables_metadata = {"record_status"} if "record_status" in variables else set()
        data = data_variables - data_variables_location_boundaries - data_variables_metadata

        # Map enum members to variable sets
        selection_map = {
            xarray_variable_set.all: variables,
            xarray_variable_set.dimensions: dimensions,
            xarray_variable_set.dimensions_without_coordinates: dimensions_without_coordinates,
            xarray_variable_set.coordinates: coordinates,
            xarray_variable_set.time: time_coordinate,
            xarray_variable_set.latitude: latitude_coordinate,
            xarray_variable_set.longitude: longitude_coordinate,
            xarray_variable_set.location: location_coordinates,
            xarray_variable_set.data_variables: data_variables,
            xarray_variable_set.latitude_boundaries: data_variables_latitude_boundaries,
            xarray_variable_set.longitude_boundaries: data_variables_longitude_boundaries,
            xarray_variable_set.location_boundaries: data_variables_location_boundaries,
            xarray_variable_set.data: data,
            xarray_variable_set.metadata: data_variables_metadata,
        }
    except Exception as e:
        raise ValueError(f"Error processing dataset: {str(e)}")


    # Convert any strings to enum members (case-insensitive)
    enum_set = set()
    for v in variable_set:
        if isinstance(v, XarrayVariableSet):
            enum_set.add(v)
        elif isinstance(v, str):
            # Case-insensitive match
            try:
                enum_set.add(XarrayVariableSet(v.lower()))
            except ValueError:
                raise ValueError(f"Unknown variable set: {v!r}")
        else:
            raise TypeError(f"Invalid type for variable set: {type(v)}")

    # Collect selected variables
    selected_variables = set()
    for enum_member in enum_set:
        selected_variables.update(selection_map.get(enum_member, set()))

    # Only return variables present in the dataset
    return selected_variables & set(dataset.variables)


    # for variable in xarray_variable_set:
    #     if variable in variable_set:
    #         selected_variables.update(selection_map[variable])

    # return selected_variables


def validate_variable_set(
    variable_set_input: List[enum.Enum],
) -> list[XarrayVariableSet]:
    if not variable_set_input:
        # Use a sensible default or raise
        return [XarrayVariableSet.all]
    validated = []
    for v in variable_set_input:
        if v in XarrayVariableSet.__members__:
            validated.append(XarrayVariableSet[v])
        else:
            raise ValueError(f"Invalid variable set: {v}")

    return validated


def select_netcdf_variable_set_from_dataset(
    netcdf4_variable_set: Type[enum.Enum],
    variable_set: List[enum.Enum],
    dataset: netCDF4.Dataset,
):
    """
    The same Enum model for both : netcdf4_variable_set and xarray_variable_set
    """
    metadata_attributes = {"record_status", "bnds"}
    coordinates_data_attributes = {"lat_bnds", "lon_bnds"}
    time_coordinate = {"time"}
    dimensions_attributes = set(dataset.dimensions)  # no `coordinates` via netCDF4
    variables_attributes = set(dataset.variables)
    data_attributes = (
        variables_attributes
        - dimensions_attributes
        - coordinates_data_attributes
        - metadata_attributes
    )

    if variable_set == netcdf4_variable_set.all:
        return variables_attributes

    elif variable_set == netcdf4_variable_set.coordinates:
        return dimensions_attributes  # Same as next one ?

    # elif variable_set == netcdf4_variable_set.coordinates_without_data:
    #     return dimensions_attributes

    elif variable_set == netcdf4_variable_set.data:
        return data_attributes

    elif variable_set == netcdf4_variable_set.metadata:
        return metadata_attributes.intersection(variables_attributes)

    elif variable_set == netcdf4_variable_set.time:
        return time_coordinate

    else:
        raise ValueError("Invalid category")


class FileFormat(enum.Enum):
    NETCDF = ".nc"
    PARQUET = ".parquet"
    JSON = ".json"
    ZARR = ".zarr"  # or directory ?

    def open_dataset_options(self) -> dict:
        if self == FileFormat.NETCDF:
            # Default options for other formats
            return {"mask_and_scale": False}
        if self == FileFormat.JSON:
            return {
                "engine": "zarr",
                "backend_kwargs": {
                    "consolidated": False,
                },
                "chunks": None,
                "mask_and_scale": False,
            }
        if self == FileFormat.PARQUET:
            return {
                "engine": "kerchunk",
                "storage_options": {
                    # "skip_instance_cache": True,
                    "remote_protocol": "file",
                },
            }
        if self == FileFormat.ZARR:
            return {}

    def dataset_select_options(self, tolerance) -> dict:
        
        if self == FileFormat.PARQUET:
            return {"tolerance": tolerance}

        if self == FileFormat.ZARR:
            return {"tolerance": tolerance}
        
        else:
            # Default or no additional options for other formats
            return {}


def get_file_format(file_path: Path) -> FileFormat:
    """
    Get the format from the filename extension.
    """
    file_extension = file_path.suffix.lower()
    return FileFormat(file_extension)

"""Docstring for the diagnose.py module.

Functions to inspect the metadata, the structure
and specifically diagnose and validate
the chunking shapes of NetCDF files.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
import xarray as xr
from .log import logger
from .models import (
    XarrayVariableSet,
    select_xarray_variable_set_from_dataset,
    validate_variable_set,
)
from math import ceil, prod


def count_chunks(
    dataset: xr.Dataset,
    variable: str,
):
    """
    """
    # Add chunk count calculation
    shape = dataset[variable].shape
    chunks = dataset[variable].chunking()
    
    if chunks and chunks != "contiguous":
        chunks_per_dim = [ceil(s/c) for s, c in zip(shape, chunks)]
        number_of_chunks = prod(chunks_per_dim)

    else:
        number_of_chunks = 1
            
    return number_of_chunks


def detect_chunking_shapes(
    file_path: Path,
    variable_set: list[XarrayVariableSet] = [XarrayVariableSet.all],
    # ) -> Tuple[Dict[str, Set[int]], str]:
) -> Tuple[Dict, Dict, str]:
    """
    Detect the chunking shapes of variables within single NetCDF file.

    Parameters
    ----------
    file_path: Path
        Path to input file
    variable_set: XarrayVariableSet
        Name of the set of variables to query. See also docstring of
        XarrayVariableSet

    Returns
    -------
    Tuple[Dict, str]
    # Tuple[Dict[str, Set[int]], str]
        A tuple containing a dictionary `chunking_shape` and the name of the
        input file. The nested dictionary's first level keys are variable names,
        and the second level keys are the chunking shapes encountered, with the
        associated values being sets of file names where those chunking shapes
        are found.

    Raises
    ------
    FileNotFoundError
        If the specified NetCDF file does not exist.

    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    chunking_shapes = {}
    number_of_chunks = {}
    with xr.open_dataset(file_path, engine="netcdf4") as dataset:
        variable_set = validate_variable_set(variable_set)
        selected_variables = select_xarray_variable_set_from_dataset(
            XarrayVariableSet, variable_set, dataset
        )
        for variable in selected_variables:
            chunking_shape = dataset[variable].encoding.get("chunksizes")
            var = dataset[variable]
            shape = var.shape
            if chunking_shape and chunking_shape != "contiguous":
                # Review Me ! ----------------------
                # if variable not in chunking_shapes:
                #     chunking_shapes[variable] = set()
                # Review Me ! ----------------------


                # Review Me ! ----------------------
                # chunking_shapes[variable].add(tuple(chunking_shape))
                # Review Me ! ----------------------

                # Calculate number of chunks
                chunks_per_dimension = [ceil(dimension_size / chunk) for dimension_size, chunk in zip(shape, chunking_shape)]
                chunks = prod(chunks_per_dimension)

            else:
                chunks = 1

            chunking_shapes[variable] = chunking_shape
            number_of_chunks[variable] = chunks

    return chunking_shapes, number_of_chunks, file_path.name


def detect_chunking_shapes_parallel(
    file_paths: List[Path],
    variable_set: list[XarrayVariableSet] = list[XarrayVariableSet.all],
) -> dict:
    """
    Detect and aggregate the chunking shapes of variables within a set of
    multiple NetCDF files in parallel.

    Parameters
    ----------
    file_paths: List[Path]
        A list of file paths pointing to the NetCDF files to be scanned.

    Returns
    -------
    dict
        A nested dictionary where the first level keys are variable names, and the
        second level keys are the chunking shapes encountered, with the associated
        values being sets of file names where those chunking shapes are found.
    """
    aggregated_chunking_shapes = {}
    with ProcessPoolExecutor() as executor:
        # futures = [
        #     executor.submit(detect_chunking_shapes, file_path, variable_set.value)
        #     for file_path in file_paths
        # ]
        futures = [
            executor.submit(detect_chunking_shapes, file_path, [variable_set])
            for file_path in file_paths
        ]

        for future in as_completed(futures):
            try:
                chunking_shapes, number_of_chunks, file_name = future.result()
                logger.info(f"Scanned file: {file_name}")

                for variable, chunking_shape in chunking_shapes.items():
                    chunks = number_of_chunks[variable]
                    key = (
                        tuple(chunking_shape)
                        if chunking_shape and chunking_shape != "contiguous"
                        else "contiguous"
                    )
                    if variable not in aggregated_chunking_shapes:
                        aggregated_chunking_shapes[variable] = {}
                        logger.info(
                            f"Initial chunk sizes set for {variable} in {file_name}"
                        )
                    if key not in aggregated_chunking_shapes[variable]:
                        aggregated_chunking_shapes[variable][key] = {
                                'files': set(),
                                'chunks': chunks
                        }
                    # if chunking_shape not in aggregated_chunking_shapes[variable]:
                    #     aggregated_chunking_shapes[variable][chunking_shape] = set()
                        logger.info(
                            f"New chunking shape {chunking_shape} found for variable {variable} in {file_name}"
                        )
                    # aggregated_chunking_shapes[variable][chunking_shape].add(file_name)
                    aggregated_chunking_shapes[variable][key]['files'].add(file_name)

            except Exception as e:
                logger.error(f"Error processing file: {e}")

    return aggregated_chunking_shapes

"""Multi-threaded CSV writer, much faster than :meth:`pandas.DataFrame.to_csv`,
with full support for `dask <http://dask.org/>`_ and `dask distributed
<http://distributed.dask.org/>`_.
"""
from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import xarray
from dask.base import tokenize
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from rich import print
from xarray_extras.kernels import csv as kernels

__all__ = ("to_csv",)


def to_csv(
    x: xarray.DataArray,
    path: str | Path,
    *,
    nogil: bool = True,
    **kwargs,
):
    """Print DataArray to CSV.

    When x has numpy backend, this function is functionally equivalent to (but
    much) faster than)::

        x.to_pandas().to_csv(path_or_buf, **kwargs)

    When x has dask backend, this function returns a dask delayed object which
    will write to the disk only when its .compute() method is invoked.

    Formatting and optional compression are parallelised across all available
    CPUs, using one dask task per chunk on the first dimension. Chunks on other
    dimensions will be merged ahead of computation.

    :param x:
        :class:`~xarray.DataArray` with one or two dimensions
    :param str path:
        Output file path
    :param bool nogil:
        If True, use accelerated C implementation. Several kwargs won't be
        processed correctly (see limitations below). If False, use pandas
        to_csv method (slow, and does not release the GIL).
        nogil=True exclusively supports float and integer values dtypes (but
        the coords can be anything). In case of incompatible dtype, nogil
        is automatically switched to False.
    :param kwargs:
        Passed verbatim to :meth:`pandas.DataFrame.to_csv` or
        :meth:`pandas.Series.to_csv`

    **Limitations**

    - Fancy URIs are not (yet) supported.
    - compression='zip' is not supported. All other compression methods (gzip,
      bz2, xz) are supported.
    - When running with nogil=True, the following parameters are ignored:
      columns, quoting, quotechar, doublequote, escapechar, chunksize, decimal

    **Distributed computing**

    This function supports `dask distributed`_, with the caveat that all workers
    must write to the same shared mountpoint and that the shared filesystem
    must strictly guarantee **close-open coherency**, meaning that one must be
    able to call write() and then close() on a file descriptor from one host
    and then immediately afterwards open() from another host and see the output
    from the first host. Note that, for performance reasons, most network
    filesystems do not enable this feature by default.

    Alternatively, one may write to local mountpoints and then manually collect
    and concatenate the partial outputs.
    """
    if not isinstance(x, xarray.DataArray):
        raise ValueError("first argument must be a DataArray")

    # Health checks
    if not isinstance(path, Path):
        try:
            path = Path(path)
        except:
            raise ValueError("path_or_buf must be a file path")

    if x.ndim not in (1, 2):
        raise ValueError(
            "cannot convert arrays with %d dimensions into " "pandas objects" % x.ndim
        )

    if nogil and x.dtype.kind not in "if":
        nogil = False

    # Extract row and columns indices
    indices = [x.get_index(dim) for dim in x.dims]
    if x.ndim == 2:
        index, columns = indices
    else:
        index = indices[0]
        columns = None

    compression = kwargs.pop("compression", "infer")
    compress = _compress_func(path, compression)
    mode = kwargs.pop("mode", "w")
    if mode not in "wa":
        raise ValueError('mode: expected w or a; got "%s"' % mode)

    # Fast exit for numpy backend
    if not x.chunks:
        bdata = kernels.to_csv(x.values, index, columns, True, nogil, kwargs)
        if compress:
            bdata = compress(bdata)
        with open(path, mode + "b") as fh:
            fh.write(bdata)
        return None

    # Merge chunks on all dimensions beyond the first
    x = x.chunk((x.chunks[0],) + tuple((s,) for s in x.shape[1:]))

    # Manually define the dask graph
    tok = tokenize(x.data, index, columns, compression, path, kwargs)
    name1 = "to_csv_encode-" + tok
    name2 = "to_csv_compress-" + tok
    name3 = "to_csv_write-" + tok
    name4 = "to_csv-" + tok

    dsk: dict[str | tuple, tuple] = {}

    assert x.chunks
    assert x.chunks[0]
    offset = 0
    for i, size in enumerate(x.chunks[0]):
        # Slice index
        index_i = index[offset : offset + size]
        offset += size

        x_i = (x.data.name, i) + (0,) * (x.ndim - 1)

        # Step 1: convert to CSV and encode to binary blob
        if i == 0:
            # First chunk: print header
            dsk[name1, i] = (kernels.to_csv, x_i, index_i, columns, True, nogil, kwargs)
        else:
            kwargs_i = kwargs.copy()
            kwargs_i["header"] = False
            dsk[name1, i] = (kernels.to_csv, x_i, index_i, None, False, nogil, kwargs_i)

        # Step 2 (optional): compress
        if compress:
            prevname = name2
            dsk[name2, i] = compress, (name1, i)
        else:
            prevname = name1

        # Step 3: write to file
        if i == 0:
            # First chunk: overwrite file if it already exists
            dsk[name3, i] = kernels.to_file, path, mode + "b", (prevname, i)
        else:
            # Next chunks: wait for previous chunk to complete and append
            dsk[name3, i] = (kernels.to_file, path, "ab", (prevname, i), (name3, i - 1))

    # Rename final key
    dsk[name4] = dsk.pop((name3, i))

    hlg = HighLevelGraph.from_collections(name4, dsk, (x,))
    return Delayed(name4, hlg)


def _compress_func(
    path: Path,
    compression: str | None,
) -> Callable[[bytes], bytes] | None:
    if compression == "infer":
        compression = path.suffix[1:].lower()
        if compression == "gz":
            compression = "gzip"
        elif compression == "csv":
            compression = None

    if compression is None:
        return None
    elif compression == "gzip":
        import gzip

        return gzip.compress
    elif compression == "bz2":
        import bz2

        return bz2.compress
    elif compression == "xz":
        import lzma

        return lzma.compress
    elif compression == "zip":
        raise NotImplementedError("zip compression is not supported")
    else:
        raise ValueError("Unrecognized compression: %s" % compression)


def write_metadata_dictionary_to_csv(
    dictionary: dict,
    output_filename: Path,
) -> None:
    """
    Write a metadata dictionary to a CSV file.

    Parameters
    ----------
    dictionary:
        A dictionary containing the metadata.
    output_filename: Path
        Path to the output CSV file.

    Returns
    -------
    None

    """
    if not dictionary:
        raise ValueError("The given dictionary is empty!")

    headers = [
        "File Name",
        "File Size",
        "Variable",
        "Shape",
        "Type",
        "Compression",
        "Read time",
    ]

    with open(output_filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        file_name = dictionary.get("File name", "")
        file_size = dictionary.get("File size", "")

        for variable, metadata in dictionary.get("Variables", {}).items():
            if "Compression" in metadata:
                from .print import format_compression

                compression_details = format_compression(metadata["Compression"])
            row = [
                file_name,
                file_size,
                variable,
                metadata.get("Shape", ""),
                metadata.get("Type", ""),
                metadata.get("Scale", ""),
                metadata.get("Offset", ""),
                compression_details["Filters"] if compression_details else None,
                compression_details["Level"] if compression_details else None,
                metadata.get("Shuffling", ""),
                metadata.get("Read time", ""),
            ]
            writer.writerow(row)
    print(f"Output written to [code]{output_filename}[/code]")


def write_nested_dictionary_to_csv(
    nested_dictionary: dict,
    output_filename: Path,
) -> None:
    """ """
    if not nested_dictionary:
        raise ValueError("The given dictionary is empty!")

    with open(output_filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "File",
                "Size",
                "Variable",
                "Shape",
                "Chunks",
                "Cache",
                "Elements",
                "Preemption",
                "Type",
                "Scale",
                "Offset",
                "Compression",
                "Level",
                "Shuffling",
                # "Repetitions",
                "Read time",
            ]
        )

        for file_name, file_data in nested_dictionary.items():
            for variable, metadata in file_data.get("Variables", {}).items():
                row = [
                    file_data.get("File name", ""),
                    file_data.get("File size", ""),
                    variable,
                    metadata.get("Shape", ""),
                    metadata.get("Chunks", ""),
                    metadata.get("Cache", ""),
                    metadata.get("Elements", ""),
                    metadata.get("Preemption", ""),
                    metadata.get("Type", ""),
                    metadata.get("Scale", ""),
                    metadata.get("Offset", ""),
                    metadata.get("Compression", ""),
                    metadata.get("Level", ""),
                    metadata.get("Shuffling", ""),
                    metadata.get("Read time", ""),
                ]
                writer.writerow(row)
    print(f"Output written to [code]{output_filename}[/code]")

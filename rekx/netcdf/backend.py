from pathlib import Path
from rekx.backend import RechunkingBackendBase
from rekx.log import logger


class NetCDF4Backend(RechunkingBackendBase):
    def rechunk(
        self,
        input_filepath: Path,
        output_filepath: Path,
        time: int = None,
        lat: int = None,
        lon: int = None,
    ) -> None:
        """Rechunk data stored in a NetCDF4 file.

        Notes
        -----
        Text partially quoted from

        https://unidata.github.io/netcdf4-python/#netCDF4.Dataset.createVariable :

        The function `createVariable()` available through the `netcdf4-python`
        python interface to the netCDF C library, features the optional keyword
        `chunksizes` which can be used to manually specify the HDF5 chunk sizes for
        each dimension of the variable.

        A detailed discussion of HDF chunking and I/O performance is available at
        https://support.hdfgroup.org/HDF5/doc/Advanced/Chunking/. The default
        chunking scheme in the netcdf-c library is discussed at
        https://docs.unidata.ucar.edu/nug/current/netcdf_perf_chunking.html.

        Basically, the chunk size for each dimension should match as closely as
        possible the size of the data block that users will read from the file.
        `chunksizes` cannot be set if `contiguous=True`.
        """
        # Check if any chunking has been requested
        if time is None and lat is None and lon is None:
            logger.info(
                f"No chunking requested for {input_filepath}. Exiting function."
            )
            return

        # logger.info(f"Rechunking of {input_filepath} with chunk sizes: time={time}, lat={lat}, lon={lon}")
        new_chunks = {"time": time, "lat": lat, "lon": lon}
        with nc.Dataset(input_filepath, mode="r") as input_dataset:
            with nc.Dataset(output_filepath, mode="w") as output_dataset:
                for name in input_dataset.ncattrs():
                    output_dataset.setncattr(name, input_dataset.getncattr(name))
                for name, dimension in input_dataset.dimensions.items():
                    output_dataset.createDimension(
                        name, (len(dimension) if not dimension.isunlimited() else None)
                    )
                for name, variable in input_dataset.variables.items():
                    # logger.debug(f"Processing variable: {name}")
                    if name in new_chunks:
                        chunk_size = new_chunks[name]
                        import dask.array as da

                        if chunk_size is not None:
                            # logger.debug(f"Chunking variable `{name}` with chunk sizes: {chunk_size}")
                            x = da.from_array(
                                variable, chunks=(chunk_size,) * len(variable.shape)
                            )
                            output_dataset.createVariable(
                                name,
                                variable.datatype,
                                variable.dimensions,
                                zlib=True,
                                complevel=4,
                                chunksizes=(chunk_size,) * len(variable.shape),
                            )
                            output_dataset[name].setncatts(input_dataset[name].__dict__)
                            output_dataset[name][:] = x
                        else:
                            # logger.debug(f"No chunk sizes specified for `{name}`, copying as is.")
                            output_dataset.createVariable(
                                name, variable.datatype, variable.dimensions
                            )
                            output_dataset[name].setncatts(input_dataset[name].__dict__)
                            output_dataset[name][:] = variable[:]
                    else:
                        # logger.debug(f"Variable `{name}` not in chunking list, copying as is.")
                        output_dataset.createVariable(
                            name, variable.datatype, variable.dimensions
                        )
                        output_dataset[name].setncatts(input_dataset[name].__dict__)
                        output_dataset[name][:] = variable[:]

        # logger.info(f"Completed rechunking from {input_filepath} to {output_filepath}")

import enum
from abc import ABC, abstractmethod
from pathlib import Path
from rekx.chunk import rechunk_netcdf_via_xarray
from rekx.constants import VERBOSE_LEVEL_DEFAULT
from rekx.nccopy.constants import (
    COMPRESSION_FILTER_DEFAULT,
    COMPRESSION_LEVEL_DEFAULT,
    SHUFFLING_DEFAULT,
)


class RechunkingBackendBase(ABC):
    @abstractmethod
    def rechunk(
        self,
        input_filepath,
        output_directory,
        **kwargs,
    ):
        pass


class XarrayBackend(RechunkingBackendBase):
    def rechunk(
        self,
        input_filepath: Path,
        variables: list[str],
        output_filepath: Path,
        time: int | None = None,
        latitude: int | None = None,
        longitude: int | None = None,
        min_longitude: float | None = None,
        min_x: int | None = None,
        max_longitude: float | None = None,
        max_x: int | None = None,
        min_latitude: float | None = None,
        min_y: int | None = None,
        max_latitude: float | None = None,
        max_y: int | None = None,
        every_nth_timestamp: int | None = None,
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
        engine: str = 'netcdf4',
        dry_run: bool = False,
        verbose: int = VERBOSE_LEVEL_DEFAULT, 
        **kwargs
    ):
        """
        """
        message = f"Rechunk via Xarray\n   - from {input_filepath}\n   - to {output_filepath}\n   - with chunks (time={time}, lat={latitude}, lon={longitude})"
        if not dry_run:
            rechunk_netcdf_via_xarray(
                input_filepath=input_filepath,
                output_filepath=output_filepath,
                time=time,
                latitude=latitude,
                longitude=longitude,
                min_longitude=min_longitude,
                min_x=min_x,
                max_longitude=max_longitude,
                max_x=max_x,
                min_latitude=min_latitude,
                min_y=min_y,
                max_latitude=max_latitude,
                max_y=max_y,
                every_nth_timestamp=every_nth_timestamp,
                mask_and_scale=mask_and_scale,
                drop_other_variables=drop_other_variables,
                fix_unlimited_dimensions=fix_unlimited_dimensions,
                # spatial_symmetry=spatial_symmetry,
                # variable_set=variable_set,
                # cache_size=cache_size,
                # cache_elements=cache_elements,
                # cache_preemption=cache_preemption,
                compression=compression,
                compression_level=compression_level,
                shuffling=shuffling,
                # memory=memory,
                mode=mode,
                overwrite_output=overwrite_output,
                engine=engine,
            )
        return message


@enum.unique
class RechunkingBackend(str, enum.Enum):
    all = "all"
    netcdf4 = "netCDF4"
    xarray = "xarray"
    nccopy = "nccopy"

    @classmethod
    def default(cls) -> "RechunkingBackend":
        """Default rechunking backend to use"""
        return cls.nccopy

    def get_backend(self) -> RechunkingBackendBase:
        """Array type associated to a backend."""

        if self.name == "nccopy":
            return nccopyBackend()

        elif self.name == "netcdf4":
            return NetCDF4Backend()

        elif self.name == "xarray":
            return XarrayBackend()

        else:
            raise ValueError(f"No known backend for {self.name}.")

from pathlib import Path
from typing_extensions import List
from rekx.nccopy.constants import (
    FIX_UNLIMITED_DIMENSIONS_DEFAULT,
    CACHE_SIZE_DEFAULT,
    CACHE_ELEMENTS_DEFAULT,
    CACHE_PREEMPTION_DEFAULT,
    COMPRESSION_FILTER_DEFAULT,
    COMPRESSION_LEVEL_DEFAULT,
    SHUFFLING_DEFAULT,
    RECHUNK_IN_MEMORY_DEFAULT,
)

class nccopyBackend(RechunkingBackendBase):
    def rechunk(
        self,
        input_filepath: Path,
        variables: List[str],
        output_directory: Path,
        time: int | None = None,
        latitude: int | None = None,
        longitude: int | None = None,
        fix_unlimited_dimensions: bool = FIX_UNLIMITED_DIMENSIONS_DEFAULT,
        cache_size: int | None = CACHE_SIZE_DEFAULT,
        cache_elements: int | None = CACHE_ELEMENTS_DEFAULT,
        cache_preemption: float | None = CACHE_PREEMPTION_DEFAULT,
        compression: str = COMPRESSION_FILTER_DEFAULT,
        compression_level: int = COMPRESSION_LEVEL_DEFAULT,
        shuffling: bool | None = SHUFFLING_DEFAULT,
        memory: bool = RECHUNK_IN_MEMORY_DEFAULT,
        dry_run: bool = False,  # return command as a string ?
    ):  # **kwargs):
        """
        Options considered for ``nccopy`` :
        [ ] [-k kind_name]
        [ ] [-kind_code]
        [x] [-d n]  # deflate
        [x] [-s]  # shuffling
        [x] [-c chunkspec]  # chunking sizes
        [x] [-u] Convert unlimited size input dimensions to fixed size output dimensions. May speed up variable-at-a-time access, but slow down record-at-a-time access.
        [x] [-w]  # read and process data in-memory, write out in the end
        [ ] [-[v] var1,...]
        [x] [-[V] var1,...]
        [ ] [-[g|G] grp1,...]
        [ ] [-m bufsize]
        [x] [-h chunk_cache]  #
        [x] [-e cache_elems]  # Number of elements in cache
        [ ] [-r]
        [x] infile
        [x] outfile
        """
        variable_option = f"-V {','.join(variables)}" if variables else "" # it's a capital V
        chunking_shape = (
            f"-c time/{time},lat/{latitude},lon/{longitude}"
            if all([time, latitude, longitude])
            else ""
        )
        fixing_unlimited_dimensions = f"-u" if fix_unlimited_dimensions else ""
        compression_options = f"-d {compression_level}" if compression == "zlib" else ""
        shuffling_option = f"-s" if shuffling and compression_level > 0 else ""
        cache_size_option = f"-h {cache_size} " if cache_size != CACHE_SIZE_DEFAULT else False  # cache size in bytes
        cache_elements_option = f"-e {cache_elements}" if cache_elements != CACHE_ELEMENTS_DEFAULT else False
        memory_option = f"-w" if memory else ""

        # Collect all non-empty options into a list
        options = [
            variable_option,
            chunking_shape,
            fixing_unlimited_dimensions,
            compression_options,
            shuffling_option,
            cache_size_option,
            cache_elements_option,
            memory_option,
            input_filepath,
        ]
        # Build the command by joining non-empty options
        command = "nccopy " + " ".join(filter(bool, options)) + " "  # space before output filename

        # Build the output file path
        output_filename = f"{Path(input_filepath).stem}"
        output_filename += f"_{time}"
        output_filename += f"_{latitude}"
        output_filename += f"_{longitude}"
        output_filename += f"_{compression}"
        output_filename += f"_{compression_level}"
        if shuffling and compression_level > 0:
            output_filename += f"_shuffled"
        output_filename += f"{Path(input_filepath).suffix}"
        output_directory.mkdir(parents=True, exist_ok=True)
        output_filepath = output_directory / output_filename
        command += f"{output_filepath}"

        if dry_run:
            return command

        else:
            args = shlex.split(command)
            subprocess.run(args)

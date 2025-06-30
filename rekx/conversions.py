import numpy as np
import typer


def convert_to_radians(
    ctx: typer.Context, param: typer.CallbackParam, angle: float
) -> float:
    """Convert floating point angular measurement from degrees to radians."""
    if ctx.resilient_parsing:
        return
    if type(angle) != float:
        raise typer.BadParameter("Input should be a float!")

    return np.radians(angle)

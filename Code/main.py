from stabilization import stabilize
from background_substraction import background_substraction
from matting import matting_and_tracking


def run_all_parts():
    stabilize()
    background_substraction()
    matting_and_tracking()

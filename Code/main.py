from stabilization import stabilize
from background_substraction import background_substraction
from matting import matting_and_tracking
import time
import json


def run_all_parts():
    json_file = "Outputs/timing.json"
    data = {
        "time_to_stabilize": 0,
        "time_to_binary": 0,
        "time_to_alpha": 0,
        "time_to_matted": 0,
        "time_to_output": 0,
    }
    start_time = time.time()

    stabilize()
    
    stab_end_time = time.time()
    data["time_to_stabilize"] = stab_end_time-start_time

    background_substraction()

    bs_end_time = time.time()
    data["time_to_binary"] = bs_end_time-start_time

    matting_and_tracking()

    tracking_end_time = time.time()

    data["time_to_alpha"] = tracking_end_time-start_time
    data["time_to_matted"] = tracking_end_time-start_time
    data["time_to_output"] = tracking_end_time-start_time

    with open(json_file, 'w') as file:
        json.dump(json_file, file)


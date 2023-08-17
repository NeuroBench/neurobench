"""
"""

def check_data(metric, key_list, run_data):
    missing_keys = []

    for key in key_list:
        if key not in run_data:
            missing_keys.append(key)

    if len(missing_keys) > 0:
        raise ValueError("{} missing required tracked keys in run_data: {}".format(metric, missing_keys))

    return


# example
def model_size(run_data):
    check_data("model_size", ["model"], run_data)

    model = run_data["model"]
    return model.size()
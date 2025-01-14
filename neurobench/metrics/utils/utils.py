def convert_to_class_name(metric_name):
    """
    Convert a metric name from a string to a class name.

    Args:
        metric_name (str): The metric name in snake_case or other formatting.

    Returns:
        str: The corresponding class name in CamelCase.

    """
    # Convert snake_case to CamelCase
    if metric_name in ["sMAPE", "r2", "mse"]:
        return metric_name.upper()
    return "".join(word.title() for word in metric_name.split("_"))

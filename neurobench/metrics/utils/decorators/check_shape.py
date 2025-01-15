from functools import wraps


def check_shapes(func):
    """Decorator to check that the shapes of predictions and labels are the same before
    executing the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Assuming `preds` and `labels` are the first two positional arguments
        if len(args) >= 2:
            preds, data = args[2:]
            if preds.shape != data[1].shape:
                raise ValueError(
                    f"In {func.__name__}: 'preds' and 'labels' must have the same shape. "
                    f"Got preds.shape={preds.shape} and data[1].shape={data[1].shape}."
                )
        return func(*args, **kwargs)

    return wrapper

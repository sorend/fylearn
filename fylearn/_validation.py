"""Shared validation helpers."""



def has_nan_classes(classes):
    """Check whether the class array contains NaN values.

    Uses the ``x != x`` idiom so that it also works for non-numeric (e.g. str)
    class arrays, for which NaN is not applicable.
    """
    return any(x != x for x in classes)

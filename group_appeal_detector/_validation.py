from .exceptions import InputTypeError, InputValueError

_VALID_DEVICE_TYPES = {"cpu", "cuda", "mps"}


def validate_device(device: str) -> None:
    """Validates a torch device string, e.g. ``cpu``, ``cuda``, ``cuda:0``, ``mps``."""
    if not isinstance(device, str):
        raise InputTypeError(
            f"Expected a string for device, got {type(device).__name__}."
        )
    device_type = device.split(":", 1)[0]
    if device_type not in _VALID_DEVICE_TYPES:
        raise InputValueError(
            f"Unsupported device '{device}'. Expected one of "
            f"{sorted(_VALID_DEVICE_TYPES)} (optionally suffixed with an index, e.g. 'cuda:0')."
        )


def validate_str(value, name: str) -> None:
    """Validates that ``value`` is a string."""
    if not isinstance(value, str):
        raise InputTypeError(
            f"Expected a string for {name}, got {type(value).__name__}."
        )


def validate_str_list(values, name: str) -> None:
    """Validates that ``values`` is a list of strings."""
    if not isinstance(values, list):
        raise InputTypeError(
            f"Expected a list for {name}, got {type(values).__name__}."
        )
    for i, v in enumerate(values):
        if not isinstance(v, str):
            raise InputTypeError(
                f"Expected all elements of {name} to be strings, "
                f"got {type(v).__name__} at index {i}."
            )


def validate_pairs(pairs, name: str = "pairs") -> None:
    """Validates that ``pairs`` is a list of ``(text, target_group)`` string tuples."""
    if not isinstance(pairs, list):
        raise InputTypeError(f"Expected a list for {name}, got {type(pairs).__name__}.")
    for i, item in enumerate(pairs):
        if not (isinstance(item, tuple) and len(item) == 2):
            raise InputTypeError(
                f"Expected each element of {name} to be a (text, target_group) tuple, "
                f"got {type(item).__name__} at index {i}."
            )
        text, target_group = item
        if not isinstance(text, str):
            raise InputTypeError(
                f"Expected a string for text in {name}[{i}], got {type(text).__name__}."
            )
        if not isinstance(target_group, str):
            raise InputTypeError(
                f"Expected a string for target_group in {name}[{i}], "
                f"got {type(target_group).__name__}."
            )


def validate_positive_int(value, name: str) -> None:
    """Validates that ``value`` is a positive (>= 1) integer."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise InputTypeError(f"Expected an int for {name}, got {type(value).__name__}.")
    if value < 1:
        raise InputValueError(f"{name} must be a positive integer, got {value}.")

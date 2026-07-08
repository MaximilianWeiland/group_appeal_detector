"""Custom exception types raised by the group_appeal_detector package.

All exceptions defined here inherit from :class:`GroupAppealDetectorError`,
so callers can catch that single class to handle any package-specific
failure. Each subclass also inherits from the relevant built-in exception
(``TypeError`` or ``ValueError``), so existing code that catches those
built-ins keeps working unchanged.
"""


class GroupAppealDetectorError(Exception):
    """Base class for all errors raised by group_appeal_detector."""


class InputTypeError(GroupAppealDetectorError, TypeError):
    """Raised when an argument has an unexpected type."""


class InputValueError(GroupAppealDetectorError, ValueError):
    """Raised when an argument has an invalid value."""


class ModelLoadError(GroupAppealDetectorError):
    """Raised when a pretrained model, tokenizer, or checkpoint fails to load."""

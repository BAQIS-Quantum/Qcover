"""Exceptions for errors raised by Qcover."""

from typing import Optional
import warnings


class QcoverError(Exception):
    """Base class for errors raised by Qcover."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class GraphTypeError(QcoverError):
    """Raised when a sequence subscript is out of range."""

    def __init__(self, msg):   #, *args
        """Set the error message."""
        # warnings.warn(
        # "The problem is transformed into a dense graph, which is difficult to be solved effectively by Qcover",
        #     stacklevel=2,
        # )
        # self.message = "The problem is transformed into a dense graph, " \
        #                "which is difficult to be solved effectively by Qcover"
        self.name = "GraphTypeError: "
        self.message = msg
        super().__init__(msg)

    def __str__(self):
        return self.name + self.message
        # print("The problem is transformed into a dense graph, which is difficult to be solved effectively by Qcover")


class ArrayShapeError(QcoverError):
    """Raised when a sequence subscript is out of range."""

    def __init__(self, msg):   #, *args
        """Set the error message."""
        # warnings.warn(
        # "The problem is transformed into a dense graph, which is difficult to be solved effectively by Qcover",
        #     stacklevel=2,
        # )
        # self.message = "The problem is transformed into a dense graph, " \
        #                "which is difficult to be solved effectively by Qcover"
        self.name = "ArrayShapeError: "
        self.message = msg
        super().__init__(msg)

    def __str__(self):
        return self.name + self.message
        # print("The problem is transformed into a dense graph, which is difficult to be solved effectively by Qcover")


class UserConfigError(QcoverError):
    """Raised when an error is encountered reading a user config file."""
    def __init__(self, msg):
        self.name = "UserConfigError: "
        self.message = msg
        super().__init__(msg)

    def __str__(self):
        return self.name + self.message


class OptimizerConfigError(QcoverError):
    def __init__(self, msg):
        self.name = "OptimizerConfigError: "
        self.message = msg
        super().__init__(msg)

    def __str__(self):
        return self.name + self.message
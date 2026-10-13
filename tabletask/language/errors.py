"""Exceptions raised when a task cannot be sampled or run."""


class UnsupportedError(RuntimeError):
    """A valid task needs something this simulator does not provide."""

    def __init__(self, operation: str, missing):
        self.operation = operation
        self.missing = tuple(sorted(missing))
        super().__init__(f"unsupported {operation}: {', '.join(self.missing)}")


class SamplingError(RuntimeError):
    """No valid start was found within the sampling budget.

    This does not prove that the start is impossible.
    """

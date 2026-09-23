class JobFailedError(Exception):
    """Raised when a training job fails after all retries."""

    def __init__(self, message: str, last_config: str, attempts: int):
        super().__init__(message)
        self.last_config = last_config
        self.attempts = attempts


class MaxRetriesExceeded(JobFailedError):
    """Raised when 3 retry attempts are exhausted."""

    pass


class ConfigurationError(Exception):
    """Raised when the training config is invalid."""

    pass
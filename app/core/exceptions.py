class VoiceAPIError(Exception):
    """Base exception for voice API."""
    pass


class TaskNotFoundError(VoiceAPIError):
    """Raised when a task ID is not found in the job store."""
    pass


class UnsupportedFormatError(VoiceAPIError):
    """Raised when uploaded file format is not supported."""
    pass


class FileTooLargeError(VoiceAPIError):
    """Raised when uploaded file exceeds size limit."""
    pass

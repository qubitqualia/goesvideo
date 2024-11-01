class InvalidArgumentError(Exception):
    "Raised when invalid arguments are passed to class constructors and/or methods"

    def __init__(self, message):
        if message:
            super().__init__(message)


class UnavailableDataError(Exception):
    def __init__(self, message):
        if message:
            super().__init__(message)


class GenericDownloadError(Exception):
    def __init__(self, message):
        if message:
            super().__init__(message)


class UnspecifiedFontPathError(Exception):
    def __init__(self, message):
        if message:
            super().__init__(message)


class MismatchedBoundsError(Exception):
    def __init__(self, message):
        if message:
            super().__init__(message)


class MismatchedCRS(Exception):
    def __init__(self, message):
        if message:
            super().__init__(message)


class MismatchedImageSize(Exception):
    def __init__(self, message):
        if message:
            super().__init__(message)

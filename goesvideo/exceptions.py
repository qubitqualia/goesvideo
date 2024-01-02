class InvalidArgumentError(Exception):
    "Raised when invalid arguments are passed to class constructors and/or methods"

    def __init__(self, message):
        if message:
            super().__init__(message)

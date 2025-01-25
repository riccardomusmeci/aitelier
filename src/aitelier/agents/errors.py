
class StateStepError(Exception):
    """Raised when an invalid state step is attempted"""
    pass

class ToolExecutionError(Exception):
    """Raised when a tool execution fails"""
    pass

class MaxRetryError(Exception):
    """Raised when the maximum number of retries is reached"""
    pass

class MaxItersError(Exception):
    """Raised when the maximum number of iterations is reached"""
    pass

class ParsingToolError(Exception):
    """Raised when there is an error parsing the tool"""
    pass

class ParsingArgsError(NameError):
    """Raised when there is an error parsing the arguments"""
    pass

class ToolNotFoundError(Exception):
    """Raised when a tool is not found"""
    pass
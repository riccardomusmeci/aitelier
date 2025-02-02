from typing import Any, Dict, List

class StateStepError(Exception):
    """Raised when an invalid state step is attempted"""
    def __init__(self, message: str) -> None:
        self.message = f"""You answer did not provide the step transition correctly. 
This was your last answer: {message}.
Please provide the correct format for the step transition and try again."""
        
    def __repr__(self) -> str:
        return self.message

class ToolExecutionError(Exception):
    """Raised when a tool execution fails"""
    def __init__(self, tool: str, args: Dict[str, Any], error: str) -> None:
        self.message = f"""An error occurred while executing the tool {tool} with the following arguments: {args}
The error message is: {error}

Please check the tool's documentation and the arguments you provided and try again."""

    def __repr__(self) -> str:
        return self.message

class MaxRetryError(Exception):
    """Raised when the maximum number of retries is reached"""
    pass

class MaxItersError(Exception):
    """Raised when the maximum number of iterations is reached"""
    pass

class ParsingToolError(Exception):
    """Raised when there is an error parsing the tool"""
    
    def __init__(self, response: str) -> None:
        self.message = f"""An error occurred while parsing the tool from the response: {response}.

You have to be sure that the your answer is provided in the correct format, like in this example: 
    <tool>tool_name</tool> <args> {{'arg1': value1, 'arg2': value2, ...}} </args>
Also always remember to put the dictionary key name between single quotes, like in this example: {{'arg1': value1, 'arg2': value2, ...}}. 
Finally, remember to respect the data type in the args dictionary, for instance if value1 is a string, it must be enclosed in double quotes, if value2 is a list, it must be enclosed in square brackets, etc."""
        
    def __repr__(self) -> str:
        return self.message

class ParsingArgsError(NameError):
    """Raised when there is an error parsing the arguments"""
    pass

class ToolNotFoundError(Exception):
    """Raised when a tool is not found"""
    
    def __init__(self, tool_name: str, available_tools: List[str]):
        self.message = f"""You selected the tool '{tool_name}', but it is not available in the list of available tools: {available_tools}.
Pick one of tha available tools and try again."""

    def __repr__(self) -> str:
        return self.message
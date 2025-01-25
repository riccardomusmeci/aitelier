import inspect
from typing import Callable, Any

class Tool:
    """A decorator that wraps a function and adds additional functionality to it.
    
    Args:
        function (Callable): the function to be decorated
    """
    def __init__(self, function: Callable):
        self.tool = function

    @property
    def description(self) -> str:
        """Return the tool's code as Python script (definition, args, source code)
        
        Returns:
            str: the tool's code as Python script
        """
        code = inspect.getsource(self.tool).replace("@Tool", "")
                
        return code
    
    @property
    def name(self) -> str:
        """Return the tool's name.
        
        Returns:
            str: the tool's name
        """
        return self.tool.__name__
    
    @property
    def docstring(self) -> str:
        """Return the tool's docstring.
        
        Returns:
            Optional[str]: the tool's docstring
        """
        if self.tool.__doc__ is None:
            return "No docstring available."
        else:
            return self.tool.__doc__
    
    @property
    def args(self) -> str:
        """Return the tool's arguments.
        
        Returns:
            str: the tool's arguments
        """
        return str(inspect.signature(self.tool))

    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool with the given arguments.
        
        Returns:
            Any: the result of the tool's execution
        """
        try:
            result = self.tool(*args, **kwargs)
        except Exception as e:
            result = str(e)
        return result

    def __call__(self, *args, **kwargs) -> Any:
        """Make the decorator itself callable to maintain the expected behavior of a decorator.
        
        Returns:
            Any: the result of the tool's execution
        """
        return self.execute(*args, **kwargs)

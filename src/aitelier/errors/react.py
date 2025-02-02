from typing import Optional

class ReActParsingToolError(Exception):
    """Raised when there is an error parsing the tool"""
    
    def __init__(self, response: str, stop_word: Optional[str] = None):
        self.message = f"""
While parsing tool got the error "ParsingToolError".
Your previous answer was: {response}
You have to be sure that the your answer is provided in the correct format, like in this example: 
    <state>Act</state><content><tool>tool_name</tool> <args> {{'arg1': value1, 'arg2': value2, ...}} </args></content> {stop_word if stop_word else ""}
Also always remember to put the dictionary key name between single quotes, like in this example: {{'arg1': value1, 'arg2': value2, ...}}. 
Finally, remember to respect the data type in the args dictionary, for instance if value1 is a string, it must be enclosed in double quotes, if value2 is a list, it must be enclosed in square brackets, etc."""

    def __repr__(self):
        return self.message

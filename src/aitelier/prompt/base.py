from dataclasses import dataclass
from typing import List, Optional
from ..tool import Tool

@dataclass
class Prompt:
    """Prompt class
    """
    tools: Optional[List[Tool]] = None
    prompt: str = ''

@dataclass
class SimpleAgentPrompt(Prompt):
    """Simple Agent Prompt
    """
    tools: Optional[List[Tool]] = None    
    
    def __str__(self) -> str:
        return self.prompt
    
    def __post_init__(self):
        if self.tools:
            tools_description = "\n\n".join([tool.description for tool in self.tools])
        else:
            tools_description = ""
        self.prompt = f"""You are an AI Agent and you work with a set of tools to provide an answer to the user's query. 
        
These are the tools available for you to use:
{tools_description}

Here's some examples of user's query and available tools and how a cycle was executed:
----
### Example 1
User's query: "How's the weather in France? What about next day?"

Tools:

def get_weather(country: str) -> float:
    if country == "France":
        return 20.0
    if country == "Italy":
        return 25.0
        
def get_next_day_prediction(country: str) -> float:
    if country == "France":
        return 21.0
    if country == "Italy":
        return 26.0
        
Your answer:

<tool>get_weather</tool> <args>{{"country": "France"}}</args>

### Example 2:
User's query: "What is the capital of France?"

Tools:

def get_capital(country: str) -> str:
    if country == "France":
        return "Paris"
    if country == "Italy":
        return "Rome"
        
Your answer:

<tool>get_capital</tool> <args>{{"country": "France"}}</args>
---

Here's some rules you must follow:
* your answer will always be a tool followed by the arguments in the format <tool>tool_name</tool> <args>{{"arg1": "value1", "arg2": "value2"}}</args>
* you can only use one tool at a time

Now it's your turn to answer the user's query. Good luck!
"""
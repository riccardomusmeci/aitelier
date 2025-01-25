from typing import Dict, Any, Optional, List, Union
from ..model import Model
from ..tool import Tool
from .base import AgentContext
from ..prompt import SimpleAgentPrompt, Prompt # type: ignore
from .base import State, StateType

class SimpleAgentStateType(StateType):
    AGENT = "AGENT"
    END = "END"
    ERROR = "ERROR"

class EndState(State):
    """End state represents the end of the agent.

    Args:
        final_response (str): the final response
    """
    def __init__(self, final_response: str) -> None:
        self.final_response = final_response
        self.state_type = SimpleAgentStateType.END
        
    def execute(self, context: AgentContext) -> "EndState":
        """Execute the End state.
        
        Args:
            context (AgentContext): the agent context
            
        Returns:
            EndState: the End state
        """
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return the metadata of the End state.
        
        Returns:
            Dict[str, Any]: the metadata
        """
        return {"response": self.final_response}    

class ErrorState(State):
    """Error state represents an error in the agent.

    Args:
        error_message (str): the error message
    """
    def __init__(self, error_message: str) -> None:
        self.error_message = error_message
        self.state_type = SimpleAgentStateType.ERROR
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return the metadata of the Error state.
        
        Returns:
            Dict[str, Any]: the metadata
        """
        return {"error_message": self.error_message}

class AgentState(State):
    
    def __init__(self) -> None:
        self.state_type = SimpleAgentStateType.AGENT
        self.tool_name = None
        self.tool_args = None
        self.input_tokens: List[int] = []
        self.output_tokens: List[int] = []
        self.generation_time: List[float] = []
    
    def execute(self, context: AgentContext) -> Union[EndState, ErrorState]:
        """Execute the Agent state.
        
        Args:
            context (AgentContext): the agent context
        """
        
        response = context.model.generate(
            context.memory,
            max_tokens=context.max_tokens,
            stop_word=context.stop_word
        )
        self.input_tokens.append(context.model.input_tokens[-1])
        self.output_tokens.append(context.model.output_tokens[-1])
        self.generation_time.append(context.model.generation_time[-1])
        try:
            self.tool_name = response.split("<tool>")[-1].split("</tool>")[0] # type: ignore
            self.tool_args = eval(response.split("<args>")[-1].split("</args>")[0]) # type: ignore
        except Exception as e:
            context.add_to_memory(
                "system", 
                f"Error parsing tool data: {e}. Your answer is {response}. Fix it to continue."
            )
            return ErrorState(f"Error parsing tool data: {e}. Your answer is {response}. Fix it to continue.")        
        try:
            tool = context.tools[self.tool_name] # type: ignore
        except KeyError:
            context.add_to_memory(
                "system", 
                f"Tool {self.tool_name} not found. Your answer is {response}. Fix it to continue."
            )
            return ErrorState(f"Tool {self.tool_name} not found. Your answer is {response}. Fix it to continue.")
        try:
            result = tool(**self.tool_args) # type: ignore
        except Exception as e:
            context.add_to_memory(
                "system", 
                f"Error executing tool {self.tool_name} with args {self.tool_args}: {e}. Your answer is {response}. Fix it to continue."
            )
            return ErrorState(f"Error executing tool {self.tool_name}: {e}. Your answer is {response}. Fix it to continue.")
                
        return EndState(result)
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "generation_time": self.generation_time
        }

class Agent:
    """A simple agent that can interact with tools.
    
    Args:
        model (Model): large language model
        tools (List[Tool]): the tools available for the agent
        system_prompt (Prompt, optional): the system prompt to use. Defaults to SimpleAgentPrompt.
        stop_word (Optional[str], optional): the stop word to use. Defaults to None.
        max_tokens (int, optional): the maximum number of tokens to generate. Defaults to 1024.
    """
    
    def __init__(
        self,
        model: Model,
        tools: List[Tool],
        system_prompt: Prompt = SimpleAgentPrompt, # type: ignore
        stop_word: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> None:

        self.system = system_prompt(tools) # type: ignore
        self.context = AgentContext(
            memory=[{"role": "system", "content": self.system.prompt}],
            model=model,
            tools={tool.name: tool for tool in tools},
            stop_word=stop_word,
            max_tokens=max_tokens,
        )
        self.state = AgentState()
        self.output_state = None
        self.stop_word = stop_word
 
    def __call__(self, message: str):
        """Start the Simple Agent with a message from the user.

        Args:
            message (str): the user message
        """
        print("\n************** Starting Agent **************\n")
        print(f"User: {message}")
        # first step is always Think
        self.context.add_to_memory("user", f"{message}")
        self.output_state = self.state.execute(self.context) # type: ignore
        if isinstance(self.output_state, EndState):
            print(f"Agent Answer: {self.output_state.final_response}")
        elif isinstance(self.output_state, ErrorState):
            print(f"Agent Error: {self.output_state.error_message}")
        
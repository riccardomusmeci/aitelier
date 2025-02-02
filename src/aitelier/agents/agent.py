from typing import List, Dict, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from ._base import StateType, AgentState, StartState, EndState, FSMAgent, AgentContext
from ..tool import Tool
from ..model import Model
from ..prompt.agent import Prompt, AgentPrompt
from ..errors import (
    ParsingToolError, ToolNotFoundError, ToolExecutionError
)

class AgentStateType(StateType):
    LLM = "LLM"

# Default ReAct state transitions
AGENT_VALID_STEPS = {
    AgentStateType.START: {AgentStateType.LLM},
    AgentStateType.LLM: {AgentStateType.LLM, AgentStateType.END, AgentStateType.ERROR},
    AgentStateType.ERROR: {AgentStateType.LLM},
    AgentStateType.END: {},
}

@dataclass
class AgentStartState(StartState):
    
    def execute(self, context: AgentContext) -> "AgentLLMState":
        """Execute the start state and return the LLM state.

        Args:
            context (AgentContext): the agent context

        Returns:
            LLMState: the LLM state
        """
        context.add_to_memory("user", self.message)
        return AgentLLMState()

@dataclass
class AgentLLMState(AgentState):
    state_type = AgentStateType.LLM
    tool_tag: str = "tool"
    args_tag: str = "args"
    end_tag: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def _update_metadata(self, context: AgentContext):
        """Update the metadata with the latest inference data.

        Args:
            context (AgentContext): the agent context
        """
        if "input_tokens" not in self.metadata:
            self.metadata["input_tokens"] = []
            self.metadata["output_tokens"] = []
            self.metadata["inference_time"] = []
            
        self.metadata["input_tokens"].append(context.model.input_tokens[-1])
        self.metadata["output_tokens"].append(context.model.output_tokens[-1])
        self.metadata["inference_time"].append(context.model.inference_time[-1])
    
    def _parse_response(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """Extract the tool name and arguments from the response.

        Args:
            response (str): the response from LLM
            
        Raises:
            ParsingToolError: if there is an error parsing the tool

        Returns:
            Tuple[str, Dict[str, Any]]: tool name and tool arguments
        """
        state_response = response.split(self.end_tag)[0] if self.end_tag else response
        state_response = state_response.strip()
        
        try:
            tool_name = state_response.split(f"<{self.tool_tag}>")[-1].split(f"</{self.tool_tag}>")[0]
            tool_args = eval(state_response.split(f"<{self.args_tag}>")[-1].split(f"</{self.args_tag}>")[0])
        except Exception:
            raise ParsingToolError(response=response)
        return tool_name, tool_args
    
    def execute(self, context: AgentContext) -> Union[EndState, "AgentErrorState", "AgentLLMState"]:
        """Execute the LLM state.

        Args:
            context (AgentContext): the agent context

        Returns:
            Union[EndState, ErrorState, LLMState]: the next state
        """
        response = context.model.generate(
            context.memory,
            max_tokens=context.max_tokens,
            stop_word=context.stop_word
        )
        
        # update state stats
        self._update_metadata(context)
        
        context.add_to_memory("assistant", response)
        
        # 1) parse the response - if error, return AgentErrorState with ParsingToolError
        try:
            # this should be the last memory content ("Act: ...")
            tool_name, tool_args = self._parse_response(response) # type: ignore
        except ParsingToolError:
            self.metadata["error"] = f"LLM answer: {response}\nError: ParsingToolError"
            return AgentErrorState(ParsingToolError(response=response))
        
        # this means the agent was not able to find a valid tool
        if tool_name == "None":
            result = response
        else:
            # 2) check if the tool exists
            try:
                tool = context.tools[tool_name] # type: ignore
            except KeyError:
                available_tools = list(context.tools.keys())
                self.metadata["error"] = f"LLM answer: {response}\nError: ToolNotFoundError"
                return AgentErrorState(ToolNotFoundError(tool_name, available_tools))
            
            # 3) run the tool with the arguments
            try:
                result = tool(**tool_args) # type: ignore
            except Exception as e:
                self.metadata["error"] = f"LLM answer: {response}\nError: ToolExecutionError ({e})"
                return AgentErrorState(ToolExecutionError(tool_name, tool_args, str(e)))
            # 4) validate transition before returning the next state
            context.validate_step(self.state_type, AgentStateType.END) # type: ignore
            self.metadata = {"tool": tool_name, "args": tool_args, "result": result}
        # 5) return the Observe state
        return EndState(result)

@dataclass
class AgentErrorState(AgentState):
    """Error state represents an error in the agent.

    Args:
        error (Exception): the error
    """
    
    error: Exception
    state_type: str = AgentStateType.ERROR
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.error = str(self.error.message)
        
    def execute(self, context: AgentContext) -> AgentLLMState:
        """Execute the Error state.

        Args:
            context (AgentContext): the agent context

        Returns:
            ThinkState: the next state
        """
        # I need to add the error to the memory (Error: ...)
        context.add_to_memory("assistant", f"Error: {self.error}")
        self.metadata["error"] = self.error
        return AgentLLMState()

class Agent(FSMAgent):
    
    def __init__(
        self, 
        model: Model,
        tools: List[Tool],
        system_prompt: Prompt = AgentPrompt, # type: ignore
        valid_steps: Dict[str, Set[str]] = AGENT_VALID_STEPS, # type: ignore
        start_state: StartState = AgentStartState, # type: ignore
        stop_word: Optional[str] = None,
        max_iters: int = 10,
        max_tokens: int = 1024,
        max_retries: int = 3
    ) -> None:
        super().__init__(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            valid_steps=valid_steps,
            start_state=start_state,
            stop_word=stop_word,
            max_iters=max_iters,
            max_tokens=max_tokens,
            max_retries=max_retries
        )
        

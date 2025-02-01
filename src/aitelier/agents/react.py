from typing import Dict, Any, List, Union, Set, Tuple, Optional
from dataclasses import dataclass, field
from ._base import AgentContext, AgentState, StartState, StateType, EndState, FSMAgent
from ..model import Model
from ..tool import Tool
from ..prompt import ReActSystemPrompt, Prompt
from ..errors import ToolExecutionError, MaxRetryError, ToolNotFoundError, ReActParsingToolError, StateStepError

################################### State Types ###################################
class ReActStateType(StateType):
    THINK = "THINK"
    ACT = "ACT"
    OBSERVE = "OBSERVE"

################################### Valide Steps ###################################

# Default stop word
REACT_STOP_WORD = "PAUSE"

# Default ReAct state transitions
REACT_VALID_STEPS = {
    ReActStateType.START: {ReActStateType.THINK},
    ReActStateType.THINK: {ReActStateType.THINK, ReActStateType.ACT, ReActStateType.END, ReActStateType.ERROR},
    ReActStateType.ACT: {ReActStateType.OBSERVE, ReActStateType.ERROR},
    ReActStateType.OBSERVE: {ReActStateType.THINK, ReActStateType.ACT, ReActStateType.ERROR},
    ReActStateType.END: set(),
    ReActStateType.ERROR: {ReActStateType.THINK}
}

################################### Agent States ###################################

@dataclass
class ReActStartState(StartState):
    """Start state represents the start of the agent.
    """
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    state_type: str = ReActStateType.START
        
    def execute(self, context: AgentContext) -> "ReActThinkState":
        """Execute the Start state.

        Args:
            context (AgentContext): the agent context

        Returns:
            ThinkState: the next state
        """
        context.add_to_memory(role="user", content=self.message)
        self.metadata["message"] = self.message
        return ReActThinkState()

@dataclass
class ReActThinkState(AgentState):
    
    retry_count: int = 0
    state_type: str = ReActStateType.THINK
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def _update_metadata(self, context: AgentContext):
        """Update the statistics of the Think state.

        Args:
            context (AgentContext): agent context
        """
        if "input_tokens" not in self.metadata:
            self.metadata["input_tokens"] = []
            self.metadata["output_tokens"] = []
            self.metadata["llm_time"] = []
            
        self.metadata["input_tokens"].append(context.model.input_tokens[-1])
        self.metadata["output_tokens"].append(context.model.output_tokens[-1])
        self.metadata["llm_time"].append(context.model.inference_time[-1])
    
    def _get_next_step(self, response: str) -> str:
        """Get the next step from the response.

        Args:
            response (str): the response

        Returns:
            str: the next step
        """
        return response.strip().split(":")[0]
        
    def execute(self, context: AgentContext) -> Union["ReActActState", "EndState", "ReActThinkState", "ReActErrorState"]:
        """Execute the Think state.

        Args:
            context (AgentContext): the agent context

        Raises:
            Exception: if the maximum number of retries is exceeded

        Returns:
            Union[ActState, EndState, ThinkState, ErrorState]: the next state
        """
        if self.retry_count >= context.max_retries:
            raise MaxRetryError(f"Max retries {context.max_retries} reached. Ending the agent.")
        
        response = context.model.generate(
            messages=context.memory,
            stop_word=context.stop_word, 
            max_tokens=context.max_tokens
        )
        context.add_to_memory("assistant", response)
        # update state stats
        self._update_metadata(context)
        
        # get next step
        next_step = self._get_next_step(response).lower()
        
        # check the next step
        if next_step == "think":
            context.validate_step(self.state_type, ReActStateType.THINK) # type: ignore
            return ReActThinkState(self.retry_count + 1)
        elif next_step == "act":
            context.validate_step(self.state_type, ReActStateType.ACT)
            return ReActActState(end_tag=context.stop_word)
        elif next_step == "end":
            context.validate_step(self.state_type, ReActStateType.END)
            return EndState(response)
        else:
            context.validate_step(self.state_type, ReActStateType.ERROR)
            error_message = f"Expected 'Think', 'Act', or 'End' at the beginning of the answer, got: {response}\nYour next answer must start with 'Think:', 'Act:', or 'End:'."
            self.metadata["error"] = "LLM answer: {response}\nError: {error_message}"
            return ReActErrorState(
                StateStepError(error_message)
            )
                

@dataclass
class ReActErrorState(AgentState):
    """Error state represents an error in the agent.

    Args:
        error (Exception): the error
    """
    
    error: Exception
    state_type: str = ReActStateType.ERROR
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.error = str(self.error.message)
        
    def execute(self, context: AgentContext) -> ReActThinkState:
        """Execute the Error state.

        Args:
            context (AgentContext): the agent context

        Returns:
            ThinkState: the next state
        """
        # I need to add the error to the memory (Error: ...)
        context.add_to_memory("assistant", f"{self.state_type}: {self.error}")
        self.metadata["error"] = self.error
        return ReActThinkState(retry_count=0)


@dataclass
class ReActActState(AgentState):
    """Act state represents the execution of a tool.
    """
    state_type: str = ReActStateType.ACT
    start_tag: str = "Act:"
    tool_tag: str = "tool"
    args_tag: str = "args"
    end_tag: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
        
    def _parse_response(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """Extract the tool name and arguments from the response.

        Args:
            response (str): the response from LLM
            
        Raises:
            ReActParsingToolError: if there is an error parsing the tool

        Returns:
            Tuple[str, Dict[str, Any]]: tool name and tool arguments
        """
        state_response = response.split(self.end_tag)[0] if self.end_tag else response
        state_response = state_response.replace(self.start_tag, "").strip()
        try:
            tool_name = state_response.split(f"<{self.tool_tag}>")[-1].split(f"</{self.tool_tag}>")[0]
            tool_args = eval(state_response.split(f"<{self.args_tag}>")[-1].split(f"</{self.args_tag}>")[0])
        except Exception:
            raise ReActParsingToolError(response=response, stop_word=self.end_tag)
        return tool_name, tool_args
        
    def execute(self, context: AgentContext) -> Union["ReActObserveState", "ReActErrorState"]:
        """Execute the Act state.
        
        Args:
            context (AgentContext): the agent context
            
        Raises:
            ToolExecutionError: if the tool execution fails
            
        Returns:
            Union[ObserveState, ErrorState]: the next state
        """
        # last LLM response (the one with Act: ...)
        response = context.memory[-1]["content"]
        
        # 1) parse the response - if error, return ErrorState with ReActParsingToolError
        try:
            # this should be the last memory content ("Act: ...")
            tool_name, tool_args = self._parse_response(response) # type: ignore
        except ReActParsingToolError:
            self.metadata["error"] = f"LLM answer: {response}\nError: ReActParsingToolError"
            return ReActErrorState(
                ReActParsingToolError(response=response, stop_word=self.end_tag)
            )
        
        # 2) check if the tool exists
        try:
            tool = context.tools[tool_name] # type: ignore
        except KeyError:
            available_tools = list(context.tools.keys())
            self.metadata["error"] = f"LLM answer: {response}\nError: ToolNotFoundError"
            return ReActErrorState(ToolNotFoundError(tool_name, available_tools))
        
        # 3) run the tool with the arguments
        try:
            result = tool(**tool_args) # type: ignore
        except Exception as e:
            self.metadata["error"] = f"LLM answer: {response}\nError: ToolExecutionError ({e})"
            return ReActErrorState(ToolExecutionError(tool_name, tool_args, str(e)))
        
        # 4) validate transition before returning the next state
        context.validate_step(self.state_type, ReActStateType.OBSERVE) # type: ignore
        self.metadata = {"tool": tool_name, "args": tool_args}
        
        # 5) return the Observe state
        return ReActObserveState(result)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return the metadata of the Act state.

        Returns:
            Dict[str, Any]: the metadata
        """
        return self.metadata
@dataclass
class ReActObserveState(AgentState):
    """Observe state represents the observation of the tool execution.

    Args:
        result (str): the result of the tool execution
        state_type (str, optional): the state type. Defaults to ReActStateType.OBSERVE.
    """
    result: str
    state_type: str = ReActStateType.OBSERVE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def execute(self, context: AgentContext) -> ReActThinkState:
        """Execute the Observe state.
        
        Args:
            context (AgentContext): the agent context
            
        Returns:
            Union[ThinkState, ErrorState]: the next state
        """
        content = f"Observe: {self.result}"
        context.add_to_memory("assistant", content)
        context.validate_step(self.state_type, ReActStateType.THINK) # type: ignore
        self.metadata["result"] = self.result
        return ReActThinkState()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return the metadata of the Observe state.

        Returns:
            Dict[str, Any]: the metadata
        """
        return self.metadata

################################### ReAct Agent ###################################
class ReActAgent(FSMAgent):
    
    def __init__(
        self,
        model: Model,
        tools: List[Tool],
        system_prompt: Prompt = ReActSystemPrompt, # type: ignore
        valid_steps: Dict[str, Set[str]] = REACT_VALID_STEPS, # type: ignore
        start_state: StartState = ReActStartState, # type: ignore
        stop_word: Optional[str] = REACT_STOP_WORD,
        max_iters: int = 20,
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
    
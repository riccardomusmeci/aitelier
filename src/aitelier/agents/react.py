from typing import Dict, Any, List, Union, Set, Tuple, Optional
from ..model import Model
from ..tool import Tool
from datetime import datetime
from .base import AgentContext, AgentStep, State, StateType
from ..prompt import ReActSystemPrompt, Prompt
from .errors import ToolExecutionError, MaxRetryError, ParsingToolError, ToolNotFoundError

# Default ReAct state types
class ReActStateType(StateType):
    THINK = "THINK"
    ACT = "ACT"
    OBSERVE = "OBSERVE"
    END = "END"
    ERROR = "ERROR"

# default stop word
DEFAULT_STOP_WORD = "PAUSE"

# Default ReAct state transitions
DEFAULT_REACT_VALID_STEPS = {
    ReActStateType.THINK: {ReActStateType.THINK, ReActStateType.ACT, ReActStateType.END, ReActStateType.ERROR},
    ReActStateType.ACT: {ReActStateType.OBSERVE, ReActStateType.ERROR},
    ReActStateType.OBSERVE: {ReActStateType.THINK, ReActStateType.ACT, ReActStateType.ERROR},
    ReActStateType.END: set(),
    ReActStateType.ERROR: {ReActStateType.END}
}

class ErrorState(State):
    """Error state represents an error in the agent.

    Args:
        error (Exception): the error
    """
    def __init__(self, error: Exception) -> None:
        self.error = str(error)
        self.state_type = ReActStateType.ERROR
        
    def execute(self, context: AgentContext) -> Union["ThinkState"]:
        """Execute the Error state.

        Args:
            context (AgentContext): the agent context

        Returns:
            ThinkState: the next state
        """
        context.add_to_memory("assistant", f"{self.state_type}: {self.error}")
        return ThinkState(retry_count=0)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return the metadata of the Error state.
        
        Returns:
            Dict[str, Any]: the metadata
        """
        return {"error_type": type(self.error).__name__, "error_message": str(self.error)}

class ThinkState(State):
    
    def __init__(self, retry_count: int = 0):
        """Initialize the Think state.

        Args:
            retry_count (int, optional): the number of retries. Defaults to 0.
        """
        self.retry_count = retry_count
        self.state_type = ReActStateType.THINK
        self.input_tokens: List[int] = []
        self.output_tokens: List[int] = []
        self.generation_time: List[float] = []
        
    def execute(self, context: AgentContext) -> Union["ActState", "EndState", "ThinkState", "ErrorState"]:
        """Execute the Think state.

        Args:
            context (AgentContext): the agent context

        Raises:
            Exception: if the maximum number of retries is exceeded

        Returns:
            Union[ActState, EndState, ThinkState, ErrorState]: the next state
        """
        if self.retry_count >= context.max_retries:
            raise MaxRetryError(f"Max retries {context.max_retries} reached.")
        
        response = context.model.generate(
            messages=context.memory,
            stop_word=context.stop_word, 
            max_tokens=context.max_tokens
        )
        
        self.input_tokens.append(context.model.input_tokens[-1])
        self.output_tokens.append(context.model.output_tokens[-1])
        self.generation_time.append(context.model.generation_time[-1])
        # possible transitions: Think, Act, or End
        if response.strip().startswith("Think:"):
            context.validate_step(self.state_type, ReActStateType.THINK) # type: ignore
            context.add_to_memory("assistant", response)
            return ThinkState(self.retry_count + 1)
        elif response.strip().startswith("Act:"): 
            context.validate_step(self.state_type, ReActStateType.ACT) # type: ignore
            context.add_to_memory("assistant", response)
            return ActState()
        elif response.strip().startswith("End:"):
            context.validate_step(self.state_type, ReActStateType.END) # type: ignore
            context.add_to_memory("assistant", response)
            return EndState(response)
        elif response.strip().startswith("Think:"):
            context.validate_step(self.state_type, ReActStateType.THINK) # type: ignore
            context.add_to_memory("assistant", response)
            return ThinkState(self.retry_count + 1)
        else:
            context.validate_step(self.state_type, ReActStateType.ERROR) # type: ignore
            return ErrorState(
                Exception(f"""Expected 'Think', 'Act', or 'End' at the beginning of the answer, got: "{response}". Your next answer must start with 'Think:', 'Act:', or 'End:'.""")
            )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return the metadata of the Think state.

        Returns:
            Dict[str, Any]: the metadata
        """
        return {
            "retry_count": self.retry_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "generation_time": self.generation_time
        }

class ActState(State):
    """Act state represents the execution of a tool.
    """
    def __init__(self) -> None:
        self.state_type = ReActStateType.ACT
        self.tool_name, self.tool_args = None, None
        
    def _extract(self, content: str, stop_word: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Extract the tool name and arguments from the content.

        Args:
            content (str): the content
            stop_word (Optional[str], optional): the stop word. Defaults to None.
            
        Raises:
            ParsingToolError: if there is an error parsing the tool

        Returns:
            Tuple[str, Dict[str, Any]]: tool name and tool arguments
        """
        if stop_word:
            act_content = content.split(stop_word)[0].replace("Act: ", "")
        else:
            act_content = content.replace("Act: ", "")
        try:
            tool_name = act_content.split("<tool>")[-1].split("</tool>")[0]
            tool_args = eval(act_content.split("<args>")[-1].split("</args>")[0])
        except Exception as e:
            raise ParsingToolError(
                f"""While parsing tool got this error {str(e)}.
Your previous answer was: {act_content}
You have to be sure that the Act state is provided in the correct format, like in this example: <tool>tool_name</tool> <args> {{'arg1': value1, 'arg2': value2, ...}} </args> {stop_word if stop_word else ""}.
Also always remember to put the dictionary key name between single quotes, like in this example: {{'arg1': value1, 'arg2': value2, ...}}. Finally, remember to respect the data type in the args dictionary, for instance if value1 is a string, it must be enclosed in double quotes, if value2 is a list, it must be enclosed in square brackets, etc."""
            )
        return tool_name, tool_args
        
    def execute(self, context: AgentContext) -> Union["ObserveState", "ErrorState"]:
        """Execute the Act state.
        
        Args:
            context (AgentContext): the agent context
            
        Raises:
            ToolExecutionError: if the tool execution fails
            
        Returns:
            Union[ObserveState, ErrorState]: the next state
        """
        try:
            # this should be the last memory content ("Act: ...")
            memory_content = context.memory[-1]["content"]
            self.tool_name, self.tool_args = self._extract(memory_content, context.stop_word) # type: ignore
        except ParsingToolError as e:
            return ErrorState(e)
        # check if the tool exists
        if self.tool_name not in context.tools:
            return ErrorState(
                ToolNotFoundError("Selected tool not found between tools. Fix it.")
            )
        tool = context.tools[self.tool_name] # type: ignore
        try:
            result = tool(**self.tool_args) # type: ignore
        except Exception as e:
            return ErrorState(
                ToolExecutionError(f"Tool execution failed: {str(e)}")
            )
        # validate transition before returning the next state
        context.validate_step(self.state_type, ReActStateType.OBSERVE) # type: ignore
        return ObserveState(result)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return the metadata of the Act state.

        Returns:
            Dict[str, Any]: the metadata
        """
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
        }

class ObserveState(State):
    """Observe state represents the observation of the tool execution.

    Args:
        observation (str): the observation
    """
    
    def __init__(self, observation: str) -> None:
        self.observation = observation
        self.state_type = ReActStateType.OBSERVE
        
    def execute(self, context: AgentContext) -> Union["ThinkState", "ErrorState"]:
        """Execute the Observe state.
        
        Args:
            context (AgentContext): the agent context
            
        Returns:
            Union[ThinkState, ErrorState]: the next state
        """
        try:
            content = f"Observe: {self.observation}"
            context.add_to_memory("assistant", content)
            context.validate_step(self.state_type, ReActStateType.THINK) # type: ignore
        except Exception as e:
            return ErrorState(e)
        return ThinkState()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return the metadata of the Observe state.

        Returns:
            Dict[str, Any]: the metadata
        """
        return {"observation": self.observation}

class EndState(State):
    """End state represents the end of the agent.

    Args:
        final_response (str): the final response
    """
    def __init__(self, final_response: str) -> None:
        self.final_response = final_response
        self.state_type = ReActStateType.END
        
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

class ReActAgent:
    """A ReAct Agent is a sequence of states that represent the interaction between the user and the agent in the Reason and Act cycle.
    
    Args:
        model (Model): large language model
        tools (List[Tool]): the tools available for the agent
        system_prompt (Prompt, optional): the system prompt to use. Defaults to ReActSystemPrompt.
        valid_steps (Dict[StateType, Set[StateType]], optional): valid state transitions. Defaults to DEFAULT_REACT_VALID_STEPS.
        stop_word (str, optional): the stop word to use. Defaults to DEFAULT_STOP_WORD.
        max_iters (int, optional): the maximum number of iterations. Defaults to 20.
        max_tokens (int, optional): the maximum number of tokens to generate. Defaults to 1024.
        max_retries (int, optional): the maximum number of retries in the Think state. Defaults to 3.
    """
    
    def __init__(
        self,
        model: Model,
        tools: List[Tool],
        system_prompt: Prompt = ReActSystemPrompt, # type: ignore
        valid_steps: Dict[StateType, Set[StateType]] = DEFAULT_REACT_VALID_STEPS, # type: ignore
        stop_word: str = DEFAULT_STOP_WORD,
        max_iters: int = 20,
        max_tokens: int = 1024,
        max_retries: int = 3
    ) -> None:

        self.system = system_prompt(tools) # type: ignore
        self.context = AgentContext(
            memory=[{"role": "system", "content": self.system.prompt}],
            model=model,
            tools={tool.name: tool for tool in tools},
            stop_word=stop_word,
            max_tokens=max_tokens,
            max_retries=max_retries,
            valid_steps=valid_steps
        )
        self.stop_word = stop_word
        self.max_iters = max_iters
        self.agent_steps: List[AgentStep] = []
        
    def _record_step(
        self, 
        from_state: ReActStateType, 
        to_state: ReActStateType, 
        metadata: Dict[str, Any],
    ) -> None:
        """Record a step in the agent.

        Args:
            from_state (ReActStateType): the state to move from
            to_state (ReActStateType): the state to move to
            metadata (Dict[str, Any]): the metadata
        """
        self.agent_steps.append(
            AgentStep(
                from_state=from_state, 
                to_state=to_state,
                timestamp=datetime.now(),
                metadata=metadata
            )
        )
        
    def _print_last_step(self):
        """Print the last step in the agent."""
        if self.agent_steps:
            print(f"\n---------- Step {self.context.step_iter} ----------")
            print(f"> Agent state: {self.agent_steps[-1].from_state}")
            # print(f"> Agent last role: {self.context.memory[-1]['role']}")
            print(f"> {self.context.memory[-1]['content']}")
            print(f"> Agent next state: {self.agent_steps[-1].to_state}")
        
    def __call__(self, message: str):
        """Start the ReAct agent with a message from the user.

        Args:
            message (str): the user message
        """
        print("\n************** Starting ReAct Agent **************\n")
        print(f"User: {message}")
        # first step is always Think
        self.context.add_to_memory("user", f"{message}")
        state = ThinkState()
        while not isinstance(state, EndState):        
            self.context.step_iter += 1
            # recording the state transition
            from_state = state.state_type
            # executing the state
            state = state.execute(self.context) # type: ignore
            # recording the step
            self._record_step(
                from_state=from_state, # type: ignore
                to_state=state.state_type, # type: ignore
                metadata=state.get_metadata()
            )
            # print the last step
            self._print_last_step()
            # TODO: improve this part with ending with Error state
            if self.context.step_iter == self.max_iters and not isinstance(state, EndState):
                print(f"[ERROR] Max iterations {self.max_iters} reached. Last state: {state.state_type}. Ending the agent.")
        print("\n************** ReAct Agent Ended **************\n")
        
        

        
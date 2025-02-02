from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from ..model import Model
from ..tool import Tool
from ..prompt import Prompt
from ..errors import MaxRetryError, StateStepError


################################### Agent Context ###################################

@dataclass
class AgentContext:
    memory: List[Dict[str, str]]
    model: Model
    tools: Dict[str, Tool]
    stop_word: Optional[str] = None
    max_tokens: int = 1024
    max_retries: int = 3
    valid_steps: Optional[Dict[str, Set[str]]] = None
    
    def add_to_memory(self, role: str, content: str):
        """Add a new entry to the memory.

        Args:
            role (str): who is speaking
            content (str): what is being said
        """
        if self.stop_word and not content.strip().endswith(self.stop_word):
            content += f" {self.stop_word}"
        content = content.strip()
        self.memory.append({"role": role, "content": content})
        
    def validate_step(self, from_state: str, to_state: str) -> bool:
        """Validate the step from one state to

        Args:
            from_state (str): state to move from
            to_state (str): state to move to

        Raises:
            StateStepError: if the step is invalid

        Returns:
            bool: True if the step is valid
        """
        if not self.valid_steps: # no check needed
            return True
        if from_state not in self.valid_steps:
            raise StateStepError(f"Invalid step from {from_state} to {to_state}")
        if to_state not in self.valid_steps[from_state]:
            raise StateStepError(
                f"Invalid step from {from_state} to {to_state}. Valid steps are: {self.valid_steps[from_state]}"
            )
        return False


################################### Agent States ###################################

@dataclass    
class StateType:
    START: str = "START"
    ERROR: str = "ERROR"
    END: str = "END"
    
class AgentState(ABC):
    
    @abstractmethod
    def execute(self, context: AgentContext) -> Any:
        raise NotImplementedError      

@dataclass
class StartState(AgentState):
    """Start state represents the beginning of the agent.

    Args:
        message (str): the user message
    """
    message: str
    state_type: str = StateType.START
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, context: AgentContext) -> AgentState:
        """Execute the Start state.

        Args:
            context (AgentContext): the agent context

        Raises:
            NotImplementedError: not implemented

        Returns:
            AgentState: the next state
        """
        raise NotImplementedError

@dataclass
class EndState(AgentState):
    """End state represents the end of the agent.

    Args:
        response (str): the final response
    """
    response: str
    state_type: str = StateType.END
    metadata: Dict[str, Any] = field(default_factory=dict)
        
    def execute(self, context: AgentContext) -> "EndState":
        """Execute the End state.
        
        Args:
            context (AgentContext): the agent context
            
        Returns:
            EndState: the End state
        """
        self.metadata["response"] = self.response
        return self


################################ Base Agent ################################

class FSMAgent:
    """The aitelier Finite State Machine Agent (FSM Agent).
    
    Args:
        model (Model): the model to use
        tools (List[Tool]): the list of tools
        system_prompt (Prompt): the system prompt
        valid_steps (Dict[str, Set[str]]): the valid steps for the agent
        start_state (StartState): the start state
        stop_word (Optional[str]): the stop word
        max_iters (int): the maximum number of iterations
        max_tokens (int): the maximum number of tokens
        max_retries (int): the maximum number of retries
    """
    
    def __init__(
        self,
        model: Model,
        tools: List[Tool],
        system_prompt: Prompt, # type: ignore
        valid_steps: Dict[str, Set[str]], # type: ignore
        start_state: StartState,
        stop_word: Optional[str] = None,
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
        self.start_state = start_state
        self.max_iters = max_iters
        self.states: List[AgentState] = []

    def _print_progress(self):
        """Print the progress of the agent (the last step).
        """
        if len(self.states) == 1:
            from_state = None
        else:
            from_state = self.states[-2].state_type
        print(f"******** Agent Step {len(self.states)} ********")
        if from_state:
            print(f"FSM: '{from_state}' -> '{self.states[-1].state_type}'")
        else:
            print(f"FSM: '{self.states[-1].state_type}'")
        print(f"""Agent: "{self.context.memory[-1]['content']}" """)
        
    def __call__(self, message: str):
        """Start the agent with a message from the user.

        Args:
            message (str): the user message
        """
        # let's start the agent and the stats
        self.states.append(
            self.start_state(message=message) # type: ignore
        )
        # loop over the states
        step_iter = 0
        while not isinstance(self.states[-1], EndState):
            # 1. check if the max iterations is reached
            if step_iter == self.max_iters:
                print(f"[ERROR] Max iterations {self.max_iters} reached. Ending the agent.")
                break 
            # 2. call the state execution with the context
            try:
                new_state = self.states[-1].execute(self.context) # type: ignore
            # if MaxRetryError -> LLM is not able to generate a valid response)
            except MaxRetryError:
                print(f"[ERROR] Reached max retries {self.context.max_retries} with LLM (no valid answer generated). Ending the agent.")
                break
            # 3. print current state
            self._print_progress()
            # 4. record the new state
            self.states.append(new_state)

            # 5. update step_iter
            step_iter += 1
        
        # 6. print final state if only EndState
        if isinstance(self.states[-1], EndState):
            self._print_progress()

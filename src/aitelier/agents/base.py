from dataclasses import dataclass
from ..model import Model
from ..tool import Tool
from abc import ABC
from .errors import StateStepError
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

class StateType(ABC):
    state_type: str = "STATE"
    pass
      
@dataclass        
class AgentStep:
    from_state: StateType
    to_state: StateType
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class AgentContext:
    memory: List[Dict[str, str]]
    model: Model
    tools: Dict[str, Tool]
    stop_word: Optional[str] = None
    max_tokens: int = 1024
    max_retries: int = 3
    valid_steps: Optional[Dict[StateType, Set[StateType]]] = None
    step_iter: int = 0
    
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
        
    def validate_step(self, from_state: StateType, to_state: StateType) -> bool:
        """Validate the step from one state to

        Args:
            from_state (StateType): state to move from
            to_state (StateType): state to move to

        Raises:
            StateStepError: if the step is invalid

        Returns:
            bool: True if the step is valid
        """
        if not self.valid_steps: # no check needed
            return True
        if from_state not in self.valid_steps:
            raise StateStepError(f"Invalid state {from_state} in valid_steps")
        if to_state not in self.valid_steps[from_state]:
            raise StateStepError(
                f"Invalid step from {from_state.state_type} to {to_state.state_type}"
            )
        return False
            
class State:
    def execute(self, context: AgentContext) -> 'State':
        raise NotImplementedError
    
    def get_metadata(self) -> Dict[str, Any]:
        return {}
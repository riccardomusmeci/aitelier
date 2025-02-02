# How to create Custom Agent as FSM - Guidelines

This guide explains how to create a custom agent using our Finite State Machine (FSM) implementation. The FSM approach allows you to create structured, predictable agent behaviors by defining explicit states and transitions between them.

## Basic Concepts

The agent FSM implementation consists of several key components:

1. **States**: Classes that inherit from `AgentState` and define specific behaviors
2. **Transitions**: Valid movements between states defined in `valid_steps`
3. **Context**: The `AgentContext` that maintains the agent's memory and tools
4. **Tools**: Functions that the agent can use to interact with the environment

## Implementation Steps

### 1. Define Your State Types

First, create an enum-like class to define your state types:

```python
class CustomStateType(StateType):
    START = "START"
    CUSTOM = "CUSTOM"  # Your custom states
    ERROR = "ERROR"
    END = "END"
```

### 2. Define Valid Transitions

Create a dictionary that defines valid transitions between states:

```python
CUSTOM_VALID_STEPS = {
    CustomStateType.START: {CustomStateType.CUSTOM},
    CustomStateType.CUSTOM: {CustomStateType.END, CustomStateType.ERROR},
    CustomStateType.ERROR: {CustomStateType.CUSTOM},
    CustomStateType.END: {},
}
```

### 3. Create Custom States

Implement your custom states by inheriting from `AgentState`:

```python
@dataclass
class CustomState(AgentState):
    state_type = CustomStateType.CUSTOM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, context: AgentContext) -> Union[EndState, "CustomErrorState", "CustomState"]:
        # Your state logic here
        try:
            # Process input, use tools, etc.
            result = self._process_input(context)
            return EndState(result)
        except Exception as e:
            return CustomErrorState(e)
```

### 4. Create Your Custom Agent

Inherit from the `FSMAgent` class to create your custom agent:

```python
class CustomAgent(FSMAgent):
    def __init__(
        self,
        model: Model,
        tools: List[Tool],
        system_prompt: Prompt = CustomPrompt,
        valid_steps: Dict[str, Set[str]] = CUSTOM_VALID_STEPS,
        start_state: StartState = CustomStartState,
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
```

## Example Usage

Here's a complete example of how to create and use a custom agent:

```python
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field

# 1. Define your tools
@dataclass
class CustomTool(Tool):
    def __call__(self, **kwargs):
        # Tool implementation
        pass

# 2. Create your states
@dataclass
class CustomStartState(StartState):
    def execute(self, context: AgentContext) -> "CustomState":
        context.add_to_memory("user", self.message)
        return CustomState()

@dataclass
class CustomState(AgentState):
    state_type = CustomStateType.CUSTOM
    
    def execute(self, context: AgentContext):
        # Your state logic
        pass

# 3. Initialize and use your agent
tools = [CustomTool()]
model = Model()  # Your LLM model
agent = CustomAgent(
    model=model,
    tools=tools,
    system_prompt=CustomPrompt(),
    max_iters=5
)

# 4. Run the agent
result = agent.run("Your input message")
```

## Error Handling

The FSM implementation includes built-in error handling through the `ErrorState`. You can customize error handling by:

1. Defining custom exceptions
2. Creating a custom error state
3. Implementing recovery logic in the error state's `execute` method

```python
@dataclass
class CustomErrorState(AgentState):
    error: Exception
    state_type = CustomStateType.ERROR
    
    def execute(self, context: AgentContext) -> CustomState:
        # Log error, implement recovery logic
        context.add_to_memory("assistant", f"Error: {self.error}")
        return CustomState()
```

## Best Practices

1. **State Design**:
   - Keep states focused on a single responsibility
   - Use meaningful names for state types
   - Include appropriate metadata in states for debugging

2. **Error Handling**:
   - Define specific exceptions for different error cases
   - Implement graceful recovery mechanisms
   - Log errors and state transitions for debugging

3. **Context Management**:
   - Keep the context updated with relevant information
   - Use the memory system to maintain conversation history
   - Validate state transitions using the context

4. **Tool Integration**:
   - Design tools to be stateless and reusable
   - Include proper input validation in tools
   - Document tool parameters and return values


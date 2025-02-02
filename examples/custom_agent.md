# How to create Custom Agent as FSM - Practical Example

This guide explains how to create a custom agent using the Finite State Machine (FSM) implementation. The FSM approach allows you to create structured, predictable agent behaviors by defining explicit states and transitions between them.

## Basic Concepts

The agent FSM implementation consists of several key components:

1. **States**: classes that inherit from `AgentState` and define specific behaviors
2. **Prompt**: a system prompt that guides the agent's responses
3. **Steps**: valid movements between states
4. **Context**: maintains the agent's memory and tools (class `AgentContext`)
5. **Tools**: functions that the agent can use to interact with the environment

## Implementation Steps

### 1. Define Your State Types

First, create a class to define your state types:

```python
from aitelier.agents import StateType

class MyAgentStateType(StateType):
    START = "START"
    AGENT = "AGENT"  # it's the LLM
    ERROR = "ERROR"
    END = "END"
```

### 2. Define Valid Transitions

Create a dictionary that defines valid transitions between states:

```python
MY_AGENT_VALID_STEPS = {
    MyAgentStateType.START: {MyAgentStateType.AGENT},
    MyAgentStateType.AGENT: {MyAgentStateType.END, MyAgentStateType.ERROR},
    MyAgentStateType.ERROR: {MyAgentStateType.AGENT},
    MyAgentStateType.END: {},
}
```

### 3. Define System Prompt
It's very important to define a system prompt that will be used by the agent to generate the response. The system prompt should be a class that inherits from the `Prompt`.

```python
from aitelier.prompt import Prompt
from aitelier.tool import Tool
from dataclasses import dataclass

@dataclass
class MyAgentPrompt(Prompt):
    """Agent Prompt
    """
    tools: Optional[List[Tool]] = None    
    
    def __str__(self) -> str:
        return self.prompt
    
    def __post_init__(self):
        if self.tools:
            tools_description = "\n\n".join([tool.description for tool in self.tools])
        else:
            tools_description = ""
        self.prompt = f"""You are an agent and you work with a set of tools to provide an answer to the user's query. 
        
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
* if you get some errors from previous iterations, you must fix your answer according to the suggestions
* if the available tools cannot be used to solve the task, your answer must be structured as follows: <tool>None</tool> <args>{{}}</args>. Then you have to continue by saying why you can't solve the task based on the tools you have. The user needs an explanation of why you can't solve the task.
    
Now it's your turn to answer the user's query. Good luck!
"""
```

Some tips to write a good system prompt for AI Agents to get predictable responses:
- be very specific with how the LLM must response
- imho, use the XML tags to delimit the LLM answer so that it's easier to parse the response and use a stop word at the end of the answer (e.g. PAUSE)
- stop words are already implemented in the `FSMAgent` class (you can change it if you want)


### 4. Implement Custom States

Before implementing the custom states, it's important to introduce some predefined states in the `aitelier` library:
* `StartState`: the initial state of the agent that processes the input message and identifies the first step within the FSM - usually is a step that calls the LLM
* `EndState`: the final state of the agent that returns the result to the user - most of the time you don't need to redefine this state
* `AgentState`: the base class for all custom states

All the custom states must implement the `execute` method, which defines the logic of the state and returns the next state based on the context. Moreover, they need to have a field called `state_type` that defines the type of the state.

Let's implement the custom states:

```python
from aitelier.agents import AgentState, AgentContext, EndState, StartState
from aiteler.errors import ParsingToolError, ToolNotFoundError, ToolExecutionError
from dataclasses import dataclass, field # usually I use dataclasses for states - I like them :)

@dataclass
class MyAgentStartState(StartState): # go checkout the StartState class in the aitelier library and see how it's implemented
    state_type = MyAgentStateType.START
    
    def execute(self, context: AgentContext) -> "MyAgentState":
        # Your start state logic here
        context.add_to_memory("user", self.message) # mandatory to add the user input to the memory 
        self.metadata["message"] = self.message
        return MyAgentState()

@dataclass
class MyAgentState(AgentState):
    state_type = MyAgentStateType.AGENT
    tool_tag: str = "tool"
    args_tag: str = "args"
    end_tag: Optional[str] = None
    
    def _parse_response(self, response: str) -> Tuple[str, Dict[str, Any]]:
        state_response = response.split(self.end_tag)[0] if self.end_tag else response
        state_response = state_response.strip()
        try:
            tool_name = state_response.split(f"<{self.tool_tag}>")[-1].split(f"</{self.tool_tag}>")[0]
            tool_args = eval(state_response.split(f"<{self.args_tag}>")[-1].split(f"</{self.args_tag}>")[0])
        except Exception:
            raise ParsingToolError(response=response)
        return tool_name, tool_args
    
    def execute(self, context: AgentContext) -> Union[EndState, "MyAgentErrorState", "MyAgentState"]:
        """Execute the LLM state.

        Args:
            context (AgentContext): the agent context

        Returns:
            Union[EndState, ErrorState, LLMState]: the next state
        """
        response = context.model.generate(
            context.memory, context.max_tokens, context.stop_word
        )
        context.add_to_memory("assistant", response)        
        # 1) parse the response - if error, return AgentErrorState with ParsingToolError
        if tool_name
        try:
            tool_name, tool_args = self._parse_response(response)
        except ParsingToolError:
            return AgentErrorState(ParsingToolError(response=response))
        # 2) check if the tool exists - if not, return AgentErrorState with ToolNotFoundError
        if tool_name == "None":
            context.validate_step(self.state_type, AgentStateType.END) # type: ignore
            return EndState(response)
        try:
            tool = context.tools[tool_name] # type: ignore
        except KeyError:
            available_tools = list(context.tools.keys())
            return AgentErrorState(ToolNotFoundError(tool_name, available_tools))
        # 3) run the tool with the arguments
        try:
            result = tool(**tool_args) # type: ignore
        except Exception as e:
            return AgentErrorState(ToolExecutionError(tool_name, tool_args, str(e)))
        
        # 4) validate transition before returning the next state
        context.validate_step(self.state_type, AgentStateType.END) # type: ignore
        return EndState(result)

@dataclass
class AgentErrorState(AgentState):
    error: Exception
    state_type: str = MyAgentStateType.ERROR
    
    def __post_init__(self):
        self.error = str(self.error.message)
        
    def execute(self, context: AgentContext) -> MyAgentState:
        # I need to add the error to the memory (Error: ...)
        context.add_to_memory("assistant", f"Error: {self.error}")
        return MyAgentState()
```

### 5. Create Your Custom Agent

Inherit from the `FSMAgent` class to create your custom agent:

```python
from atelier.agents import FSMAgent

class MyCustomAgent(FSMAgent):
    def __init__(
        self,
        model: Model,
        tools: List[Tool],
        system_prompt: Prompt = MyAgentPrompt,
        valid_steps: Dict[str, Set[str]] = MY_AGENT_VALID_STEPS, 
        start_state: StartState = MyAgentStartState,
        stop_word: Optional[str] = None,
        max_iters: int = 10,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            valid_steps=valid_steps,
            start_state=start_state,
            stop_word=stop_word,
            max_iters=max_iters,
            max_tokens=max_tokens
        )
```

## 6. Wrapping up

Here's a complete example of how to create and use a custom agent to interact with the Hugging Face Hub to get the best model in some AI task:

```python
from huggingface_hub import list_models
from aitelier.tool import Tool
from aitelier.model import LLM, Claude

@Tool
def most_download(task: str) -> str:
    """Return the most downloaded model of a given task on the Hugging Face Hub. It returns the name of the checkpoint.

    Args:
        task: The task for which
    """
    try:
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
    except Exception:
        return f"No model found for the {task} task. It is not available on the Hugging Face Hub."
    return model.id

model = LLM("llama_3_2_3b_instruct") # from mlx-llm
model = Claude(...) # from Anthropic API

agent = MyAgent(
    model=model,
    tools=[most_download]
)
agent("What is the most downloaded model for the text generation task?")
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

4. **LLM Choice**:
   - Smaller models tend to be faster but less accurate
   - Bigger and free with models mlx-llm are good
   - Paid and bigger models are the best


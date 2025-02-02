# aitelier 🎨🤖
An atelier for AI Agents running on your Mac thanks to [mlx-llm](https://https://github.com/riccardomusmeci/mlx-llm).

## How to install it
```bash
pip install aitelier
```

## Why `aitelier`? 

`aitelier` is not just another agent library — it’s my personal playground to learn how AI agents work

* **Learning by doing**  🛠️: I want to understand every part of building AI agents. Instead of just using existing frameworks, I’m creating this library to figure out how things work from the ground up.

* **Exploration, not duplication** 🧐: `aitelier` isn’t meant to compete with other agent libraries. My goal is to learn, not to replace. Each update happens when I discover a new concept or method I want to explore.
	
* **Deep understanding through practice** 🧠: building from scratch helps me go beyond the surface. By recreating different paradigms, I can focus on truly understanding the details and improving my AI development skills.

`aitelier` will grow as I dive into new ideas and challenge myself to learn more. 🎯

I wrote an article about the motivation behind `aitelier` and how I implemented the first version of the library. You can read it [here](https://reminiscent-puffin-1cb.notion.site/WTF-are-AI-Agents-Let-s-build-aitelier-17a43b7c0ffb807e8a1bf8f890c1ab2b?pvs=74).

## Agents as FSMs
In `aitelier`, agents are implemented as Finite State Machines (FSMs). Each agent has a set of states and transitions that define how the agent processes the input and produces the output.

Currently, `aitelier` supports two types of agents:
- **Agent**: A basic agent that takes a query and returns a response based on the available tools
- **ReAct Agent**: An agent the follows the Reasoning and Act paradigm [paper](https://arxiv.org/abs/2210.03629) with available tools


## Agent Example
```python
from aitelier.model import LLM
from aitelier.agents import Agent
from aitelier.tool import Tool

@Tool
def multiply(a: float, b: float) -> float:
    return a * b

model = LLM("llama_3_2_3b_instruct")
agent = Agent(model=model, tools=[multiply])
agent("What is 3 multiplied by 4?")
```

## ReAct Agent
The ReAct agent is a more advanced agent that follows the Reasoning and Act paradigm. The agent goes through the Think, Act, Observe states to produce the final answer based on a set of tools.

<div style="text-align: center;">
    <img src="static/react.gif" alt="ReAct Agent in action" width="600">
</div>

This is how you can use the ReAct Agent in `aitelier`:
```python
from aitelier.model import LLM
from aitelier.agents import ReActAgent
from aitelier.tool import Tool

@Tool
def divide(a: float, b: float) -> float:
    if b == 0:
        return "Division by zero is not allowed"
    return a / b

model = LLM("deepseek_r1_distill_llama_8b")
agent = ReActAgent(model=model, tools=[divide])
agent("What is 10 divided by 2?")
```

## Supported Models
`aitelier` natively supports [mlx-llm](https://github.com/riccardomusmeci/mlx-llm) and Claude (with own API key)

## Examples
1) [How to create Custom Agent as FSM - Guideline](examples/custom_agent_guideline.md)

## Known Issues
- `aitelier` is in active development, so expect bugs and breaking changes
- The library is currently only tested on macOS
- All the agents depend on the LLM model - LLMs with fewer parameters are way more buggy than the larger ones since their behavior is less predictable - my advice is to use the largest models available
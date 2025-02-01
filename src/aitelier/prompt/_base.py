from dataclasses import dataclass
from typing import List, Optional
from ..tool import Tool

class Prompt:
    """Prompt class
    """
    tools: Optional[List[Tool]] = None
    prompt: str = ''


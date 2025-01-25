from mlx_llm.model import (
    create_model, 
    create_tokenizer,
    list_models, 
    quantize
)
import mlx.core as mx
from typing import Dict, Optional, List, Generator, Tuple
import logging
import time
from abc import ABC, abstractmethod
from anthropic import Anthropic
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_QUANTIZE_ARGS = {"group_size": 64, "bits": 8}

class Model(ABC):
    
    input_tokens: List[int] = []
    output_tokens: List[int] = []
    generation_time: List[float] = []
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], stop_word: Optional[str] = None, max_tokens: int = 1024) -> str:
        pass
    
    @abstractmethod
    def stream(self, messages: List[Dict[str, str]], stop_word: Optional[str] = None, max_tokens: int = 1024) -> Generator:
        pass

# TODO: support to TokenUsage
class LLM(Model):
    """Large Language Model class.
    
    Args:
        model_name (str): model name
        model_args (dict, optional): model arguments. Defaults to {}.
        quantize_args (dict, optional): quantize arguments. Defaults to DEFAULT_QUANTIZE_ARGS.
        debug (bool, optional): if True, print debug information. Defaults to False.
    """
    def __init__(
        self,
        model_name: str,
        model_args: dict = {},
        quantize_args: dict = DEFAULT_QUANTIZE_ARGS,
        debug: bool = False
    ):
        assert (model_name in list_models()), f"Model {model_name} not found. Available models: {list_models()}"
        self.debug = debug
        try:
            self.model = create_model(model_name, **model_args)
            self.model = quantize(self.model, **quantize_args) if quantize_args else self.model
            self.tokenizer = create_tokenizer(model_name)
            self.model_name = model_name
            if self.debug: 
                print(f"[DEBUG] Model {model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model {model_name}.\nError: {e}")
            raise e
        
    def _tokenize_prompt(self, messages: List[Dict[str, str]]) -> mx.array:
        """Prepare the prompt for the model.
        
        Args:
            messages (List[Dict[str, str]]): list of messages
        Returns:
            mx.array: tokenized prompt
        """
        chat_messages = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.debug:
            print("[DEBUG] ***************")
            print(f"PROMPT\n{chat_messages}")
            print("[DEBUG] ***************")
        tokens = self.tokenizer.encode(chat_messages)
        self.input_tokens.append(len(tokens))
        tokens = mx.array(tokens)
        return tokens
        
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        stop_word: Optional[str] = None,
        max_tokens: int = 1024
    ) -> str:
        """Generate answer from the given prompt.

        Args:
            messages (List[Dict[str, str]]): list of messages
            stop_word (Optional[str]): stop word. Defaults to None.
            max_tokens (int, optional): max number of tokens to generate. Defaults to 1024.

        Returns:
            str: answer
        """
        input_tokens = self._tokenize_prompt(messages)
        start_time = time.time()
        token_count = 0
        output_tokens = []
        alert_word = []
        for token in self.model.generate(input_tokens):
            output_tokens.append(token.item())
            token_count += 1
            # check if the last tokens are a stop word
            last_token_decoded = self.tokenizer.decode([output_tokens[-1]]).strip()
            if stop_word:
                if last_token_decoded in stop_word: 
                    alert_word.append(last_token_decoded)
                    if "".join(alert_word).strip() == stop_word:
                        break
            if len(output_tokens) >= max_tokens or output_tokens[-1] == self.tokenizer.eos_token_id:
                break
        answer = self.tokenizer.decode(
            output_tokens[:-1] if output_tokens[-1] == self.tokenizer.eos_token_id else output_tokens
        )
        if stop_word:
            answer = answer.replace(stop_word, "")
        self.output_tokens.append(len(output_tokens))
        total_time = time.time() - start_time
        tokens_per_second = token_count / total_time
        self.generation_time.append(total_time)
        if self.debug:
            print(f"[DEBUG] Total tokens generated: {token_count}")
            print(f"[DEBUG] Total time taken: {total_time:.2f} seconds")
            print(f"[DEBUG] Tokens per second: {tokens_per_second:.2f}")
        return answer
    
    def stream(
        self, 
        messages: List[Dict[str, str]],
        stop_word: Optional[str] = None,
        max_tokens: int = 1024
    ) -> Generator:
        """Stream the generated answer from agent.

        Args:
            messages (List[Dict[str, str]]): list of messages
            stop_word (Optional[str]): stop word. Defaults to None.
            max_tokens (int, optional): max number of tokens to generate. Defaults to 1024.

        Yields:
            Generator: piece of agent's answer
        """
        input_tokens = self._tokenize_prompt(messages)
        start_time = time.time()
        token_count = 0
        output_tokens = []
        alert_word = []
        for token in self.model.generate(input_tokens):
            output_tokens.append(token.item())
            token_count += 1
            last_token_decoded = self.tokenizer.decode([output_tokens[-1]]).strip()
            if stop_word: 
                if last_token_decoded in stop_word: 
                    alert_word.append(last_token_decoded)
                    if "".join(alert_word).strip() == stop_word:
                        break
            if len(output_tokens) >= max_tokens:
                break
            if output_tokens[-1] == self.tokenizer.eos_token_id:
                break
            yield self.tokenizer.decode([output_tokens[-1]])
        
        # STATS IN DEBUG MODE
        total_time = time.time() - start_time
        self.output_tokens.append(len(output_tokens))
        self.generation_time.append(total_time)

class Claude(Model):
    """Claude wrapper for the Anthropic API.
    
    Args:
        api_key (str): Anthropic API key
        model_name (str): Claude model name
    """
    
    def __init__(self, api_key: str, model_name: str):
        """Initialize the Claude wrapper with your API key."""
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
        
    def _prepare_messages(self, messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        """Prepare messages for the Claude API.
        
        Args:
            messages (List[Dict[str, str]]): List of messages
        
        Returns:
            Tuple[str, List[Dict[str, str]]]: Tuple with system message and filtered messages
        """
        system = ""
        filtered_messages = []
        for message in messages:
            message["content"] = message["content"].strip()
            if message.get('role') == 'system':
                system = message['content']
            else:
                filtered_messages.append(message)
        return system, filtered_messages

    def generate(
        self, 
        messages: List[Dict[str, str]], 
        stop_word: Optional[str] = None,
        max_tokens: int = 1024
    ) -> str:
        """Generate a response from Claude given a list of messages.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with 'content' key
            stop_word (Optional[str]): Stop word to end generation. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.
            
        Returns:
            str: Claude's response text
        """
        
        # extract system messages
        system, messages = self._prepare_messages(messages)

        t_start = time.time()
        response = self.client.messages.create(
            model=self.model_name,
            system=system,
            messages=messages,
            stop_sequences=[stop_word] if stop_word else None,
            max_tokens=max_tokens
        )
        
        # Record token usage
        self.input_tokens.append(response.usage.input_tokens)
        self.output_tokens.append(response.usage.output_tokens)
        self.generation_time.append(time.time() - t_start)
        
        return response.content[0].text

    def stream(
        self, 
        messages: List[Dict[str, str]], 
        stop_word: Optional[str] = None,
        max_tokens: int = 1024
    ) -> Generator[str, None, None]:
        """Stream a response from Claude given a list of messages.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with 'content' key
            stop_word (Optional[str]): Stop word to end generation. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.
            
        Yields:
            str: Chunks of Claude's response text
        """
        system, messages = self._prepare_messages(messages)
        t_start = time.time()
        input_tokens = 0
        output_tokens = 0
        with self.client.messages.stream(
            model=self.model_name,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            stop_sequences=[stop_word] if stop_word else None
        ) as stream:
            for chunk in stream:
                if hasattr(chunk, 'message') and chunk.type == 'message_start':
                    input_tokens = chunk.message.usage.input_tokens
                if hasattr(chunk, 'type'):
                    if chunk.type == 'content_block_delta':
                        yield chunk.delta.text
                    elif chunk.type == "message_stop":
                        output_tokens = chunk.message.usage.output_tokens
                        break
        
        # Record token usage after stream completes
        self.input_tokens.append(input_tokens)
        self.output_tokens.append(output_tokens)
        self.generation_time.append(time.time() - t_start)
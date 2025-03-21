import asyncio, torch, threading, queue
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, EngineArgs
from typing import Optional
from transformers import AutoTokenizer
from .decoder import tokens_decoder_sync

class OrpheusModel:    
    def __init__(self, 
            model_name, 
            dtype=torch.bfloat16,

            seed: int = 0,
            max_model_len: Optional[int] = None,
            cpu_offload_gb: float = 0, # GiB
            gpu_memory_utilization: float = 0.90,
            quantization: Optional[str] = None,
            max_seq_len_to_capture: int = 8192,
            enforce_eager: Optional[bool] = None
        ):
        """
        Initialize the Orpheus Text-to-Speech engine.
        
        This class provides high-quality text-to-speech synthesis using neural models.
        It handles model loading, voice selection, and audio generation with various
        configuration options for performance and quality.
        
        Parameters
        ----------
        model_name : str
            Name or path of the model to use (e.g., "medium-3b" or a custom model path).
        dtype : torch.dtype, default=torch.bfloat16
            Data type for model computation. BFloat16 offers good performance balance.
        seed : int, default=0
            Random seed for reproducible generation.
        max_model_len : Optional[int], default=None
            Maximum sequence length the model can process.
        cpu_offload_gb : float, default=0
            Amount of model weights (in GiB) to offload to CPU when GPU memory is limited.
        gpu_memory_utilization : float, default=0.90
            Target GPU memory utilization (0.0 to 1.0).
        quantization : Optional[str], default=None
            Quantization method for reduced memory usage ("int8", "int4", etc.).
        max_seq_len_to_capture : int, default=8192
            Maximum sequence length for KV cache optimization.
        enforce_eager : Optional[bool], default=None
            Whether to disable CUDA graph optimizations for debugging.
            
        Attributes
        ----------
        model_name : str
            The resolved model path after parameter mapping.
        engine : AsyncLLMEngine
            The underlying inference engine.
        available_voices : list
            Available voice identifiers ("zoe", "zac", etc.).
        tokeniser : AutoTokenizer
            Tokenizer for text processing.
            
        Examples
        --------
        >>> model = OrpheusModel("medium-3b")
        >>> audio = model.generate_speech(prompt="Hello world!", voice="zoe")
        """

        self.model_name = self._map_model_params(model_name)
        self.dtype = dtype
        self.engine = self._setup_engine(seed, max_model_len, cpu_offload_gb, gpu_memory_utilization, quantization, max_seq_len_to_capture, enforce_eager)
        self.available_voices = ["zoe", "zac","jess", "leo", "mia", "julia", "leah"]
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)

    
    def _map_model_params(self, model_name):
        model_map = {
            # "nano-150m":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # }, 
            # "micro-400m":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # }, 
            # "small-1b":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # },
            "medium-3b":{
                "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            },
        }
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
        if (model_name  in unsupported_models):
            raise ValueError(f"Model {model_name} is not supported. Only medium-3b is supported, small, micro and nano models will be released very soon")
        elif model_name in model_map:
            return model_map[model_name]["repo_id"]
        else:
            return model_name
        
    def _setup_engine(
            self,
            seed: int = 0,
            max_model_len: Optional[int] = None,
            cpu_offload_gb: float = 0, # GiB
            gpu_memory_utilization: float = 0.90,
            quantization: Optional[str] = None,
            max_seq_len_to_capture: int = 8192,
            enforce_eager: Optional[bool] = None
        ):
        """
            Sets up and initializes the LLM engine with specified configuration.
            Args:
                seed (int, optional): Random seed for reproducibility. Defaults to 0.
                max_model_len (Optional[int], optional): Maximum sequence length the model can handle. Defaults to None.
                cpu_offload_gb (float, optional): Amount of memory in GiB to offload to CPU. Defaults to 0.
                gpu_memory_utilization (float, optional): Fraction of GPU memory to utilize. Defaults to 0.90 (90%).
                quantization (Optional[str], optional): Quantization method to use. Defaults to None.
                max_seq_len_to_capture (int, optional): Maximum sequence length to capture for optimization. Defaults to 8192.
                enforce_eager (Optional[bool], optional): Whether to enforce eager execution. Defaults to None.
            Returns:
                AsyncLLMEngine: Initialized language model engine ready for inference.
        """
        
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
            max_model_len=max_model_len,
            cpu_offload_gb=cpu_offload_gb,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
            max_seq_len_to_capture=max_seq_len_to_capture,
            enforce_eager=enforce_eager,
            seed=seed
        )

        return AsyncLLMEngine.from_engine_args(engine_args)
    
    def validate_voice(self, voice):
        if voice:
            if voice not in self.engine.available_voices:
                raise ValueError(f"Voice {voice} is not available for model {self.model_name}")
    
    def _format_prompt(self, prompt, voice="tara", model_type="larger"):
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            else:
                return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokeniser(adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokeniser.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokeniser(prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokeniser.decode(all_input_ids[0])
                return prompt_string


    def generate_tokens_sync(self, prompt, voice=None, request_id="req-001", temperature=0.6, top_p=0.8, max_tokens=1200, stop_token_ids = [49158], repetition_penalty=1.3):
        prompt_string = self._format_prompt(prompt, voice)
        print(prompt)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,  # Adjust max_tokens as needed.
            stop_token_ids = stop_token_ids, 
            repetition_penalty=repetition_penalty, 
        )

        token_queue = queue.Queue()

        async def async_producer():
            async for result in self.engine.generate(prompt=prompt_string, sampling_params=sampling_params, request_id=request_id):
                # Place each token text into the queue.
                token_queue.put(result.outputs[0].text)
            token_queue.put(None)  # Sentinel to indicate completion.

        def run_async():
            asyncio.run(async_producer())

        thread = threading.Thread(target=run_async)
        thread.start()

        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

        thread.join()
    
    def generate_speech(self, **kwargs):
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs))
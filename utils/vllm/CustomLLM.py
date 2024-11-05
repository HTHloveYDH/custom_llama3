import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.weight_utils import load_tensor_parallel_weights
from transformers import AutoTokenizer

from models.ModelArgs import ModelArgs
from utils.vllm.CustomModel import CustomModel


class CustomLLM:
    def __init__(self, model_path:str, tokenizer_path:str, params:dict):
        # load model (CustomModel)
        self.model = CustomLLM.load_custom_model(model_path, params)
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # initialize VLLM
        self.llm = LLM(model=self.model, tokenizer=self.tokenizer)
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7,
                 top_p: float = 0.95, stream: bool = False):
        # set generation parameters (SamplingParams)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_tokens=[self.tokenizer.eos_token],
            stream=stream
        )
        # generate text
        if stream:
            return self._stream_generate(prompt, sampling_params)
        else:
            outputs = self.llm.generate(prompt, sampling_params)
            return outputs[0].text
    
    def _stream_generate(self, prompt, sampling_params):
        generator = self.llm.generate(prompt, sampling_params)
        for output in generator:
            yield output.text

    @staticmethod
    def load_custom_model(model_path:str, params:dict):
        # create configuration
        model_args = ModelArgs(**params)
        # initialize model
        model = CustomModel(model_args)
        # load model weights
        checkpoint = torch.load(model_path)
        # load weights mapping
        weight_map = {
            'word_embeddings.weight': checkpoint['embeddings.word_embeddings.weight'],
            'position_embeddings.weight': checkpoint['embeddings.position_embeddings.weight'],
            'ln_f.weight': checkpoint['ln_f.weight'],
            'ln_f.bias': checkpoint['ln_f.bias'],
            'lm_head.weight': checkpoint['lm_head.weight'],
        }
        # 加载Transformer层权重
        for i in range(model_args.n_layers):
            layer_weights = {
                f'layers.{i}.attention.query.weight': 
                    checkpoint[f'transformer.layers.{i}.attention.query.weight'],
                f'layers.{i}.attention.key.weight':
                    checkpoint[f'transformer.layers.{i}.attention.key.weight'],
                f'layers.{i}.attention.value.weight':
                    checkpoint[f'transformer.layers.{i}.attention.value.weight'],
                # ... 其他层权重
            }
            weight_map.update(layer_weights)
        # 加载权重
        load_tensor_parallel_weights(model, weight_map)
        return model

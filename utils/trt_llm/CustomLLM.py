import tensorrt_llm
from tensorrt_llm.runtime import Session, ModelConfig
import numpy as np
from transformers import AutoTokenizer

from models.ModelArgs import ModelArgs


class CustomLLM:
    def __init__(self, engine_path:str, tokenizer_path: str, params:dict):
        model_args = ModelArgs(**params)
        # load configuration
        self.config = ModelConfig(
            max_batch_size=model_args.max_batch_size,
            max_input_len=model_args.max_seq_len,
            max_output_len=model_args.max_seq_len,
            vocab_size=model_args.vocab_size
        )
        # create session
        self.session = Session.from_serialized_engine(engine_path, self.config)
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    def generate(self, prompt:str, max_length:int, temperature:float, top_p:float):
        # encode input text to tokens
        tokens = self.tokenizer.encode(prompt)
        input_len = np.array([len(tokens)], dtype=np.int32)
        # sampling configuration
        sampling_config = tensorrt_llm.runtime.SamplingConfig(
            end_id=self.tokenizer.eos_token_id, pad_id=self.tokenizer.pad_token_id,
            temperature=temperature, top_p=top_p
        )
        # text generation
        output_ids = self.session.generate(
            tokens, input_len, sampling_config, max_length
        )
        # decode output tokens to text
        generated_text = self.tokenizer.decode(output_ids[0])
        return generated_text
    
    def _stream_generate(self, prompt:str, max_length:int, temperature:float, top_p:float):
        # encode input text to tokens
        tokens = self.tokenizer.encode(prompt)
        # sampling configuration
        sampling_config = tensorrt_llm.runtime.SamplingConfig(
            end_id=self.tokenizer.eos_token_id, pad_id=self.tokenizer.pad_token_id,
            temperature=temperature, top_p=top_p
        )
        generator = self.session.generate_stream(
            tokens, sampling_config, max_length=max_length
        )
        # text generation
        generated_text = ""
        for token in generator:
            # decode output tokens to text
            token_text = self.tokenizer.decode([token])
            generated_text += token_text
            yield token_text
import tensorrt as trt
from tensorrt_llm import Tensor
from tensorrt_llm.builder import Builder
from tensorrt_llm.network import net_guard

from models.ModelArgs import ModelArgs
from utils.trt_llm.CustomModel import CustomModel


def build_engine(model_args:ModelArgs, weights_path:str, engine_path:str):
    # create config
    builder = Builder()
    builder_config = builder.create_builder_config()
    # set precision
    builder_config.set_flag(trt.BuilderFlag.FP16)
    # create network
    network = builder.create_network()
    with net_guard(network):
        model = CustomModel(model_args)
        # 加载权重
        model.load_state_dict(torch.load(weights_path))
        # 定义输入
        tokens = Tensor(name='tokens', dtype=trt.int32, shape=[-1, -1])
        # forward pass
        logits = model(tokens)
        logits.mark_output('logits', trt.float32)
    # build engine
    engine = builder.build_engine(network, builder_config)
    # save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    return engine
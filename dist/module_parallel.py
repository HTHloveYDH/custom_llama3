from dist.data_parallel import data_parallelize
from models.Transformer import Transformer as Llama
from models.DPOLlama import DPOLlama


def module_parallelize_llama(model, pp_mesh):
    return model

def module_parallelize(model, tp_mesh, training:bool):
    if isinstance(model, Llama):
        model = module_parallelize_llama(model, tp_mesh, training)
    elif isinstance(model, DPOLlama):
        # TODO:
        model.llm = module_parallelize_llama(model.llm, tp_mesh, training)
    return model

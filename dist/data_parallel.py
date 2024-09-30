from models.Transformer import Transformer as Llama
from models.DPOLlama import DPOLlama


def data_parallelize_llama(model, pp_mesh):
    return model

def data_parallelize(model, tp_mesh, training:bool):
    if isinstance(model, Llama):
        model = data_parallelize_llama(model, tp_mesh, training)
    elif isinstance(model, DPOLlama):
        # TODO:
        model.llm = data_parallelize_llama(model.llm, tp_mesh, training)
    return model

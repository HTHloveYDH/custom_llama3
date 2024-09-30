from models.Transformer import Transformer as Llama
from models.DPOLlama import DPOLlama


def llama_PP(model, pp_mesh):
    return model

def TP(model, tp_mesh, training:bool):
    if isinstance(model, Llama):
        model = llama_PP(model, tp_mesh, training)
    elif isinstance(model, DPOLlama):
        # TODO:
        model.llm = llama_PP(model.llm, tp_mesh, training)
    return model

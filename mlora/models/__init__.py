from .modeling_llama import LlamaForCausalLM

model_dict = {
    "llama": LlamaForCausalLM,
}


def from_pretrained(llm_model, **kwargs):
    if llm_model.config.model_type == "hunyuan":  # 混元模型同样使用Llama实现
        return LlamaForCausalLM.from_pretrained(llm_model, **kwargs)

    if llm_model.config.model_type in model_dict:
        return model_dict[llm_model.config.model_type].from_pretrained(llm_model, **kwargs)
    else:
        raise RuntimeError(
            f"Model {llm_model.config.model_type} not supported.")


__all__ = [
    "LlamaForCausalLM",
    "from_pretrained",
]

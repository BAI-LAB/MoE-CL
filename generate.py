import fire
import torch

import mlora


def inference_callback(cur_pos, outputs):
    print(f"Position: {cur_pos}")
    for adapter_name, output in outputs.items():
        print(f"{adapter_name} output: {output[0]}")


def main(base_model: str="/apdcephfs_qy3/share_300998916/joyyyhuang/model/Llama-2-7b-hf",
         instruction: str="who are you?",
         input: str = None,
         template: str = None,
         max_seq_len: int = None,
         stream: bool = False,
         device: str = f"{mlora.get_backend().device_name()}:0"):

    model = mlora.LLMModel.from_pretrained(base_model,
                                           device=device,
                                           attn_impl="eager",
                                           use_sliding_window=False,
                                           bits=None,
                                           load_dtype=torch.bfloat16)
    tokenizer = mlora.Tokenizer(base_model)

    model.load_adapter_weight(
        path="/apdcephfs_qy3/share_300998916/joyyyhuang/test/results/moe-cl/tencent/order1/shipinhao_finetuned", 
        adapter_name="moe-cl"
    )
    adapter_name = "moe-cl"

    instruction = f"""Please classify the following text, 0 indicates normal text, 1 indicates non-compliant text: 
Title: 哈哈, ParentComment: 哈哈, SubComment: 她笑了。
Result: 
"""
    print(instruction)
    generate_paramas = mlora.GenerateConfig(
        adapter_name=adapter_name,
        prompt_template=template,
        prompts=[(instruction, input)])

    if max_seq_len is None:
        max_seq_len = model.config_.max_seq_len_

    output = mlora.generate(
        model, tokenizer, [generate_paramas], max_gen_len=max_seq_len, stream_callback=inference_callback if stream else None)

    for prompt in output[adapter_name]:
        print(f"\n{'='*10}\n")
        print(prompt)
        print(f"\n{'='*10}\n")


if __name__ == "__main__":
    fire.Fire(main)

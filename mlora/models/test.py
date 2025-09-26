from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.models.llama.modeling_llama as modeling_llama
from transformers.activations import ACT2FN

from mlora.backends import _backend, get_backend
from mlora.common import (CHECKPOINT_CLASSES, FeedForward, Linear,
                          LLMAttention, LLMDecoder, LLMFeedForward,
                          LLMForCausalLM, LLMModelArgs, Masks,
                          MultiLoraBatchData, _flash_attn_available,
                          apply_rotary_emb, get_unpad_data,
                          precompute_rope_angle,
                          prepare_4d_causal_attention_mask, repeat_kv,
                          scaled_dot_product_attention)
from mlora.utils import copy_parameters

if _flash_attn_available:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import (index_first_axis, pad_input,
                                         unpad_input)


class test:
    def __init__(self):
        pass

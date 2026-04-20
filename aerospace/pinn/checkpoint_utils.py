"""PINN checkpoint 键名与 torch.compile 的兼容性。"""

from __future__ import annotations


_COMPILE_PREFIX = "_orig_mod."


def normalize_pinn_state_dict(state_dict: dict) -> dict:
    """去掉 torch.compile 包装器在 state_dict 上附加的 ``_orig_mod.`` 前缀。

    训练时若启用了 ``torch.compile``，保存的权重键名会带此前缀；推理侧通常
    构造的是未 compile 的 ``nn.Module``，直接 ``load_state_dict`` 会报错。
    """
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if keys and keys[0].startswith(_COMPILE_PREFIX):
        return {
            k[len(_COMPILE_PREFIX) :]: v
            for k, v in state_dict.items()
            if k.startswith(_COMPILE_PREFIX)
        }
    return state_dict

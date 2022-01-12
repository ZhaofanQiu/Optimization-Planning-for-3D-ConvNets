"""
Select model, transfer pre-train weights (2D), remove fc layer
By Zhaofan Qiu
zhaofanqiu@gmail.com
"""
import sys

model_dict = {}
transfer_dict = {}


def get_model_by_name(net_name, **kwargs):
    return model_dict.get(net_name)(**kwargs)


def transfer_weights(net_name, state_dict, early_stride):
    if transfer_dict[net_name] is None:
        raise NotImplementedError
    else:
        return transfer_dict[net_name](state_dict, early_stride)


def remove_fc(net_name, state_dict):
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)

    state_dict.pop('fc_g.weight', None)
    state_dict.pop('fc_g.bias', None)

    state_dict.pop('fc_dual.weight', None)
    state_dict.pop('fc_dual.bias', None)

    state_dict.pop('fc_dist.weight', None)
    state_dict.pop('fc_dist.bias', None)
    return state_dict


# from https://github.com/rwightman/pytorch-image-models/
def register_model(fn):
    mod = sys.modules[fn.__module__]
    model_name = fn.__name__

    # add entries to registry dict/sets
    assert model_name not in model_dict
    model_dict[model_name] = fn
    if hasattr(mod, 'transfer_weights'):
        transfer_dict[model_name] = mod.transfer_weights
    else:
        transfer_dict[model_name] = None
    return fn

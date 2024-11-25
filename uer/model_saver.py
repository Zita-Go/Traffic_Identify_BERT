import torch
import loralib as lora
import copy

# def save_model(model, model_path, lora_r=False):
#     if lora_r:
#         if hasattr(model, "module"):
#             torch.save(lora.lora_state_dict(model.module), model_path)
#         else:
#             torch.save(lora.lora_state_dict(model), model_path)
#     else:
#         # 检查是否用了多卡GPU
#         if hasattr(model, "module"):
#             torch.save(model.module.state_dict(), model_path)
#         else:
#             torch.save(model.state_dict(), model_path)

def save_model(model, model_path, lora_r=False):
    if lora_r:
        if hasattr(model, "module"):
            torch.save(merge_model(model.module), model_path)
        else:
            torch.save(merge_model(model), model_path)
    else:
        # 检查是否用了多卡GPU
        if hasattr(model, "module"):
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)

def merge_model(model):
    params = model.state_dict()
    lora_layer = set()
    model_dict = {}
    for k, v in params.items():
        if 'lora_' in k:
            lora_layer.add(k.split('lora_')[0])
    for k, v in params.items():
        layer = k.split('weight')[0]
        if 'weight' in k and layer in lora_layer:

            if 'output' in k:
                scaling = 20
            else:
                scaling = 2
            model_dict[k] = v + (params[layer+'lora_B'] @ params[layer+'lora_A']) * scaling
        elif 'lora_' in k:
            continue
        else:
            model_dict[k] = v
    return model_dict
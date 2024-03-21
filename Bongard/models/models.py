import torch


models = {}
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    if name is None:
        return None
    model = models[name](**kwargs)

    # if torch.cuda.is_available():
    #     model = model.cuda()
    return model


def load(model_sv, name=None, **kwargs):
    if name is None:
        name = 'model'
    model = make(name, **kwargs)
    # state_dict = torch.load(pretrain_enc_pth, map_location='cpu')
    # missing_keys, unexpected_keys = model.encoder.encoder.load_state_dict(state_dict, strict=False)
    missing_keys, unexpected_keys = model.load_state_dict(model_sv)
    return model


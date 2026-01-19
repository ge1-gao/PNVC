from . import iscas26, pnvc


def create_compress_model(logger, model, cfgs, model_cfg=None, stage=None):
    ctype = cfgs["type"]
    if ctype.upper() in ("NVRC", "BASE", "ISCAS26"):
        return iscas26.build_compress_model(logger, model, cfgs["config"], model_cfg=model_cfg, stage=stage)
    if ctype.lower() in ("pnvc",):
        return pnvc.build_compress_model(logger, model, cfgs["config"], model_cfg=model_cfg, stage=stage)
    else:
        raise ValueError(f"Unknown compress model type: {cfgs['type']}")

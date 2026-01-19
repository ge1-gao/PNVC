from . import hinerv


def create_model(args, logger, input):
    model_type = args["type"]
    if model_type in ("HiNeRV", "PNVC"):
        # PNVC uses the HiNeRV backbone with ModMixer decoder via config
        return hinerv.build_model(args["config"], logger, input)
    raise ValueError(f"Unknown model type: {model_type}")

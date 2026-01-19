def create_model(args, logger, input):
    if args["type"] == "HiNeRV":
        from .ops import hinerv  # lazy import to avoid heavy deps on module import

        return hinerv.build_model(args["config"], logger, input)
    else:
        raise ValueError(f"Unknown model type: {args['type']}")

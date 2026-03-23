__all__ = [
    'train_model',
    'predict',
    'InferenceManager'
]


def __getattr__(name):
    if name == 'train_model':
        from .training import train_model
        return train_model

    if name in {'predict', 'InferenceManager'}:
        from .inference import predict, InferenceManager
        return {'predict': predict, 'InferenceManager': InferenceManager}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

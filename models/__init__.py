__all__ = ['get_densenet169', 
           'get_efficientnetv2', 
           'get_convnext', 
           'get_swin_transformer', 
           'get_hybrid_model', 
           'get_ensemble_model', 
           'get_net'
           ]


def __getattr__(name):
    exports = {
        'get_densenet169',
        'get_efficientnetv2',
        'get_convnext',
        'get_swin_transformer',
        'get_hybrid_model',
        'get_ensemble_model',
        'get_net',
    }
    if name in exports:
        from .model import (
            get_densenet169,
            get_efficientnetv2,
            get_convnext,
            get_swin_transformer,
            get_hybrid_model,
            get_ensemble_model,
            get_net,
        )
        return {
            'get_densenet169': get_densenet169,
            'get_efficientnetv2': get_efficientnetv2,
            'get_convnext': get_convnext,
            'get_swin_transformer': get_swin_transformer,
            'get_hybrid_model': get_hybrid_model,
            'get_ensemble_model': get_ensemble_model,
            'get_net': get_net,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

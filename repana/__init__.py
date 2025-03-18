__all__ = [
    'ControlModel',
    'ControlVector', 'ReadingVector', 'ReadingContrastVector', 'PCAContrastVector',
    'Dataset', 'evaluate', 'eval_kld', 'eval_entropy', 'eval_prob_mass'
]

# Define lazy loading functions
def __getattr__(name):
    if name == 'ControlModel':
        from .controlModel import ControlModel
        return ControlModel
    elif name in ('ControlVector', 'ReadingVector', 'ReadingContrastVector', 'PCAContrastVector'):
        from .controlVector import ControlVector, ReadingVector, ReadingContrastVector, PCAContrastVector
        globals()[name] = locals()[name]
        return globals()[name]
    elif name in ('Dataset', 'evaluate', 'eval_kld', 'eval_entropy', 'eval_prob_mass'):
        from .utils import Dataset, evaluate, eval_kld, eval_entropy, eval_prob_mass
        globals()[name] = locals()[name]
        return globals()[name]
    raise AttributeError(f"module has no attribute '{name}'")
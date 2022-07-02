from .formulatrinity import FormulaTrinitySegmentation

class DanishSegmentation(FormulaTrinitySegmentation):
    def __init__(self, root='datasets/danish', **kwargs):
        super().__init__(root=root, **kwargs)

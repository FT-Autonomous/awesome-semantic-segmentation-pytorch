from .formulatrinity import FormulaTrinitySegmentation

class OverfitSegmentation(FormulaTrinitySegmentation):
    def __init__(self, root='datasets/overfit', **kwargs):#
        super().__init__(root=root, **kwargs)

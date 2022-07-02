from .formulatrinity import FormulaTrinitySegmentation

class UGhentSegmentation(FormulaTrinitySegmentation):
    def __init__(self, root='datasets/ughent', **kwargs):#
        super().__init__(root=root, **kwargs)

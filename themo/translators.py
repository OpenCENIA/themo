import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers

# models: helsinski (opus), T5, nllb
# they must allow en -> es and es -> en translation

class GenericTranslator(nn.Module):
    def __init__(self) -> None:
        pass

class T5(TranslatorMetaClass):
    def __init__(self) -> None:
        super().__init__()

class NLLB(TranslatorMetaClass):
    def __init__(self) -> None:
        super().__init__()

class OPUS(TranslatorMetaClass):
    def __init__(self) -> None:
        super().__init__()


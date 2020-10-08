from tape import ProteinModel, ProteinConfig

class CPCConfig(ProteinConfig):
    """ Specify parameters using Sacred,
        load JSON as dictionary and directly
        modify the `__dict__` attribute;
        need this config class to keep TAPE happy
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CPCAbstractModel(ProteinModel):
    """ Abstract base model to keep TAPE happy
    """
    config_class = CPCConfig
    base_model_prefix = 'cpc'

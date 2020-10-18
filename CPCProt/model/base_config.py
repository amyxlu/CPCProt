from tape import ProteinModel, ProteinConfig

class CPCProtConfig(ProteinConfig):
    """ Config class (needed for the TAPE framework).
    If specifying parameters using sacred,
    load the JSON as a dictionary, and directly
    modify the `__dict__` attribute.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(**kwargs)

class CPCAbstractModel(ProteinModel):
    """ Abstract base model needed for the TAPE framework
    """
    config_class = CPCProtConfig
    base_model_prefix = 'cpc'

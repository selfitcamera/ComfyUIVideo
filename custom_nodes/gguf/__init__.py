try:
    import comfy.utils
except ImportError:
    pass
else:
    from .pig import NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# in case any node falsely calls gguf as site-package; still make it works
from .gguf_connector.reader import *

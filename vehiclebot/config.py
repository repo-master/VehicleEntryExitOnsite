
from decouple import config as deconf, UndefinedValueError
from yaml import load, Node
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
    
import os
import logging

BASE = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

#Custom loader constructors
def constr_config_external(loader : Loader, node : Node):
    key_src = loader.construct_scalar(node)
    def load_env_key_or_log(key : str, cast = str):
        try:
            return deconf(key, cast=cast)
        except UndefinedValueError:
            logger.exception("%s variable is not set in the environment variables, but was requested to source this value from the environment variables. Please set this value correctly." % key)
            return

    #Only allow access to predefined keys in environment vars
    if key_src == 'SITE_ID':
         return load_env_key_or_log("SITE_ID", int)
    elif key_src == 'SITE_KEY':
        return load_env_key_or_log("SITE_KEY")
    else:
        raise ValueError("Incorrect source \"%s\" for env. Please only use one of %s, or use values in config file directly." % (
            key_src,
            ','.join(['SITE_ID', 'SITE_KEY'])
        ))

def constr_base_append(loader : Loader, node : Node):
    pass

Loader.add_constructor('!env', constr_config_external)
Loader.add_constructor('!base', constr_base_append)

def load_config(cfg_file : os.PathLike):
    '''
    Simple wrapper to load config from .yaml file
    '''
    with open(cfg_file, 'r') as f:
        return load(f.read(), Loader)


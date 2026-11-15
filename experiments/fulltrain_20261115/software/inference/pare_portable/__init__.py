"""Explicit, path-independent inference. Importing never opens a data artifact."""
from .spec import ModelSpec
from .api import load_model,score_queries
from .io import documents_from_cache,validate_input_contract
__all__=['ModelSpec','load_model','score_queries','documents_from_cache','validate_input_contract']

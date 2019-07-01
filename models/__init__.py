'''
Initialize the models module
'''
from models.transformer import Transformer
from models.parse import ParseTransformer
from models.probe_transformer import ProbeTransformer
from models.new_transformer import NewTransformer

MODELS = {
    'transformer': Transformer,
    'parse_transformer': ParseTransformer,
    'probe_transformer': ProbeTransformer,
    'new_transformer': NewTransformer
}

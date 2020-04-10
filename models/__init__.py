'''
Initialize the models module
'''
from models.transformer import Transformer
from models.parse import ParseTransformer
from models.new_transformer import NewTransformer

MODELS = {
    'transformer': Transformer,
    'parse_transformer': ParseTransformer,
    'new_transformer': NewTransformer,
}

'''
Initialize the actions module
'''
from actions.train import Trainer
from actions.evaluate import Evaluator
from actions.translate import Translator
from actions.probe import Prober
from actions.probe_train import ProbeTrainer
from actions.probe_evaluate import ProbeEvaluator
from actions.probe_new_translate import ProbeNewTranslator
from actions.probe_attn_stats import ProbeStatsGetter

class Pass(object):
    ''' Action that does nothing... '''
    def __init__(self, *args, **kwargs):
        ''' Do nothing '''
        pass

    def __call__(self, *args, **kwargs):
        ''' Do nothing '''
        pass


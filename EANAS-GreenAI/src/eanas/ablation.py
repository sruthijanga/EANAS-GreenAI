"""Control variants with identical search budgets and final device evaluation."""
from .search import HybridSearcher

class AblationSearcher(HybridSearcher):
    def __init__(self,*args,variant='full',**kwargs):
        super().__init__(*args,**kwargs)
        if variant not in ('full','without_rl','without_ea','without_hardware_feedback','without_multiobjective'):
            raise ValueError('Unknown ablation variant')
        self.enable_rl=variant!='without_rl'
        self.enable_ea=variant!='without_ea'
        self.use_hardware_feedback=variant!='without_hardware_feedback'
        self.use_multiobjective=variant!='without_multiobjective'

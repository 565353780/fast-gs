from base_gs_trainer.Config.config import (
    ParamGroup,
    BaseModelParams,
    BasePipelineParams,
    BaseOptimizationParams,
)


class ModelParams(BaseModelParams, ParamGroup):
    def __init__(self, parser, sentinel=False):
        BaseModelParams.__init__(self)

        ParamGroup.__init__(self, parser, "Loading Parameters", sentinel)
        return

class PipelineParams(BasePipelineParams, ParamGroup):
    def __init__(self, parser):
        BasePipelineParams.__init__(self)

        self.separate_sh = True
        self.antialiasing = False

        ParamGroup.__init__(self, parser, "Pipeline Parameters")
        return

class OptimizationParams(BaseOptimizationParams, ParamGroup):
    def __init__(self, parser):
        BaseOptimizationParams.__init__(self)

        # fastgs parameters
        self.loss_thresh = 0.1
        self.grad_abs_thresh = 0.0012
        self.highfeature_lr = 0.005
        self.lowfeature_lr = 0.0025
        self.grad_thresh = 0.0002
        self.dense = 0.001
        self.mult = 0.5      # multiplier for the compact box to control the tile number of each splat

        ParamGroup.__init__(self, parser, "Optimization Parameters")
        return

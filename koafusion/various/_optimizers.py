from torch import optim


def CustomWarmupStaticDecayLR(optimizer, epochs_warmup, epochs_static, epochs_decay,
                              warmup_factor=0.1, decay_factor=0.9, **kwargs):
    def fn(epoch):
        end_w = epochs_warmup
        end_s = end_w + epochs_static

        if epoch <= end_w:
            ## Linear
            return warmup_factor + (1. - warmup_factor) * epoch / float(epochs_warmup)

            ## Exponential
            # r = (1. / warmup_factor) ** (1. / epochs_warmup) - 1
            # return warmup_factor * (1. + r) ** epoch

            ## Sigmoid
            # a = 1. / warmup_factor - 1
            # b = np.log(1. / (9 * a)) / (-epochs_warmup)
            # c = 1.
            # return c / (1 + a * np.exp(-epoch * b))
        elif end_w < epoch <= end_s:
            return 1.
        else:
            return decay_factor ** (epoch - end_s)

    return optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=fn)


def CustomWarmupMultiStepLR(optimizer, epochs_warmup, mstep_milestones,
                            warmup_factor=0.1, mstep_factor=0.1, **kwargs):
    def fn(epoch):
        end_w = epochs_warmup
        end_m = [end_w + e for e in mstep_milestones]

        if epoch <= end_w:
            # linear
            return warmup_factor + (1. - warmup_factor) * epoch / float(epochs_warmup)
        else:
            after_milestones = [epoch >= e for e in end_m]
            return mstep_factor ** sum(after_milestones)

    return optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=fn)


dict_optimizers = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
    'RMSprop': optim.RMSprop,
}

dict_schedulers = {
    "LambdaLR": optim.lr_scheduler.LambdaLR,
    "MultiplicativeLR": optim.lr_scheduler.MultiplicativeLR,
    "StepLR": optim.lr_scheduler.StepLR,
    "MultiStepLR": optim.lr_scheduler.MultiStepLR,
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "CyclicLR": optim.lr_scheduler.CyclicLR,
    "OneCycleLR": optim.lr_scheduler.OneCycleLR,
    "CosineAnnealingWarmRestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "CustomWarmupStaticDecayLR": CustomWarmupStaticDecayLR,
    "CustomWarmupMultiStepLR": CustomWarmupMultiStepLR,
}

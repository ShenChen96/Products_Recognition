import torch

def build_optimizer(options, model):
    if options["optimizer"] == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=options['lr'], momentum=0.9, weight_decay=0.00001)
    else:
        err_msg = 'Do not support the optimizer named {} now'.format(
            options["optimizer"])
        logger.error(err_msg)
        raise NotImplementedError(err_msg)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=options['decay_step'], gamma=options['decay_gamma'])
    
    return optimizer, scheduler
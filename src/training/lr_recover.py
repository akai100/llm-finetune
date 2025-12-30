def reduce_lr(optimizer, factor=0.5):
    for group in optimizer.param_groups:
        group["lr"] *= factor

# training/quantization/qat.py
def enable_fake_quant(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            module.fake_quant = True

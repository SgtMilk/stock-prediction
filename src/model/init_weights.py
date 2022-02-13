from torch.nn.init import normal_, constant_

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        normal_(m.weight.data, 0.0, 0.02)
        constant_(m.bias.data, 0)
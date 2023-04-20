manifest = (
    ('R56N_', 'resnet56_noshort', 70),
    ('R56_', 'resnet56', 20),
    ('tn', 'vgg9', 0),
    ('D121_', 'densenet121', 120),
    ('LE_', 'lenet', 0),
)

def find_arch(proj: str):
    for p in manifest:
        if proj.startswith(p[0]):
            return p[1]
    raise NameError('Unknown Network Architecture')
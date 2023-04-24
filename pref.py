manifest = (
    ('R56N_', 'resnet56_noshort', 70),
    ('R56_', 'resnet56', 20),
    ('tn', 'vgg9', 0),
    ('D121_', 'densenet121', 120),
    ('LE_', 'lenet', 0),
    ('CN12_', 'cnn12', 0),
    ('CN24_', 'cnn24', 0),
    ('CN36_', 'cnn36', 0),
    ('CN48_', 'cnn48', 0),
    ('CN96_', 'cnn96', 0),
    ('CN48x2_', 'cnn48x2', 0),
    ('CN48x3_', 'cnn48x3', 0),
    ('EF_', 'effnet_s', 0),
)

def find_arch(proj: str):
    for p in manifest:
        if proj.startswith(p[0]):
            return p[1]
    raise NameError('Unknown Network Architecture')
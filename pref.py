manifest = (
    ('R56N_', 'resnet56_noshort', 70), # 3.26M
    ('R56_', 'resnet56', 20), # 3.28M
    ('R20N_', 'resnet20_noshort', 0), # 1.03M
    ('R20_', 'resnet20', 0), # 1.04M
    ('R110N_', 'resnet110_noshort', 0), # 6.62M
    ('R110_', 'resnet110', 0), # 6.63M
    ('tn', 'vgg9', 0), # 10.6M
    ('V16_', 'vgg16', 0), # 57.2M
    ('D121_', 'densenet121', 120), # 26.8M
    ('LE_', 'lenet', 0), # 324K
    ('CN12_', 'cnn12', 0), # 3.01M
    ('CN24_', 'cnn24', 0), # 6.02M
    ('CN36_', 'cnn36', 0), # 9.02M
    ('CN48_', 'cnn48', 0), # 12.0M
    ('CN96_', 'cnn96', 0), # 24.0M
    ('CN48x2_', 'cnn48x2', 0), # 3.25M
    ('CN48x3_', 'cnn48x3', 0), # 1.22M
    ('EF_', 'effnet_s', 0), # 78.1M
)

def find_arch(proj: str):
    for p in manifest:
        if proj.startswith(p[0]):
            return p[1]
    raise NameError('Unknown Network Architecture')
from mrcnn.config import Config

labels = [
    {'index': 0, 'name': 'background'},
    {'index': 1, 'name': 'signature'},
    {'index': 2, 'name': 'stamp'},
    {'index': 3, 'name': 'handwriting'},  # should not be used any more
    {'index': 4, 'name': 'driving'},
    {'index': 5, 'name': 'personalid'},
    {'index': 6, 'name': 'passport'},
]

labelmap_for_signatures = []
labelmap_for_stamps = []
labelmap_for_references = []

for l in labels:
    if l['index'] in [1]:  # signatures
        sign = l.copy()
        sign['index'] = len(labelmap_for_signatures) + 1
        labelmap_for_signatures.append(sign)
    if l['index'] in [2]:  # stamps
        stamp = l.copy()
        stamp['index'] = len(labelmap_for_stamps) + 1
        labelmap_for_stamps.append(stamp)
    if l['index'] in [1, 4, 5, 6]:  # references
        ref = l.copy()
        ref['index'] = len(labelmap_for_references) + 1
        labelmap_for_references.append(ref)

############################################################
#  Configurations
############################################################


class BaseConfig(Config):
    """Configuration for training on the signature and stamp dataset.
    Derives from the base Config class and overrides some values.
    """

    def __init__(self, name, num_classes):
        self.NAME = name
        self.NUM_CLASSES = num_classes
        super().__init__()

    NAME = "base_config"
    BACKBONE = "resnet50"
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 500
    DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.85
    LEARNING_RATE = 0.002
    VALIDATION_STEPS = 100


class SignatureConfig(BaseConfig):
    def __init__(self):
        super().__init__(
            name='signatures',
            num_classes=len(labelmap_for_signatures) + 1
        )


class StampConfig(BaseConfig):
    def __init__(self):
        super().__init__(
            name='stamps',
            num_classes=len(labelmap_for_stamps) + 1
        )


class ReferenceConfig(BaseConfig):
    def __init__(self):
        super().__init__(
            name='references',
            num_classes=len(labelmap_for_references) + 1
        )


def get_labels_by_mode(mode):
    assert mode in ['signatures', 'stamps', 'references']

    labels = None

    if mode == 'signatures':
        labels = labelmap_for_signatures
    elif mode == 'stamps':
        labels = labelmap_for_stamps
    elif mode == 'references':
        labels = labelmap_for_references

    return labels


def get_config_by_mode(mode):
    assert mode in ['signatures', 'stamps', 'references']

    config = None

    if mode == 'signatures':
        config = SignatureConfig()
    elif mode == 'stamps':
        config = StampConfig()
    elif mode == 'references':
        config = ReferenceConfig()

    return config

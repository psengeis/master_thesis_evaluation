from mrcnn import model as modellib
from custom_config import get_config_by_mode, get_labels_by_mode
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from pathlib import Path

tf1.disable_v2_behavior()

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

sign_model = None
stamp_model = None
ref_model = None
LOGS_DIR = 'C:\\Users\\senge\\Documents\\github\\_models'


def load_model(weights, config):

    # global LOGS_DIR

    graph = None
    session = None
    model = None

    graph = tf1.Graph()
    with graph.as_default():
        session = tf1.Session()
        with session.as_default():
            model = modellib.MaskRCNN(mode="inference",
                                      config=config,
                                      model_dir=LOGS_DIR
                                      )
            model.load_weights(weights, by_name=True)

    return graph, session, model


def get_model_by_mode(detection_mode):

    global sign_model
    global stamp_model
    global ref_model

    assert detection_mode in ['signatures', 'stamps', 'references']

    graph = None
    session = None
    model = None

    if detection_mode == 'signatures':

        if sign_model == None:
            sign_model = load_model(
                weights='_models\\signatures.h5',
                config=get_config_by_mode(detection_mode)
            )

        graph, session, model = sign_model

    elif detection_mode == 'stamps':

        if stamp_model == None:
            stamp_model = load_model(
                weights='_models\\stamps.h5',
                config=get_config_by_mode(detection_mode)
            )

        graph, session, model = stamp_model

    elif detection_mode == 'references':

        if ref_model == None:
            ref_model = load_model(
                weights='_models\\references.h5',
                config=get_config_by_mode(detection_mode)
            )

        graph, session, model = ref_model

    return (graph, session, model)


def detect(image, detection_mode='signatures'):
    graph, session, model = get_model_by_mode(detection_mode)

    detections = []

    with graph.as_default():
        with session.as_default():
            detections = model.detect([image], verbose=0)[0]

    return detections

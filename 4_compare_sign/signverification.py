from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import Sequence
from tensorflow_addons.losses import ContrastiveLoss
import wandb
from wandb.keras import WandbCallback
from Config import Config


def InitConfig(useWandb: bool):
    config = Config(useWandb)
    config["input_dim"] = (440, 220, 1)
    config["img_shape"] = (440, 220, 1)
    config["epochs"] = 30
    config["batch_size"] = 32
    config["plot_info"] = True
    config["embeddingDim"] = 16
    config["learning_rate"] = 0.0001

    return config


class SigVerSiameseCNN(keras.Model):

    def __init__(self, config, **kwargs):
        self.config = config
        self.buildModel()

    def buildFeatureExtractor(self):
       # specify the inputs for the feature extractor network
        inputs = Input(self.config["input_dim"])

        # define the first set of CONV => RELU => POOL => DROPOUT layers
        x = Conv2D(32, (11, 11), padding="same", strides=(
            1, 1),  activation="relu")(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x)
        x = Conv2D(64, (5, 5), padding="same", activation="relu",)(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(128, (3, 3), padding="same", activation="relu",)(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(256, (3, 3), padding="same", activation="relu",)(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(512, (3, 3), padding="same", activation="relu",)(x)
        x = Dropout(0.3)(x)

        # prepare the final outputs
        pooledOutput = GlobalAveragePooling2D()(x)
        outputs = Dense(self.config["embeddingDim"])(pooledOutput)

        model = Model(inputs, outputs)

        if self.config["plot_info"]:
            print("**** featureExtractorNetwork ****")
            model.summary()

        return model

    def buildModel(self, **kwargs):

        input_ImgA = Input(shape=self.config["img_shape"])
        input_ImgB = Input(shape=self.config["img_shape"])

        featureExtractorNetwork = self.buildFeatureExtractor()

        # using exactly the same network twice
        imgAcnn = featureExtractorNetwork(input_ImgA)
        imgBcnn = featureExtractorNetwork(input_ImgB)

        distance = K.sqrt(
            K.sum(K.square(imgAcnn - imgBcnn), axis=1, keepdims=True))

        super(SigVerSiameseCNN, self).__init__(
            inputs=[input_ImgA, input_ImgB], outputs=distance, **kwargs)
        cLoss = ContrastiveLoss(margin=1)

        self.compile(optimizer=keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
                     loss=cLoss)

        if self.config["plot_info"]:
            print("**** Model Setup ****")
            self.summary()

    def train(self, dataloader: Sequence):

        print("starting SigVerSiameseCNN training ...")

        self.fit(dataloader,
                 epochs=self.config["epochs"],
                 batch_size=self.config["batch_size"],
                 #    use_multiprocessing = True,
                 #  callbacks=[WandbCallback()]
                 )

        print("successfully trained SigVerSiameseCNN !!!")

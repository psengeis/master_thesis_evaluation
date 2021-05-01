import wandb
import pickle


class Config:

    @staticmethod
    def loadInstanceFromFile(filepath: str):
        with open(filepath, 'rb') as f:
            privateDict = pickle.load(f)
            config = Config()
            config.privateDict = privateDict
            return config

    def __init__(self, useWandb: bool = False):
        self.privateDict = {}
        self.useWandb = useWandb
        if self.useWandb:
            self.wandbConfig = wandb.config

    def __setitem__(self, key, item):
        self.privateDict[key] = item

        if self.useWandb:
            self.wandbConfig[key] = item

    def __getitem__(self, key):
        print("__getitem__{}".format(key))
        return self.privateDict[key]

    def saveToFile(self, path: str):
        with open(path, 'wb+') as outfile:
            pickle.dump(self.privateDict, outfile)

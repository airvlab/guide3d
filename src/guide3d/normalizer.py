from abc import ABC, abstractmethod


class Normalizer(ABC):
    @abstractmethod
    def normalize(self, data):
        pass

    @abstractmethod
    def unnormalize(self, data):
        pass

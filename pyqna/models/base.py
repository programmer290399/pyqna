"""
This file is meant to hold all the base classes for various types of model classes.
"""
from typing import Dict
from abc import ABC, abstractmethod


class QnAModelBase(ABC):
    @abstractmethod
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.is_trained = False

    @abstractmethod
    def train(self, dataset) -> None:
        """
        Train the model using the dataset given.
        """
        raise NotImplementedError()

    @abstractmethod
    def test(self, dataset) -> None:
        """
        Test the model using the dataset given.
        """
        raise NotImplementedError()


class ReadingComprehensionModel(QnAModelBase, ABC):
    @abstractmethod
    def __init__(self, config: Dict) -> None:
        super().__init__()

    @abstractmethod
    def get_answer(self, context, question) -> None:
        """
        Return the inference from model using the given context and question.
        """
        raise NotImplementedError()


class OpenDomainModel(QnAModelBase, ABC):
    @abstractmethod
    def __init__(self, config: Dict) -> None:
        super().__init__()

    @abstractmethod
    def get_answer(self, question) -> None:
        """
        Return the inference from model using the given question.
        """
        raise NotImplementedError()

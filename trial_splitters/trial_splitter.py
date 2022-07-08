from __future__ import annotations
from typing import Tuple, List

from abc import ABC, abstractmethod

class TrialSplitter(ABC): 
    """An abstract class for a trial splitter.
    Provides a way to query for train/test splits on trials
    Subclasses should extend this interace and implement the methods
    """
    @abstractmethod
    def __iter__(self) -> TrialSplitter:
        raise NotImplementedError

    @abstractmethod
    def __next__(self) -> Tuple[List[int], List[int]]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
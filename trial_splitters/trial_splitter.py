from __future__ import annotations
from typing import Tuple, List


class TrialSplitter: 
    """
    An interface for a trial splitter. 
    Provides a way to query for train/test splits on trials
    Subclasses should extend this interace and implement the methods
    """
    def __iter__(self) -> TrialSplitter:
        raise NotImplementedError

    def __next__(self) -> Tuple[List[int], List[int]]:
        raise NotImplementedError
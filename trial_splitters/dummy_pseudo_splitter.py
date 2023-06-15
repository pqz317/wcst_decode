class DummyPseudoSplitter:
    """
    Splits trials into train/test sets randomly on each iteration, 
    Used for pseudo_pop dataframe which has already labelled splits
    """
    def __init__(self, pseudo_pop, num_runs: int) -> None:
        self.pseudo_pop = pseudo_pop
        self.split_num = 0
        self.num_runs = num_runs

    def __iter__(self):
        self.split_num = 0
        return self

    def __next__(self):
        if self.split_num < self.num_runs:
            train_trials = self.pseudo_pop[(self.pseudo_pop.SplitNum == self.split_num) & (self.pseudo_pop.Type == "Train")].PseudoTrialNumber.values
            test_trials = self.pseudo_pop[(self.pseudo_pop.SplitNum == self.split_num) & (self.pseudo_pop.Type == "Test")].PseudoTrialNumber.values
            self.split_num += 1
            return (train_trials, test_trials)
        raise StopIteration

    def __len__(self) -> int:
        return self.num_runs
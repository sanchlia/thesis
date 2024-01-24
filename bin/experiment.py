class Experiment:
    def __init__(self, meta):
        self.meta = meta
        self.fold = self.meta["fold"]
        self.pre_processing = self.meta["pre-processing"]
        self.threshold = self.meta["threshold"]
        self.fair_learning = self.meta["learning"]
        self.probability_path = self.meta["probability_path"]
        self.metrics = self.meta['metrics']
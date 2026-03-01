from typing import List

from tqdm import tqdm
from .Metric import Metric
from evaluation.Dataset import Dataset
from pipelines.Vanilla import Vanilla

class Evaluator:
    def __init__(self, pipeline: Vanilla, metrics: List[Metric]):
        self.pipeline = pipeline
        self.metrics = metrics
        
    def run(self, dataset: Dataset):
        for metric in self.metrics:
            metric.reset()

        for i, sample in tqdm(enumerate(dataset), desc=f"Evaluating {type(self.pipeline).__name__} pipeline:"):
            output = self.pipeline.inpaint(sample['image'], sample['mask'], sample['prompt'])
            
            for metric in self.metrics:
                metric.update(sample['image'], sample['mask'], sample['prompt'], output)

            if (i % 5 == 0):
                with open(f'{type(self.pipeline).__name__}_checkpoints.txt', 'w+') as file:
                    file.write({ metric.get_name: metric.compute() for metric in self.metrics })

        return { metric.get_name: metric.compute() for metric in self.metrics }

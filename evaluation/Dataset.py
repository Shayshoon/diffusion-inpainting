from abc import ABC, abstractmethod

from datasets import load_dataset

class Dataset(ABC):
    @abstractmethod
    def __init__(self, samples_count = 250):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, i):
        pass

class PipeDataset(Dataset):
    def __init__(self, samples_count = 250):
        self.images  = load_dataset('paint-by-inpaint/PIPE', split="test", streaming=True)
        self.masks  = load_dataset('paint-by-inpaint/PIPE_Masks', split="test", streaming=True)
        self.images = self.images.take(samples_count)
        self.masks = self.masks.take(samples_count)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image_sample = self.images[i]
        mask_sample = self.masks[i]

        source = image_sample['target_img'].convert('RGB')
        target = image_sample['target_img'].convert('RGB')
        mask = mask_sample['mask'].convert('L')
        prompt = image_sample['Instruction_Class']
        id = image_sample['img_id']

        return {
            "source": source,
            "target": target,
            "mask": mask,
            "prompt": prompt,
            "id": id,
        }

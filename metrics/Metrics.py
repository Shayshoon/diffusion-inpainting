from datasets import load_dataset
from huggingface_hub import login

# from KID import KID
# from BackgroundPreservation import MSE
from typing import Any, Callable, Dict, List, Optional, Union, Literal

# metrics = {"KID": KID, "MSE": MSE}



def run_metric(metric: Literal["KID", "MSE"]):
    dataset  = load_dataset('paint-by-inpaint/PIPE', split="test", streaming=True)
    # dataset_masks  = load_dataset('paint-by-inpaint/PIPE_Masks', split="test", streaming=True)

    # example_test = dataset['test'][0]
    # print(example_test)

    example_test_mask = dataset[0]
    sample = next(iter(dataset))
    sample['source_img'].show()
    sample['target_img'].show()
    n=0
    for i in dataset:
        if n >= 5:
            return
        n += 1
        print(i)
# {'source_img': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x26592BCF350>, 'target_img': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x26592BCF010>, 'Instruction_VLM-LLM': '', 'Instruction_Class': 'add a fire hydrant', 'Instruction_Ref_Dataset': '', 'object_location': '', 'target_img_dataset': 'COCO', 'img_id': '150265', 'ann_id': '417517'}
# {'mask': <PIL.JpegImagePlugin.JpegImageFile image mode=L size=512x512 at 0x23764D38F90>, 'target_img_dataset': 'COCO', 'img_id': '150265', 'ann_id': '417517'}

    match metric:
        case "KID":

            return
        case "MSE":

            return

if __name__ == "__main__":
    run_metric("MSE")
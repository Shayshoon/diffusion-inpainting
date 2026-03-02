import os
from PIL import Image

def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        return "Error: The file was not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"
        

def mask_pair_generator(directory):
    """
    Yields (source_pil, mask_pil, prompt, filename) for every sample.
    """
    files = os.listdir(directory)
    base_images = [f for f in files if ".mask." not in f and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in base_images:
        name_part, ext = os.path.splitext(filename)
        mask_filename = f"{name_part}.mask{ext}"
        prompt_filename = f"{name_part}.txt"
        
        src_path = os.path.join(directory, filename)
        mask_path = os.path.join(directory, mask_filename)
        prompt_path = os.path.join(directory, prompt_filename)

        src_obj = Image.open(src_path)
        mask_obj = Image.open(mask_path) if os.path.exists(mask_path) else get_square()
        prompt = read_file(prompt_path) if os.path.exists(prompt_path) else ""
            
        yield src_obj, mask_obj, prompt, filename

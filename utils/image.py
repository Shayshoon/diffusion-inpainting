from PIL import Image, ImageOps, ImageDraw, ImageFont

def overlay_mask(source_img, mask_img, color=(255, 0, 0), alpha=0.5):
    src = source_img.convert("RGBA")
    mask = mask_img.convert("L").resize(src.size)
    mask = ImageOps.invert(mask)
    color_layer = Image.new("RGBA", src.size, color + (255,))
    masked_image = Image.composite(color_layer, src, mask)
    combined = Image.blend(src, masked_image, alpha)
    
    return combined

def create_comparison_canvas(source_img, mask_img, 
                             result_img=None,
                             text_label="Comparison View",
                             color=(255, 0, 0), alpha=0.5):
    
    src = source_img.convert("RGBA")
    w, h = src.size
    src_with_mask = overlay_mask(source_img, mask_img, color=color, alpha=alpha)
    
    if result_img:
        result_image = result_img.convert("RGBA")

    padding = 15
    text_height = 50
    
    num_images = 3 if result_img else 2
    total_width = (w * num_images) + (padding * (num_images - 1))
    total_height = h + text_height
    
    # 3. Create Canvas
    canvas = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))

    # 4. Paste Images
    canvas.paste(src, (0, 0))
    canvas.paste(src_with_mask, (w + padding, 0))
    if result_img:
        canvas.paste(result_image, (2*w + 2*padding, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("./assets/Roboto-Regular.ttf", 24)
    left, top, right, bottom = draw.textbbox((0, 0), text_label, font=font)
    text_width = right - left
    
    x_pos = (total_width - text_width) // 2
    y_pos = h + (text_height // 4)
    
    draw.text((x_pos, y_pos), text_label, fill=(0, 0, 0, 255), font=font)

    return canvas

def get_square():
    mask = Image.new("L", (512, 512), 255)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([150, 150, 350, 350], fill=0)
    
    return mask
    
def get_grid(width=512, height=512):
    """
    Generates a grayscale mask where:
    - Every second row (y=1, 3, 5...) is black.
    - Every second column (x=1, 3, 5...) is black.
    - Other pixels remain white.
    """
    # Create a white image (255) in 'L' (grayscale) mode
    mask = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(mask)
    
    # Obscure every second line (rows 1, 3, 5, ...)
    for y in range(1, height, 2):
        draw.line([(0, y), (width - 1, y)], fill=0)
        
    # Obscure every second column (columns 1, 3, 5, ...)
    for x in range(1, width, 2):
        draw.line([(x, 0), (x, height - 1)], fill=0)
        
    return mask

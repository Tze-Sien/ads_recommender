"""Image processing utilities for the ads gallery application."""

import base64
from io import BytesIO
from PIL import Image
from config.gallery_config import IMAGE_SIZE


def decode_image(image_bytes):
    """Decode base64 image to PIL Image."""
    try:
        if isinstance(image_bytes, bytes):
            image = Image.open(BytesIO(image_bytes))
        else:
            # If it's already a string, try to decode base64
            if isinstance(image_bytes, str):
                image_data = base64.b64decode(image_bytes)
                image = Image.open(BytesIO(image_data))
            else:
                return None
        return image
    except Exception:
        return None


def resize_image_to_square(image, size=IMAGE_SIZE):
    """Resize image to square for Instagram-like display while maintaining aspect ratio."""
    # Calculate the aspect ratio
    original_width, original_height = image.size
    target_width, target_height = size

    # Calculate scaling factor to fit the image within the target size
    scale_factor = min(target_width / original_width, target_height / original_height)

    # Calculate new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image maintaining aspect ratio
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a new white background image
    background = Image.new("RGB", size, (255, 255, 255))

    # Calculate position to center the resized image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Paste the resized image onto the white background
    background.paste(resized_image, (x_offset, y_offset))

    return background


def create_image_placeholder():
    """Create HTML for image placeholder when image is not available."""
    return """
    <div style='
        height: 300px; 
        background: linear-gradient(45deg, #f0f0f0, #e0e0e0); 
        display: flex; 
        align-items: center; 
        justify-content: center;
        border-radius: 10px;
        margin-bottom: 10px;
    '>
        <span style='color: #999; font-size: 14px;'>No Image</span>
    </div>
    """

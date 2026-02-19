"""Save base64 image
"""

import base64

def save_base64_image(base64_string: str, output_path: str = "images/segmentation_image.png"):
    """Save base64 image into file

    Args:
        base64_string (str): The string to convert
        output_path (str, optional): The path to the file. Defaults to "images".

    Returns:
        str: the output path
    """
    # cleaning image
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    # decode image
    img_data = base64.b64decode(base64_string)
    
    # write file
    with open(output_path, "wb") as f:
        f.write(img_data)
        
    print(f"Image successfully saved to: {output_path}")
    
    return output_path
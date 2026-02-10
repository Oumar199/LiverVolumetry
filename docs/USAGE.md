# Usage

This document describes how to install and use the LiverVolumetry package both locally and on the RunPod serverless API.

## Installation

### Local Package Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Oumar199/LiverVolumetry.git
   cd LiverVolumetry
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Use the package in your script:
   ```python
   from liver_volumetry import LiverVolumetry
   # Example of usage
   lv = LiverVolumetry()
   lv.run(image_data)
   ```

### RunPod Serverless API Calls
1. Ensure you have the RunPod account set up and access to the API.
2. Prepare the binary image data to send:
   ```python
   import requests
   import base64

   with open('image.jpg', 'rb') as image_file:
       binary_image = base64.b64encode(image_file.read()).decode('utf-8')
   ```
3. Make a POST request to the RunPod API:
   ```python
   response = requests.post('https://api.runpod.io/liver_volumetry', json={'image_data': binary_image})
   print(response.json())
   ```

## Examples
### Local Example
```python
# Here is an example of an application using the local package
from liver_volumetry import LiverVolumetry

# Create an instance of the LiverVolumetry class
lv = LiverVolumetry()

# Assuming `image_data` is already prepared
result = lv.run(image_data)
print('Liver volume:', result)
```

### Serverless Example
```python
import requests
import base64

# Load image data
with open('image.jpg', 'rb') as image_file:
    binary_image = base64.b64encode(image_file.read()).decode('utf-8')

# Call the RunPod API
response = requests.post('https://api.runpod.io/liver_volumetry', json={'image_data': binary_image})

# Print the response
print(response.json())
```

## Conclusion
Using the LiverVolumetry package can streamline the process of calculating liver volumes both locally and via a serverless API.

#!/usr/bin/env python3

import requests

def test_api_with_file():
    """Test the API with a real image file."""
    try:
        with open('tomas_phone.jpg', 'rb') as f:
            files = {'image': ('tomas_phone.jpg', f, 'image/jpeg')}
            data = {'max_new_tokens': 100}
            
            response = requests.post(
                'http://localhost:9000/describe',
                files=files,
                data=data,
                timeout=15
            )
            
            print(f"Status code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Description: {result.get('description', 'No description')}")
                print(f"Inference time: {result.get('inference_time', 'N/A')}")
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_api_with_file()
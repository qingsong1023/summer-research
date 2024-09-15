import os
import time
import requests
from tqdm import tqdm
from requests.exceptions import RequestException, ChunkedEncodingError, ConnectionError
from urllib3.exceptions import ProtocolError

def download_file(url, file_path):
    headers = {}
    initial_size = 0
    
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        response = requests.head(url)
        total_size = int(response.headers.get('content-length', 0))
        
        if file_size >= total_size:
            print(f"File already downloaded: {file_path}")
            return
        
        headers['Range'] = f'bytes={file_size}-'
        initial_size = file_size

    try:
        with requests.get(url, headers=headers, stream=True) as response:
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0)) + initial_size
            mode = 'ab' if response.status_code == 206 else 'wb'

            with open(file_path, mode) as f:
                with tqdm(total=total_size, initial=initial_size, unit='B', unit_scale=True, unit_divisor=1024) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
        print("Download completed.")
    except (ChunkedEncodingError, ConnectionError, ProtocolError, RequestException) as e:
        print(f"Error occurred: {e}. Failed to download file.")

if __name__ == "__main__":
    URL = 'https://s3.unistra.fr/camma_public/datasets/m2cai16/m2cai16-workflow/m2cai16-workflow.zip'
    output_dir = 'data' 
    os.makedirs(output_dir, exist_ok=True)

    outfile = os.path.join(output_dir, 'm2cai16-workflow.zip')

    download_file(URL, outfile)



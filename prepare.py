import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os

URL = "https://s3.unistra.fr/camma_public/datasets/m2cai16/m2cai16-workflow/m2cai16-workflow.zip" ## according to the dataset webset
CHUNK_SIZE = 2 ** 20

def download_file(url, outfile):
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    print("Downloading archive to {}".format(outfile))
    with session.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = int(r.headers.get("content-length", 0)) / 10 ** 6
        progress_bar = tqdm(unit="MB", total=total_length)
        with open(outfile, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                progress_bar.update(len(chunk) / 10 ** 6)
                f.write(chunk)
        progress_bar.close()

output_directory = "data"
os.makedirs(output_directory, exist_ok=True)
outfile = os.path.join(output_directory, "m2cai16-workflow.zip")  ## change by the different name of datasets

download_file(URL, outfile)

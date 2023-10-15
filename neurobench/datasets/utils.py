import os
import sys
import re
import hashlib
import urllib
from urllib.parse import urlparse
from torch.utils.model_zoo import tqdm

USER_AGENT = "neurobench"


def _save_response_content(
    content,
    destination,
    length,
):
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            pbar.update(len(chunk))


def _urlretrieve(url, filename, chunk_size=1024*32):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
        _save_response_content(iter(lambda: response.read(chunk_size), b""), filename, length=response.length)


def _get_redirect_url(url, max_hops=3):
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

    for _ in range(max_hops + 1):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            if response.url == url or response.url is None:
                return url

            url = response.url
    else:
        raise RecursionError(
            f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect points to {url}."
        )


def calculate_md5(fpath, chunk_size=1024*1024):
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_url(url, file_path, md5=None, max_redirect_hops=3):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        file_path (str): Full path to the file to be downloaded.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # check if file is already present locally
    if check_integrity(file_path, md5):
        print("Using downloaded and verified file: " + file_path)
        return

    # expand redirect chain if needed
    url = _get_redirect_url(url, max_hops=max_redirect_hops)

    # # check if file is located on Google Drive
    # file_id = _get_google_drive_file_id(url)
    # if file_id is not None:
    #     return download_file_from_google_drive(file_id, root, filename, md5)

    # download the file
    try:
        print("Downloading " + url + " to " + file_path)
        _urlretrieve(url, file_path)
    except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Failed download. Trying https -> http instead. Downloading " + url + " to " + file_path)
            _urlretrieve(url, file_path)
        else:
            raise e

    # check integrity of downloaded file
    if not check_integrity(file_path, md5):
        raise RuntimeError("File not found or corrupted.")

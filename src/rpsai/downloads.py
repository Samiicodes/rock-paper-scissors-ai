'''Downloads pinned by SHA-256, so that every run uses the same data.'''

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path

from rpsai.config import Download

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1 << 20


def sha256_of(path: Path) -> str:
    '''Return the hex SHA-256 of a file.'''
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(download: Download, directory: Path) -> Path:
    '''Download a file unless a copy with the right checksum is already there.'''
    target = directory / download.file
    if target.exists() and sha256_of(target) == download.sha256:
        logger.info('Found %s', target.as_posix())
        return target
    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    logger.info('Downloading %s', download.url)
    request = urllib.request.Request(download.url, headers={'User-Agent': 'rpsai-data-download'})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as handle:
        shutil.copyfileobj(response, handle, CHUNK_BYTES)
    actual = sha256_of(partial)
    if actual != download.sha256:
        partial.unlink()
        raise ValueError(f'checksum mismatch for {download.file}: got {actual}')
    partial.replace(target)
    logger.info('Saved %s', target.as_posix())
    return target

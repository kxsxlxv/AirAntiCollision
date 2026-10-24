"""Fetch selected original TartanAviation ZIP members without the labelled AVI.

Standard library only. Range responses and ZIP CRCs are checked. Pickles are
downloaded as data and never executed. Source video is not resampled.
"""
import argparse
import io
import json
import hashlib
from pathlib import Path
import struct
import subprocess
import shutil
import time
import urllib.request
import zipfile
import zlib

SAMPLE = '1_2023-02-22-15-21-49'
BASE = 'https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanaviation-vision/'


def request_range(url, start, end):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    response = urllib.request.urlopen(req, timeout=60)
    expected = f'bytes {start}-{end}/'
    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(expected):
        response.close()
        raise ValueError('Server did not honor the exact requested byte range')
    return response


def members(url, size):
    start = max(0, size - 262144)
    with request_range(url, start, size - 1) as r:
        tail = r.read()
    class Tail(io.BytesIO):
        def seek(self, offset, whence=0):
            if whence == 0:
                offset -= start
            return super().seek(offset, whence) + start
        def tell(self):
            return super().tell() + start
    with zipfile.ZipFile(Tail(tail)) as archive:
        return archive.infolist()


def fetch_member(url, info, output):
    dest = output / Path(info.filename).name
    if dest.exists() and dest.stat().st_size == info.file_size:
        crc = 0
        with dest.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                crc = zlib.crc32(chunk, crc)
        if crc == info.CRC:
            return dest
    with request_range(url, info.header_offset, info.header_offset + 29) as r:
        header = r.read()
    if header[:4] != b'PK\x03\x04':
        raise ValueError('Invalid ZIP local header')
    name_size, extra_size = struct.unpack_from('<HH', header, 26)
    start = info.header_offset + 30 + name_size + extra_size
    compressed = dest.with_suffix(dest.suffix + '.compressed.part')
    for attempt in range(4):
        done = compressed.stat().st_size if compressed.exists() else 0
        if done == info.compress_size:
            break
        if done > info.compress_size:
            raise ValueError('Partial member is larger than expected')
        try:
            with request_range(url, start + done, start + info.compress_size - 1) as r, compressed.open('ab') as f:
                last = time.monotonic()
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if time.monotonic() - last > 10:
                        print(f'{dest.name}: {done / 1e6:.1f}/{info.compress_size / 1e6:.1f} MB', flush=True)
                        last = time.monotonic()
            if done != info.compress_size:
                raise IOError('Incomplete range response')
            break
        except (OSError, ValueError) as error:
            if attempt == 3:
                raise
            print(f'Resuming after {error}', flush=True)
    if info.compress_type == 9:
        # Preserve one original Deflate64 member inside a small valid ZIP. The
        # installed 7-Zip handles it; do not install or change Python packages.
        seven = shutil.which('7z') or 'C:/Program Files/7-Zip/7z.exe'
        if not Path(seven).exists():
            raise ValueError('Deflate64 requires an existing 7-Zip executable')
        name = dest.name.encode('utf-8')
        local = struct.pack('<IHHHHHIIIHH', 0x04034b50, 21, 0, 9, 0, 0,
                            info.CRC, info.compress_size, info.file_size, len(name), 0) + name
        central = struct.pack('<IHHHHHHIIIHHHHHII', 0x02014b50, 21, 21, 0, 9, 0, 0,
                              info.CRC, info.compress_size, info.file_size, len(name), 0, 0, 0, 0, 0, 0) + name
        mini = dest.with_suffix(dest.suffix + '.member.zip')
        with mini.open('wb') as f, compressed.open('rb') as src:
            f.write(local)
            shutil.copyfileobj(src, f)
            f.write(central)
            f.write(struct.pack('<IHHHHIIH', 0x06054b50, 0, 0, 1, 1, len(central), len(local) + info.compress_size, 0))
        temporary = dest.with_suffix(dest.suffix + '.part')
        with temporary.open('wb') as dst:
            subprocess.run([seven, 'x', '-so', str(mini)], stdout=dst, stderr=subprocess.PIPE, check=True)
        crc = 0
        with temporary.open('rb') as src:
            for chunk in iter(lambda: src.read(1 << 20), b''):
                crc = zlib.crc32(chunk, crc)
        if temporary.stat().st_size != info.file_size or crc != info.CRC:
            raise ValueError('Deflate64 member size/CRC mismatch')
        temporary.replace(dest)
        compressed.unlink()
        mini.unlink()
        return dest
    if info.compress_type not in (0, 8):
        raise ValueError(f'Unsupported ZIP method {info.compress_type}')
    decoder = zlib.decompressobj(-15) if info.compress_type == 8 else None
    temporary = dest.with_suffix(dest.suffix + '.part')
    crc = size = 0
    with compressed.open('rb') as src, temporary.open('wb') as dst:
        for chunk in iter(lambda: src.read(1 << 20), b''):
            data = decoder.decompress(chunk) if decoder else chunk
            dst.write(data)
            size += len(data)
            crc = zlib.crc32(data, crc)
        if decoder:
            data = decoder.flush()
            dst.write(data)
            size += len(data)
            crc = zlib.crc32(data, crc)
            if not decoder.eof or decoder.unused_data:
                raise ValueError('Invalid deflate stream')
    if size != info.file_size or crc != info.CRC:
        raise ValueError('Extracted member size/CRC mismatch')
    temporary.replace(dest)
    compressed.unlink()
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/tartan_sample')
    parser.add_argument('--sample', default=SAMPLE, help='Official vision archive stem')
    parser.add_argument('--video', action='store_true', help='Also fetch original MP4 (~933 MB)')
    parser.add_argument('--adsb-date', help='Also fetch one raw KAGC day, MM-DD-YY (2022 archive)')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.sample or any(c not in '0123456789_-' for c in args.sample):
        parser.error('Invalid archive stem')
    with urllib.request.urlopen(BASE + '?format=json&prefix=' + args.sample + '.zip', timeout=30) as r:
        objects = json.load(r)
    obj = next(x for x in objects if x['name'] == args.sample + '.zip')
    url = BASE + obj['name']
    manifest = {'source_url': url, 'object': obj, 'members': [], 'fetched': []}
    for info in members(url, obj['bytes']):
        manifest['members'].append({'name': info.filename, 'size': info.file_size,
                                    'compressed_size': info.compress_size, 'crc32': f'{info.CRC:08x}'})
        if info.is_dir() or info.filename.endswith('.avi'):
            continue
        if info.filename.endswith('.mp4') and not args.video:
            continue
        path = fetch_member(url, info, output)
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        manifest['fetched'].append({'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size})
        print(f'Verified {path.name}: {path.stat().st_size} bytes SHA256 {digest}', flush=True)
    (output / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.adsb_date:
        if not __import__('re').fullmatch(r'\d{2}-\d{2}-22', args.adsb_date):
            parser.error('--adsb-date must be MM-DD-22')
        base = BASE.replace('tartanaviation-vision/', 'tartanaviation-adsb/')
        with urllib.request.urlopen(base + '?format=json&prefix=kagc/raw/2022.zip', timeout=30) as r:
            obj = next(x for x in json.load(r) if x['name'] == 'kagc/raw/2022.zip')
        info = next(x for x in members(base + obj['name'], obj['bytes'])
                    if x.filename == args.adsb_date + '/1.csv')
        raw_dir = output / 'adsb'
        raw_dir.mkdir(exist_ok=True)
        path = fetch_member(base + obj['name'], info, raw_dir)
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        (raw_dir / 'download_manifest.json').write_text(json.dumps({
            'source_url': base + obj['name'], 'member': info.filename,
            'bytes': path.stat().st_size, 'crc32': f'{info.CRC:08x}', 'sha256': digest}, indent=2), encoding='utf-8')
        print(f'Verified raw ADS-B {path}', flush=True)


if __name__ == '__main__':
    main()

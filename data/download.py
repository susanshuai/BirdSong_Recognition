"""Download public CC-BY-4.0 flight-call datasets from Zenodo.

Zenodo throttles each connection (~150 KB/s), so every file is fetched in
parallel byte ranges and stitched together. Re-running resumes unfinished parts.
"""
import json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = {"3666782": "birdvox_anafcc", "1205569": "birdvox_full_night"}
PART = 32 * 1024 * 1024
lock = threading.Lock()


def get_part(url, path, a, b):
    part = f"{path}.part{a}"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have >= b - a + 1:
        return
    for attempt in range(20):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a + have}-{b}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as fh:
                while chunk := r.read(1 << 16):
                    fh.write(chunk)
            if os.path.getsize(part) >= b - a + 1:
                return
        except Exception as e:
            print("retry", os.path.basename(path), a, e, flush=True)
            time.sleep(min(120, 5 * 2 ** attempt))
        have = os.path.getsize(part) if os.path.exists(part) else 0
    raise RuntimeError(f"failed {path} {a}")


def main():
    jobs = []
    for rec, folder in RECORDS.items():
        os.makedirs(os.path.join(HERE, folder), exist_ok=True)
        meta = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}"))
        for f in meta["files"]:
            path = os.path.join(HERE, folder, f["key"])
            if os.path.exists(path) and os.path.getsize(path) == f["size"]:
                continue
            jobs.append((f["links"]["self"], path, f["size"]))
    with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 48) as ex:
        futs = {}
        for url, path, size in jobs:
            futs[path] = (size, [ex.submit(get_part, url, path, a, min(size, a + PART) - 1)
                                 for a in range(0, size, PART)])
        for path, (size, fs) in futs.items():
            for fu in fs:
                fu.result()
            with open(path, "wb") as out:
                for a in range(0, size, PART):
                    with open(f"{path}.part{a}", "rb") as fh:
                        out.write(fh.read())
            for a in range(0, size, PART):
                os.remove(f"{path}.part{a}")
            assert os.path.getsize(path) == size
            print("done", os.path.relpath(path, HERE), size, flush=True)
    print("ALL_DOWNLOADS_DONE", flush=True)


if __name__ == "__main__":
    main()

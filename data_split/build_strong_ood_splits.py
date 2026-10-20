# Strong-OOD four-mode split verifier: MD5 check against checksums.json
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1] 
NEUTRAL = WORKSPACE / "dataset" / "strong_ood_splits"

OOD_MODES = ("solvent", "solute", "scaffold", "temperature")
ARTIFACT_NAMES = ("split_indices.csv", "split_manifest.json")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_verify(out_root: Path) -> bool:
    ck_path = out_root / "checksums.json"
    if not ck_path.is_file():
        print(f"[verify] missing {ck_path}")
        return False
    ck = json.loads(ck_path.read_text(encoding="utf-8"))
    ok = True
    for rel, expected in ck["md5"].items():
        path = out_root / rel
        if not path.is_file():
            print(f"[verify] MISSING {rel}")
            ok = False
            continue
        actual = md5(path)
        flag = "OK " if actual == expected else "FAIL"
        if actual != expected:
            ok = False
        print(f"[verify] {flag} {rel}")
    print(f"[verify] {'all OK' if ok else 'mismatches found'}: {out_root}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Strong-OOD four-mode split verifier")
    p.add_argument("--verify", action="store_true", help="Verify on-disk artifacts against checksums.json MD5")
    args = p.parse_args()

    if not args.verify:
        p.print_help()
        sys.exit(0)

    sys.exit(0 if run_verify(NEUTRAL) else 1)


if __name__ == "__main__":
    main()

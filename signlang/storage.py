"""
Profiles: each person keeps their own sign set (hands differ a lot from
person to person). Stored as profiles/<NAME>.json.

Profile format (version 3):
    {"version": 3, "name": "DEFAULT", "signs": [...],
     "stats": {label: {...practice stats...}}, "history": {word: count}}

Older files ({"A": [...]} and {"version": 2, "signs": [...]}) are converted
automatically. They were recorded without mirroring, so they get mirrored
to match the current camera view.
"""

import glob
import json
import os
import re
import time

from . import config as cfg


def sanitize_name(name):
    name = re.sub(r"[^A-Z0-9_-]", "", name.strip().upper().replace(" ", "_"))
    return name[:24] or cfg.DEFAULT_PROFILE


def profile_path(name):
    return os.path.join(cfg.PROFILES_DIR, f"{sanitize_name(name)}.json")


def empty_profile(name):
    return {"version": 3, "name": sanitize_name(name), "signs": [], "stats": {}, "history": {}}


# -- Legacy conversion ---------------------------------------------------------------

def _mirror_xy(xy):
    return [-v if i % 2 == 0 else v for i, v in enumerate(xy)]


def _is_v3(data):
    return (isinstance(data, dict) and isinstance(data.get("signs"), list)
            and all("kind" in s for s in data["signs"]))


def convert_legacy(data, mirror=cfg.MIRROR):
    """Convert the v1 / v2 formats into a list of v3 signs."""
    if isinstance(data, dict) and isinstance(data.get("signs"), list):
        old = data["signs"]                                       # v2
    else:
        old = [{"label": k, "hands": [v], "rel": None} for k, v in data.items()]   # v1
    signs = []
    for s in old:
        hands = [{"xy": _mirror_xy(v) if mirror else list(v), "z": None, "side": None}
                 for v in s["hands"]]
        rel = s.get("rel")
        if rel is not None and mirror:
            rel = [-rel[0], rel[1]]
        signs.append({"label": s["label"], "kind": "static", "n_hands": len(hands),
                      "samples": [{"hands": hands, "rel": rel}]})
    return signs


def profile_from_data(data, name):
    if _is_v3(data):
        prof = empty_profile(data.get("name", name))
        prof.update({k: data[k] for k in ("signs", "stats", "history") if k in data})
        return prof
    prof = empty_profile(name)
    prof["signs"] = convert_legacy(data)
    return prof


# -- Load / save ----------------------------------------------------------------------

def load_profile(name):
    name = sanitize_name(name)
    path = profile_path(name)
    if os.path.exists(path):
        with open(path) as f:
            return profile_from_data(json.load(f), name)
    prof = empty_profile(name)
    if name == cfg.DEFAULT_PROFILE and os.path.exists(cfg.LEGACY_FILE):
        with open(cfg.LEGACY_FILE) as f:
            prof["signs"] = convert_legacy(json.load(f))
        print(f"Imported {len(prof['signs'])} sign(s) from {cfg.LEGACY_FILE}")
        save_profile(prof)
    return prof


def save_profile(profile):
    os.makedirs(cfg.PROFILES_DIR, exist_ok=True)
    path = profile_path(profile["name"])
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profile, f, indent=1)
    os.replace(tmp, path)          # atomic: never leaves a half-written file


def list_profiles():
    names = [os.path.splitext(os.path.basename(p))[0]
             for p in glob.glob(os.path.join(cfg.PROFILES_DIR, "*.json"))]
    return sorted(n for n in names if not n.startswith("_"))


def _last_file():
    return os.path.join(cfg.PROFILES_DIR, "_last.txt")


def load_last_profile_name():
    try:
        with open(_last_file()) as f:
            return sanitize_name(f.read())
    except OSError:
        return cfg.DEFAULT_PROFILE


def save_last_profile_name(name):
    os.makedirs(cfg.PROFILES_DIR, exist_ok=True)
    with open(_last_file(), "w") as f:
        f.write(sanitize_name(name))


# -- Editing signs --------------------------------------------------------------------

def find_sign(profile, label):
    return next((s for s in profile["signs"] if s["label"] == label), None)


def add_sign(profile, new_sign):
    """
    Add a recorded sign. If the word already exists with the same type and
    hand count, the new samples are added to it (keeping the newest
    MAX_SAMPLES); otherwise it's replaced. Returns "added"/"merged"/"replaced".
    """
    existing = find_sign(profile, new_sign["label"])
    if existing is None:
        profile["signs"].append(new_sign)
        return "added"
    if existing["kind"] == new_sign["kind"] and existing["n_hands"] == new_sign["n_hands"]:
        existing["samples"] = (existing["samples"] + new_sign["samples"])[-cfg.MAX_SAMPLES:]
        return "merged"
    profile["signs"].remove(existing)
    profile["signs"].append(new_sign)
    return "replaced"


def delete_sign(profile, label):
    before = len(profile["signs"])
    profile["signs"] = [s for s in profile["signs"] if s["label"] != label]
    profile["stats"].pop(label, None)
    return len(profile["signs"]) < before


# -- Export / import --------------------------------------------------------------------

def export_profile(profile):
    os.makedirs(cfg.EXPORTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(cfg.EXPORTS_DIR, f"{profile['name']}-{stamp}.json")
    with open(path, "w") as f:
        json.dump({"version": 3, "name": profile["name"], "signs": profile["signs"]}, f, indent=1)
    return path


def resolve_import_path(text):
    """Find a file from what the user typed (full path, or a name in exports/)."""
    text = text.strip()
    candidates = [text, text + ".json",
                  os.path.join(cfg.EXPORTS_DIR, text),
                  os.path.join(cfg.EXPORTS_DIR, text + ".json")]
    for c in candidates:
        if os.path.isfile(c):
            return c
    for folder in (cfg.EXPORTS_DIR, "."):
        matches = sorted(p for p in glob.glob(os.path.join(folder, "*.json"))
                         if os.path.basename(p).lower().startswith(text.lower()))
        if matches:
            return matches[-1]                  # newest export with that prefix
    return None


def import_signs(profile, path):
    """Merge signs from a file into the profile. Returns (added, merged, skipped)."""
    with open(path) as f:
        incoming = profile_from_data(json.load(f), "import")["signs"]
    added = merged = skipped = 0
    for s in incoming:
        existing = find_sign(profile, s["label"])
        if existing and (existing["kind"] != s["kind"] or existing["n_hands"] != s["n_hands"]):
            skipped += 1                       # don't overwrite a different sign
            continue
        result = add_sign(profile, s)
        added += result == "added"
        merged += result == "merged"
    return added, merged, skipped

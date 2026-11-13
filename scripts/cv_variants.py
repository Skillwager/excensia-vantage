"""One cross-validation fold, several variants of the method at once. External matches only.

    python scripts/cv_variants.py --spec scripts/specs/third_simulator.json --data-tag v16 \
        --test J03WMX metrica1

scripts/cv.py fits one method per run. This fits several in one run of a fold, so that they are
compared on exactly the same held-out recordings, and it is how the paired comparisons in
section 5 of the technical document were made. Each variant is written to runs/cv/<tag>/ in the
form cv.py writes a fold (prob_*.npz and summary_<test>.json), so scripts/compare_cv.py and
`cv.py summary` read it unchanged. For the same settings the fit is the fit cv.py makes: the
same rows in the same order, the same weights, and with it the same model, bit for bit.

The spec is a JSON list of variants:

    {"tag": "new"}                                              every feature of the tables
    {"tag": "base", "state": {"only": "@v10"}, "boundary": {"only": "@v10"}}
    {"tag": "old", "state": {"train_tag": "v10"}, "boundary": {"train_tag": "v10"}}
    {"tag": "wide", "state": "reuse:new", "boundary": {"width": 9}}

"state" and "boundary" are either a dict of settings, or "reuse:<tag>" to take that model's
answers from another variant of this run or from saved files of an earlier run. Settings:
train_tag (which cached tables to fit on), copies (fit on the first so many simulated recordings
of every half only), only / drop (feature names, prefixes ending in *, or "@<tag>" for the
features of that tag's tables), rounds, params, seed, and for the frame classifier short_weight,
short_seconds, metrica_weight, change_weight; for the boundary model width, metrica_weight.

"state" / "boundary" may also be "model:<tag>": the models fitted for that variant of this run,
asked again about this variant's own input. With "norm": true the held-out recordings are first
brought to the reference smoothness (vantage/smoothing.py), as the run command does.

Fitted models are kept in runs/cv/_models/ under a name made from their settings, their tables
and their training matches, and read back from there unless a training table is newer, so a run
that is started again does not fit them twice. A summary records which code made it, and a fold
whose summary was made by other code is not skipped: the variant has to be given a new tag.

--gens lists the simulator generations the held-out matches are recorded with (g3: as the
provided recording's tracker treats its paths; g1 and g2: the two earlier generations, see
vantage/external.py). A variant scored under another generation than g2 is written to
<tag>_<generation>.

The training tables are held in files and mapped, not in memory, so a fold needs about 2 GB of
memory where cv.py needs 6.
"""

import argparse
import ctypes
import gc
import hashlib
import json
import os
import shutil
import time

import numpy as np

from vantage import _official_metrics as official
from vantage import external, model, smoothing
from vantage.external import ALL_MATCHES, ROOT, labelled_halves, source_of
from vantage.labels import change_classes

import cv as cvlib


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


# -- the official overlap search, faster ---------------------------------------------------------
_slow_overlaps = official.spell_overlaps


def fast_overlaps(gt_spells, pred_spells):
    """official.spell_overlaps for spells in time order: the same pairs, found by one sweep."""
    out = []
    first, n = 0, len(pred_spells)
    for i, (gt_start, gt_stop, gt_code) in enumerate(gt_spells):
        while first < n and pred_spells[first][1] <= gt_start:
            first += 1
        j = first
        while j < n and pred_spells[j][0] < gt_stop:
            pred_start, pred_stop, pred_code = pred_spells[j]
            if pred_code == gt_code and pred_stop > gt_start:
                intersection = min(gt_stop, pred_stop) - max(gt_start, pred_start)
                union = max(gt_stop, pred_stop) - min(gt_start, pred_start)
                out.append((intersection / float(union), i, j))
            j += 1
    out.sort(key=lambda item: (-item[0], item[1], item[2]))
    return out


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended", ctypes.c_ulonglong)]


def free_commit_gb():
    try:
        status = _MemoryStatus()
        status.length = ctypes.sizeof(_MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.avail_page / 2.0 ** 30
    except Exception:
        return 99.0


def wait_for_memory(need_gb, what):
    """Hold until the PC has `need_gb` of memory it can still promise. Says so once."""
    told = False
    while free_commit_gb() < need_gb:
        if not told:
            log("waiting for memory before {}: {:.1f} GB free, {} wanted".format(
                what, free_commit_gb(), need_gb))
            told = True
        time.sleep(30)


def table_names(tag, match):
    with np.load(external.train_table_path(tag, match, 1, 0, "own")) as data:
        return [str(name) for name in data["names"]]


def mapped_training(tag, matches, variants, folder):
    """external.load_training with the table in a mapped file."""
    paths = [(match, external.train_table_path(tag, match, period, variant, view))
             for match in matches for period in (1, 2) for variant in range(variants)
             for view in ("own", "swap")]
    copy_of = [variant for match in matches for period in (1, 2) for variant in range(variants)
               for view in ("own", "swap")]
    sizes = []
    for _, path in paths:
        with np.load(path) as data:
            sizes.append(len(data["target"]))
    with np.load(paths[0][1]) as data:
        names = [str(name) for name in data["names"]]
        width = data["table"].shape[1]
        target_type, change_type, run_type = data["target"].dtype, data["change"].dtype, data["run"].dtype
        change_rows = data["change"].shape[0]
    total = int(sum(sizes))
    os.makedirs(folder, exist_ok=True)
    file = os.path.join(folder, "table_{}.npy".format(tag))
    table = np.lib.format.open_memmap(file, mode="w+", dtype=np.float32, shape=(total, width))
    out = {"target": np.empty(total, dtype=target_type),
           "change": np.empty((change_rows, total), dtype=change_type),
           "run": np.empty(total, dtype=run_type),
           "metrica": np.empty(total, dtype=bool),
           "copy": np.empty(total, dtype=np.int8)}
    start = 0
    made_by = set()
    for (match, path), size, copy in zip(paths, sizes, copy_of):
        out["copy"][start:start + size] = copy
        with np.load(path) as data:
            made_by.add(str(data["generation"]) if "generation" in data.files else "not recorded")
            if [str(name) for name in data["names"]] != names:
                raise SystemExit("feature names differ between tables: " + path)
            table[start:start + size] = data["table"]
            out["target"][start:start + size] = data["target"]
            out["change"][:, start:start + size] = data["change"]
            out["run"][start:start + size] = data["run"]
        out["metrica"][start:start + size] = source_of(match) == "metrica"
        start += size
    table.flush()
    del table
    if len(made_by) > 1:
        raise SystemExit("the tables of {} come from different simulators: {}".format(tag, sorted(made_by)))
    out["generation"] = made_by.pop()
    out["table"] = np.load(file, mmap_mode="r")
    out["names"] = names
    return out


def column_subset(table, columns, folder, name):
    """A mapped copy of some columns of the mapped table."""
    file = os.path.join(folder, name + ".npy")
    out = np.lib.format.open_memmap(file, mode="w+", dtype=np.float32,
                                    shape=(table.shape[0], len(columns)))
    step = 200000
    for start in range(0, table.shape[0], step):
        out[start:start + step] = table[start:start + step][:, columns]
    out.flush()
    del out
    return np.load(file, mmap_mode="r")


def first_copies(training, copies, folder, name):
    """The rows of the first `copies` simulated recordings of every half, in the same order, with
    the table in a mapped file of its own. It is the set cv.py fits on with --variants <copies>."""
    keep = training["copy"] < copies
    rows = np.flatnonzero(keep)
    file = os.path.join(folder, name + ".npy")
    table = np.lib.format.open_memmap(file, mode="w+", dtype=np.float32,
                                      shape=(len(rows), training["table"].shape[1]))
    step = 200000
    for start in range(0, len(rows), step):
        table[start:start + step] = training["table"][rows[start:start + step]]
    table.flush()
    del table
    out = {key: training[key][keep] for key in ("target", "run", "metrica", "copy")}
    out["generation"] = training["generation"]
    out["change"] = training["change"][:, keep]
    out["table"] = np.load(file, mmap_mode="r")
    out["names"] = training["names"]
    return out


def code_mark():
    """A short hash of the package and of this script: what made a result."""
    import vantage
    digest = hashlib.sha1()
    folder = os.path.dirname(os.path.abspath(vantage.__file__))
    files = sorted(os.path.join(folder, name) for name in os.listdir(folder) if name.endswith(".py"))
    for file in files + [os.path.abspath(__file__)]:
        with open(file, "rb") as handle:
            digest.update(handle.read().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:12]


def kept_names(names, drop, only, match):
    """The features kept, in table order. `drop` and `only` are lists of names or prefixes ending
    in *, or "@<tag>" for the features of that tag's tables."""
    def expand(patterns):
        if patterns is None:
            return None
        if isinstance(patterns, str):
            patterns = [patterns]
        out = []
        for pattern in patterns:
            out += table_names(pattern[1:], match) if pattern.startswith("@") else [pattern]
        return out

    def hit(name, patterns):
        for pattern in patterns:
            if pattern.endswith("*"):
                if name.startswith(pattern[:-1]):
                    return True
            elif name == pattern:
                return True
        return False

    drop, only = expand(drop) or [], expand(only)
    for pattern in drop + (only or []):
        if not any(hit(name, [pattern]) for name in names):
            raise SystemExit("no feature is called " + pattern)
    return [name for name in names if not hit(name, drop) and (only is None or hit(name, only))]


def state_weights(training, settings):
    weight = external.row_weights(training, settings.get("metrica_weight", 1.0),
                                  settings.get("short_weight", 4.0),
                                  settings.get("short_seconds", 8.0))
    near = settings.get("change_weight")
    if near:
        frames, factor = near           # rows within `frames` of any change count `factor` times
        distance = np.min(training["change"], axis=0)
        boost = np.where(distance <= frames, np.float32(factor), np.float32(1.0))
        weight = boost if weight is None else (weight * boost).astype(np.float32)
    return weight


def fit_state(training, table, names, settings):
    return model.fit(table, training["target"], names, seed=settings.get("seed", 0),
                     rounds=settings.get("rounds", model.ROUNDS),
                     params=settings.get("params") or None, weight=state_weights(training, settings))


def fit_boundary(training, table, names, settings):
    params = dict(model.BOUNDARY_PARAMS)
    params.update(settings.get("params") or {})
    target = change_classes(training["change"], settings.get("width", model.BOUNDARY_WIDTH))
    weight = external.row_weights(training, settings.get("metrica_weight", 1.0))
    return model.fit(table, target, names, seed=settings.get("seed", 0),
                     rounds=settings.get("rounds", 300), params=params, weight=weight)


def saved(tag, match, condition):
    """The saved (state, boundary) answers of an earlier run for one cell, per half."""
    path = os.path.join(ROOT, "runs", "cv", tag, "prob_{}_{}.npz".format(match, condition))
    with np.load(path) as data:
        count = sum(1 for key in data.files if key.startswith("y"))
        return ([data["p{}".format(i)].astype(np.float64) for i in range(count)],
                [data["b{}".format(i)].astype(np.float64) if "b{}".format(i) in data.files else None
                 for i in range(count)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--data-tag", default="g2")
    parser.add_argument("--variants", type=int, default=6)
    parser.add_argument("--gens", nargs="+", default=["g2"])
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="score these recording conditions only (a quick trial)")
    parser.add_argument("--work", default=os.path.join(ROOT, "runs", "cv_tmp"))
    parser.add_argument("--threads", type=int, default=model.PARAMS["num_threads"],
                        help="threads for fitting and predicting (the fit is the same for any number)")
    args = parser.parse_args()
    model.PARAMS["num_threads"] = args.threads
    with open(args.spec) as handle:
        spec = json.load(handle)
    test = list(args.test)
    name = "-".join(test)

    def out_tag(tag, gen):
        return tag if gen == "g2" else "{}_{}".format(tag, gen)

    mark = code_mark()

    def done(variant):
        for gen in variant.get("gens", args.gens):
            path = os.path.join(ROOT, "runs", "cv", out_tag(variant["tag"], gen), "summary_{}.json".format(name))
            if not os.path.isfile(path):
                return False
            with open(path) as handle:
                made_by = json.load(handle).get("code")
            if made_by is not None and made_by != mark:
                raise SystemExit("{} was made by other code ({}, now {}): give the variant a new tag".format(
                    path, made_by, mark))
        return True

    todo = [variant for variant in spec if not done(variant)]
    if not todo:
        log("all variants already there for", name)
        return
    train = [match for match in ALL_MATCHES if match not in test]
    log("code from", os.path.dirname(os.path.abspath(model.__file__)), "data root", ROOT)
    folder = os.path.join(args.work, "{}_{}".format(name, os.getpid()))
    official.spell_overlaps = fast_overlaps

    import lightgbm
    trainings = {}          # data tag -> mapped training set

    def training_for(tag):
        if tag not in trainings:
            wait_for_memory(3.0, "loading the {} tables".format(tag))
            trainings[tag] = mapped_training(tag, train, args.variants, folder)
            log("train tag {} (simulator {}): {} rows x {} features from {}".format(
                tag, trainings[tag]["generation"], len(trainings[tag]["table"]),
                len(trainings[tag]["names"]), train))
        return trainings[tag]

    # -- fit every model that is asked for; a model is keyed by its settings ---------------
    fitted = {}             # key -> (booster, feature names)
    tables = {}             # (tag, copies, tuple(columns)) -> mapped table
    fewer = {}              # (tag, copies) -> the training set of the first copies only

    def store(key, settings=None):
        # named after the tables the model is really fitted on, so that a variant that names its
        # own tables is found again whatever the run's default tag is
        tag = (settings or {}).get("train_tag", args.data_tag)
        digest = hashlib.sha1("|".join([key, tag, str(args.variants)] + train).encode()).hexdigest()[:16]
        return os.path.join(ROOT, "runs", "cv", "_models", name, digest + ".txt")

    newest = {}             # data tag -> when its newest training table was written

    def fresh(key, settings):
        """A stored model is read back only if none of its training tables is newer than it."""
        tag = settings.get("train_tag", args.data_tag)
        if tag not in newest:
            newest[tag] = max(os.path.getmtime(external.train_table_path(tag, match, period, variant, view))
                              for match in train for period in (1, 2)
                              for variant in range(args.variants) for view in ("own", "swap"))
        return os.path.getmtime(store(key, settings)) > newest[tag]

    def get(kind, settings):
        key = kind + json.dumps(settings, sort_keys=True)
        if (key not in fitted and os.path.isfile(store(key, settings)) and os.path.isfile(store(key, settings) + ".json")
                and fresh(key, settings)):
            with open(store(key, settings) + ".json") as handle:
                fitted[key] = (lightgbm.Booster(model_file=store(key, settings)), json.load(handle)["features"])
            log("read {} {} from {}".format(kind, json.dumps(settings, sort_keys=True), store(key, settings)))
        if key not in fitted:
            tag = settings.get("train_tag", args.data_tag)
            training = training_for(tag)
            copies = int(settings.get("copies", args.variants))
            if copies < args.variants:
                if (tag, copies) not in fewer:
                    fewer[(tag, copies)] = first_copies(training, copies, folder,
                                                        "copies{}_{}".format(copies, tag))
                    log("first {} copies of {}: {} rows".format(copies, tag, len(fewer[(tag, copies)]["table"])))
                training = fewer[(tag, copies)]
            names = training["names"]
            used = kept_names(names, settings.get("drop"), settings.get("only"), train[0])
            if len(used) == len(names):
                table = training["table"]
            else:
                columns = [names.index(n) for n in used]
                table_key = (tag, copies, tuple(columns))
                if table_key not in tables:
                    tables[table_key] = column_subset(training["table"], columns, folder,
                                                      "sub{}".format(len(tables)))
                table = tables[table_key]
            wait_for_memory(3.0, "a fit")
            started = time.time()
            fit = fit_state if kind == "state" else fit_boundary
            booster = fit(training, table, used, settings)
            log("fit {} {} on {} features in {:.0f}s".format(
                kind, json.dumps(settings, sort_keys=True), len(used), time.time() - started))
            os.makedirs(os.path.dirname(store(key, settings)), exist_ok=True)
            booster.save_model(store(key, settings) + ".tmp")
            with open(store(key, settings) + ".json", "w") as handle:
                json.dump({"kind": kind, "settings": settings, "features": used, "train": train,
                           "data_tag": tag, "variants": args.variants}, handle)
            os.replace(store(key, settings) + ".tmp", store(key, settings))
            del booster
            fitted[key] = (lightgbm.Booster(model_file=store(key, settings)), used)
        return fitted[key]

    by_tag = {variant["tag"]: variant for variant in spec}

    def resolve(kind, entry):
        if isinstance(entry, str) and entry.startswith("model:"):
            return resolve(kind, by_tag[entry.split(":", 1)[1]].get(kind, {}))
        return entry if isinstance(entry, str) else get(kind, entry)

    plan = []
    for variant in todo:
        plan.append((variant, resolve("state", variant.get("state", {})),
                     resolve("boundary", variant.get("boundary", {}))))

    # -- score: the features of every held-out cell are built once per generation --------------
    grid = cvlib.decode_grid(True)
    checked = False
    windows = {}            # (generation, condition) -> moving averages applied by 'norm'
    for gen in args.gens:
        variants = [(v, s, b) for v, s, b in plan if gen in v.get("gens", args.gens)]
        if not variants:
            continue
        os.environ["VANTAGE_SIM"] = gen
        results = {v["tag"]: {} for v, _, _ in variants}
        for condition in (args.conditions or external.CONDITIONS):
            for match in test:
                halves = [external.condition_half(half, match, condition)
                          for half in labelled_halves(match)]
                from vantage.features import build
                built_for = {}

                def built(norm):
                    if norm not in built_for:
                        out = []
                        for half in halves:
                            if norm:
                                half, window = smoothing.to_reference(half)
                                windows.setdefault((gen, condition), []).append(window)
                            own_names, own = build(half)
                            _, mirrored = build(half.swapped())
                            out.append((own_names, own, mirrored))
                        built_for[norm] = out
                    return built_for[norm]

                def predict(entry, which, norm):
                    booster, used = entry
                    out = []
                    for own_names, own, mirrored in built(norm):
                        columns = [own_names.index(n) for n in used]
                        a = model.predict_table(booster, own[:, columns])
                        b = model.predict_table(booster, mirrored[:, columns])
                        if which == "state":
                            b = b[:, [1, 0, 2]]
                        out.append((0.5 * (a + b)).astype(np.float64))
                    return out

                answers, cache = {}, {}
                for variant, state, boundary in variants:
                    tag = variant["tag"]
                    parts = []
                    for which, entry in (("state", state), ("boundary", boundary)):
                        if isinstance(entry, str):
                            source = entry.split(":", 1)[-1]
                            if entry == "none":
                                parts.append([None] * len(halves))
                            elif source in answers:
                                parts.append(answers[source][0 if which == "state" else 1])
                            else:
                                parts.append(saved(out_tag(source, gen), match,
                                                   condition)[0 if which == "state" else 1])
                        else:
                            norm = bool(variant.get("norm"))
                            key = (which, id(entry[0]), norm)
                            if key not in cache:
                                cache[key] = predict(entry, which, norm)
                            parts.append(cache[key])
                    answers[tag] = (parts[0], parts[1])
                    out_dir = os.path.join(ROOT, "runs", "cv", out_tag(tag, gen))
                    os.makedirs(out_dir, exist_ok=True)
                    payload, triples = {}, []
                    for index, half in enumerate(halves):
                        payload["y{}".format(index)] = half.labels
                        payload["p{}".format(index)] = parts[0][index].astype(np.float32)
                        if parts[1][index] is not None:
                            payload["b{}".format(index)] = parts[1][index].astype(np.float32)
                        triples.append((half.labels, parts[0][index], parts[1][index]))
                    np.savez_compressed(os.path.join(out_dir, "prob_{}_{}.npz".format(match, condition)),
                                        **payload)
                    scores = cvlib.grid_scores(triples, grid)
                    if not checked:
                        # once per run: the fast overlap search gives the official numbers
                        official.spell_overlaps = _slow_overlaps
                        slow = cvlib.grid_scores(triples, grid[:6])
                        official.spell_overlaps = fast_overlaps
                        if slow != scores[:6]:
                            raise SystemExit("the fast overlap search disagrees with the official one")
                        checked = True
                    results[tag]["{}|{}".format(condition, match)] = scores
                    best = int(np.argmax([s["spellF1"] for s in scores]))
                    log("{:<12} {} {:<8} {:<9} best spellF1 {:.3f} macroF1 {:.3f} poss {}/{}".format(
                        tag, gen, condition, match, scores[best]["spellF1"], scores[best]["macroF1"],
                        scores[best]["n_pred"], scores[best]["n_true"]))
        for variant, _, _ in variants:
            tag = variant["tag"]
            path = os.path.join(ROOT, "runs", "cv", out_tag(tag, gen), "summary_{}.json".format(name))
            with open(path, "w") as handle:
                json.dump({"test": test, "train": train, "generation": gen, "code": mark,
                           "data_tag": args.data_tag, "edges": os.environ.get("VANTAGE_EDGES", ""),
                           "grid": [{field: getattr(config, field) for field in cvlib.GRID_FIELDS}
                                    for config in grid],
                           "results": results[tag]}, handle)
    for (gen, condition), used in sorted(windows.items()):
        log("brought to the reference: {} {} with moving averages {}".format(gen, condition, used))
    del tables, fitted, plan, trainings, fewer
    gc.collect()
    shutil.rmtree(folder, ignore_errors=True)
    log("fold done", name)


if __name__ == "__main__":
    main()

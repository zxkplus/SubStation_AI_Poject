#!/usr/bin/env python3
"""One-shot organizer: copy raw annotated data into ROI分割数据/<category>/.

Source data is copied only (never moved or deleted). Each matched image/json
pair is written as <N>.<ext> + <N>.json, continuing per-category numbering.
"""
import json
import os
import shutil
import sys
import zipfile

RAW_BASE = "/media/industai/data1/SEG_DATA/raw_data"
OUT_BASE = "/media/industai/data1/SEG_DATA/ROI分割数据"

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
IMG_EXT_SET = set(IMG_EXT)


def stem(name):
    return os.path.splitext(os.path.basename(name))[0]


def list_images_in_dir(path):
    out = {}
    for f in os.listdir(path):
        if os.path.splitext(f)[1].lower() in IMG_EXT_SET:
            out[stem(f)] = os.path.join(path, f)
    return out


def list_json_in_dir(path):
    out = {}
    for f in os.listdir(path):
        if f.lower().endswith(".json"):
            out[stem(f)] = os.path.join(path, f)
    return out


def read_json_zip(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.lower().endswith(".json") and not n.endswith("/"):
                out[stem(n)] = z.read(n)
    return out


def read_mixed_zip(path):
    imgs, jsons = {}, {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith("/"):
                continue
            if os.path.splitext(n)[1].lower() in IMG_EXT_SET:
                imgs[stem(n)] = z.read(n)
            elif n.lower().endswith(".json"):
                jsons[stem(n)] = z.read(n)
    return imgs, jsons


def next_index(category):
    cat_dir = os.path.join(OUT_BASE, category)
    if not os.path.isdir(cat_dir):
        return 0
    mx = -1
    for f in os.listdir(cat_dir):
        s, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXT_SET or ext.lower() == ".json":
            if s.isdigit():
                mx = max(mx, int(s))
    return mx + 1


def build_batches():
    R = RAW_BASE
    batches = []

    for i in range(1, 13):
        img = os.path.join(R, "202601210067_20260528", f"表计{i}")
        if i == 1:
            img_dir = os.path.join(img, "1")
        elif i == 12:
            img_dir = os.path.join(img, "12-1966jpg")
        else:
            img_dir = os.path.join(img, f"{i}-jpg")
        if i == 1:
            jz = os.path.join(img, "1-json.zip")
        elif i == 12:
            jz = os.path.join(img, "12-1996json.zip")
        else:
            jz = os.path.join(img, f"{i}-json.zip")
        batches.append(dict(label=f"20260121 表计{i}", target="表计",
                            img_dir=img_dir, json_zip=jz))

    simple = [
        ("伸缩节-264", "264-jpg", "264-json", "伸缩节"),
        ("升高坐-322", "322-jpg", "322-json", "升高坐"),
        ("呼吸器-1208", "1208-jpg", "1208-json", "呼吸器"),
        ("机构箱-4291", "4291-jpg", "4291-json", "机构箱"),
        ("继电器-384", "384-jpg", "384-json", "继电器"),
    ]
    for cat, idir, jdir, target in simple:
        top = os.path.join(R, "202602250035_20260310", cat)
        batches.append(dict(label=f"20260225 {cat}", target=target,
                            img_dir=os.path.join(top, idir),
                            json_dir=os.path.join(top, jdir)))

    for n in range(1, 12):
        leaf = os.path.join(R, "202602250035_20260310", "隔离开关-11210", str(n))
        batches.append(_leaf_batch(f"20260225 隔离{n}", leaf, "隔离开关"))

    jg = os.path.join(R, "202605070034_20260604", "伸缩节（继电器）")
    rel_sub = [
        ("1-伸缩节736", "1-伸缩节736-json.zip", "伸缩节"),
        ("2-气体继电器460", "2-继电器460-json.zip", "继电器"),
        ("3-伸缩节126", "3-伸缩节126-json.zip", "伸缩节"),
        ("4-气体继电器119", "4-继电器119-json.zip", "继电器"),
    ]
    for sub, jz, target in rel_sub:
        batches.append(dict(label=f"20260507 伸缩节/继电器 {sub}", target=target,
                            img_dir=os.path.join(jg, "jpg", sub),
                            json_zip=os.path.join(jg, "json", jz)))

    for n in range(1, 5):
        leaf = os.path.join(R, "202605070034_20260604", "均压环-3999", str(n))
        batches.append(_leaf_batch(f"20260507 均压环{n}", leaf, "均压环"))

    for n in range(1, 6):
        top = os.path.join(R, "202605070034_20260604", f"引线接头-{n}")
        batches.append(dict(label=f"20260507 引线接头{n}", target="引线接头",
                            img_dir=os.path.join(top, f"{n}-jpg"),
                            json_zip=os.path.join(top, f"{n}-json.zip")))

    top = os.path.join(R, "202605070034_20260604", "散热器修改数据-1395")
    batches.append(dict(label="20260507 散热器修改数据", target="散热器",
                        img_dir=os.path.join(top, "1395张-jpg", "散热器"),
                        json_zip=os.path.join(top, "1395张-json.zip")))

    top = os.path.join(R, "202606110064_20260626", "伸缩节")
    batches.append(dict(label="20260611 伸缩节", target="伸缩节",
                        img_dir=os.path.join(top, "伸缩节-1126jpg"),
                        json_zip=os.path.join(top, "伸缩节-1126json.zip")))

    for n in range(1, 5):
        leaf = os.path.join(R, "202606110064_20260626", "机构箱", str(n))
        batches.append(_leaf_batch(f"20260611 机构箱{n}", leaf, "机构箱"))

    for n in range(1, 12):
        leaf = os.path.join(R, "202606110064_20260626", "隔离开关", str(n))
        batches.append(_leaf_batch(f"20260611 隔离{n}", leaf, "隔离开关"))

    batches.append(dict(
        label="20260804 呼吸器1000",
        target="呼吸器",
        mixed_zip=os.path.join(R, "202608040012_20260806",
                               "数据外协247批次-呼吸器1000张.zip")))

    # 20260708 油枕
    yz = os.path.join(R, "202607080073_20260727")
    batches.append(dict(
        label="20260708 油枕1", target="油枕",
        img_dir=os.path.join(yz, "1", "油枕1-1000jpg"),
        json_zip=os.path.join(yz, "1", "油枕1-1000json.zip")))
    for n in range(2, 6):
        zname = f"油枕{n}-1000张.zip" if n < 5 else "油枕5-1427张.zip"
        batches.append(dict(
            label=f"20260708 油枕{n}", target="油枕",
            mixed_zip=os.path.join(yz, zname)))

    return batches


def _leaf_batch(label, leaf, target):
    """A numeric leaf dir contains one image dir + one json zip."""
    img_dir = None
    json_zip = None
    for ent in os.listdir(leaf):
        p = os.path.join(leaf, ent)
        if os.path.isdir(p) and not ent.lower().endswith(".zip"):
            img_dir = p
        elif ent.lower().endswith(".zip"):
            json_zip = p
    return dict(label=label, target=target, img_dir=img_dir, json_zip=json_zip)


def run(batches):
    report = []
    total_ok = 0
    counters = {}

    for b in batches:
        target = b["target"]
        counters.setdefault(target, next_index(target))

        imgs = {}
        jsons = {}
        if b.get("mixed_zip"):
            imgs, jsons = read_mixed_zip(b["mixed_zip"])
        else:
            imgs = list_images_in_dir(b["img_dir"])
            if b.get("json_dir"):
                jsons = list_json_in_dir(b["json_dir"])
            elif b.get("json_zip"):
                jsons = read_json_zip(b["json_zip"])

        matched = sorted(set(imgs) & set(jsons))
        only_img = sorted(set(imgs) - set(jsons))
        only_json = sorted(set(jsons) - set(imgs))

        ok = 0
        for s in matched:
            n = counters[target]
            counters[target] += 1
            img_src = imgs[s]
            json_src = jsons[s]
            if isinstance(img_src, str):
                ext = os.path.splitext(img_src)[1].lower()
            else:
                ext = _ext_from_mixed(b["mixed_zip"], s)
            img_name = f"{n}{ext}"
            cat_dir = os.path.join(OUT_BASE, target)
            os.makedirs(cat_dir, exist_ok=True)
            dest_img = os.path.join(cat_dir, img_name)
            dest_json = os.path.join(cat_dir, f"{n}.json")
            if os.path.exists(dest_img) or os.path.exists(dest_json):
                raise FileExistsError(f"target exists: {dest_img} / {dest_json}")
            _write_image(img_src, dest_img)
            _write_json(json_src, dest_json, img_name)
            ok += 1

        total_ok += ok
        report.append((b["label"], target, ok, len(only_img), len(only_json)))

    _print_report(report, total_ok)


def _ext_from_mixed(zip_path, s):
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if stem(n) == s and os.path.splitext(n)[1].lower() in IMG_EXT_SET:
                return os.path.splitext(n)[1].lower()
    raise FileNotFoundError(f"image not found in mixed zip for stem {s}")


def _write_image(src, dest):
    if isinstance(src, str):
        shutil.copyfile(src, dest)
    else:
        with open(dest, "wb") as fh:
            fh.write(src)


def _write_json(src, dest, image_name):
    if isinstance(src, str):
        with open(src, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = json.loads(src.decode("utf-8"))
    if isinstance(data, dict):
        data["imagePath"] = image_name
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _print_report(report, total_ok):
    print(f"total copied: {total_ok}")
    by_cat = {}
    for label, target, ok, oi, oj in report:
        by_cat[target] = by_cat.get(target, 0) + ok
        print(f"{label}: target={target} ok={ok} img_only={oi} json_only={oj}")
    print("--- per category ---")
    for cat in sorted(by_cat):
        print(f"{cat}: {by_cat[cat]}")
    print("--- on-disk per category ---")
    for cat in sorted(os.listdir(OUT_BASE)):
        d = os.path.join(OUT_BASE, cat)
        if os.path.isdir(d):
            img = sum(1 for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT_SET)
            js = sum(1 for f in os.listdir(d) if f.lower().endswith(".json"))
            print(f"{cat}: imgs={img} json={js} match={img == js}")


def main():
    if "--dry-run" in sys.argv:
        for b in build_batches():
            print(b["label"], "->", b["target"],
                  b.get("img_dir"), b.get("json_zip") or b.get("json_dir") or b.get("mixed_zip"))
        return
    run(build_batches())


if __name__ == "__main__":
    main()

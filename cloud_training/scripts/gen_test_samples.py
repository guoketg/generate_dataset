"""gen_test_samples.py —— PIL 合成订单截图/防伪码图/瑕疵图(PRD 17 5.3.5)。

数量参数化: --n-e2e 20(e2e 测试) --n-train 800(训练数据消费,PRD 18)。
关键设计:每张合成图同时落盘元数据 meta.jsonl(图内 ground truth),
供训练态执行器 ocr/vl_describe「真实执行」与 gen_questions 出题消费。

幂等:各目录 meta.jsonl 存在且行数达标则跳过该类。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # cloud_training/
DATA = _ROOT / "data"

import qrcode
from PIL import Image, ImageDraw, ImageFont

SEED = 42
STATUS_CN = ["已发货", "运输中", "派送中", "已签收", "已拒收"]
DEFECT_TYPES = [
    ("开胶", "鞋面右侧有明显开胶，位于接缝处，长约 3cm"),
    ("划痕", "表面有一道明显划痕，位于正面中央，长约 5cm"),
    ("污渍", "表面有无法擦除的深色污渍，位于左下角，约 4x4cm"),
    ("变形", "外壳受挤压变形，位于顶部边角"),
    ("破损", "边角破裂露出内部材料，位于右下角"),
]
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "arial.ttf",
]


def load_font(size: int):
    for fp in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(fp, size)
        except OSError:
            continue
    print("[WARN] no CJK font found, Chinese text may render as boxes", flush=True)
    return ImageFont.load_default()


def gen_order_screenshot(order_id: str, product_title: str, price: float,
                         status_cn: str, path: Path, font, big_font):
    """订单截图:订单号/商品/金额/状态(元数据即 OCR ground truth)"""
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "订单详情", fill="black", font=big_font)
    draw.text((20, 80), f"订单号：{order_id}", fill="black", font=font)
    draw.text((20, 120), f"商品：{product_title[:40]}", fill="black", font=font)
    draw.text((20, 160), f"金额：￥{price:.2f}", fill="red", font=font)
    draw.text((20, 200), f"状态：{status_cn}", fill="blue", font=font)
    img.save(path)


def gen_anti_fake_image(code: str, product_id: int, path: Path, font):
    """防伪码图:二维码 + 文字标签"""
    qr = qrcode.make(code).resize((300, 300))
    img = Image.new("RGB", (400, 420), "white")
    img.paste(qr, (50, 20))
    draw = ImageDraw.Draw(img)
    draw.text((50, 340), f"防伪码：{code}", fill="black", font=font)
    draw.text((50, 370), f"商品 ID：{product_id}", fill="black", font=font)
    img.save(path)


def gen_defect_image(product_img_path: Path, defect_type: tuple, path: Path):
    """瑕疵图:商品图 + 红框标注(红框位置即瑕疵位置,元数据落盘)"""
    img = Image.open(product_img_path).convert("RGB").resize((400, 400))
    draw = ImageDraw.Draw(img)
    x, y = random.randint(50, 270), random.randint(50, 270)
    draw.rectangle([x, y, x + 80, y + 80], outline="red", width=4)
    img.save(path, quality=88)
    return {"x": x, "y": y, "w": 80, "h": 80}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-e2e", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=800)
    args = ap.parse_args()
    n_total = args.n_e2e + args.n_train
    rng = random.Random(SEED)
    random.seed(SEED)

    t0 = time.time()
    products = [json.loads(l) for l in open(DATA/"products.jsonl", encoding="utf-8")]
    anti_fakes = [json.loads(l) for l in open(DATA/"anti_fake.jsonl", encoding="utf-8")]
    font, big_font = load_font(20), load_font(28)

    # 1) 订单截图(e2e 前 n_e2e 张命名 e2e_,其余 train_)
    d = DATA/"images/orders"
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / "meta.jsonl"
    if meta_path.exists():
        n_exist = sum(1 for _ in open(meta_path, encoding="utf-8"))
    else:
        n_exist = 0
    if n_exist < n_total:
        metas = ([json.loads(l) for l in open(meta_path, encoding="utf-8")]
                 if meta_path.exists() else [])
        for i in range(n_exist, n_total):
            p = rng.choice(products)
            status_cn = rng.choice(STATUS_CN)
            prefix = "e2e" if i < args.n_e2e else "train"
            fname = f"{prefix}_order_{i:04d}.png"
            gen_order_screenshot(f"ORD{p['product_id']:08d}", p["title"],
                                 p["price"], status_cn, d / fname, font, big_font)
            metas.append({"file": fname, "type": "order_screenshot",
                          "order_id": f"ORD{p['product_id']:08d}",
                          "product_id": p["product_id"],
                          "product_title": p["title"], "price": p["price"],
                          "status_cn": status_cn, "split": prefix})
            if (i + 1) % 200 == 0:
                print(f"[orders] {i + 1}/{n_total}", flush=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in metas)
    print(f"[orders] done ({n_total})", flush=True)

    # 2) 防伪码图
    d = DATA/"images/anti_fake"
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / "meta.jsonl"
    n_exist = (sum(1 for _ in open(meta_path, encoding="utf-8"))
               if meta_path.exists() else 0)
    if n_exist < n_total:
        metas = ([json.loads(l) for l in open(meta_path, encoding="utf-8")]
                 if meta_path.exists() else [])
        for i in range(n_exist, n_total):
            af = rng.choice(anti_fakes)
            prefix = "e2e" if i < args.n_e2e else "train"
            fname = f"{prefix}_af_{i:04d}.png"
            gen_anti_fake_image(af["code"], af["product_id"], d / fname, font)
            metas.append({"file": fname, "type": "anti_fake",
                          "code": af["code"], "product_id": af["product_id"],
                          "is_genuine": af["is_genuine"], "split": prefix})
            if (i + 1) % 200 == 0:
                print(f"[anti_fake] {i + 1}/{n_total}", flush=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in metas)
    print(f"[anti_fake] done ({n_total})", flush=True)

    # 3) 瑕疵图(基于商品图加红框;依赖 Unsplash 商品图,缺图时记录跳过)
    d = DATA/"images/defects"
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / "meta.jsonl"
    product_imgs = sorted(DATA/"images/products".glob("*.jpg"))
    if not product_imgs:
        print("[defects] WARN: no product images yet, run "
              "fetch_product_images.py first; skip defects", flush=True)
    else:
        n_exist = (sum(1 for _ in open(meta_path, encoding="utf-8"))
                   if meta_path.exists() else 0)
        if n_exist < n_total:
            metas = ([json.loads(l) for l in open(meta_path, encoding="utf-8")]
                     if meta_path.exists() else [])
            for i in range(n_exist, min(n_total, len(product_imgs))):
                src = product_imgs[i % len(product_imgs)]
                dtype, ddesc = rng.choice(DEFECT_TYPES)
                prefix = "e2e" if i < args.n_e2e else "train"
                fname = f"{prefix}_defect_{i:04d}.jpg"
                box = gen_defect_image(src, (dtype, ddesc), d / fname)
                metas.append({"file": fname, "type": "defect",
                              "defect_type": dtype, "defect_desc": ddesc,
                              "box": box, "src_image": src.name, "split": prefix})
                if (i + 1) % 200 == 0:
                    print(f"[defects] {i + 1}/{n_total}", flush=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in metas)
        print(f"[defects] done ({min(n_total, len(product_imgs))})", flush=True)

    print(f"[gen_test_samples] DONE elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

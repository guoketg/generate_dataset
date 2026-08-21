"""translate_products.py —— 翻译+清洗产品数据

功能：
1. 将英文标题翻译成中文
2. 将真实品牌名称替换为通用描述
3. 保留配件的适配信息
4. 生成清洗后的产品数据

设计原则：
- 不依赖外部翻译API（网络受限）
- 使用规则+映射字典进行翻译
- 保留商品的核心信息
- 适合开源项目使用
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # cloud_training/
DATA = _ROOT / "data"

# 真实品牌→中文名称映射（保留品牌，只做翻译）
BRAND_MAPPING = {
    # 手机品牌
    "Apple": "苹果",
    "Samsung": "三星",
    "Huawei": "华为",
    "Xiaomi": "小米",
    "Sony": "索尼",
    "LG": "LG",
    "Google": "谷歌",
    "OnePlus": "一加",
    "Oppo": "OPPO",
    "Vivo": "vivo",
    
    # 电脑品牌
    "Dell": "戴尔",
    "HP": "惠普",
    "Lenovo": "联想",
    "Asus": "华硕",
    "Acer": "宏碁",
    "Microsoft": "微软",
    # 产品线名保留原样
    "MacBook": "MacBook",
    "ThinkPad": "ThinkPad",
    
    # 平板（产品线名保留原样）
    "iPad": "iPad",
    "Galaxy Tab": "Galaxy Tab",
    "Surface": "Surface",
    
    # 手机型号（产品线名保留原样）
    "iPhone": "iPhone",
    "Galaxy S": "Galaxy S",
    "Galaxy Note": "Galaxy Note",
    "Pixel": "Pixel",
    
    # 耳机（产品线名保留原样）
    "AirPods": "AirPods",
    "Galaxy Buds": "Galaxy Buds",
    "Pixel Buds": "Pixel Buds",
    
    # 其他设备（产品线名保留原样）
    "Kindle": "Kindle",
    "Apple Watch": "Apple Watch",
    "Galaxy Watch": "Galaxy Watch",
    "PlayStation": "PlayStation",
    "Xbox": "Xbox",
    "Nintendo Switch": "Nintendo Switch",
    
    # 电商平台
    "Amazon": "亚马逊",
    "Alexa": "Alexa",
    "Echo": "Echo",
    "Fire TV": "Fire TV",
    "Fire Tablet": "Fire Tablet",
}

# 商品类别→中文翻译
CATEGORY_TRANSLATION = {
    # 电子产品
    "Headphones": "耳机",
    "Earbuds": "耳塞",
    "Earphones": "耳机",
    "Speaker": "音箱",
    "Microphone": "麦克风",
    "Camera": "摄像头",
    "Webcam": "网络摄像头",
    "Charger": "充电器",
    "Cable": "数据线",
    "Adapter": "适配器",
    "Power Bank": "充电宝",
    "Battery": "电池",
    "Case": "保护壳",
    "Cover": "保护套",
    "Screen Protector": "屏幕保护膜",
    "Protector": "保护膜",
    "Stand": "支架",
    "Mount": "支架",
    "Holder": "支架",
    "Dock": "扩展坞",
    "Hub": "集线器",
    
    # 电脑配件
    "Keyboard": "键盘",
    "Mouse": "鼠标",
    "Monitor": "显示器",
    "Printer": "打印机",
    "Scanner": "扫描仪",
    "Hard Drive": "硬盘",
    "SSD": "固态硬盘",
    "Memory": "内存",
    "RAM": "内存",
    "Graphics Card": "显卡",
    "Motherboard": "主板",
    "Power Supply": "电源",
    "Cooling Fan": "散热风扇",
    "散热器": "散热器",
    
    # 家居用品
    "Lamp": "台灯",
    "Light": "灯",
    "Fan": "风扇",
    "Heater": "加热器",
    "Humidifier": "加湿器",
    "Air Purifier": "空气净化器",
    "Vacuum": "吸尘器",
    "Mop": "拖把",
    "Broom": "扫帚",
    
    # 厨房用品
    "Blender": "搅拌机",
    "Mixer": "搅拌机",
    "Toaster": "烤面包机",
    "Coffee Maker": "咖啡机",
    "Kettle": "电水壶",
    "Rice Cooker": "电饭煲",
    "Microwave": "微波炉",
    "Oven": "烤箱",
    
    # 运动用品
    "Yoga Mat": "瑜伽垫",
    "Dumbbell": "哑铃",
    "Resistance Band": "弹力带",
    "Jump Rope": "跳绳",
    "Water Bottle": "水杯",
    
    # 服装
    "T-Shirt": "T恤",
    "Shirt": "衬衫",
    "Pants": "裤子",
    "Shorts": "短裤",
    "Jacket": "夹克",
    "Coat": "外套",
    "Shoes": "鞋子",
    "Socks": "袜子",
    "Hat": "帽子",
    "Gloves": "手套",
    "Scarf": "围巾",
    
    # 美妆
    "Lipstick": "口红",
    "Foundation": "粉底",
    "Mascara": "睫毛膏",
    "Eyeshadow": "眼影",
    "Perfume": "香水",
    "Cream": "面霜",
    "Lotion": "乳液",
    "Serum": "精华",
    
    # 母婴
    "Diaper": "纸尿裤",
    "Baby Food": "婴儿食品",
    "Toy": "玩具",
    "Stroller": "婴儿车",
    "Car Seat": "安全座椅",
    
    # 图书
    "Book": "图书",
    "Notebook": "笔记本",
    "Pen": "笔",
    "Pencil": "铅笔",
    "Marker": "马克笔",
    
    # 其他
    "Sticker": "贴纸",
    "Decal": "贴花",
    "Tape": "胶带",
    "Glue": "胶水",
    "Scissors": "剪刀",
    "Knife": "刀",
    "Tool": "工具",
    "Kit": "套装",
    "Set": "套装",
    "Pack": "装",
    "Piece": "件",
}

# 型号/规格→中文
SPEC_TRANSLATION = {
    "ft": "英尺",
    "inch": "英寸",
    "mm": "毫米",
    "cm": "厘米",
    "m": "米",
    "g": "克",
    "kg": "千克",
    "ml": "毫升",
    "L": "升",
    "W": "瓦",
    "V": "伏",
    "A": "安",
    "mAh": "毫安时",
    "GB": "GB",
    "TB": "TB",
    "MHz": "MHz",
    "GHz": "GHz",
}


def translate_brand(brand: str) -> str:
    """将真实品牌替换为通用描述"""
    for eng, chn in BRAND_MAPPING.items():
        if eng.lower() in brand.lower():
            return chn
    return brand


def translate_title(title: str) -> str:
    """将英文标题翻译成中文（规则方法）"""
    # 1. 替换真实品牌（按长度排序，优先匹配长词）
    sorted_brands = sorted(BRAND_MAPPING.items(), key=lambda x: len(x[0]), reverse=True)
    for eng, chn in sorted_brands:
        title = re.sub(re.escape(eng), chn, title, flags=re.IGNORECASE)
    
    # 2. 翻译常见商品类别（按长度排序）
    sorted_categories = sorted(CATEGORY_TRANSLATION.items(), key=lambda x: len(x[0]), reverse=True)
    for eng, chn in sorted_categories:
        title = re.sub(r'\b' + re.escape(eng) + r'\b', chn, title, flags=re.IGNORECASE)
    
    # 3. 翻译规格单位
    for eng, chn in SPEC_TRANSLATION.items():
        title = re.sub(r'(\d+)\s*' + re.escape(eng) + r'\b', r'\1' + chn, title)
    
    # 4. 翻译通用功能词汇（保留产品型号如 Pro、Air、Plus 等）
    common_words = {
        "Charging": "充电",
        "Wireless": "无线",
        "Bluetooth": "蓝牙",
        "Waterproof": "防水",
        "Portable": "便携",
        "Smart": "智能",
        "Digital": "数码",
        "Gaming": "游戏",
        "Office": "办公",
        "Home": "家用",
        "Car": "车载",
        "Phone": "手机",
        "Tablet": "平板",
        "Laptop": "笔记本",
        "Desktop": "台式机",
        "Headphones": "耳机",
        "Earbuds": "耳塞",
        "Speaker": "音箱",
        "Charger": "充电器",
        "Cable": "数据线",
        "Case": "保护壳",
        "Cover": "保护套",
        "Screen Protector": "屏幕保护膜",
        "Protector": "保护膜",
        "Stand": "支架",
        "Mount": "支架",
        "Holder": "支架",
        "Dock": "扩展坞",
        "Hub": "集线器",
        "Keyboard": "键盘",
        "Mouse": "鼠标",
        "Monitor": "显示器",
        "Printer": "打印机",
        "Hard Drive": "硬盘",
        "SSD": "固态硬盘",
        "Memory": "内存",
        "RAM": "内存",
        "Graphics Card": "显卡",
        "Motherboard": "主板",
        "Power Supply": "电源",
        "Cooling Fan": "散热风扇",
        "散热器": "散热器",
        "Lamp": "台灯",
        "Light": "灯",
        "Fan": "风扇",
        "Heater": "加热器",
        "Humidifier": "加湿器",
        "Air Purifier": "空气净化器",
        "Vacuum": "吸尘器",
    }
    for eng, chn in common_words.items():
        title = re.sub(r'\b' + re.escape(eng) + r'\b', chn, title, flags=re.IGNORECASE)
    
    # 5. 清理多余空格
    title = re.sub(r'\s+', ' ', title).strip()
    
    return title


def clean_description(desc: str) -> str:
    """清洗描述文本，移除英文残留"""
    # 移除英文括号内容
    desc = re.sub(r'\([^)]*\)', '', desc)
    # 移除英文单词（保留中文）
    desc = re.sub(r'[a-zA-Z]+', '', desc)
    # 清理多余空格和标点
    desc = re.sub(r'\s+', ' ', desc).strip()
    desc = re.sub(r'，+', '，', desc)
    desc = re.sub(r'。+', '。', desc)
    return desc


def main():
    # 加载原始数据
    products = [json.loads(l) for l in open(DATA/"products.jsonl", encoding="utf-8")]
    
    print(f"[translate] 原始产品数量: {len(products)}", flush=True)
    
    # 翻译+清洗
    translated = []
    brand_count = 0
    
    for p in products:
        # 使用中文描述作为主要标题（如果存在）
        # 原始数据中 description 已经是中文
        desc_cn = p.get("description", "")
        
        # 如果描述为空，才尝试翻译英文标题
        if not desc_cn or len(desc_cn) < 10:
            title_cn = translate_title(p["title"])
        else:
            title_cn = desc_cn
        
        # 检测是否包含真实品牌
        has_real_brand = False
        for eng in BRAND_MAPPING.keys():
            if eng.lower() in p["title"].lower():
                has_real_brand = True
                brand_count += 1
                break
        
        # 翻译品牌名
        brand_cn = translate_brand(p["brand"])
        
        # 构建翻译后的产品
        p_translated = {
            "product_id": p["product_id"],
            "title": title_cn,
            "category": p["category"],
            "brand": brand_cn,
            "model": p["model"],
            "price": p["price"],
            "platform": p["platform"],
            "description": desc_cn,
            "attributes": p.get("attributes", {}),
            "original_title": p["title"],  # 保留原始标题供参考
            "has_real_brand": has_real_brand,
        }
        translated.append(p_translated)
    
    # 保存翻译后的数据
    output_path = DATA/"products_cn.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in translated)
    
    print(f"[translate] 翻译完成: {len(translated)} 个产品", flush=True)
    print(f"[translate] 包含真实品牌: {brand_count} 个 ({brand_count/len(translated)*100:.1f}%)", flush=True)
    print(f"[translate] 输出文件: {output_path}", flush=True)
    
    # 打印示例
    print("\n[translate] 翻译示例:", flush=True)
    for i in [0, 1, 2, 10, 50, 100]:
        if i < len(translated):
            p = translated[i]
            print(f"  原始标题: {p['original_title'][:60]}...", flush=True)
            print(f"  中文标题: {p['title'][:60]}...", flush=True)
            print(f"  品牌: {p['brand']}", flush=True)
            print(flush=True)


if __name__ == "__main__":
    main()

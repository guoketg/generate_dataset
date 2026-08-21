"""gen_taobao_screenshots.py —— 基于真实电商UI结构的合成截图

生成三种类型的截图：
1. 物流详情页（6 节点时间轴）
2. 订单详情页（已签收状态，包含店铺、商品、金额、订单信息）
3. 订单列表页（多个订单卡片）

设计原则：
- 基于真实电商截图的文字描述（通用化，不包含特定平台品牌）
- 包含足够的信息供 OCR/VL 模型识别
- 生成元数据供训练数据消费
- 适合开源项目使用
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # cloud_training/
DATA = _ROOT / "data"

from PIL import Image, ImageDraw, ImageFont

SEED = 42

# 字体候选
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "arial.ttf",
]

# 颜色定义（淘宝风格）
COLORS = {
    "orange": (255, 120, 0),       # 淘宝橙
    "orange_light": (255, 240, 220),  # 浅橙底
    "green": (0, 180, 0),          # 绿色标签
    "green_light": (230, 255, 230),  # 浅绿底
    "red": (255, 60, 60),          # 红色价格
    "gray": (150, 150, 150),       # 灰色文字
    "gray_light": (240, 240, 240),  # 浅灰底
    "gray_dark": (100, 100, 100),  # 深灰文字
    "black": (30, 30, 30),         # 近黑色文字
    "white": (255, 255, 255),
    "bg": (245, 245, 245),         # 页面背景
}

# 快递公司列表
EXPRESS_COMPANIES = ["中通快递", "圆通速递", "韵达快递", "申通快递", "顺丰速运", "京东物流"]

# 城市列表（发货地→中转地→目的地）- 丰富化
CITY_ROUTES = [
    # 华中地区
    ("合肥瑶海", "鄂州转运中心", "武汉武昌", "湖北工业大学"),
    ("长沙岳麓", "武汉转运中心", "武汉洪山", "华中师范大学"),
    ("郑州金水", "许昌转运中心", "洛阳涧西", "洛阳理工学院"),
    ("南昌青山湖", "九江转运中心", "武汉江汉", "江汉大学"),
    # 华东地区
    ("杭州萧山", "金华转运中心", "上海浦东", "复旦大学"),
    ("苏州工业园", "无锡转运中心", "南京鼓楼", "南京大学"),
    ("宁波鄞州", "温州转运中心", "杭州西湖", "浙江大学"),
    ("厦门思明", "泉州转运中心", "福州鼓楼", "福建师范大学"),
    # 华南地区
    ("深圳宝安", "东莞转运中心", "广州天河", "华南理工大学"),
    ("广州白云", "佛山转运中心", "珠海香洲", "中山大学"),
    ("南宁青秀", "柳州转运中心", "桂林秀峰", "广西师范大学"),
    # 华北地区
    ("北京朝阳", "天津转运中心", "石家庄桥西", "河北师范大学"),
    ("青岛崂山", "济南转运中心", "烟台芝罘", "鲁东大学"),
    # 西南地区
    ("成都武侯", "重庆转运中心", "贵阳南明", "贵州大学"),
    ("昆明盘龙", "大理转运中心", "丽江古城", "云南大学"),
    # 西北地区
    ("西安雁塔", "咸阳转运中心", "兰州城关", "兰州大学"),
]

# 状态节点模板
STATUS_TEMPLATES = {
    "已签收": [
        {"status": "已签收", "desc": "包裹已从代收点取出"},
        {"status": "待取件", "desc": "快件已在{station}暂放"},
        {"status": "派送中", "desc": "{courier}快递员正在为您派件"},
        {"status": "运输中", "desc": "快件已到达{station}"},
        {"status": "运输中", "desc": "快件已从{transit}发出"},
        {"status": "已揽件", "desc": "快件已由{origin}揽收"},
    ],
    "运输中": [
        {"status": "运输中", "desc": "快件已从{transit}发出"},
        {"status": "已揽件", "desc": "快件已由{origin}揽收"},
    ],
    "派送中": [
        {"status": "派送中", "desc": "{courier}快递员正在为您派件"},
        {"status": "运输中", "desc": "快件已到达{station}"},
        {"status": "运输中", "desc": "快件已从{transit}发出"},
        {"status": "已揽件", "desc": "快件已由{origin}揽收"},
    ],
}

# 商品类别（丰富化）
PRODUCT_CATEGORIES = [
    {"name": "食品", "titles": [
        "星空莓野吐司全麦面包", "黄山薄脆烧饼", "滇式宣威火腿酥皮",
        "有机燕麦片500g", "每日坚果混合装", "进口巧克力礼盒",
        "即食鸡胸肉低脂", "冻干水果脆片", "手工曲奇饼干",
    ]},
    {"name": "服饰", "titles": [
        "纯棉T恤男款", "运动鞋女款透气", "冬季加厚羽绒服",
        "休闲牛仔裤直筒", "连帽卫衣宽松款", "防晒衣轻薄透气",
        "羊毛衫圆领毛衣", "运动短裤速干", "真皮腰带头层牛皮",
    ]},
    {"name": "数码", "titles": [
        "无线蓝牙耳机", "手机壳防摔", "充电器快充",
        "机械键盘青轴", "游戏鼠标无线", "USB-C数据线",
        "移动电源20000mAh", "智能手表运动版", "降噪耳机头戴式",
    ]},
    {"name": "家居", "titles": [
        "保温杯不锈钢", "收纳盒塑料", "台灯护眼LED",
        "四件套纯棉", "乳胶枕头护颈", "扫地机器人智能",
        "空气净化器家用", "电饭煲IH加热", "破壁机多功能",
    ]},
    {"name": "美妆", "titles": [
        "口红丝绒质地", "粉底液持久控油", "眼影盘多色",
        "面膜补水保湿", "洗发水氨基酸", "身体乳滋润",
    ]},
    {"name": "母婴", "titles": [
        "婴儿奶粉3段", "纸尿裤透气", "儿童积木益智",
        "宝宝辅食机", "婴儿推车轻便", "儿童保温杯",
    ]},
    {"name": "运动", "titles": [
        "瑜伽垫加厚", "跑步鞋减震", "哑铃可调节",
        "运动护膝专业", "健身手套男女", "跳绳计数",
    ]},
    {"name": "图书", "titles": [
        "编程入门教程", "小说畅销书", "儿童绘本故事",
        "考试辅导资料", "摄影教程", "心理学书籍",
    ]},
]

# 店铺名称（通用化，不包含特定平台品牌）
SHOP_NAMES = [
    "暴肌独角兽官方自营店", "寻食记食品专营店", "小米官方旗舰店",
    "华为官方旗舰店", "Nike官方旗舰店", "优衣库官方旗舰店",
    "三只松鼠旗舰店", "良品铺子旗舰店", "百草味旗舰店",
    "格力电器旗舰店", "美的官方旗舰店", "海尔官方旗舰店",
    "苹果官方旗舰店", "三星官方旗舰店", "索尼官方旗舰店",
    "戴森官方旗舰店", "飞利浦官方旗舰店", "松下官方旗舰店",
]

# 收货地址数据（随机组合）
ADDRESSES = [
    ("南李路28号", "马克林", "1592414"),
    ("解放大道128号", "张伟", "1381234"),
    ("中山路56号", "李娜", "1398765"),
    ("人民路88号", "王芳", "1365678"),
    ("建设大道168号", "刘洋", "1354321"),
    ("和平路99号", "陈静", "1369876"),
    ("长江路15号", "赵强", "1376543"),
    ("南京路200号", "孙丽", "1387654"),
    ("北京路88号", "周杰", "1394567"),
    ("上海路168号", "吴敏", "1361234"),
    ("广州路56号", "郑浩", "1378901"),
    ("深圳路128号", "王磊", "1385678"),
    ("杭州路99号", "李静", "1392345"),
    ("成都路15号", "张伟", "1367890"),
    ("武汉路200号", "刘芳", "1374567"),
    ("长沙路88号", "陈强", "1389012"),
    ("西安路168号", "赵丽", "1396789"),
    ("重庆路56号", "孙杰", "1363456"),
    ("天津路128号", "周敏", "1370123"),
    ("南京路99号", "吴浩", "1384567"),
]

# 手机号前缀
PHONE_PREFIXES = ["138", "139", "136", "137", "135", "159", "158", "188", "189", "186"]


def load_font(size: int):
    for fp in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(fp, size)
        except OSError:
            continue
    print("[WARN] no CJK font found, Chinese text may render as boxes", flush=True)
    return ImageFont.load_default()


def draw_rounded_rect(draw, xy, radius, fill=None, outline=None, width=1):
    """绘制圆角矩形"""
    x1, y1, x2, y2 = xy
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill, outline=None)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill, outline=None)
    draw.pieslice([x1, y1, x1 + 2*radius, y1 + 2*radius], 180, 270, fill=fill)
    draw.pieslice([x2 - 2*radius, y1, x2, y1 + 2*radius], 270, 360, fill=fill)
    draw.pieslice([x1, y2 - 2*radius, x1 + 2*radius, y2], 90, 180, fill=fill)
    draw.pieslice([x2 - 2*radius, y2 - 2*radius, x2, y2], 0, 90, fill=fill)
    if outline:
        draw.arc([x1, y1, x1 + 2*radius, y1 + 2*radius], 180, 270, fill=outline, width=width)
        draw.arc([x2 - 2*radius, y1, x2, y1 + 2*radius], 270, 360, fill=outline, width=width)
        draw.arc([x1, y2 - 2*radius, x1 + 2*radius, y2], 90, 180, fill=outline, width=width)
        draw.arc([x2 - 2*radius, y2 - 2*radius, x2, y2], 0, 90, fill=outline, width=width)
        draw.line([x1 + radius, y1, x2 - radius, y1], fill=outline, width=width)
        draw.line([x1 + radius, y2, x2 - radius, y2], fill=outline, width=width)
        draw.line([x1, y1 + radius, x1, y2 - radius], fill=outline, width=width)
        draw.line([x2, y1 + radius, x2, y2 - radius], fill=outline, width=width)


def draw_status_bar(draw, width, font_small):
    """绘制顶部状态栏（时间、信号、电池）"""
    draw.text((20, 10), "10:28", fill=COLORS["black"], font=font_small)
    # 信号图标（简化的竖线）
    for i in range(4):
        h = 8 + i * 3
        draw.rectangle([width - 80 + i*8, 20 - h, width - 80 + i*8 + 5, 20], fill=COLORS["black"])
    # WiFi 图标（简化的扇形）
    draw.arc([width - 50, 5, width - 30, 25], 200, 340, fill=COLORS["black"], width=2)
    draw.arc([width - 45, 10, width - 35, 20], 200, 340, fill=COLORS["black"], width=2)
    # 电池图标
    draw.rectangle([width - 25, 8, width - 8, 22], outline=COLORS["black"], width=1)
    draw.rectangle([width - 25, 8, width - 12, 22], fill=COLORS["green"])
    draw.rectangle([width - 8, 12, width - 5, 18], fill=COLORS["black"])


def draw_title_bar(draw, width, title, font_title):
    """绘制标题栏（返回箭头、标题、客服图标）"""
    # 返回箭头
    draw.line([(20, 50), (35, 40)], fill=COLORS["black"], width=2)
    draw.line([(20, 50), (35, 60)], fill=COLORS["black"], width=2)
    draw.line([(20, 50), (50, 50)], fill=COLORS["black"], width=2)
    # 标题
    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, 38), title, fill=COLORS["black"], font=font_title)
    # 客服图标（简化的耳机）
    draw.ellipse([width - 50, 38, width - 30, 58], outline=COLORS["black"], width=2)
    draw.arc([width - 45, 42, width - 35, 52], 180, 0, fill=COLORS["black"], width=2)


def gen_logistics_detail(order_id: str, product_title: str, price: float,
                         status: str, path: Path, fonts: dict) -> dict:
    """生成物流详情页截图"""
    width, height = 400, 800
    img = Image.new("RGB", (width, height), COLORS["white"])
    draw = ImageDraw.Draw(img)
    
    # 1. 状态栏
    draw_status_bar(draw, width, fonts["small"])
    
    # 2. 标题栏
    draw_title_bar(draw, width, "物流详情", fonts["title"])
    
    # 3. 运单信息区
    y = 80
    draw.rectangle([15, y, width - 15, y + 100], fill=COLORS["gray_light"])
    draw.text((25, y + 10), f"快递公司：{random.choice(EXPRESS_COMPANIES)}", 
              fill=COLORS["black"], font=fonts["normal"])
    draw.text((25, y + 40), f"运单号：{order_id}", fill=COLORS["black"], font=fonts["normal"])
    
    # 状态高亮
    status_color = COLORS["orange"] if status == "已签收" else COLORS["green"]
    draw.text((25, y + 70), status, fill=status_color, font=fonts["bold"])
    
    # 4. 物流时间轴
    y = 200
    route = random.choice(CITY_ROUTES)
    nodes = STATUS_TEMPLATES.get(status, STATUS_TEMPLATES["运输中"])
    
    # 时间轴竖线
    draw.line([(50, y), (50, y + len(nodes) * 80)], fill=COLORS["gray"], width=2)
    
    for i, node in enumerate(nodes):
        node_y = y + i * 80
        # 节点圆点
        dot_color = COLORS["orange"] if i == 0 else COLORS["gray"]
        draw.ellipse([42, node_y - 8, 58, node_y + 8], fill=dot_color)
        
        # 时间（从当前时间往前推）
        hours_ago = i * 6 + random.randint(0, 2)
        day = 16 - (hours_ago // 24)
        hour = (12 - hours_ago % 24) % 24
        time_str = f"08-{day:02d} {hour:02d}:{random.randint(10, 59):02d}"
        draw.text((70, node_y - 15), time_str, fill=COLORS["gray_dark"], font=fonts["small"])
        
        # 状态和描述
        desc = node["desc"].format(
            station=route[3] + "代收点",
            transit=route[1],
            origin=route[0],
            courier=random.choice(["张", "李", "王", "刘"]) + "师傅",
        )
        draw.text((70, node_y + 5), node["status"], fill=COLORS["black"], font=fonts["normal"])
        draw.text((70, node_y + 30), desc, fill=COLORS["gray"], font=fonts["small"])
    
    img.save(path, quality=95)
    return {
        "file": path.name,
        "type": "logistics_detail",
        "order_id": order_id,
        "product_title": product_title,
        "price": price,
        "status": status,
        "express_company": EXPRESS_COMPANIES[0],
        "route": route,
    }


def gen_order_detail(order_id: str, product_title: str, price: float,
                     status: str, path: Path, fonts: dict) -> dict:
    """生成订单详情页截图（已签收状态）"""
    width, height = 400, 900
    img = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    
    # 1. 状态栏
    draw_status_bar(draw, width, fonts["small"])
    
    # 2. 标题栏
    draw_title_bar(draw, width, "订单详情", fonts["title"])
    
    # 3. 顶部状态区
    y = 70
    draw.rectangle([15, y, width - 15, y + 80], fill=COLORS["orange"])
    draw.text((25, y + 10), "已签收,待确认收货", fill=COLORS["white"], font=fonts["bold"])
    draw.text((25, y + 40), "还剩2天22小时自动确认", fill=COLORS["white"], font=fonts["normal"])
    
    # 4. 物流状态卡片
    y = 160
    draw.rectangle([15, y, width - 15, y + 60], fill=COLORS["white"])
    draw.ellipse([25, y + 15, 45, y + 35], fill=COLORS["orange"])
    draw.text((55, y + 10), "已签收 包裹已从代收点取出", fill=COLORS["black"], font=fonts["normal"])
    # 随机收货地址
    addr, name, phone = random.choice(ADDRESSES)
    phone_suffix = f"{random.randint(1000, 9999)}"
    draw.text((55, y + 35), f"收货地址：{addr} {name} {phone}{phone_suffix}", 
              fill=COLORS["gray"], font=fonts["small"])
    
    # 5. 店铺信息
    y = 230
    draw.rectangle([15, y, width - 15, y + 50], fill=COLORS["white"])
    shop = random.choice(SHOP_NAMES)
    draw.text((25, y + 15), shop, fill=COLORS["black"], font=fonts["bold"])
    draw.text((width - 100, y + 15), "进店逛逛 >", fill=COLORS["orange"], font=fonts["small"])
    
    # 6. 商品信息
    y = 290
    draw.rectangle([15, y, width - 15, y + 100], fill=COLORS["white"])
    # 商品图占位（灰色矩形）
    draw.rectangle([25, y + 10, 100, y + 90], fill=COLORS["gray_light"])
    draw.text((35, y + 40), "商品图", fill=COLORS["gray"], font=fonts["small"])
    # 商品信息（限制长度避免重叠）
    max_title_len = 15  # 最多显示15个字符
    display_title = product_title[:max_title_len] + "..." if len(product_title) > max_title_len else product_title
    draw.text((110, y + 10), display_title, fill=COLORS["black"], font=fonts["normal"])
    draw.text((110, y + 35), f"规格：默认", fill=COLORS["gray"], font=fonts["small"])
    draw.text((110, y + 55), "×1", fill=COLORS["gray"], font=fonts["small"])
    # 价格（右对齐，避免与标题重叠）
    price_text = f"¥{price:.2f}"
    price_bbox = draw.textbbox((0, 0), price_text, font=fonts["bold"])
    price_width = price_bbox[2] - price_bbox[0]
    draw.text((width - 25 - price_width, y + 10), price_text, fill=COLORS["red"], font=fonts["bold"])
    
    # 7. 服务标签
    y = 400
    for i, tag in enumerate(["退货宝", "15天价保", "破损包退"]):
        x = 25 + i * 100
        draw_rounded_rect(draw, [x, y, x + 80, y + 25], 5, fill=COLORS["green_light"])
        draw.text((x + 10, y + 5), tag, fill=COLORS["green"], font=fonts["small"])
    
    # 8. 金额明细
    y = 440
    draw.rectangle([15, y, width - 15, y + 120], fill=COLORS["white"])
    draw.text((25, y + 10), "商品总价", fill=COLORS["black"], font=fonts["normal"])
    draw.text((width - 100, y + 10), f"共1件 ¥{price:.2f}", fill=COLORS["black"], font=fonts["normal"])
    
    draw.text((25, y + 40), "平台优惠", fill=COLORS["black"], font=fonts["normal"])
    discount = price * 0.15
    draw.text((width - 100, y + 40), f"-¥{discount:.2f}", fill=COLORS["red"], font=fonts["normal"])
    
    draw.line([(25, y + 70), (width - 25, y + 70)], fill=COLORS["gray_light"])
    draw.text((25, y + 80), "实付款", fill=COLORS["black"], font=fonts["bold"])
    actual_price = price - discount
    draw.text((width - 120, y + 80), f"¥{actual_price:.2f}", fill=COLORS["orange"], font=fonts["title"])
    
    # 9. 订单信息
    y = 570
    draw.rectangle([15, y, width - 15, y + 120], fill=COLORS["white"])
    draw.text((25, y + 10), "订单信息", fill=COLORS["black"], font=fonts["bold"])
    draw.text((25, y + 40), f"订单号：{order_id}", fill=COLORS["black"], font=fonts["normal"])
    draw.text((width - 80, y + 40), "复制", fill=COLORS["orange"], font=fonts["small"])
    draw.text((25, y + 70), "支付方式：微信支付", fill=COLORS["gray"], font=fonts["normal"])
    draw.text((25, y + 100), "创建时间：2026-08-13 22:22:08", fill=COLORS["gray"], font=fonts["small"])
    
    # 10. 底部操作栏
    y = height - 60
    draw.rectangle([0, y, width, height], fill=COLORS["white"])
    draw.line([(0, y), (width, y)], fill=COLORS["gray_light"])
    # 客服按钮
    draw.text((25, y + 20), "客服", fill=COLORS["gray"], font=fonts["normal"])
    # 查看物流按钮
    draw_rounded_rect(draw, [width - 220, y + 10, width - 140, y + 45], 5, 
                      outline=COLORS["orange"])
    draw.text((width - 210, y + 18), "查看物流", fill=COLORS["orange"], font=fonts["normal"])
    # 确认收货按钮
    draw_rounded_rect(draw, [width - 120, y + 10, width - 20, y + 45], 5, 
                      fill=COLORS["orange"])
    draw.text((width - 105, y + 18), "确认收货", fill=COLORS["white"], font=fonts["bold"])
    
    img.save(path, quality=95)
    return {
        "file": path.name,
        "type": "order_detail",
        "order_id": order_id,
        "product_title": product_title,
        "price": price,
        "actual_price": actual_price,
        "discount": discount,
        "shop": shop,
        "status": status,
    }


def gen_order_list(n_orders: int, path: Path, fonts: dict) -> dict:
    """生成订单列表页截图（多个订单卡片）"""
    width, height = 400, 800
    img = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    
    # 1. 状态栏
    draw_status_bar(draw, width, fonts["small"])
    
    # 2. 标题栏（带搜索框）
    y = 40
    draw.rectangle([15, y, width - 15, y + 40], fill=COLORS["white"])
    draw.text((25, y + 10), "搜索订单", fill=COLORS["gray"], font=fonts["normal"])
    draw.ellipse([width - 40, y + 10, width - 25, y + 30], outline=COLORS["orange"], width=2)
    
    # 3. 筛选标签（通用化）
    y = 90
    tags = ["全部订单", "实物商品", "虚拟商品", "生活服务"]
    for i, tag in enumerate(tags):
        x = 25 + i * 90
        color = COLORS["orange"] if i == 1 else COLORS["gray"]
        draw.text((x, y), tag, fill=color, font=fonts["normal"])
        if i == 1:
            draw.line([(x, y + 25), (x + 60, y + 25)], fill=COLORS["orange"], width=2)
    
    # 4. 状态筛选
    y = 120
    status_tags = ["全部", "待发货", "待收货", "退款/售后", "评价"]
    for i, tag in enumerate(status_tags):
        x = 25 + i * 75
        if i == 2:  # 当前选中
            draw_rounded_rect(draw, [x - 5, y, x + 60, y + 30], 5, fill=COLORS["orange_light"])
        draw.text((x, y + 5), tag, fill=COLORS["black"] if i == 2 else COLORS["gray"], 
                  font=fonts["small"])
    
    # 5. 订单卡片
    y = 160
    orders_data = []
    for i in range(n_orders):
        card_y = y + i * 130
        if card_y + 120 > height - 60:
            break
        
        # 卡片背景
        draw.rectangle([15, card_y, width - 15, card_y + 120], fill=COLORS["white"])
        
        # 店铺信息
        shop = random.choice(SHOP_NAMES)
        draw.text((25, card_y + 10), shop, fill=COLORS["black"], font=fonts["normal"])
        draw.text((width - 100, card_y + 10), "已签收", fill=COLORS["orange"], font=fonts["small"])
        
        # 商品信息
        cat = random.choice(PRODUCT_CATEGORIES)
        title = random.choice(cat["titles"])
        draw.rectangle([25, card_y + 35, 80, card_y + 90], fill=COLORS["gray_light"])
        draw.text((35, card_y + 55), "商品图", fill=COLORS["gray"], font=fonts["small"])
        # 限制标题长度避免重叠
        max_title_len = 12
        display_title = title[:max_title_len] + "..." if len(title) > max_title_len else title
        draw.text((90, card_y + 35), display_title, fill=COLORS["black"], font=fonts["normal"])
        draw.text((90, card_y + 55), "×1", fill=COLORS["gray"], font=fonts["small"])
        
        # 价格（右对齐）
        price = round(random.uniform(15, 200), 2)
        price_text = f"¥{price:.2f}"
        price_bbox = draw.textbbox((0, 0), price_text, font=fonts["normal"])
        price_width = price_bbox[2] - price_bbox[0]
        draw.text((width - 25 - price_width, card_y + 35), price_text, fill=COLORS["red"], font=fonts["normal"])
        
        # 物流状态
        draw.rectangle([25, card_y + 80, width - 25, card_y + 100], fill=COLORS["gray_light"])
        draw.text((35, card_y + 83), "已签收 包裹已从代收点取出", fill=COLORS["gray"], font=fonts["small"])
        
        # 操作按钮
        draw.text((25, card_y + 105), "更多", fill=COLORS["gray"], font=fonts["small"])
        btn_labels = ["查看物流", "申请售后", "确认收货"]
        for j, label in enumerate(btn_labels):
            bx = width - 220 + j * 75
            if j == 2:  # 确认收货
                draw_rounded_rect(draw, [bx, card_y + 100, bx + 65, card_y + 118], 3, 
                                  fill=COLORS["orange"])
                draw.text((bx + 5, card_y + 103), label, fill=COLORS["white"], font=fonts["small"])
            else:
                draw_rounded_rect(draw, [bx, card_y + 100, bx + 65, card_y + 118], 3, 
                                  outline=COLORS["orange"])
                draw.text((bx + 5, card_y + 103), label, fill=COLORS["orange"], font=fonts["small"])
        
        orders_data.append({
            "shop": shop,
            "product_title": title,
            "price": price,
            "status": "已签收",
        })
    
    # 6. 底部导航栏（通用化）
    y = height - 50
    draw.rectangle([0, y, width, height], fill=COLORS["white"])
    draw.line([(0, y), (width, y)], fill=COLORS["gray_light"])
    nav_items = ["首页", "消息", "我的", "购物车"]
    for i, item in enumerate(nav_items):
        nx = 50 + i * 85
        draw.text((nx, y + 15), item, fill=COLORS["gray"] if i != 0 else COLORS["orange"], 
                  font=fonts["small"])
    
    img.save(path, quality=95)
    return {
        "file": path.name,
        "type": "order_list",
        "n_orders": len(orders_data),
        "orders": orders_data,
    }


def main():
    random.seed(SEED)
    
    # 加载字体
    fonts = {
        "small": load_font(14),
        "normal": load_font(18),
        "bold": load_font(20),
        "title": load_font(24),
    }
    
    # 加载产品数据（优先使用中文数据）
    cn_path = DATA/"products_cn.jsonl"
    if cn_path.exists():
        products = [json.loads(l) for l in open(cn_path, encoding="utf-8")]
        print(f"[gen_screenshots] 使用中文产品数据: {len(products)} 个", flush=True)
    else:
        products = [json.loads(l) for l in open(DATA/"products.jsonl", encoding="utf-8")]
        print(f"[gen_screenshots] 使用英文产品数据: {len(products)} 个", flush=True)
    
    # 输出目录
    out_dir = DATA/"images/taobao_screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metas = []
    n_total = 1000  # 增加到 1000 张截图
    
    t0 = time.time()
    
    for i in range(n_total):
        # 随机选择商品类别和标题（确保纯中文）
        cat = random.choice(PRODUCT_CATEGORIES)
        product_title = random.choice(cat["titles"])
        price = round(random.uniform(15, 500), 2)
        order_id = f"{random.randint(1000000000, 9999999999):018d}"
        status = random.choice(["已签收", "运输中", "派送中", "待发货"])
        
        # 4 种截图类型均匀分布
        if i % 4 == 0:
            # 物流详情页
            fname = f"logistics_{i:04d}.png"
            meta = gen_logistics_detail(
                order_id, product_title, price, status,
                out_dir / fname, fonts
            )
        elif i % 4 == 1:
            # 订单详情页
            fname = f"order_{i:04d}.png"
            meta = gen_order_detail(
                order_id, product_title, price, status,
                out_dir / fname, fonts
            )
        elif i % 4 == 2:
            # 订单列表页
            fname = f"list_{i:04d}.png"
            meta = gen_order_list(random.randint(2, 4), out_dir / fname, fonts)
        else:
            # 再随机选一种（增加多样性）
            choice = random.choice(["logistics", "order", "list"])
            if choice == "logistics":
                fname = f"logistics_{i:04d}.png"
                meta = gen_logistics_detail(
                    order_id, product_title, price, status,
                    out_dir / fname, fonts
                )
            elif choice == "order":
                fname = f"order_{i:04d}.png"
                meta = gen_order_detail(
                    order_id, product_title, price, status,
                    out_dir / fname, fonts
                )
            else:
                fname = f"list_{i:04d}.png"
                meta = gen_order_list(random.randint(2, 5), out_dir / fname, fonts)
        
        metas.append(meta)
        
        if (i + 1) % 100 == 0:
            print(f"[taobao_screenshots] {i + 1}/{n_total} ({time.time()-t0:.0f}s)", flush=True)
    
    # 保存元数据
    meta_path = out_dir / "meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in metas)
    
    print(f"[taobao_screenshots] DONE {n_total} images in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

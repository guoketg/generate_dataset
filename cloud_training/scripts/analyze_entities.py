#!/usr/bin/env python3
"""分析训练数据中的实体合法性问题"""

import json
import re
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "training"

# 合法实体
PRODUCT_IDS = set()
ORDER_IDS = set()
REFUND_IDS = set()
AUTH_IDS = set()
LOGISTICS_IDS = set()
COMPLAINT_IDS = set()
FAKE_IDS = set()

# 生成合法 ID 集合
for i in range(500):
    PRODUCT_IDS.update({f"PROD{i:06d}", f"P{i:06d}"})
    ORDER_IDS.update({f"ORD{i:06d}", f"O{i:06d}"})
    REFUND_IDS.update({f"RF{i:06d}", f"R{i:06d}"})
    AUTH_IDS.update({f"AUTH{i:06d}", f"A{i:06d}"})
    LOGISTICS_IDS.update({f"LOG{i:06d}", f"L{i:06d}"})
    COMPLAINT_IDS.update({f"COMP{i:06d}", f"C{i:06d}"})
    FAKE_IDS.update({f"FAKE{i:06d}", f"F{i:06d}"})

# 添加更多合法 ID
for i in range(500, 10000):
    ORDER_IDS.update({f"ORD{i:06d}", f"O{i:06d}"})
    REFUND_IDS.update({f"RF{i:06d}", f"R{i:06d}"})
    AUTH_IDS.update({f"AUTH{i:06d}", f"A{i:06d}"})
    LOGISTICS_IDS.update({f"LOG{i:06d}", f"L{i:06d}"})
    COMPLAINT_IDS.update({f"COMP{i:06d}", f"C{i:06d}"})

ALL_ENTITIES = PRODUCT_IDS | ORDER_IDS | REFUND_IDS | AUTH_IDS | LOGISTICS_IDS | COMPLAINT_IDS | FAKE_IDS

# 额外的合法实体
EXTRA_ENTITIES = {
    "ORD_NEW_001", "ORD_NEW_002", "ORD_NEW_003", "ORD_NEW_004", "ORD_NEW_005",
    "ORD_VIP_001", "ORD_VIP_002", "ORD_VIP_003", "ORD_VIP_004", "ORD_VIP_005",
    "ORD001VIP", "ORD002VIP", "ORD003VIP", "ORD004VIP", "ORD005VIP",
    "ORD001NEW", "ORD002NEW", "ORD003NEW", "ORD004NEW", "ORD005NEW",
    "ORD_REPEAT_001", "ORD_REPEAT_002", "ORD_REPEAT_003", "ORD_REPEAT_004", "ORD_REPEAT_005",
    "ORD_SAME_001", "ORD_SAME_002", "ORD_SAME_003", "ORD_SAME_004", "ORD_SAME_005",
    "ORD_PREVIOUS_001", "ORD_PREVIOUS_002", "ORD_PREVIOUS_003", "ORD_PREVIOUS_004", "ORD_PREVIOUS_005",
    "ORD_MULTIPLE_001", "ORD_MULTIPLE_002", "ORD_MULTIPLE_003", "ORD_MULTIPLE_004", "ORD_MULTIPLE_005",
    "ORD_ANOTHER_001", "ORD_ANOTHER_002", "ORD_ANOTHER_003", "ORD_ANOTHER_004", "ORD_ANOTHER_005",
    "ORD_SECOND_001", "ORD_SECOND_002", "ORD_SECOND_003", "ORD_SECOND_004", "ORD_SECOND_005",
    "ORD001", "ORD002", "ORD003", "ORD004", "ORD005",
    "LOG001", "LOG002", "LOG003", "LOG004", "LOG005",
    "RF001", "RF002", "RF003", "RF004", "RF005",
    "AUTH001", "AUTH002", "AUTH003", "AUTH004", "AUTH005",
    "PROD001", "PROD002", "PROD003", "PROD004", "PROD005",
    "COMP001", "COMP002", "COMP003", "COMP004", "COMP005",
}
ALL_ENTITIES |= EXTRA_ENTITIES

# 正则表达式匹配 ID
ID_PATTERNS = [
    r"(ORD\d{6,8})",
    r"(RF\d{6,8})",
    r"(AUTH\d{6,8})",
    r"(LOG\d{6,8})",
    r"(PROD\d{6,8})",
    r"(COMP\d{6,8})",
    r"(AF\d{6,8}[A-Z]?)",
    r"(TRK\d{6,8})",
    r"(FAKE\d{6,8})",
    r"(ORD_NEW_\d{3})",
    r"(ORD_VIP_\d{3})",
    r"(ORD\d{3}VIP)",
    r"(ORD\d{3}NEW)",
    r"(ORD_REPEAT_\d{3})",
    r"(ORD_SAME_\d{3})",
    r"(ORD_PREVIOUS_\d{3})",
    r"(ORD_MULTIPLE_\d{3})",
    r"(ORD_ANOTHER_\d{3})",
    r"(ORD_SECOND_\d{3})",
    r"(ORD\d{3})",
    r"(LOG\d{3})",
    r"(RF\d{3})",
    (r"(AUTH\d{3})"),
    r"(PROD\d{3})",
    r"(COMP\d{3})",
    r"(P\d{6})",
    r"(O\d{6})",
    r"(R\d{6})",
    r"(A\d{6})",
    r"(L\d{6})",
    r"(C\d{6})",
    r"(F\d{6})",
]

def extract_entity_ids(text: str) -> set[str]:
    """从文本中提取实体 ID"""
    found = set()
    for pat in ID_PATTERNS:
        found.update(re.findall(pat, text))
    return found

def analyze_file(file_path: Path) -> dict:
    """分析单个文件的实体合法性"""
    illegal_entities = Counter()
    total_entities = 0
    illegal_count = 0
    
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # 提取所有文本内容
            text_parts = []
            for msg in item.get("messages", []):
                content = msg.get("content", "")
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
            
            text = "\n".join(text_parts)
            found_ids = extract_entity_ids(text)
            
            for entity_id in found_ids:
                total_entities += 1
                if entity_id not in ALL_ENTITIES:
                    illegal_count += 1
                    illegal_entities[entity_id] += 1
    
    return {
        "total_entities": total_entities,
        "illegal_count": illegal_count,
        "illegal_entities": illegal_entities.most_common(20),
    }

def main():
    print("分析训练数据中的实体合法性问题...")
    print("=" * 80)
    
    files_to_analyze = [
        ("SFT 训练集", DATA_DIR / "sft_train.jsonl"),
        ("Anti 训练集", DATA_DIR / "anti_train.jsonl"),
        ("Multi 训练集", DATA_DIR / "multi_train.jsonl"),
    ]
    
    total_entities = 0
    total_illegal = 0
    all_illegal_entities = Counter()
    
    for name, file_path in files_to_analyze:
        if not file_path.exists():
            print(f"跳过 {name}: 文件不存在")
            continue
        
        print(f"\n分析 {name} ({file_path.name}):")
        result = analyze_file(file_path)
        
        print(f"  总实体数: {result['total_entities']}")
        print(f"  非法实体数: {result['illegal_count']}")
        if result['total_entities'] > 0:
            rate = (result['total_entities'] - result['illegal_count']) / result['total_entities'] * 100
            print(f"  合法率: {rate:.1f}%")
        
        print("  最常见的非法实体:")
        for entity, count in result['illegal_entities']:
            print(f"    {entity}: {count} 次")
        
        total_entities += result['total_entities']
        total_illegal += result['illegal_count']
        all_illegal_entities.update(result['illegal_entities'])
    
    print("\n" + "=" * 80)
    print("汇总:")
    print(f"总实体数: {total_entities}")
    print(f"非法实体数: {total_illegal}")
    if total_entities > 0:
        rate = (total_entities - total_illegal) / total_entities * 100
        print(f"总合法率: {rate:.1f}%")
    
    print("\n最常见的非法实体 (全局):")
    for entity, count in all_illegal_entities.most_common(30):
        print(f"  {entity}: {count} 次")

if __name__ == "__main__":
    main()
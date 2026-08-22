#!/usr/bin/env python3
"""
冒烟测试脚本：验证 LLM 生成功能在各种情况下是否可用
测试覆盖：11种工具 + 合成图片 + 电商图片
保存完整轨迹：用户查询 + 工具调用 + 工具返回 + 助手回答
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 添加脚本目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from gen_training_data import TrainingDataGenerator


def test_llm_generation():
    """测试 LLM 生成功能"""
    print("=" * 80)
    print("冒烟测试：LLM 生成功能验证（保存完整轨迹）")
    print("=" * 80)
    
    # 创建生成器实例
    print("\n[1/7] 初始化 TrainingDataGenerator...")
    generator = TrainingDataGenerator()
    print("✅ 初始化成功")
    
    # 测试结果统计
    results = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "tests": []
    }
    
    # 测试用例（复杂问题 + 丰富工具返回）
    test_cases = [
        # 1. 物流查询（单工具）- 复杂问题
        {
            "name": "物流查询（单工具）- 复杂问题",
            "user_query": "我昨天下的订单ORD00001042，现在到哪了？预计什么时候能到？我看同款商品在京东上更便宜，能帮我比比价吗？",
            "route": "logistics_single",
            "params": {"order_id": "ORD00001042"},
            "gold_chain": [
                {"tool": "query_logistics", "args": {"order_id": "ORD00001042"}}
            ],
            "observations": [{
                "order_id": "ORD00001042",
                "status_cn": "运输中",
                "trajectory": [
                    {"location": "上海浦东仓库", "time": "2025-01-15 14:30", "action": "包裹已从上海浦东仓库发出"},
                    {"location": "上海转运中心", "time": "2025-01-15 22:15", "action": "包裹到达上海转运中心，正在分拣"},
                    {"location": "南京转运中心", "time": "2025-01-16 06:45", "action": "包裹到达南京转运中心"},
                    {"location": "北京分拣中心", "time": "2025-01-16 15:20", "action": "包裹到达北京分拣中心，正在派送中"}
                ],
                "estimated_delivery": "2025-01-17 18:00",
                "carrier": "顺丰速运",
                "tracking_number": "SF1234567890"
            }]
        },
        # 2. 物流查询（多工具）- 图片+物流+比价
        {
            "name": "物流查询（多工具）- 图片+物流+比价",
            "user_query": "我买了一双鞋，订单号是ORD00001043，能帮我看看物流到哪了吗？另外我拍了个照片，能帮我看看是不是正品？同款在淘宝上多少钱？",
            "route": "logistics_multi",
            "params": {"order_id": "ORD00001043"},
            "gold_chain": [
                {"tool": "query_logistics", "args": {"order_id": "ORD00001043"}},
                {"tool": "vl_describe", "args": {"image": "shoe_photo.jpg"}},
                {"tool": "price_compare", "args": {"product_id": 12346}}
            ],
            "observations": [{
                "order_id": "ORD00001043",
                "status_cn": "已签收",
                "trajectory": [
                    {"location": "广州白云仓库", "time": "2025-01-14 09:00", "action": "包裹已从广州白云仓库发出"},
                    {"location": "广州转运中心", "time": "2025-01-14 18:30", "action": "包裹到达广州转运中心"},
                    {"location": "武汉转运中心", "time": "2025-01-15 08:15", "action": "包裹到达武汉转运中心"},
                    {"location": "北京分拣中心", "time": "2025-01-16 03:45", "action": "包裹到达北京分拣中心"},
                    {"location": "朝阳区配送站", "time": "2025-01-16 10:20", "action": "包裹到达朝阳区配送站"},
                    {"location": "用户地址", "time": "2025-01-16 14:30", "action": "包裹已签收，签收人：本人"}
                ],
                "signed_time": "2025-01-16 14:30",
                "signed_by": "本人",
                "carrier": "中通快递",
                "tracking_number": "ZT9876543210"
            }, {
                "description": "这是一双白色Nike Air Max运动鞋，鞋面有轻微折痕，鞋底磨损正常，鞋盒完整，有防伪标签。",
                "brand": "Nike",
                "model": "Air Max",
                "color": "白色",
                "size": "42",
                "condition": "全新"
            }, {
                "platform": "淘宝",
                "min_price": "899",
                "max_price": "1299",
                "average_price": "1050",
                "products": [
                    {"title": "Nike Air Max 运动鞋 白色 42码", "price": "899", "sales": "2341", "rating": "4.8"},
                    {"title": "Nike Air Max 2025 新款 白色", "price": "1099", "sales": "1567", "rating": "4.9"},
                    {"title": "Nike Air Max 经典款 白色 42", "price": "999", "sales": "3456", "rating": "4.7"},
                    {"title": "Nike Air Max 运动鞋 男款 白色", "price": "1299", "sales": "890", "rating": "4.6"}
                ]
            }]
        },
        # 3. 商品搜索 - 复杂需求
        {
            "name": "商品搜索 - 复杂需求",
            "user_query": "我想买个手机，预算5000左右，主要拍照和玩游戏，最好续航好一点，能帮我推荐几款吗？",
            "route": "search_single",
            "params": {"category": "手机", "budget": "5000", "use_case": "拍照,游戏,续航"},
            "gold_chain": [
                {"tool": "text_search", "args": {"query": "手机 拍照 游戏 续航 5000元"}}
            ],
            "observations": [{
                "results": [
                    {
                        "title": "iPhone 15 Pro 256GB 深空黑",
                        "price": "7999",
                        "sales": "12345",
                        "rating": "4.9",
                        "features": ["A17 Pro芯片", "4800万像素主摄", "钛金属边框"],
                        "battery": "4500mAh",
                        "camera_score": "98"
                    },
                    {
                        "title": "小米14 Ultra 16GB+512GB 黑色",
                        "price": "5999",
                        "sales": "8765",
                        "rating": "4.8",
                        "features": ["骁龙8 Gen3", "徕卡四摄", "5300mAh电池"],
                        "battery": "5300mAh",
                        "camera_score": "96"
                    },
                    {
                        "title": "vivo X100 Pro 12GB+256GB 蓝色",
                        "price": "4999",
                        "sales": "6543",
                        "rating": "4.7",
                        "features": ["天玑9300", "蔡司影像", "5400mAh电池"],
                        "battery": "5400mAh",
                        "camera_score": "94"
                    },
                    {
                        "title": "OPPO Find X7 Ultra 16GB+512GB",
                        "price": "5499",
                        "sales": "4321",
                        "rating": "4.8",
                        "features": ["骁龙8 Gen3", "哈苏影像", "5600mAh电池"],
                        "battery": "5600mAh",
                        "camera_score": "95"
                    }
                ],
                "total_count": 4,
                "price_range": "4999-7999",
                "recommendation": "根据您的需求，推荐小米14 Ultra或vivo X100 Pro，性价比最高"
            }]
        },
        # 4. 防伪码验证（正品）- 复杂场景
        {
            "name": "防伪码验证（正品）- 复杂场景",
            "user_query": "我在拼多多上买了个化妆品，防伪码是AF00001042K，但包装有点破损，能帮我查查是不是正品？如果是假的怎么办？",
            "route": "authenticity_single",
            "params": {"code": "AF00001042K", "platform": "拼多多", "category": "化妆品"},
            "gold_chain": [
                {"tool": "authenticity_check", "args": {"code": "AF00001042K"}}
            ],
            "observations": [{
                "code": "AF00001042K",
                "is_genuine": True,
                "product_name": "兰蔻小黑瓶精华液 50ml",
                "brand": "兰蔻",
                "production_date": "2024-12-15",
                "expiry_date": "2027-12-15",
                "batch_number": "LN20241215A",
                "manufacturer": "欧莱雅（中国）有限公司",
                "verification_count": 3,
                "last_verification": "2025-01-16 10:30",
                "verification_source": "兰蔻官方防伪系统"
            }]
        },
        # 5. 防伪码验证（假货）- 复杂场景
        {
            "name": "防伪码验证（假货）- 复杂场景",
            "user_query": "朋友送了我个奢侈品包包，防伪码是AF00001043K，但我觉得价格太便宜了，能帮我验验真伪吗？如果是假的我该怎么维权？",
            "route": "authenticity_single",
            "params": {"code": "AF00001043K", "category": "奢侈品"},
            "gold_chain": [
                {"tool": "authenticity_check", "args": {"code": "AF00001043K"}}
            ],
            "observations": [{
                "code": "AF00001043K",
                "is_genuine": False,
                "product_name": "LV Neverfull 中号手袋",
                "brand": "路易威登",
                "suspicious_points": [
                    "防伪码格式不符合官方标准",
                    "生产日期与批次号不匹配",
                    "材质手感与正品有差异"
                ],
                "suggestion": "建议联系路易威登官方客服进行进一步鉴定",
                "official_contact": "400-658-8800"
            }]
        },
        # 6. 退款状态查询 - 复杂场景
        {
            "name": "退款状态查询 - 复杂场景",
            "user_query": "我的退款RF00000001申请了一个星期了，怎么还没到账？是不是出了什么问题？能帮我催催吗？",
            "route": "refund_status_single",
            "params": {"refund_id": "RF00000001"},
            "gold_chain": [
                {"tool": "query_refund", "args": {"refund_id": "RF00000001"}}
            ],
            "observations": [{
                "refund_id": "RF00000001",
                "state_cn": "退款处理中",
                "state": "processing",
                "order_id": "ORD00001045",
                "refund_amount": "299.00",
                "refund_reason": "商品质量问题",
                "apply_time": "2025-01-10 14:30",
                "audit_time": "2025-01-11 09:15",
                "audit_result": "审核通过",
                "estimated_arrival": "2025-01-18 24:00",
                "refund_method": "原支付方式",
                "processing_notes": "银行处理中，预计1-3个工作日到账",
                "customer_service_notes": "已联系银行加急处理"
            }]
        },
        # 7. 退款创建 - 复杂场景
        {
            "name": "退款创建 - 复杂场景",
            "user_query": "我买的鞋子ORD00001044尺码不对，想退货退款，但商家说已经发货了，怎么办？能帮我申请退款吗？",
            "route": "refund_single",
            "params": {"order_id": "ORD00001044", "reason": "尺码不对", "status": "已发货"},
            "gold_chain": [
                {"tool": "create_refund_ticket", "args": {"order_id": "ORD00001044", "reason": "尺码不对", "refund_type": "退货退款"}}
            ],
            "observations": [{
                "refund_id": "RF00000003",
                "state": "pending_return",
                "order_id": "ORD00001044",
                "refund_amount": "599.00",
                "refund_reason": "尺码不对",
                "refund_type": "退货退款",
                "return_address": "北京市朝阳区xx路xx号 退货仓",
                "return_deadline": "2025-01-25 24:00",
                "return_tracking_required": True,
                "notes": "请在收到商品后7天内寄回，运费由买家承担",
                "merchant_response": "同意退货退款，请按地址寄回"
            }]
        },
        # 8. 价格比较 - 复杂场景
        {
            "name": "价格比较 - 复杂场景",
            "user_query": "我想买个戴森吹风机，能帮我比比京东、淘宝、拼多多的价格吗？哪个平台最便宜？有没有优惠券？",
            "route": "price_compare_single",
            "params": {"product_id": 12345, "product_name": "戴森吹风机"},
            "gold_chain": [
                {"tool": "price_compare", "args": {"product_id": 12345}}
            ],
            "observations": [{
                "product_name": "戴森 Supersonic 吹风机 HD08 紫红色",
                "platforms": [
                    {
                        "platform": "京东",
                        "price": "2990",
                        "original_price": "3290",
                        "discount": "300元优惠券",
                        "stock": "有货",
                        "delivery": "次日达",
                        "rating": "4.9",
                        "sales": "12345"
                    },
                    {
                        "platform": "淘宝",
                        "price": "2890",
                        "original_price": "3190",
                        "discount": "300元店铺券",
                        "stock": "有货",
                        "delivery": "3天内",
                        "rating": "4.8",
                        "sales": "8765"
                    },
                    {
                        "platform": "拼多多",
                        "price": "2790",
                        "original_price": "3090",
                        "discount": "百亿补贴",
                        "stock": "仅剩5件",
                        "delivery": "48小时内",
                        "rating": "4.7",
                        "sales": "5432"
                    }
                ],
                "lowest_price": "2790",
                "lowest_platform": "拼多多",
                "recommendation": "拼多多价格最低，但库存有限；京东次日达最快，建议根据需求选择"
            }]
        },
        # 9. 图片描述（合成图片）- 复杂场景
        {
            "name": "图片描述（合成图片）- 复杂场景",
            "user_query": "我买了个包，拍了个照片，能帮我看看有没有质量问题？如果有的话能申请退款吗？",
            "route": "vl_describe_single",
            "params": {},
            "gold_chain": [
                {"tool": "vl_describe", "args": {"image": "bag_photo.jpg"}}
            ],
            "observations": [{
                "description": "图片显示一个红色真皮手提包，表面有3处明显划痕（分别位于正面、侧面和底部），拉链处有轻微磨损，金属扣件有氧化痕迹。整体成色约8成新。",
                "quality_issues": [
                    {"location": "正面", "issue": "划痕", "severity": "中度"},
                    {"location": "侧面", "issue": "划痕", "severity": "轻微"},
                    {"location": "底部", "issue": "划痕", "severity": "严重"},
                    {"location": "拉链", "issue": "磨损", "severity": "轻微"},
                    {"location": "金属扣件", "issue": "氧化", "severity": "轻微"}
                ],
                "brand": "Coach",
                "model": "Tabby 26",
                "material": "真皮",
                "color": "红色",
                "estimated_condition": "8成新"
            }]
        },
        # 10. 图片描述（电商图片）- 复杂场景
        {
            "name": "图片描述（电商图片）- 复杂场景",
            "user_query": "我在网上看到这个商品图片，能帮我分析一下这是什么品牌？大概多少钱？质量怎么样？",
            "route": "vl_describe_single",
            "params": {},
            "gold_chain": [
                {"tool": "vl_describe", "args": {"image": "product_image.jpg"}}
            ],
            "observations": [{
                "description": "商品图片显示一双白色运动鞋，鞋面采用网面材质，鞋底有Air Max气垫技术，鞋舌处有Nike标志，鞋盒完整，有防伪标签。",
                "brand": "Nike",
                "model": "Air Max 270",
                "color": "白色",
                "material": "网面+合成革",
                "technology": "Air Max气垫",
                "estimated_price": "899-1299",
                "quality_assessment": "做工精细，材质优良，气垫饱满，整体质量较好",
                "authenticity_indicators": [
                    "鞋盒标签清晰",
                    "防伪码可查询",
                    "鞋标字体规范",
                    "气垫透明度正常"
                ]
            }]
        },
        # 11. 转人工 - 复杂场景
        {
            "name": "转人工 - 复杂场景",
            "user_query": "我买的手机屏幕有裂痕，商家说是人为损坏不给退，但我收到就这样了，能帮我投诉吗？我要找人工客服！",
            "route": "transfer_to_human",
            "params": {"issue": "商品质量问题", "merchant_refusal": True},
            "gold_chain": [
                {"tool": "transfer_to_human", "args": {"reason": "商品质量问题纠纷，商家拒绝退款，用户要求人工介入"}}
            ],
            "observations": [{
                "reason": "商品质量问题纠纷，商家拒绝退款，用户要求人工介入",
                "issue_type": "质量纠纷",
                "order_id": "ORD00001046",
                "product": "iPhone 15 Pro",
                "problem": "屏幕裂痕",
                "merchant_response": "人为损坏，拒绝退款",
                "user_claim": "收到时已损坏",
                "priority": "高",
                "estimated_wait_time": "5分钟",
                "available_channels": ["在线客服", "电话客服", "投诉专线"]
            }]
        }
    ]
    
    # 执行测试
    print("\n[2/7] 开始执行测试用例...")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  测试 {i}/{len(test_cases)}: {test_case['name']}")
        print(f"    用户查询: {test_case['user_query']}")
        results["total"] += 1
        
        try:
            # 构建完整轨迹（模拟实际数据生成）
            # 1. 用户查询
            user_message = {"role": "user", "content": test_case["user_query"]}
            
            # 2. 助手第一次回复（工具调用）
            tool_calls = []
            for j, step in enumerate(test_case["gold_chain"]):
                tool_calls.append({
                    "id": f"call_{j}",
                    "type": "function",
                    "function": {
                        "name": step["tool"],
                        "arguments": json.dumps(step["args"], ensure_ascii=False)
                    }
                })
            
            assistant_message_1 = {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls
            }
            
            # 3. 工具返回结果
            tool_messages = []
            for j, obs in enumerate(test_case["observations"]):
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{j}",
                    "content": json.dumps(obs, ensure_ascii=False)
                })
            
            # 4. 助手最终回答
            final_answer = generator._generate_final_answer(
                test_case["route"],
                test_case["params"],
                test_case["observations"],
                use_llm=True
            )
            
            assistant_message_2 = {
                "role": "assistant",
                "content": final_answer
            }
            
            # 构建完整轨迹
            full_trajectory = [user_message, assistant_message_1] + tool_messages + [assistant_message_2]
            
            # 检查结果
            if final_answer and len(final_answer) > 10:
                results["success"] += 1
                status = "✅ 成功"
                print(f"    {status}")
                print(f"    最终回答: {final_answer[:100]}...")
                
                # 记录测试结果（包含完整轨迹）
                results["tests"].append({
                    "name": test_case["name"],
                    "user_query": test_case["user_query"],
                    "status": "success",
                    "final_answer": final_answer,
                    "answer_length": len(final_answer),
                    "full_trajectory": full_trajectory,
                    "tool_calls": test_case["gold_chain"],
                    "observations": test_case["observations"]
                })
            else:
                results["failed"] += 1
                status = "❌ 失败（回答太短或为空）"
                print(f"    {status}")
                print(f"    最终回答: {final_answer}")
                
                results["tests"].append({
                    "name": test_case["name"],
                    "user_query": test_case["user_query"],
                    "status": "failed",
                    "final_answer": final_answer,
                    "error": "回答太短或为空",
                    "full_trajectory": full_trajectory,
                    "tool_calls": test_case["gold_chain"],
                    "observations": test_case["observations"]
                })
                
        except Exception as e:
            results["failed"] += 1
            status = f"❌ 失败（异常: {e}）"
            print(f"    {status}")
            
            results["tests"].append({
                "name": test_case["name"],
                "user_query": test_case["user_query"],
                "status": "failed",
                "error": str(e)
            })
    
    # 测试多样性
    print("\n[3/7] 测试多样性（同一查询生成多个回答）...")
    diversity_results = []
    test_route = "logistics_single"
    test_params = {"order_id": "ORD00001045"}
    test_observations = [{
        "order_id": "ORD00001045",
        "status_cn": "派送中",
        "trajectory": [
            {"location": "配送站", "time": "2025-01-18 09:00"}
        ]
    }]
    
    for i in range(5):
        try:
            answer = generator._generate_final_answer(
                test_route,
                test_params,
                test_observations,
                use_llm=True
            )
            diversity_results.append(answer)
            print(f"  回答 {i+1}: {answer[:80]}...")
        except Exception as e:
            print(f"  回答 {i+1}: 失败 - {e}")
    
    # 检查多样性
    unique_answers = len(set(diversity_results))
    print(f"\n  多样性测试: 生成 {len(diversity_results)} 个回答，其中 {unique_answers} 个不同")
    if unique_answers >= 3:
        print("  ✅ 多样性良好")
    else:
        print("  ⚠️ 多样性不足")
    
    # 测试图片场景
    print("\n[4/7] 测试图片场景...")
    image_test_cases = [
        {
            "name": "合成图片（图片验证码）",
            "route": "ocr_single",
            "params": {},
            "observations": [{"text": "ABCD1234"}]
        },
        {
            "name": "电商图片（商品主图）",
            "route": "vl_describe_single",
            "params": {},
            "observations": [{"description": "商品图片显示一台笔记本电脑，银色外壳"}]
        },
        {
            "name": "订单截图",
            "route": "ocr_single",
            "params": {},
            "observations": [{"text": "订单号: ORD00001046"}]
        }
    ]
    
    for test_case in image_test_cases:
        print(f"\n  测试: {test_case['name']}")
        try:
            answer = generator._generate_final_answer(
                test_case["route"],
                test_case["params"],
                test_case["observations"],
                use_llm=True
            )
            if answer and len(answer) > 10:
                print(f"    ✅ 成功: {answer[:80]}...")
            else:
                print(f"    ❌ 失败: {answer}")
        except Exception as e:
            print(f"    ❌ 失败: {e}")
    
    # 测试工具组合
    print("\n[5/7] 测试工具组合...")
    tool_combinations = [
        {
            "name": "物流+图片（订单截图查询物流）",
            "route": "logistics_image",
            "params": {"order_id": "ORD00001047"},
            "observations": [{
                "order_id": "ORD00001047",
                "status_cn": "已发货",
                "trajectory": [{"location": "仓库", "time": "2025-01-18 10:00"}]
            }]
        },
        {
            "name": "搜索+比价（搜索商品后比价）",
            "route": "search_price",
            "params": {"category": "手机", "product_id": 12346},
            "observations": [{
                "results": [{"title": "小米14", "price": "3999"}]
            }, {
                "platform": "淘宝",
                "min_price": "3899"
            }]
        }
    ]
    
    for test_case in tool_combinations:
        print(f"\n  测试: {test_case['name']}")
        try:
            answer = generator._generate_final_answer(
                test_case["route"],
                test_case["params"],
                test_case["observations"],
                use_llm=True
            )
            if answer and len(answer) > 10:
                print(f"    ✅ 成功: {answer[:80]}...")
            else:
                print(f"    ❌ 失败: {answer}")
        except Exception as e:
            print(f"    ❌ 失败: {e}")
    
    # 测试边界情况
    print("\n[6/7] 测试边界情况...")
    edge_cases = [
        {
            "name": "空观察结果",
            "route": "logistics_single",
            "params": {"order_id": "ORD00001048"},
            "observations": [{}]
        },
        {
            "name": "缺失参数",
            "route": "logistics_single",
            "params": {},
            "observations": [{"order_id": "ORD00001049", "status_cn": "未知"}]
        },
        {
            "name": "长文本观察结果",
            "route": "vl_describe_single",
            "params": {},
            "observations": [{"description": "这是一段很长的描述" * 50}]
        }
    ]
    
    for test_case in edge_cases:
        print(f"\n  测试: {test_case['name']}")
        try:
            answer = generator._generate_final_answer(
                test_case["route"],
                test_case["params"],
                test_case["observations"],
                use_llm=True
            )
            if answer:
                print(f"    ✅ 成功: {answer[:80]}...")
            else:
                print(f"    ❌ 失败: 返回空")
        except Exception as e:
            print(f"    ❌ 失败: {e}")
    
    # 生成测试报告（包含完整轨迹）
    print("\n[7/7] 生成测试报告...")
    report = {
        "test_time": datetime.now().isoformat(),
        "summary": {
            "total_tests": results["total"],
            "success": results["success"],
            "failed": results["failed"],
            "success_rate": f"{results['success'] / results['total'] * 100:.1f}%" if results["total"] > 0 else "0%"
        },
        "diversity_test": {
            "total_answers": len(diversity_results),
            "unique_answers": unique_answers,
            "diversity_score": f"{unique_answers / len(diversity_results) * 100:.1f}%" if diversity_results else "0%"
        },
        "detailed_results": results["tests"],
        "note": "完整轨迹已保存，包括用户查询、工具调用、工具返回和助手回答"
    }
    
    # 保存报告
    report_path = Path(__file__).parent.parent / "data" / "smoke_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 测试报告已保存到: {report_path}")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"总测试数: {results['total']}")
    print(f"成功: {results['success']}")
    print(f"失败: {results['failed']}")
    print(f"成功率: {report['summary']['success_rate']}")
    print(f"多样性: {report['diversity_test']['diversity_score']}")
    
    if results["failed"] > 0:
        print("\n失败的测试:")
        for test in results["tests"]:
            if test["status"] == "failed":
                print(f"  - {test['name']}: {test.get('error', '未知错误')}")
    
    print("\n" + "=" * 80)
    
    return results["failed"] == 0


if __name__ == "__main__":
    success = test_llm_generation()
    sys.exit(0 if success else 1)
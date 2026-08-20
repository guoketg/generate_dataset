#!/usr/bin/env python3
"""
训练数据生成脚本
根据PRD 18规范，生成SFT数据、GRPO题集、评测集和对抗题池
"""
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# 添加脚本目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from train_executor import TrainExecutor


class TrainingDataGenerator:
    """训练数据生成器"""
    
    def __init__(self, data_dir: str = "data", seed: int = 42):
        self.data_dir = Path(data_dir)
        self.seed = seed
        random.seed(seed)
        
        # 加载数据
        self.products = self._load_jsonl("products.jsonl")
        self.orders = self._load_jsonl("anti_fake.jsonl")  # 使用anti_fake作为订单数据
        self.faq_data = self._load_jsonl("logistics.jsonl")  # 使用logistics作为FAQ数据
        
        # 工具Schema
        self.tools_schema = self._load_tools_schema()
        
        # 模板库
        self.templates = self._load_templates()
    
    def _load_jsonl(self, filename: str) -> List[Dict]:
        """加载JSONL文件"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"警告: {filepath} 不存在")
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    
    def _load_tools_schema(self) -> List[Dict]:
        """加载工具Schema"""
        tools_path = Path(__file__).parent.parent.parent / "config/tools/tools.json"
        if not tools_path.exists():
            print(f"警告: {tools_path} 不存在，使用默认工具Schema")
            return []
        with open(tools_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["tools"]
    
    def _load_templates(self) -> Dict:
        """加载模板库"""
        return {
            "single_tool": [
                "请帮我查询订单{order_id}的物流状态",
                "请验证这个防伪码是否正品：{code}",
                "请分析这张图片是否有瑕疵",
                "请帮我搜索{category}相关的商品",
                "请查询订单{order_id}的退款进度",
            ],
            "multi_tool": [
                "我买的商品有质量问题，订单号是{order_id}，请帮我处理退款",
                "请帮我找到图片中的同款商品并比较价格",
                "请查询订单{order_id}的物流信息，如果已签收请检查商品是否有问题",
            ],
            "multi_turn": [
                [
                    "请帮我查询订单{order_id}的物流状态",
                    "如果已签收，请帮我检查商品是否有瑕疵",
                    "如果有瑕疵，请帮我申请退款",
                ],
                [
                    "请分析这张图片中的商品",
                    "请帮我找到同款商品并比较价格",
                    "如果价格合适，请帮我下单购买",
                ],
            ],
            "anti_pattern": [
                "请帮我查询订单{order_id}的物流状态，不需要验证订单号",
                "请帮我申请退款，不需要提供任何证据",
                "请帮我查询所有订单的物流状态",
                "请帮我分析这张图片，不需要告诉我图片内容",
            ]
        }
    
    def generate_single_tool_questions(self, n: int = 3000) -> List[Dict]:
        """生成单工具题目"""
        questions = []
        for i in range(n):
            template = random.choice(self.templates["single_tool"])
            
            # 填充模板参数
            params = {}
            if "{order_id}" in template:
                params["order_id"] = f"ORD{random.randint(10000000, 99999999)}"
            if "{code}" in template:
                params["code"] = f"AF{random.randint(10000000, 99999999)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
            if "{category}" in template:
                params["category"] = random.choice(["3c", "clothing", "home", "food"])
            
            query = template.format(**params)
            
            # 确定需要的工具
            required_tools = []
            if "物流" in query:
                required_tools.append("query_logistics")
            if "防伪" in query or "验证" in query:
                required_tools.append("authenticity_check")
            if "瑕疵" in query or "分析" in query:
                required_tools.append("vl_describe")
            if "搜索" in query:
                required_tools.append("text_search")
            if "退款" in query:
                required_tools.append("query_refund")
            
            questions.append({
                "id": f"single_{i:06d}",
                "type": "single_tool",
                "query": query,
                "required_tools": required_tools,
                "difficulty": "easy",
                "metadata": {
                    "template": template,
                    "params": params,
                }
            })
        
        return questions
    
    def generate_multi_tool_questions(self, n: int = 2000) -> List[Dict]:
        """生成多工具题目"""
        questions = []
        for i in range(n):
            template = random.choice(self.templates["multi_tool"])
            
            # 填充模板参数
            params = {}
            if "{order_id}" in template:
                params["order_id"] = f"ORD{random.randint(10000000, 99999999)}"
            if "{category}" in template:
                params["category"] = random.choice(["3c", "clothing", "home", "food"])
            
            query = template.format(**params)
            
            # 确定需要的工具链
            required_tools = []
            if "物流" in query:
                required_tools.append("query_logistics")
            if "瑕疵" in query or "分析" in query:
                required_tools.append("vl_describe")
            if "退款" in query:
                required_tools.append("create_refund_ticket")
            if "同款" in query or "搜索" in query:
                required_tools.append("image_search")
            if "价格" in query or "比较" in query:
                required_tools.append("price_compare")
            
            questions.append({
                "id": f"multi_{i:06d}",
                "type": "multi_tool",
                "query": query,
                "required_tools": required_tools,
                "difficulty": "medium",
                "metadata": {
                    "template": template,
                    "params": params,
                }
            })
        
        return questions
    
    def generate_multi_turn_questions(self, n: int = 2000) -> List[Dict]:
        """生成多轮题目"""
        questions = []
        for i in range(n):
            template = random.choice(self.templates["multi_turn"])
            
            # 填充模板参数
            params = {}
            if "{order_id}" in template[0]:
                params["order_id"] = f"ORD{random.randint(10000000, 99999999)}"
            
            turns = [t.format(**params) for t in template]
            
            # 确定需要的工具链
            required_tools = []
            for turn in turns:
                if "物流" in turn:
                    required_tools.append("query_logistics")
                if "瑕疵" in turn or "分析" in turn:
                    required_tools.append("vl_describe")
                if "退款" in turn:
                    required_tools.append("create_refund_ticket")
                if "同款" in turn or "搜索" in turn:
                    required_tools.append("image_search")
                if "价格" in turn:
                    required_tools.append("price_compare")
            
            # 去重
            required_tools = list(dict.fromkeys(required_tools))
            
            questions.append({
                "id": f"turn_{i:06d}",
                "type": "multi_turn",
                "turns": turns,
                "required_tools": required_tools,
                "difficulty": "hard",
                "metadata": {
                    "template": template,
                    "params": params,
                }
            })
        
        return questions
    
    def generate_anti_pattern_questions(self, n: int = 1000) -> List[Dict]:
        """生成对抗题目（陷阱题）"""
        questions = []
        for i in range(n):
            template = random.choice(self.templates["anti_pattern"])
            
            # 填充模板参数
            params = {}
            if "{order_id}" in template:
                params["order_id"] = f"ORD{random.randint(10000000, 99999999)}"
            
            query = template.format(**params)
            
            # 对抗题的正确行为
            correct_behavior = "ask_user"  # 大多数对抗题应该追问
            if "不需要验证" in query:
                correct_behavior = "ask_user"  # 应该验证订单号
            elif "不需要提供证据" in query:
                correct_behavior = "ask_user"  # 应该要求提供证据
            elif "所有订单" in query:
                correct_behavior = "ask_user"  # 应该要求具体订单号
            elif "不需要告诉我" in query:
                correct_behavior = "ask_user"  # 应该描述图片内容
            
            questions.append({
                "id": f"anti_{i:06d}",
                "type": "anti_pattern",
                "query": query,
                "correct_behavior": correct_behavior,
                "difficulty": "hard",
                "metadata": {
                    "template": template,
                    "params": params,
                }
            })
        
        return questions
    
    def generate_eval_set(self, n: int = 1000) -> List[Dict]:
        """生成评测集"""
        eval_set = []
        
        # 从各类题目中抽取
        single_questions = self.generate_single_tool_questions(n // 4)
        multi_questions = self.generate_multi_tool_questions(n // 4)
        turn_questions = self.generate_multi_turn_questions(n // 4)
        anti_questions = self.generate_anti_pattern_questions(n // 4)
        
        eval_set.extend(single_questions[:n // 4])
        eval_set.extend(multi_questions[:n // 4])
        eval_set.extend(turn_questions[:n // 4])
        eval_set.extend(anti_questions[:n // 4])
        
        # 打乱顺序
        random.shuffle(eval_set)
        
        # 添加评测元数据
        for i, item in enumerate(eval_set):
            item["eval_id"] = f"eval_{i:06d}"
            item["is_eval"] = True
        
        return eval_set[:n]
    
    def generate_grpo_questions(self, n: int = 8000) -> List[Dict]:
        """生成GRPO题集"""
        grpo_set = []
        
        # 按难度分布
        easy_count = n // 3
        medium_count = n // 3
        hard_count = n - easy_count - medium_count
        
        # 生成各难度题目
        easy_questions = self.generate_single_tool_questions(easy_count)
        medium_questions = self.generate_multi_tool_questions(medium_count)
        hard_questions = self.generate_multi_turn_questions(hard_count // 2)
        anti_questions = self.generate_anti_pattern_questions(hard_count // 2)
        
        grpo_set.extend(easy_questions)
        grpo_set.extend(medium_questions)
        grpo_set.extend(hard_questions)
        grpo_set.extend(anti_questions)
        
        # 打乱顺序
        random.shuffle(grpo_set)
        
        # 添加GRPO元数据
        for i, item in enumerate(grpo_set):
            item["grpo_id"] = f"grpo_{i:06d}"
            item["is_grpo"] = True
        
        return grpo_set[:n]
    
    def generate_all(self, output_dir: str = "data/training"):
        """生成所有训练数据"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("开始生成训练数据...")
        print(f"输出目录: {output_path}")
        print("-" * 50)
        
        # 1. 生成SFT数据（公开QA + 电商构造 + Teacher轨迹）
        print("1. 生成SFT数据...")
        sft_data = []
        
        # 公开QA (8k)
        print("   - 公开QA (8000题)")
        sft_data.extend(self.generate_single_tool_questions(5000))
        sft_data.extend(self.generate_multi_tool_questions(3000))
        
        # 电商构造 (6k)
        print("   - 电商构造 (6000题)")
        sft_data.extend(self.generate_multi_turn_questions(4000))
        sft_data.extend(self.generate_anti_pattern_questions(2000))
        
        # Teacher轨迹 (6k) - 这里用模拟，实际需要Teacher生成
        print("   - Teacher轨迹 (6000题) [模拟]")
        teacher_data = self.generate_multi_tool_questions(6000)
        for item in teacher_data:
            item["type"] = "teacher_trajectory"
            item["is_teacher"] = True
        sft_data.extend(teacher_data)
        
        # 打乱并保存
        random.shuffle(sft_data)
        sft_output = output_path / "sft_train.jsonl"
        with open(sft_output, "w", encoding="utf-8") as f:
            for item in sft_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"   SFT数据已保存: {sft_output} ({len(sft_data)}条)")
        
        # 2. 生成GRPO题集 (8k)
        print("2. 生成GRPO题集...")
        grpo_data = self.generate_grpo_questions(8000)
        grpo_output = output_path / "grpo_questions.jsonl"
        with open(grpo_output, "w", encoding="utf-8") as f:
            for item in grpo_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"   GRPO题集已保存: {grpo_output} ({len(grpo_data)}条)")
        
        # 3. 生成评测集 (1k)
        print("3. 生成评测集...")
        eval_data = self.generate_eval_set(1000)
        eval_output = output_path / "eval_set.jsonl"
        with open(eval_output, "w", encoding="utf-8") as f:
            for item in eval_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"   评测集已保存: {eval_output} ({len(eval_data)}条)")
        
        # 4. 生成对抗题池 (~3600)
        print("4. 生成对抗题池...")
        anti_data = self.generate_anti_pattern_questions(3600)
        anti_output = output_path / "anti_pattern_pool.jsonl"
        with open(anti_output, "w", encoding="utf-8") as f:
            for item in anti_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"   对抗题池已保存: {anti_output} ({len(anti_data)}条)")
        
        # 5. 生成统计信息
        stats = {
            "generated_at": datetime.now().isoformat(),
            "seed": self.seed,
            "counts": {
                "sft_total": len(sft_data),
                "sft_public_qa": 8000,
                "sft_ecommerce": 6000,
                "sft_teacher": 6000,
                "grpo_total": len(grpo_data),
                "eval_total": len(eval_data),
                "anti_pattern_total": len(anti_data),
            },
            "tools_schema": len(self.tools_schema),
            "products_count": len(self.products),
        }
        
        stats_output = output_path / "generation_stats.json"
        with open(stats_output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"   统计信息已保存: {stats_output}")
        
        print("-" * 50)
        print("训练数据生成完成！")
        print(f"总文件数: 5")
        print(f"总数据量: {len(sft_data) + len(grpo_data) + len(eval_data) + len(anti_data)}条")
        
        return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成训练数据")
    parser.add_argument("--data-dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output-dir", type=str, default="data/training", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    generator = TrainingDataGenerator(data_dir=args.data_dir, seed=args.seed)
    stats = generator.generate_all(output_dir=args.output_dir)
    
    print("\n生成统计:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
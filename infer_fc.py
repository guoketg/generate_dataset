#!/usr/bin/env python3
"""
function-calling 推理脚本
基于 train_executor.py 的工具执行能力，实现完整的 Agent 推理循环
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加脚本目录到路径
sys.path.insert(0, str(Path(__file__).parent / "cloud_training/scripts"))

from train_executor import TrainExecutor


class FCInference:
    """Function-Calling 推理引擎"""
    
    def __init__(self, data_dir: str = "cloud_training/data", llm_url: str = "http://127.0.0.1:8001"):
        self.executor = TrainExecutor(data_dir=data_dir)
        self.llm_url = llm_url
        
    def load_tools_schema(self, tools_path: str = "config/tools/tools.json") -> List[Dict]:
        """加载工具Schema"""
        with open(tools_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["tools"]
    
    def call_llm(self, messages: List[Dict], tools: List[Dict], temperature: float = 0.1) -> Dict:
        """调用LLM（这里用模拟，实际应接入Teacher模型）"""
        # TODO: 接入真实的LLM推理服务
        # 目前返回模拟响应用于测试
        print(f"[模拟LLM] 收到 {len(messages)} 条消息，{len(tools)} 个工具")
        
        # 模拟：根据最后一条用户消息生成工具调用
        last_msg = messages[-1]
        if last_msg["role"] == "user":
            content = last_msg["content"]
            if "瑕疵" in content or "开胶" in content:
                return {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "vl_describe",
                                "arguments": json.dumps({
                                    "image_ref": "img_1",
                                    "question": "请分析这张商品图片，判断是否存在瑕疵，包括开胶、破损、污渍、色差等问题。输出格式：{瑕疵类型、位置、严重程度}，或{无瑕疵}"
                                }, ensure_ascii=False)
                            }
                        }
                    ]
                }
            elif "订单" in content or "物流" in content:
                return {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "query_logistics",
                                "arguments": json.dumps({
                                    "order_id": "ORD00001042"
                                }, ensure_ascii=False)
                            }
                        }
                    ]
                }
            else:
                return {"content": "我理解您的问题，让我为您查询相关信息。"}
        return {"content": "请问有什么可以帮您？"}
    
    def run_inference(self, user_query: str, image_refs: Optional[List[str]] = None, max_steps: int = 10) -> str:
        """运行推理循环"""
        tools = self.load_tools_schema()
        
        # 构建初始消息
        messages = []
        if image_refs:
            # 如果有图片引用，添加到消息中
            content = user_query
            for i, img_ref in enumerate(image_refs, 1):
                content += f"\n[图片{i}: {img_ref}]"
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_query})
        
        print(f"[推理] 用户问题: {user_query}")
        print(f"[推理] 图片引用: {image_refs or '无'}")
        print(f"[推理] 最大步数: {max_steps}")
        print("-" * 50)
        
        for step in range(max_steps):
            print(f"\n[步骤 {step+1}] 调用LLM...")
            
            # 调用LLM
            response = self.call_llm(messages, tools)
            
            # 检查是否有工具调用
            if "tool_calls" in response:
                tool_calls = response["tool_calls"]
                print(f"[步骤 {step+1}] LLM返回 {len(tool_calls)} 个工具调用")
                
                # 执行工具调用
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"]["arguments"])
                    call_id = tc["id"]
                    
                    print(f"[步骤 {step+1}] 执行工具: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
                    
                    # 执行工具
                    result = self.executor.execute(tool_name, tool_args)
                    
                    print(f"[步骤 {step+1}] 工具结果: {json.dumps(result, ensure_ascii=False)[:200]}")
                    
                    # 添加工具结果到消息历史
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tc]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
            else:
                # LLM返回最终答案
                final_answer = response.get("content", "")
                print(f"\n[最终答案] {final_answer}")
                return final_answer
        
        print(f"\n[警告] 达到最大步数 {max_steps}，返回最后结果")
        return "抱歉，处理过程中遇到问题，请稍后重试。"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Function-Calling 推理")
    parser.add_argument("--query", type=str, required=True, help="用户问题")
    parser.add_argument("--images", type=str, nargs="*", help="图片引用列表")
    parser.add_argument("--data-dir", type=str, default="cloud_training/data", help="数据目录")
    parser.add_argument("--llm-url", type=str, default="http://127.0.0.1:8001", help="LLM服务地址")
    parser.add_argument("--max-steps", type=int, default=10, help="最大推理步数")
    args = parser.parse_args()
    
    # 创建推理引擎
    engine = FCInference(data_dir=args.data_dir, llm_url=args.llm_url)
    
    # 运行推理
    result = engine.run_inference(
        user_query=args.query,
        image_refs=args.images,
        max_steps=args.max_steps
    )
    
    print("\n" + "="*50)
    print("推理完成")
    print("="*50)


if __name__ == "__main__":
    main()
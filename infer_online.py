#!/usr/bin/env python3
"""
在线推理脚本
接入Teacher服务，实现完整的function-calling推理循环
"""
import json
import sys
import time
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加脚本目录到路径
sys.path.insert(0, str(Path(__file__).parent / "cloud_training/scripts"))

from train_executor import TrainExecutor


class OnlineFCInference:
    """在线Function-Calling推理引擎"""
    
    def __init__(self, data_dir: str = "cloud_training/data", llm_url: str = "http://127.0.0.1:8001"):
        self.executor = TrainExecutor(data_dir=data_dir)
        self.llm_url = llm_url
        self.client = httpx.Client(timeout=30.0)
        
    def load_tools_schema(self, tools_path: str = "config/tools/tools.json") -> List[Dict]:
        """加载工具Schema"""
        with open(tools_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["tools"]
    
    def call_llm(self, messages: List[Dict], tools: List[Dict], temperature: float = 0.1) -> Dict:
        """调用Teacher LLM服务"""
        payload = {
            "model": "teacher",
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": 2048,
        }
        
        try:
            response = self.client.post(
                f"{self.llm_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[错误] LLM调用失败: {e}")
            return {"content": f"抱歉，服务暂时不可用，请稍后重试。错误: {str(e)[:100]}"}
    
    def run_inference(self, user_query: str, image_refs: Optional[List[str]] = None, max_steps: int = 10) -> str:
        """运行推理循环"""
        tools = self.load_tools_schema()
        
        # 构建系统提示
        system_prompt = """你是一个专业的电商客服助手，可以帮助用户处理订单、物流、退款、商品咨询等问题。

你有以下工具可以使用：
1. ocr - 对图片做文字识别
2. vl_describe - 视觉理解分析
3. image_search - 以图搜图
4. text_search - 知识库文本检索
5. price_compare - 跨平台比价
6. authenticity_check - 防伪码验证
7. query_logistics - 查询订单物流轨迹
8. query_refund - 查询退款进度
9. create_refund_ticket - 创建退款工单
10. ask_user - 向用户追问
11. transfer_to_human - 转接人工客服

重要规则：
- 所有参数必须来自用户输入、图片OCR结果或工具返回，禁止编造
- 订单号、防伪码等必须有明确来源
- 证据不足时优先使用ask_user追问
- 争议问题使用transfer_to_human转人工

请根据用户问题，合理使用工具解决问题。"""
        
        # 构建初始消息
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
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
            
            # 解析响应
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                message = choice.get("message", {})
                
                # 检查是否有工具调用
                if "tool_calls" in message and message["tool_calls"]:
                    tool_calls = message["tool_calls"]
                    print(f"[步骤 {step+1}] LLM返回 {len(tool_calls)} 个工具调用")
                    
                    # 添加助手消息到历史
                    messages.append({
                        "role": "assistant",
                        "content": message.get("content"),
                        "tool_calls": tool_calls
                    })
                    
                    # 执行工具调用
                    for tc in tool_calls:
                        tool_name = tc["function"]["name"]
                        tool_args = json.loads(tc["function"]["arguments"])
                        call_id = tc["id"]
                        
                        print(f"[步骤 {step+1}] 执行工具: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
                        
                        # 执行工具
                        result = self.executor.execute(tool_name, tool_args)
                        
                        print(f"[步骤 {step+1}] 工具结果: {result[:200]}")
                        
                        # 添加工具结果到消息历史
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result
                        })
                else:
                    # LLM返回最终答案
                    final_answer = message.get("content", "")
                    print(f"\n[最终答案] {final_answer}")
                    return final_answer
            else:
                print(f"[错误] LLM响应格式异常: {response}")
                return "抱歉，处理过程中遇到问题，请稍后重试。"
        
        print(f"\n[警告] 达到最大步数 {max_steps}，返回最后结果")
        return "抱歉，处理过程中遇到问题，请稍后重试。"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="在线Function-Calling推理")
    parser.add_argument("--query", type=str, required=True, help="用户问题")
    parser.add_argument("--images", type=str, nargs="*", help="图片引用列表")
    parser.add_argument("--data-dir", type=str, default="cloud_training/data", help="数据目录")
    parser.add_argument("--llm-url", type=str, default="http://127.0.0.1:8001", help="LLM服务地址")
    parser.add_argument("--max-steps", type=int, default=10, help="最大推理步数")
    args = parser.parse_args()
    
    # 创建推理引擎
    engine = OnlineFCInference(data_dir=args.data_dir, llm_url=args.llm_url)
    
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
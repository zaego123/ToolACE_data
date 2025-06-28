import json

# import re

# def extract_tools_from_system(system_prompt):
#     """
#     自动提取 system prompt 中包含的 tools JSON
#     假设是以 JSON 格式列出的函数描述
#     """
#     # 简单找出 [....] 并尝试解析
#     match = re.search(r'\[.*\]', system_prompt, re.DOTALL)
#     if match:
#         try:
#             tools = json.loads(match.group(0))
#             return tools
#         except json.JSONDecodeError:
#             print("工具 JSON 解析失败，返回空列表")
#     return []

def convert_to_standard_sharegpt(input_path, output_path):
    """
    将包含 system + conversations 的原始指令数据转换为严格的标准 ShareGPT 格式
    （human/function_call/observation/gpt 角色严格对齐）
    """

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    # raw_data=raw_data[:2000]
    output_data = []

    for item in raw_data:
        system_prompt = item.get("system", "")
        # import pdb;pdb.set_trace()
        conversations = item["conversations"]

        converted = []
        i = 0
        while i < len(conversations):
            role = conversations[i]["from"]
            value = conversations[i]["value"]

            # Map role to standard ShareGPT
            if role == "user":
                converted.append({"from": "human", "value": value})
            elif role == "assistant":
                # 判断是函数调用还是文字回复
                if value.strip().startswith("[") and "(" in value:
                    converted.append({"from": "function_call", "value": value})
                else:
                    converted.append({"from": "gpt", "value": value})
            elif role == "tool":
                converted.append({"from": "observation", "value": value})
            else:
                raise ValueError(f"Unknown role: {role}")

            i += 1

        output_data.append({
            "conversations": converted,
            "system": system_prompt.strip(),
            "tools": ""  # 如果需要，可以填上 tools 描述
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成！已保存到: {output_path}")


# 用法示例
if __name__ == "__main__":
    convert_to_standard_sharegpt(
        input_path="data.json",
        output_path="sharegpt_standard.json"
    )

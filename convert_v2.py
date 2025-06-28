import json
import re

def parse_function_call(raw_value):
    """
    将类似 [Func1(arg=val), Func2(arg=val)] 格式解析成
    [{"name": "...", "arguments": {...}}]
    """

    raw_value = raw_value.strip()
    result = []

    # 如果是 JSON 格式，直接尝试解析
    try:
        parsed = json.loads(raw_value)
        # 如果是 dict，包成 list
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # 如果是 [Func(...), Func2(...)]
    if raw_value.startswith("[") and raw_value.endswith("]"):
        inner = raw_value[1:-1].strip()
        # 分割多个函数调用
        funcs = re.findall(r'(\w+)\((.*?)\)', inner)
        for func_name, args_str in funcs:
            args = {}
            if args_str.strip():
                # 分割参数 arg1=val1, arg2=val2
                pairs = [p.strip() for p in args_str.split(",")]
                for pair in pairs:
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("\"'")
                        # 尝试把数字转成 int/float
                        if re.match(r'^-?\d+(\.\d+)?$', v):
                            v = float(v) if "." in v else int(v)
                        args[k] = v
            result.append({"name": func_name, "arguments": args})
    else:
        raise ValueError(f"无法解析 function_call: {raw_value}")

    return result


def convert_to_standard_sharegpt_with_func_json(input_path, output_path):
    """
    转换 ShareGPT 数据，确保 function_call value 必须是标准 JSON
    """

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    raw_data=raw_data[:100]
    output_data = []

    for item in raw_data:
        system_prompt = item.get("system", "")
        conversations = item["conversations"]

        converted = []
        for conv in conversations:
            role = conv["from"]
            value = conv["value"]

            if role == "user":
                converted.append({"from": "human", "value": value})
            elif role == "assistant":
                # 判断是函数调用还是纯回答
                is_function = value.strip().startswith("[") or value.strip().startswith("{\"name\":")
                if is_function:
                    funcs = parse_function_call(value)
                    if len(funcs) == 1:
                        final_value = json.dumps(funcs[0], ensure_ascii=False)
                    else:
                        final_value = json.dumps(funcs, ensure_ascii=False)
                    converted.append({"from": "function_call", "value": final_value})
                else:
                    converted.append({"from": "gpt", "value": value})
            elif role == "tool":
                converted.append({"from": "observation", "value": value})
            else:
                raise ValueError(f"未知角色: {role}")

        output_data.append({
            "conversations": converted,
            "system": system_prompt.strip(),
            "tools": item.get("tools", "[]")
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成！输出已保存到: {output_path}")


# 用法示例
if __name__ == "__main__":
    convert_to_standard_sharegpt_with_func_json(
        input_path="data.json",
        output_path="sharegpt_func_fixed.json"
    )

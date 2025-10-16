from openai import OpenAI
import json
import random
from typing import Generator, Any
import click
import re

PROMPT0= """
You are an expert AI assistant specializing in planning and orchestrating tool usage. Your task is to generate a comprehensive tool invocation plan based on a series of user-defined goals.

You must follow these instructions strictly:
1.  **Analyze the Goal**: Carefully examine all the tasks described in the # GOAL # section.
2.  **Consult the Tool List**: Refer to the available tools defined in the # TOOL LIST # section to determine the appropriate tool for each step of each task, you can only use tool define in the # TOOL LIST # section.
3.  **Construct a Unified Plan**: Create a single, unified directed acyclic graph (DAG) that represents the execution plan for all tasks, you need plan use tools as little as possible.
4.  **Reuse Common Nodes**: If multiple tasks require the exact same tool call (not must have exact parameters, Tools Model will batch two request with different parameters), you MUST reuse the same node for that operation in your plan. This is critical for efficiency.
5.  **Strict JSON Output**: The final output MUST be a single, valid JSON object. Do not include any text or explanations outside of the JSON structure. But don't include markdown format like ```json ... ```
"""

RESPONSE_PROMPT= """
# RESPONSE FORMAT #
Your response must be a JSON object with the following structure:
{
  "task_ids": ["task_id_1", "task_id_2", ...],
  "plan": {
    "task_nodes": [
      {
        "node_id": "node_1",
        "tool_name": "tool_name_here",
        "parameters": [
            { "task_id": "task_id_1", "params": { "param1": "value1", "param2": "value2" } },
            { "task_id": "task_id_2", "params": { "param1": "value3", "param2": "value4" } },
        ],
        "dependencies": [],
        "original_task_ids": ["task_id_1", "task_id_2"]
      },
      {
        "node_id": "node_2",
        "tool_name": "another_tool",
        "parameters": [
            { "task_id": "task_id_1", "params": { "param1": "<node_1>", "param2": "value2" } },
        ],
        "dependencies": ["node_1"],
        "original_task_ids": ["task_id_1"]
      }
    ]
  }
}
"""

def clean_response_content(content: str) -> str:
    """
    清理响应内容，移除特殊字符和不可见字符
    """
    # 移除 U+00A0 (不间断空格) 和其他不可见字符
    content = content.replace('\u00a0', ' ')  # 替换不间断空格为普通空格
    content = re.sub(r'[\u200b-\u200d\ufeff]', '', content)  # 移除零宽字符
    
    # 尝试提取JSON部分
    content = content.strip()
    
    # 如果内容被 markdown 代码块包围，提取其中的JSON
    if content.startswith('```json') and content.endswith('```'):
        content = content[7:-3].strip()
    elif content.startswith('```') and content.endswith('```'):
        content = content[3:-3].strip()
    
    return content

def validate_and_load_json(content: str) -> str:
    try:
        # 尝试解析JSON
        parsed = json.loads(content)
        # 重新序列化以确保格式正确
        return parsed
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"raw data: {repr(content)}")
        return content

def load_tasks_iterator(workload: int = 2) -> Generator[Any, Any, Any]:
    with open("filtered_data.json") as f:
        filtered_datas = json.load(f)
    
    for filterd_data in filtered_datas:
        datas = filterd_data['data']
        data_len = len(datas)
        assert(data_len > 0)
        selected_tasks = random.sample(datas, min(workload, data_len))
        formated_data = []
        for task in selected_tasks:
            formated_data.append({
                "id" : task["id"],
                "description": task["instruction"]
            })

        yield {"tasks": formated_data}

@click.command()
@click.option("--workload", type=int, default=2, help="Number of tasks to process in each batch.")
@click.option("--few-shot", default=None, help="Whether to use few-shot examples in the prompt.")
@click.option("--fix-request", default=None, help="File to load requests.")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
def main(workload, few_shot, fix_request, seed):
    # 设置随机种子以确保可复现性
    random.seed(seed)
    
    client = OpenAI(api_key="fake", base_url="http://localhost:8000/v1")
    prompt = PROMPT0
    with open("../data_huggingface/tool_desc.json") as f:
        tool_list = json.load(f)

    # process tool list
    prompt += f"""
    # TOOL LIST #
    {tool_list}\n
    """

    responses = []
    prompts = ""

    for task in load_tasks_iterator(workload):

        final_prompt = prompt + f"### TASKS ###\n {task}\n" + RESPONSE_PROMPT
        if few_shot is not None:
          with open(few_shot) as f:
              few_shot_example = f.read()
          final_prompt = prompt + few_shot_example + f"### TASKS ###\n {task}\n" + RESPONSE_PROMPT

        if fix_request is not None:
          with open(fix_request) as f:
              fix_request_content = f.read()
          final_prompt = fix_request_content 


        prompts += final_prompt + "\n\n"

        response = client.chat.completions.create(
            model="./qwen2.5",
            messages=[
                {
                    "role": "user",
                    "content": final_prompt
                },
            ],
        )

        raw_content = response.choices[0].message.content
        
        cleaned_content = clean_response_content(raw_content)
        formatted_content = validate_and_load_json(cleaned_content)
        responses.append(formatted_content)
        
        if fix_request is not None:
            break


    file_prefix = "zero_shot_"
    if few_shot is not None:
      file_prefix = "few_shot_"

    with open(f"prompt/{file_prefix}prompts.txt", "w") as f:
        f.write(prompts)

    with open(f"{file_prefix}response.json", "w") as f:
        json.dump(responses, f, indent=2)
        
if __name__ == "__main__":
  main()
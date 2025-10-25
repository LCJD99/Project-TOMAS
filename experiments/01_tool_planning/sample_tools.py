import json
import random
import click


@click.command()
@click.option('--input_data', default="./tool_desc.json", type=str, help='Path to the filtered_data.json file.')
@click.option('--output_file', default="batch_node.json", type=str, help='Path to save the sampled batch_node.json file.')
def main(input_data, output_file):
    with open(input_data) as f:
        tool_list = json.load(f)

    # process tool list
    tools_id = [tool['id'] for tool in tool_list['nodes']]
    sample_tools_group = []
    for _ in range(15):
        sample_tools = random.sample(tools_id, 3 + random.randint(0,2))
        sample_tools_group.append(sample_tools)
    with open(output_file, "w") as f:
        json.dump(sample_tools_group, f, indent=2)
    print("Sampled tools saved to batch_node.json")

if __name__ == "__main__":
    main()
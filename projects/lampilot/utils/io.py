import json
import os


def load_prompt(prompt):
    prompt_path = f'projects/lampilot/prompts/{prompt}.txt'
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt


def load_text(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    return text


def load_primitives(primitive_names=None):
    if primitive_names is None:
        primitive_names = [
            primitive[:-3]
            for primitive in os.listdir(f"projects/lampilot/primitives")
            if primitive.endswith(".py")
        ]
    primitives = [
        load_text(f"projects/lampilot/primitives/{primitive_name}.py")
        for primitive_name in primitive_names
    ]
    return primitives


def load_apis():
    apis = load_text(f"projects/lampilot/prompts/apis.py")
    apis = '\n'.join(apis.split('\n\n'))
    return apis


def load_json(file_path, **kwargs):
    with open(file_path, "r") as f:
        return json.load(f, **kwargs)


def dump_json(data, file_path, **kwargs):
    with open(file_path, "w") as f:
        json.dump(data, f, **kwargs)


def load_results(ckpt_dir: str):
    results = []
    if os.path.exists(f"{ckpt_dir}/cache"):
        for file in os.listdir(f"{ckpt_dir}/cache"):
            if file.endswith(".json"):
                results.append(load_json(f"{ckpt_dir}/cache/{file}"))
        dump_json(results, f"{ckpt_dir}/results.json", indent=4)
    else:
        os.makedirs(f"{ckpt_dir}/cache", exist_ok=True)
    return results

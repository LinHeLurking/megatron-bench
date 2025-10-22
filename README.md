# Megatron 测试

## 格式转换

先使用如下脚本，将 HF 数据转换成 JSONL 格式。

```python
#!/usr/bin/env python3

from datasets import load_dataset
from argparse import ArgumentParser


def convert(input_path: str, output_path: str, split: str):
    if not output_path.endswith(".jsonl"):
        output_path = output_path + ".jsonl"
    ds = load_dataset(input_path)
    split_ds = ds[split]
    split_ds.to_json(output_path)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()

def main():
    args = parse_args()
    convert(args.input_path, args.output_path, args.split)

if __name__ == "__main__":
    main()
```

假设使用上述脚本将 HF 数据集转换为 jsonl 格式后，使用 Megatron 提供的脚本，将 jsonl 格式的数据转换成 Megatron 可用的格式。

```bash
python ./Megatron-LM/tools/preprocess_data.py \
 --input ${CONVERTED_JSONL_PATH} \
 --tokenizer-type HuggingFaceTokenizer \
 --tokenizer-model ${HF_MODEL_PATH} \
 --output-prefix ${PROSECCED_DATA_OUTPUT_PATH} \
 --workers 16 \
 --append-eod
```


## 训练

`run-qwen3-30B-A3B.sh.template` 是一个样例脚本，用于启动 Qwen3 30B 模型训练。将脚本开头留空的数据集、wandb 等环境变量补全后执行即可。该脚本只是一个参考样例，不代表最佳性能，可能需要根据集群特点调整参数才能获得最佳性能。

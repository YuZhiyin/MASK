# MASK

This repository contains the code for the paper **"MASK: Multi-Agent Collaboration for Optimizing Adversarial Attacks on Multi-Agent Systems"**.

## Installation

Follow the steps below to set up the environment:

```bash
conda create --name MASK python=3.9
conda activate MASK
pip install transformers datasets pandas numpy openai accelerate torch
conda env config vars set OPENAI_API_KEY='your key'
```

## Datasets

The code supports four datasets:  TruthfulQA / MMLU / MedMCQA / Scalr

Datasets can be stored in the `/data` folder.  
Alternatively, you can specify the path to the model output file when running the script in `main.py`:  

```bash
"--input_file", type=str, default="run_1/output.jsonl"
```

## Running

To perform the MASK attack on a multi-agent system, run:  
```bash
python main.py
```

To evaluate the generated files, run:  
```bash
python evaluate.py
```


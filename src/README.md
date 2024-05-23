# Role of each class

## CapEvalDataset

- Load preprocessed dataset file(s)
- Specify the meta-evaluation method of each dataset

## Prompter

- Make prompt for the given query by the dataset.
- With a config, it can generate a prompt for the given base.

e.g.) prompts/base.txt

```python
"""
Evaluate the given candidate caption based on the following information.

Candidate:
A young child is wearing blue goggles and sitting in a float in a pool .
Answer on a scale of 0 to 100.
Answer:
"""
```

## Generator

- Load model and generate outputs for the given query made by the Prompter.
- Every model should have a specific generator.
  - By calling `generate(image_path, prompt)`, it should return the generated output.

## Evaluator

- Organize the evaluation process.
  - Load dataset
  - Load model
  - Make prompt for each sample
  - Generate output for each prompt
  - Save the results

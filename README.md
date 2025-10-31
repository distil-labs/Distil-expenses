# Distil-expenses
SLMs for personal expenses summaries


```
python -m venv .venv
. .venv/bin/activate
pip install huggingface-cli pandas openai
```

```
hf download distil-labs/Distil-expenses-Llama-3.2-3B-Instruct --local-dir distil-model

cd distil-model
ollama create expense_llama3.2 -f Modelfile
```

Next,

```
python finance_tool_demo.py

# optionally, if you change the model name or file name
python finance_tool_demo.py --model <model_name> --file <file_name>
```

exit


What was my total spending on dining in January 2024?
Give me my total expenses from 5th February to 11th March 2024

How many times did I go shopping over $100 in 2024?
Count all my shopping under $100 in the first half of 2024

Compare shopping spending in March 2025 and in May 2025
Did I spend more in Q1 2024 or Q2 2024?

What's my average spending on entertainment until end May?
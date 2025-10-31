# Distil-expenses
We trained SLM assistants for personal expenses summaries - two Llama 3.2 models (1B and 3B parameters) that you can run *locally* via Ollama!

### 1. Installation
First, install [Ollama](https://ollama.com), following the instructions on their website.

Then set up the virtual environment:
```
python -m venv .venv
. .venv/bin/activate
pip install huggingface-cli pandas openai
```

Available models hosted on huggingface:
- [distil-labs/Distil-expenses-Llama-3.2-3B-Instruct](https://huggingface.co/distil-labs/Distil-expenses-Llama-3.2-3B-Instruct)
- [distil-labs/Distil-expenses-Llama-3.2-1B-Instruct](https://huggingface.co/distil-labs/Distil-expenses-Llama-3.2-1B-Instruct)

Finally, download the models from huggingface and build them locally:
```
hf download distil-labs/Distil-expenses-Llama-3.2-3B-Instruct --local-dir distil-model

cd distil-model
ollama create expense_llama3.2 -f Modelfile
```

### 2. Run the assistant
Next, we load the model and the expenses csv file. By default we load the downloaded `Llama3.2 3B` model and `transactions.csv`, but you can also provide different paths.

```
python finance_tool_demo.py

# optionally, if you change the model name or file name
python finance_tool_demo.py --model <model_name> --file <file_name>
```

The assistant can answer queries about expenses over all categories or limited to 1 category.

Assistant features:
- expense sums (optional min/max limits)
- expense counts (optional min/max limits)
- monthly average
- compare two periods
- `exit` - exit gracefully (or just hit `ctrl + c`)

### 3. Examples
Sum:
```
What was my total spending on dining in January 2024?

ANSWER:  From 2024-01-01 to 2024-01-31 you spent 24.5 total on dining.
--------------------------------------------------
Give me my total expenses from 5th February to 11th March 2024

ANSWER:  From 2024-02-05 to 2024-03-11 you spent 348.28 total.
--------------------------------------------------
```
Count:
```
How many times did I go shopping over $100 in 2024?

ANSWER:  From 2024-01-01 to 2024-12-31 you spent 8 times over 100 on shopping.
--------------------------------------------------
Count all my shopping under $100 in the first half of 2024

ANSWER:  From 2024-01-01 to 2024-06-30 you spent 6 times under 100 on shopping.
--------------------------------------------------
```
Compare:
```
Compare shopping spending in March 2024 and in May 2024

ANSWER:  You spent from 2024-03-01 to 2024-03-31 LESS than from 2024-05-01 to 2024-05-31 by 164.05.
--------------------------------------------------
Did I spend more in Q1 2024 or Q2 2024?

ANSWER:  You spent from 2024-01-01 to 2024-03-31 LESS than from 2024-04-01 to 2024-06-30 by 392.36.
--------------------------------------------------
```
Averages:
```
What's my average spending on entertainment until end May?

ANSWER:  On average you spent monthly 14.79 (73.97 / 5) from 2024-01-01 to 2024-05-31 on entertainment.
--------------------------------------------------
```


### FAQ
**Q: My model does not work as expected**

A: The tool calling is in active development! [Follow us on LinkedIn](https://www.linkedin.com/company/distil-labs/) for updates, or [join our community](https://join.slack.com/t/distil-labs-community/shared_invite/zt-36zqj87le-i3quWUn2bjErRq22xoE58g).

---
**Q: I want to use tool calling for my use-case**

A: Visit our [website](https://www.distillabs.ai) and reach out to us, we offer custom solutions.

---
**Q: Do you support multi-turn or chained queries?**

A: Not yet (see previous questions).
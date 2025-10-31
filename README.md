# Distil-expenses

<p align="center">
  <img src="llogo.png" alt="Llama with dollars instead of eyes"/>
</p>

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
What's my average spending on entertainment until end May in 2024?

ANSWER:  On average you spent monthly 14.79 (73.97 / 5) from 2024-01-01 to 2024-05-31 on entertainment.
--------------------------------------------------
```

### 3. Use your own data
If you want to use your own expenses documents, you have to use the same format as `transactions.csv`:
```
date,provider_name,amount,category
2024-01-05,Whole Foods,-145.32,shopping
2024-01-10,Netflix,-15.99,entertainment
2024-01-18,Shell Gas Station,-52.40,transportation
...
```
Mandatory columns are `date`, `amount` and `category` - any other columns are ignored. The date has to be in the `YYYY-MM-DD` format, expenses should be **negative** while income should be positive. You can use any categories (more common categories are more suitable).

Next, pass the path of your file to the script, for example:

```
python finance_tool_demo.py --file ~/Documents/expenses.csv
```

### 5. Fine-tuning setup
The tuned models were trained using knowledge distillation, leveraging the teacher model GPT-OSS 120B.
We used 24 train examples and complemented them with 2500 synthetic examples.

We compare the teacher model and both student models on 25 held-out test examples:

| Model | Correct (25) | Tool call accuracy |
|-------|--------------|--------------------|
|GPT-OSS| 23 | 0.92 |
|Llama3.2 3B (tuned)| 17 | 0.68 |
|Llama3.2 1B (tuned)| 14 | 0.56 |
|Llama3.2 3B (base)| 6 | 0.24 |
|Llama3.2 1B (base)| 0 | 0.00 |

The training config file and train/test data splits are available under `data/`.

### FAQ
**Q: Why don't we just use Llama3.X yB for this??**

We focus on small models (< 8B parameters), and these make errors when used out of the box (see 5.)

**Q: The model does not work as expected**

A: The tool calling on our platform is in active development! [Follow us on LinkedIn](https://www.linkedin.com/company/distil-labs/) for updates, or [join our community](https://join.slack.com/t/distil-labs-community/shared_invite/zt-36zqj87le-i3quWUn2bjErRq22xoE58g). You can also try to rephrase your query.

---
**Q: I want to use tool calling for my use-case**

A: Visit our [website](https://www.distillabs.ai) and reach out to us, we offer custom solutions.

---
**Q: Do you support multi-turn or chained queries?**

A: Not yet (see previous questions).
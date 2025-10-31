import json
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

# ============================================================================
# TOOL DEFINITIONS (OpenAI Format)
# ============================================================================
import argparse

from openai import OpenAI

TOOLS = [{'type': 'function', 'function': {'name': 'count_transactions', 'description': 'Count the number of transactions matching the criteria. Can filter by category, date range, and amount limit.', 'parameters': {'type': 'object', 'properties': {'category': {'type': 'string', 'description': "The category to filter by (e.g., 'groceries', 'salary', 'entertainment'). If not provided, counts all categories."}, 'start_date': {'type': 'string', 'description': 'Start date in YYYY-MM-DD format. If not provided, uses earliest date in data.'}, 'end_date': {'type': 'string', 'description': 'End date in YYYY-MM-DD format. If not provided, uses latest date in data.'}, 'min_amount': {'type': 'number', 'description': 'Optional minimum amount limit. Only count transactions with absolute value greater than or equal to this.'}, 'max_amount': {'type': 'number', 'description': 'Optional maximum amount limit. Only count transactions with absolute value less than or equal to this.'}}, 'required': []}}}, {'type': 'function', 'function': {'name': 'sum_transactions', 'description': 'Sum the total amount of transactions matching the criteria. Returns negative for expenses, positive for income (like salary). Can filter by category, date range, and amount limit.', 'parameters': {'type': 'object', 'properties': {'category': {'type': 'string', 'description': "The category to filter by (e.g., 'groceries', 'salary', 'entertainment'). If not provided, sums all categories."}, 'start_date': {'type': 'string', 'description': 'Start date in YYYY-MM-DD format. If not provided, uses earliest date in data.'}, 'end_date': {'type': 'string', 'description': 'End date in YYYY-MM-DD format. If not provided, uses latest date in data.'}, 'min_amount': {'type': 'number', 'description': 'Optional minimum amount limit. Only count transactions with absolute value greater than or equal to this.'}, 'max_amount': {'type': 'number', 'description': 'Optional maximum amount limit. Only sum transactions with absolute value less than or equal to this.'}}, 'required': []}}}, {'type': 'function', 'function': {'name': 'compare_periods', 'description': "Compare total spending or income between two time periods. Returns the sum for each period and the difference. Useful for questions like 'did I spend more in Q1 or Q2?'", 'parameters': {'type': 'object', 'properties': {'period1_start': {'type': 'string', 'description': 'Start date of first period in YYYY-MM-DD format'}, 'period1_end': {'type': 'string', 'description': 'End date of first period in YYYY-MM-DD format'}, 'period2_start': {'type': 'string', 'description': 'Start date of second period in YYYY-MM-DD format'}, 'period2_end': {'type': 'string', 'description': 'End date of second period in YYYY-MM-DD format'}, 'category': {'type': 'string', 'description': 'Optional category to filter by. If not provided, compares all transactions.'}}, 'required': ['period1_start', 'period1_end', 'period2_start', 'period2_end']}}}, {'type': 'function', 'function': {'name': 'average_monthly_expenses', 'description': 'Calculate the average monthly expenses over a time period. Only includes expenses (negative amounts), not income. Can optionally filter by category.', 'parameters': {'type': 'object', 'properties': {'start_date': {'type': 'string', 'description': 'Start date in YYYY-MM-DD format'}, 'end_date': {'type': 'string', 'description': 'End date in YYYY-MM-DD format'}, 'category': {'type': 'string', 'description': "Optional category to filter by (e.g., 'groceries', 'entertainment'). If not provided, includes all expense categories."}}, 'required': ['start_date', 'end_date']}}}]


# ============================================================================
# MODEL
# ============================================================================
class DistilLabsLLM(object):
    def __init__(self, model_name: str, api_key: str = "EMPTY", port: int = 11434):
        self.model_name = model_name
        self.client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=api_key)

    def get_prompt(
        self,
        question: str,
    ) -> list[dict[str, str]]:
        tools_string = ""
        for tool in TOOLS:
            tools_string += f"{json.dumps(tool, indent=4)}\n\n"
        
        return [
            {
                "role": "system",
                "content": f"""You have access to the following functions. To call a function, please respond with JSON for a function call.Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}.Do not use variables.

{tools_string}You are a tool-calling model working on:
<task_description>Respond with the next tool call to analyze expenses</task_description>

Solve the task in the 'question' block by generating an appropriate tool call according to the provided tool schema. Generate only the answer, do not generate anything else.""",
            },
            {
                "role": "user",
                "content": f"""Here are examples that show how this task can be solved.
In examples, tasks are in the question XML block, solutions in the answer XML block.
When solving a real task, generate only the answer, do not generate anything else.


<example>
<question>The categories are: salary, food, fun. What's my average monthly income for 2024?</question>
<answer>{{"name": "average_monthly_expenses", "parameters": {{"start_date": "2024-01-01", "end_date": "2024-12-31", "category": "salary"}}}}</answer>
</example>


<question>{question}</question>""",
            },
        ]

    def invoke(self, question: str) -> str:
        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.get_prompt(question),
            temperature=0,
            tools=TOOLS,
        )

        response = chat_response.choices[0].message
        return response.content if len(response.content.strip('\n')) else response.tool_calls[0]
    

# ============================================================================
# PYTHON FUNCTIONS
# ============================================================================

class FinanceTools:
    def __init__(self, csv_path: str):
        """Initialize with path to transactions CSV"""
        self.df = pd.read_csv(csv_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['amount'] = pd.to_numeric(self.df['amount'])
    
    def count_transactions(
        self,
        category: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """Count transactions matching criteria"""
        df = self.df.copy()
        
        # Apply filters
        if category:
            df = df[df['category'].str.lower() == category.lower()]
        
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        if min_amount is not None:
            df = df[df['amount'].abs() >= min_amount]
        
        if max_amount is not None:
            df = df[df['amount'].abs() <= max_amount]
        
        count = len(df)
        
        return {
            "count": count,
            "category": category or "all categories",
            "date_range": f"{start_date or 'beginning'} to {end_date or 'end'}",
            "min_amount_filter": min_amount,
            "max_amount_filter": max_amount
        }
    
    def sum_transactions(
        self,
        category: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """Sum transactions matching criteria"""
        df = self.df.copy()
        
        # Apply filters
        if category:
            df = df[df['category'].str.lower() == category.lower()]
        
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        if min_amount is not None:
            df = df[df['amount'].abs() >= min_amount]
        
        if max_amount is not None:
            df = df[df['amount'].abs() <= max_amount]
        
        total = df['amount'].sum()
        
        return {
            "total": round(total, 2),
            "count": len(df),
            "category": category or "all categories",
            "date_range": f"{start_date or 'beginning'} to {end_date or 'end'}"
        }
    
    def compare_periods(
        self,
        period1_start: str,
        period1_end: str,
        period2_start: str,
        period2_end: str,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare spending/income between two periods"""
        df = self.df.copy()
        
        if category:
            df = df[df['category'].str.lower() == category.lower()]
        
        # Period 1
        period1_df = df[
            (df['date'] >= pd.to_datetime(period1_start)) &
            (df['date'] <= pd.to_datetime(period1_end))
        ]
        period1_total = period1_df['amount'].sum()
        
        # Period 2
        period2_df = df[
            (df['date'] >= pd.to_datetime(period2_start)) &
            (df['date'] <= pd.to_datetime(period2_end))
        ]
        period2_total = period2_df['amount'].sum()
        
        difference = period2_total - period1_total
        
        return {
            "period1": {
                "date_range": f"{period1_start} to {period1_end}",
                "total": round(period1_total, 2),
                "count": len(period1_df)
            },
            "period2": {
                "date_range": f"{period2_start} to {period2_end}",
                "total": round(period2_total, 2),
                "count": len(period2_df)
            },
            "difference": round(difference, 2),
            "period2_vs_period1": "higher" if difference > 0 else "lower" if difference < 0 else "same",
            "category": category or "all categories"
        }
    
    def average_monthly_expenses(
        self,
        start_date: str,
        end_date: str,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate average monthly expenses"""
        df = self.df.copy()
        
        # Only expenses (negative amounts)
        df = df[df['amount'] < 0]
        
        if category:
            df = df[df['category'].str.lower() == category.lower()]
        
        # Filter date range
        df = df[
            (df['date'] >= pd.to_datetime(start_date)) &
            (df['date'] <= pd.to_datetime(end_date))
        ]
        
        # Calculate number of months
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        months = (end.year - start.year) * 12 + (end.month - start.month) + 1
        
        total_expenses = df['amount'].sum()
        avg_monthly = total_expenses / months if months > 0 else 0
        
        return {
            "average_monthly_expense": round(abs(avg_monthly), 2),
            "total_expense": round(abs(total_expenses), 2),
            "months": months,
            "category": category or "all expense categories",
            "date_range": f"{start_date} to {end_date}",
            "transaction_count": len(df)
        }


# ============================================================================
# ORCHESTRATION
# ============================================================================

def execute_tool_call(tools: FinanceTools, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool call and return the result"""
    
    if tool_name == "count_transactions":
        return tools.count_transactions(**arguments)
    elif tool_name == "sum_transactions":
        return tools.sum_transactions(**arguments)
    elif tool_name == "compare_periods":
        return tools.compare_periods(**arguments)
    elif tool_name == "average_monthly_expenses":
        return tools.average_monthly_expenses(**arguments)
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def parse_and_execute(tools: FinanceTools, llm_response: str) -> Any:
    """
    Parse LLM response and execute tool calls.
    Assumes the LLM returns JSON with tool_calls in OpenAI format.
    """
    try:
        print(llm_response)
        # TODO better
        results = []
        if not isinstance(llm_response, str):
            function_name = llm_response.function.name
            arguments = json.loads(llm_response.function.arguments)
            result = execute_tool_call(tools, function_name, arguments)
            results.append({
                "tool": function_name,
                "arguments": arguments,
                "result": result
            })
            return results

        response_data = json.loads(llm_response)
        
        # Handle OpenAI format tool calls
        if "tool_calls" in response_data:
            results = []
            for tool_call in response_data["tool_calls"]:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                result = execute_tool_call(tools, function_name, arguments)
                results.append({
                    "tool": function_name,
                    "arguments": arguments,
                    "result": result
                })
            return results
        
        return {"error": "No tool_calls found in response"}
    
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {e}"}


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="transactions.csv", required=False)
    parser.add_argument("--api-key", type=str, default="EMPTY", required=False)
    parser.add_argument("--model", type=str, default="expense_llama3.2", required=False)
    parser.add_argument("--port", type=int, default=11434, required=False)
    parser.add_argument("--port", type=int, default=11434, required=False)
    parser.add_argument("--json", action="store_true", help="Write out jsons instead of messages.")

    args = parser.parse_args()

    client = DistilLabsLLM(model_name=args.model, api_key=args.api_key, port=args.port)

    # Initialize tools with your CSV
    tools = FinanceTools(args.file)

    text = ''
    while True:
        text = input()
        if text.strip() == "exit":
            break
        api_call = client.invoke(text)
        result = parse_and_execute(tools, api_call)
        print(json.dumps(result, indent=2))

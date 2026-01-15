from typing import Dict, Any

class PromptStrategy:
    """Base class untuk strategi prompting"""
    
    def get_system_prompt(self) -> str:
        """Return system prompt"""
        raise NotImplementedError
        
    def get_user_prompt(self, schema_data: Dict[str, Any], question: str) -> str:
        """Return user prompt"""
        raise NotImplementedError

class ZeroShotStrategy(PromptStrategy):
    """
    Strategi Zero-shot: Langsung minta SQL tanpa contoh.
    Cocok untuk melihat kemampuan raw model.
    """
    
    def get_system_prompt(self) -> str:
        return """You are an expert SQL Assistant. 
Your task is to generate a SQL query to answer the user's question based on the provided database schema.
Return ONLY the SQL query. Do not include markdown or explanations."""

    def get_user_prompt(self, schema_data: Dict[str, Any], question: str) -> str:
        return f"""Database Schema:
{schema_data}

Question: {question}

Generate SQL query:"""

class FewShotStrategy(PromptStrategy):
    """
    Strategi Few-shot: Memberikan contoh input-output.
    Meningkatkan akurasi format dan logic sederhana.
    """
    
    def get_system_prompt(self) -> str:
        return """You are an expert SQL Assistant.
Your task is to generate a SQL query based on the schema and question.
Use the provided examples as a guide for style and format.
Return ONLY the SQL query."""

    def get_user_prompt(self, schema_data: Dict[str, Any], question: str) -> str:
        return f"""Database Schema:
{schema_data}

Examples:

Question: "Show all users"
SQL: SELECT * FROM users

Question: "Count products with price > 100"
SQL: SELECT COUNT(*) FROM products WHERE price > 100

Question: "Show orders for user 'John'"
SQL: SELECT * FROM orders WHERE user_id IN (SELECT id FROM users WHERE name = 'John')

Question: {question}

SQL:"""

class ChainOfThoughtStrategy(PromptStrategy):
    """
    Strategi Chain-of-Thought (CoT): Menyuruh model berpikir step-by-step.
    Meningkatkan akurasi untuk query kompleks (JOIN, Subquery).
    """
    
    def get_system_prompt(self) -> str:
        return """You are an expert SQL Query Generator with 10+ years of experience.

Your role:
- Generate precise, executable SQL queries
- Think step-by-step before answering
- Ensure queries are safe and optimized

Capabilities:
- Complex JOINs, aggregations, subqueries
- Window functions
- Date/Time manipulation"""

    def get_user_prompt(self, schema_data: Dict[str, Any], question: str) -> str:
        return f"""Given the database schema below, generate a SQL query to answer the user's question.

<database_schema>
{schema_data}
</database_schema>

Think step-by-step (Chain-of-Thought):
1. Identify which table(s) are needed
2. Determine required columns
3. Identify filtering conditions (WHERE)
4. Check for aggregations (GROUP BY) or sorting (ORDER BY)

Now generate SQL for this question:
<user_question>
{question}
</user_question>

Output format:
- Start with a clear SELECT statement
- NO markdown code blocks
- NO explanations outside the thinking process (but for this request, just give the SQL or the thinking + SQL if the parser handles it. Best to ask for SQL only at the end to simplify parsing).

IMPORTANT: Return ONLY the SQL query.
SQL Query:"""

def get_strategy(strategy_name: str = "cot") -> PromptStrategy:
    """Factory method untuk mengambil strategy"""
    strategies = {
        "zeroshot": ZeroShotStrategy(),
        "fewshot": FewShotStrategy(),
        "cot": ChainOfThoughtStrategy()
    }
    return strategies.get(strategy_name.lower(), ChainOfThoughtStrategy())

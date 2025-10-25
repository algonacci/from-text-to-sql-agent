import os
import re
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from openai import OpenAI
from langfuse.openai import openai


def show_spinner(message: str):
    """Tampilkan loading message"""
    print(f"⏳ {message}", end="", flush=True)


def load_environment():
    """Load environment variables dari .env file"""
    load_dotenv()


def setup_openai_client():
    """Setup dan return OpenAI client"""
    try:
        base_url = os.getenv("LLM_BASE_URL")
        api_key = os.getenv("LLM_API_KEY")

        if not base_url or not api_key:
            raise ValueError("LLM_BASE_URL dan LLM_API_KEY harus diset di .env file")

        return OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        # prepare Langfuse observability
        # return openai.OpenAI(
        #     base_url=base_url,
        #     api_key=api_key
        # )
    except Exception as e:
        raise RuntimeError(f"Gagal setup OpenAI client: {str(e)}")


def setup_database():
    """Setup database engine dan inspector"""
    try:
        database_url = os.getenv("DATABASE_URL")

        if not database_url:
            raise ValueError("DATABASE_URL harus diset di .env file")

        engine = create_engine(database_url)
        inspector = inspect(engine)

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return engine, inspector
    except Exception as e:
        raise RuntimeError(f"Gagal koneksi ke database: {str(e)}")


def clean_sql(generated_text: str) -> str:
    """
    Hilangkan markdown SQL block dari generated text

    Args:
        generated_text: Text yang dihasilkan oleh LLM

    Returns:
        SQL query yang sudah dibersihkan
    """
    cleaned = re.sub(r"```sql|```", "", generated_text, flags=re.IGNORECASE).strip()
    return cleaned


def classify_intent(client, question: str) -> str:
    """
    Klasifikasi intent dari pertanyaan user

    Args:
        client: OpenAI client instance
        question: Pertanyaan dari user

    Returns:
        Intent yang terdeteksi ('schema_info' atau 'query')
    """
    try:
        # Structured prompt with role, persona, CoT, and few-shot examples
        system_prompt = """You are an expert SQL Assistant with deep understanding of database queries and schema information.

Your task: Classify user questions into two categories:
1. "query" - User wants to retrieve data from database
2. "schema_info" - User wants information about database structure

Think step-by-step:
- Analyze the question intent
- Identify keywords related to data retrieval vs. structure inquiry
- Output ONLY the classification label"""

        user_prompt = f"""Classify the following question into either "query" or "schema_info".

Examples for reference:

Example 1:
Question: "Ada apa saja tabel di database?"
Analysis: User is asking about database structure (tables)
Classification: schema_info

Example 2:
Question: "Tampilkan 10 users pertama"
Analysis: User wants to retrieve data (users)
Classification: query

Example 3:
Question: "Berapa jumlah kolom di tabel products?"
Analysis: User is asking about table structure (columns)
Classification: schema_info

Example 4:
Question: "Cari semua orders dengan total > 1000000"
Analysis: User wants to retrieve specific data (orders with condition)
Classification: query

Now classify this question:
Question: "{question}"

Output format: Return ONLY one word - either "query" or "schema_info"
If unsure, default to "query"

Classification:"""

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Lower temperature for consistent classification
            max_tokens=50  # We only need one word
        )

        # save for gemini
        # result = response.choices[0].message
        result = response.choices[0].message.content

        # Fallback: ensure valid output
        if result not in ["query", "schema_info"]:
            return "query"  # Default to query if output is invalid

        return result
    except Exception as e:
        raise RuntimeError(f"Gagal klasifikasi intent: {str(e)}")


def get_database_schema(inspector):
    """
    Ambil struktur schema dari database

    Args:
        inspector: SQLAlchemy inspector instance

    Returns:
        Dictionary berisi informasi schema database
    """
    result_data = {
        "total_tables": 0,
        "tables": {}
    }

    table_names = inspector.get_table_names()
    result_data["total_tables"] = len(table_names)

    for table_name in table_names:
        table_data = {
            "columns": [],
            "primary_keys": [],
            "foreign_keys": [],
            "indexes": []
        }

        # Columns
        for col in inspector.get_columns(table_name):
            table_data["columns"].append({
                "name": col['name'],
                "type": str(col['type']),
                "nullable": col['nullable'],
                "default": col['default']
            })

        # Primary keys
        table_data["primary_keys"] = inspector.get_pk_constraint(table_name)['constrained_columns']

        # Foreign keys
        table_data["foreign_keys"] = [{
            "constrained_columns": fk['constrained_columns'],
            "referred_table": fk['referred_table'],
            "referred_columns": fk['referred_columns']
        } for fk in inspector.get_foreign_keys(table_name)]

        # Indexes
        table_data["indexes"] = [{
            "name": idx['name'],
            "columns": idx['column_names'],
            "unique": idx['unique']
        } for idx in inspector.get_indexes(table_name)]

        result_data["tables"][table_name] = table_data

    return result_data


def generate_sql_query(client, schema_data, question: str) -> str:
    """
    Generate SQL query dari pertanyaan user

    Args:
        client: OpenAI client instance
        schema_data: Dictionary schema database
        question: Pertanyaan dari user

    Returns:
        SQL query yang dihasilkan
    """
    try:
        # Structured prompt with role, persona, CoT, few-shot examples, and explicit instructions
        system_prompt = """You are an expert SQL Query Generator with 10+ years of experience in database design and optimization.

Your role:
- Generate precise, executable SQL queries based on user questions
- Ensure queries are safe (SELECT only), efficient, and follow best practices
- Never include explanations, markdown formatting, or additional text

Capabilities:
- Support for complex JOINs, aggregations, and subqueries
- Proper use of WHERE, GROUP BY, HAVING, ORDER BY clauses
- Data type awareness and implicit casting
- Query optimization for performance"""

        user_prompt = f"""Given the database schema below, generate a SQL query to answer the user's question.

<database_schema>
{schema_data}
</database_schema>

Think step-by-step (Chain-of-Thought):
1. Identify which table(s) are needed
2. Determine required columns
3. Identify any filtering conditions (WHERE)
4. Check if aggregation is needed (GROUP BY, COUNT, SUM, etc.)
5. Determine sorting requirements (ORDER BY)
6. Apply any limits if specified

Few-shot examples:

Example 1:
Question: "Tampilkan 10 users pertama"
Reasoning: Need users table, all columns, limit to 10 rows
SQL: SELECT * FROM users LIMIT 10

Example 2:
Question: "Berapa total users dengan email gmail?"
Reasoning: Need users table, filter by email pattern, count rows
SQL: SELECT COUNT(*) FROM users WHERE email LIKE '%@gmail.com%'

Example 3:
Question: "Tampilkan nama dan total orders per user, urutkan dari terbesar"
Reasoning: Need JOIN between users and orders, aggregate count, sort descending
SQL: SELECT u.name, COUNT(o.id) as total_orders FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name ORDER BY total_orders DESC

Now generate SQL for this question:
<user_question>
{question}
</user_question>

Output format requirements:
Return ONLY the SQL query
NO explanations, NO markdown blocks (```), NO extra text
Query MUST start with SELECT
Use proper SQL syntax and formatting
Ensure query is executable

If the question is unclear or impossible:
- Return: SELECT 'Error: Unable to generate query - question unclear' as message

SQL Query:"""

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Low temp for consistent SQL generation
            max_tokens=1000
        )

        query = response.choices[0].message.content.strip()
        print(f"[DEBUG] Raw LLM response: {repr(query[:200])}")  # Print first 200 chars

        # Multiple cleaning attempts
        # 1. Try clean_sql function
        query_cleaned = clean_sql(query)
        print(f"[DEBUG] After clean_sql: {repr(query_cleaned[:200])}")

        # 2. Try to find SELECT statement anywhere in the text
        import re
        # Pattern to match SELECT queries (case insensitive, multiline)
        select_pattern = r'(SELECT\s+.*?)(?:;|\Z)'
        match = re.search(select_pattern, query_cleaned, re.IGNORECASE | re.DOTALL)

        if match:
            extracted_query = match.group(1).strip()
            print(f"[DEBUG] Extracted query via regex: {repr(extracted_query[:200])}")
            return extracted_query

        # 3. Line-by-line search for SELECT
        if not query_cleaned.upper().startswith("SELECT"):
            print(f"[DEBUG] Query doesn't start with SELECT, trying line-by-line...")
            lines = query_cleaned.split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.upper().startswith("SELECT"):
                    # Try to get multi-line SELECT if needed
                    remaining_lines = '\n'.join(lines[i:])
                    select_match = re.search(select_pattern, remaining_lines, re.IGNORECASE | re.DOTALL)
                    if select_match:
                        result = select_match.group(1).strip()
                        print(f"[DEBUG] Found SELECT starting at line {i}: {repr(result[:200])}")
                        return result
                    else:
                        print(f"[DEBUG] Found SELECT at line {i}: {repr(stripped[:200])}")
                        return stripped

            # If no valid SELECT found, return error query
            print("[DEBUG] No valid SELECT statement found in response")
            print(f"[DEBUG] Full cleaned response: {repr(query_cleaned)}")
            return "SELECT 'Error: Invalid query generated' as error_message"

        return query_cleaned
    except Exception as e:
        raise RuntimeError(f"Gagal generate SQL query: {str(e)}")


def generate_schema_info(client, schema_data, question: str) -> str:
    """
    Generate informasi schema dari pertanyaan user

    Args:
        client: OpenAI client instance
        schema_data: Dictionary schema database
        question: Pertanyaan dari user

    Returns:
        Informasi schema yang dihasilkan
    """
    try:
        # Structured prompt with role, persona, CoT, few-shot examples
        system_prompt = """You are a Database Schema Expert and Documentation Specialist with deep knowledge of database design patterns.

Your role:
- Provide clear, concise information about database structure
- Explain tables, columns, relationships, and constraints
- Help users understand the database architecture
- Present information in an organized, easy-to-read format

Communication style:
- Concise and direct
- Use bullet points for clarity
- Highlight key information
- Professional tone"""

        user_prompt = f"""Given the database schema below, answer the user's question about the database structure.

<database_schema>
{schema_data}
</database_schema>

Think step-by-step (Chain-of-Thought):
1. Identify what schema information is being requested
2. Extract relevant details from the schema
3. Organize information clearly
4. Present in a user-friendly format

Few-shot examples:

Example 1:
Question: "Ada apa saja tabel di database?"
Analysis: User wants list of all tables
Response:
Database memiliki 3 tabel:
1. users - Menyimpan informasi pengguna
2. products - Menyimpan data produk
3. orders - Menyimpan transaksi pesanan

Example 2:
Question: "Apa saja kolom di tabel users?"
Analysis: User wants columns in users table
Response:
Tabel 'users' memiliki kolom berikut:
- id (Primary Key, Integer)
- name (String, Not Null)
- email (String, Unique, Not Null)
- created_at (Timestamp)

Example 3:
Question: "Bagaimana relasi antar tabel?"
Analysis: User wants foreign key relationships
Response:
Relasi antar tabel:
- orders.user_id → users.id (Many-to-One)
- orders.product_id → products.id (Many-to-One)

Now answer this question:
<user_question>
{question}
</user_question>

Output format requirements:
Provide concise, direct answer
Use bullet points or numbered lists for clarity
Highlight important information (primary keys, foreign keys, constraints)
Keep response under 200 words unless detailed explanation needed

If the question is unclear:
- Provide general overview of database structure
- List available tables and brief descriptions

Response:"""

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Moderate temp for natural language
            max_tokens=500
        )

        info = response.choices[0].message.content.strip()

        # Fallback: If response is empty or too short
        if not info or len(info) < 10:
            return f"Database memiliki {schema_data.get('total_tables', 0)} tabel. Silakan ajukan pertanyaan spesifik tentang struktur database."

        return info
    except Exception as e:
        raise RuntimeError(f"Gagal generate schema info: {str(e)}")


def execute_select_query(engine, sql_query: str):
    """
    Execute SELECT query dan return hasilnya

    Args:
        engine: SQLAlchemy engine instance
        sql_query: SQL query yang akan dieksekusi

    Returns:
        Tuple (success: bool, result: DataFrame or error message)
    """
    # Validasi query
    if not sql_query.lower().startswith("select"):
        return False, "Hanya query SELECT yang boleh dieksekusi."

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()

            if rows:
                # Konversi ke DataFrame
                df = pd.DataFrame(rows, columns=result.keys())
                return True, df
            else:
                return True, "Query berhasil, tapi tidak ada hasil."
    except Exception as e:
        return False, f"Error: {str(e)}"

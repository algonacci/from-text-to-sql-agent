import os
import re
import sys
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from openai import OpenAI


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
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME"),
            messages=[
                {
                    "role": "system",
                    "content": "Kamu adalah SQL assistant yang sangat mahir."
                },
                {
                    "role": "user",
                    "content": f"""
        1. schema_info
        Informasi mengenai schema database.

        2. query
        Pertanyaan dari user.

        Contoh 1:
        Q: Ada apa saja tabel di database?
        A: schema_info

        Contoh 2:
        Q: Tampilkan 10 users pertama
        A: query

        Klasifikasi jenis pertanyaan berikut:

        <question>
        {question}
        </question>

        Harap keluarkan hasil klasifikasinya saja, tanpa penjelasan
        Klasifikasi:
        """
                }
            ],
            temperature=0.3,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", 4096))
        )
        return response.choices[0].message.content.strip()
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
        prompt = f"""
Struktur tabel:
{schema_data}

Pertanyaan user:
"{question}"

HASIL YANG DIINGINKAN:
- Berikan HANYA SQL query yang dapat dieksekusi langsung
- JANGAN berikan penjelasan, deskripsi, atau teks tambahan
- JANGAN gunakan markdown code block (```)
- Pastikan query dimulai dengan SELECT
- Query harus valid dan dapat dijalankan

Contoh output yang benar:
SELECT * FROM users LIMIT 10
    """

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME"),
            messages=[
                {
                    "role": "system",
                    "content": "Kamu adalah SQL assistant yang sangat mahir."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()
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
        prompt = f"""
Struktur tabel:
{schema_data}

Pertanyaan user:
"{question}"

Jawab dengan format singkat dan langsung, tanpa penjelasan tambahan.
Berikan informasi deskriptif tentang struktur database.
    """

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME"),
            messages=[
                {
                    "role": "system",
                    "content": "Kamu adalah SQL assistant yang sangat mahir."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()
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

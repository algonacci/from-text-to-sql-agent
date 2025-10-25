import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request
from sqlalchemy import create_engine, inspect, text
import helpers

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Untuk session management

# Global variables untuk menyimpan client, engine, dan schema
client = None
engine = None
inspector = None
schema_data = None
current_db_url = None

def initialize_openai_client():
    """Initialize OpenAI client only"""
    global client

    try:
        client = helpers.setup_openai_client()
        print(f"✅ OpenAI client berhasil diinisialisasi!")
        return True
    except Exception as e:
        print(f"❌ Error saat inisialisasi OpenAI client: {str(e)}")
        return False

def connect_database(database_url=None):
    """Connect to database with given URL or from .env"""
    global engine, inspector, schema_data, current_db_url

    try:
        # Gunakan database_url yang diberikan atau dari .env
        if not database_url:
            database_url = os.getenv("DATABASE_URL")

        if not database_url:
            raise ValueError("DATABASE_URL tidak ditemukan")

        # Create engine dan inspector
        engine = create_engine(database_url)
        inspector = inspect(engine)

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        # Load database schema
        schema_data = helpers.get_database_schema(inspector)
        current_db_url = database_url

        print(f"✅ Database connected: {schema_data['total_tables']} tables found")
        return True, None

    except Exception as e:
        error_msg = f"Gagal koneksi ke database: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg

def build_database_url(db_type, host, user, password, db_name):
    """Build database URL dari komponen"""
    driver_map = {
        "mysql": "mysql+pymysql",
        "postgresql": "postgresql+psycopg2",
        "mariadb": "mysql+pymysql"
    }

    driver = driver_map.get(db_type, "mysql+pymysql")
    return f"{driver}://{user}:{password}@{host}/{db_name}"

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route - GET untuk tampilan, POST untuk query"""
    try:
        # Get default DB URL from .env
        default_db_url = os.getenv("DATABASE_URL", "")

        # Handle GET request - tampilkan halaman
        if request.method == "GET":
            return render_template("index.html",
                                 current_db_url=current_db_url or default_db_url,
                                 is_connected=schema_data is not None,
                                 total_tables=schema_data['total_tables'] if schema_data else 0)

        # Handle POST request - process query
        print("\n" + "="*50)
        print("🔍 Processing new query...")

        # Check if database is connected
        if schema_data is None:
            print("❌ Database not connected")
            return render_template("index.html",
                                 current_db_url=current_db_url or default_db_url,
                                 is_connected=False,
                                 error="Silakan hubungkan ke database terlebih dahulu!")

        # Ambil pertanyaan dari form
        question = request.form.get("question", "").strip()
        print(f"❓ Question: {question}")

        if not question:
            print("❌ Empty question")
            return render_template("index.html",
                                 current_db_url=current_db_url,
                                 is_connected=True,
                                 total_tables=schema_data['total_tables'],
                                 error="Pertanyaan tidak boleh kosong!")

        # Klasifikasi intent
        print("🔄 Classifying intent...")
        intent = helpers.classify_intent(client, question)
        print(f"✅ Intent: {intent}")

        # Handle berdasarkan intent
        if intent.lower() == "query":
            print("🔄 Handling query intent...")
            result = handle_query_intent(question)
            print("✅ Query intent handled")
        elif intent.lower() == "schema_info":
            print("🔄 Handling schema_info intent...")
            result = handle_schema_info_intent(question)
            print("✅ Schema info intent handled")
        else:
            print(f"⚠️  Unknown intent: {intent}")
            result = {
                "intent": intent,
                "error": "Jenis pertanyaan tidak dikenali"
            }

        print("✅ Rendering response...")
        return render_template("index.html",
                             current_db_url=current_db_url,
                             is_connected=True,
                             total_tables=schema_data['total_tables'],
                             question=question,
                             intent=intent,
                             result=result)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template("index.html",
                             current_db_url=current_db_url or default_db_url,
                             is_connected=schema_data is not None,
                             total_tables=schema_data['total_tables'] if schema_data else 0,
                             error=f"Terjadi kesalahan: {str(e)}")

@app.route("/connect", methods=["POST"])
def connect():
    """Connect to database using form input"""
    try:
        db_type = request.form.get("db_type", "mysql")
        host = request.form.get("host", "")
        user = request.form.get("user", "")
        password = request.form.get("password", "")
        db_name = request.form.get("db_name", "")

        # Build database URL
        database_url = build_database_url(db_type, host, user, password, db_name)

        # Try to connect
        success, error_msg = connect_database(database_url)

        if success:
            return render_template("index.html",
                                 current_db_url=current_db_url,
                                 is_connected=True,
                                 total_tables=schema_data['total_tables'],
                                 success_message=f"Berhasil terhubung ke database! Ditemukan {schema_data['total_tables']} tabel.")
        else:
            return render_template("index.html",
                                 current_db_url=database_url,
                                 is_connected=False,
                                 error=error_msg)

    except Exception as e:
        return render_template("index.html",
                             error=f"Terjadi kesalahan: {str(e)}")

def handle_query_intent(question: str):
    """Handle intent 'query' - generate dan execute SQL query"""
    try:
        # Generate SQL query
        print("🔄 Generating SQL query...")
        openai_output = helpers.generate_sql_query(client, schema_data, question)
        print(f"📝 Raw LLM output: {openai_output}")

        # Clean dan execute query
        sql_query = helpers.clean_sql(openai_output)
        success, result = helpers.execute_select_query(engine, sql_query)

        response = {
            "type": "query",
            "sql": sql_query,
            "success": success
        }

        if success:
            if isinstance(result, str):
                # Tidak ada hasil
                response["message"] = result
                response["data"] = None
            else:
                # Ada hasil, convert DataFrame ke HTML table
                response["data"] = result.to_html(index=False, classes="table")
                response["row_count"] = len(result)
        else:
            # Error
            response["error"] = result

        return response

    except Exception as e:
        return {
            "type": "query",
            "error": f"Error saat memproses query: {str(e)}"
        }

def handle_schema_info_intent(question: str):
    """Handle intent 'schema_info' - tampilkan informasi schema"""
    try:
        # Generate schema info
        schema_info = helpers.generate_schema_info(client, schema_data, question)

        return {
            "type": "schema_info",
            "info": schema_info
        }

    except Exception as e:
        return {
            "type": "schema_info",
            "error": f"Error saat mengambil info schema: {str(e)}"
        }

if __name__ == "__main__":
    # Initialize OpenAI client
    if initialize_openai_client():
        print("\n🚀 Starting Flask server...")
        print("💡 Anda bisa menggunakan database default dari .env atau input manual di web interface")

        # Try to connect to default database from .env (optional)
        default_db = os.getenv("DATABASE_URL")
        if default_db:
            print("\n🔌 Mencoba koneksi ke database default...")
            success, error = connect_database(default_db)
            if not success:
                print(f"⚠️  Gagal koneksi otomatis: {error}")
                print("💡 Silakan input database connection secara manual di web interface")

        app.run(debug=True)
    else:
        print("❌ Gagal menjalankan aplikasi. Silakan periksa konfigurasi OpenAI Anda.")
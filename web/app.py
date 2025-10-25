import pandas as pd
from flask import Flask, request, jsonify, render_template

from helpers import (
    load_environment,
    setup_openai_client,
    setup_database,
    get_database_schema,
    classify_intent,
    generate_sql_query,
    generate_schema_info,
    clean_sql,
    execute_select_query
)

app = Flask(__name__)

# ============================================================================
# GLOBAL VARIABLES FOR CACHING
# ============================================================================
client = None
engine = None
inspector = None
schema_data = None


def initialize_app():
    """Initialize OpenAI client, database connection, and load schema"""
    global client, engine, inspector, schema_data

    load_environment()
    client = setup_openai_client()
    engine, inspector = setup_database()
    schema_data = get_database_schema(inspector)

    print(f"✅ App initialized: {schema_data['total_tables']} tables found")


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render halaman utama"""
    return render_template('index.html')


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    API endpoint untuk memproses pertanyaan user

    Request JSON:
        {
            "question": "string"
        }

    Response JSON:
        {
            "success": bool,
            "intent": "query" | "schema_info",
            "sql_query": "string" (jika intent = query),
            "result": array | string,
            "message": "string"
        }
    """
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({
                "success": False,
                "message": "Field 'question' diperlukan"
            }), 400

        question = data['question'].strip()

        if not question:
            return jsonify({
                "success": False,
                "message": "Pertanyaan tidak boleh kosong"
            }), 400

        # Klasifikasi intent
        intent = classify_intent(client, question)

        # Handle berdasarkan intent
        if intent.lower() == "query":
            # Generate SQL query
            openai_output = generate_sql_query(client, schema_data, question)
            sql_query = clean_sql(openai_output)

            # Execute query
            success, result = execute_select_query(engine, sql_query)

            if success:
                if isinstance(result, pd.DataFrame):
                    # Convert DataFrame to dict
                    result_data = result.to_dict('records')
                    return jsonify({
                        "success": True,
                        "intent": intent,
                        "sql_query": sql_query,
                        "result": result_data,
                        "row_count": len(result_data),
                        "message": "Query berhasil dijalankan"
                    })
                else:
                    # No results
                    return jsonify({
                        "success": True,
                        "intent": intent,
                        "sql_query": sql_query,
                        "result": [],
                        "message": result
                    })
            else:
                # Error executing query
                return jsonify({
                    "success": False,
                    "intent": intent,
                    "sql_query": sql_query,
                    "message": result
                }), 400

        elif intent.lower() == "schema_info":
            # Generate schema info
            schema_info = generate_schema_info(client, schema_data, question)

            return jsonify({
                "success": True,
                "intent": intent,
                "result": schema_info,
                "message": "Informasi schema berhasil diperoleh"
            })

        else:
            return jsonify({
                "success": False,
                "message": "Intent tidak dikenali"
            }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "tables": schema_data['total_tables'] if schema_data else 0
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Initialize app
    initialize_app()

    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

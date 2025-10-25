import sys
from helpers import (
    load_environment,
    setup_openai_client,
    setup_database,
    get_database_schema,
    classify_intent,
    generate_sql_query,
    generate_schema_info,
    clean_sql,
    execute_select_query,
    show_spinner,
)


def main():
    """Main function untuk menjalankan SQL Agent"""

    try:
        # Setup environment dan services
        load_environment()
        client = setup_openai_client()
        engine, inspector = setup_database()

        # Ambil schema database
        show_spinner("Memuat schema database...")
        schema_data = get_database_schema(inspector)
        print(f"\r✅ Schema loaded: {schema_data['total_tables']} tables found\n")

        # Main loop
        while True:
            try:
                # Input pertanyaan
                user_input = input(
                    "\n💬 Masukkan pertanyaan ('exit' untuk keluar): "
                ).strip()

                if not user_input:
                    continue

                if user_input.lower() == "exit":
                    print("\n👋 Terima kasih telah menggunakan SQL Agent!")
                    break

                # Klasifikasi intent
                show_spinner("Menganalisis pertanyaan...")
                intent = classify_intent(client, user_input)
                print(f"\r🔍 Intent terdeteksi: '{intent}'")

                # Handle berdasarkan intent
                if intent.lower() == "query":
                    handle_query_intent(client, engine, schema_data, user_input)
                elif intent.lower() == "schema_info":
                    handle_schema_info_intent(client, schema_data, user_input)
                else:
                    print("⚠️ Maaf, jenis pertanyaan tidak dikenali.")

            except KeyboardInterrupt:
                print("\n\n👋 Keluar dari SQL Agent. Terima kasih!")
                sys.exit(0)
            except EOFError:
                print("\n\n👋 Keluar dari SQL Agent. Terima kasih!")
                sys.exit(0)
            except Exception as e:
                print(f"\n❌ Terjadi kesalahan: {str(e)}")
                print("💡 Silakan coba lagi atau ketik 'exit' untuk keluar.\n")

    except KeyboardInterrupt:
        print("\n\n👋 Keluar dari SQL Agent. Terima kasih!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error fatal saat startup: {str(e)}")
        sys.exit(1)


def handle_query_intent(client, engine, schema_data, question: str):
    """
    Handle intent 'query' - generate dan execute SQL query

    Args:
        client: OpenAI client instance
        engine: SQLAlchemy engine instance
        schema_data: Dictionary schema database
        question: Pertanyaan dari user
    """
    try:
        # Generate SQL query
        show_spinner("Membuat SQL query...")
        openai_output = generate_sql_query(client, schema_data, question)
        print(f"\r✅ SQL query berhasil dibuat")
        print("\n📝 Generated Query:")
        print(f"   {openai_output}\n")

        # Clean dan execute query
        sql_query = clean_sql(openai_output)
        show_spinner("Menjalankan query...")
        success, result = execute_select_query(engine, sql_query)

        if success:
            print("\r✅ Query berhasil dijalankan")
            if isinstance(result, str):
                # Tidak ada hasil
                print(f"\n💡 {result}")
            else:
                # Ada hasil, tampilkan DataFrame
                print(f"\n📊 Hasil Query ({len(result)} rows):")
                print(result.to_string(index=False))
        else:
            # Error
            print(f"\r❌ {result}")
    except Exception as e:
        print(f"\n❌ Error saat memproses query: {str(e)}")


def handle_schema_info_intent(client, schema_data, question: str):
    """
    Handle intent 'schema_info' - tampilkan informasi schema

    Args:
        client: OpenAI client instance
        schema_data: Dictionary schema database
        question: Pertanyaan dari user
    """
    try:
        # Generate schema info
        show_spinner("Mengambil informasi schema...")
        schema_info = generate_schema_info(client, schema_data, question)
        print("\r✅ Informasi berhasil diperoleh")
        print(f"\n📚 Informasi Schema:")
        print(f"{schema_info}")
    except Exception as e:
        print(f"\n❌ Error saat mengambil info schema: {str(e)}")


if __name__ == "__main__":
    main()

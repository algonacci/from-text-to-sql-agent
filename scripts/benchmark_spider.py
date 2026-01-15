import json
import os
import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from tqdm import tqdm

# Add current directory to path to import helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_helpers import setup_openai_client, get_database_schema, generate_sql_query, execute_select_query

def compare_results(df1, df2):
    """
    Compare two DataFrames strictly.
    Returns True if they contain the same data.
    """
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        return False
        
    if df1.shape != df2.shape:
        return False
        
    # Sort and reset index to ensure order doesn't matter
    # Note: This is a simplified comparison. 
    # For robust Spider evaluation, we usually use the official evaluation script.
    # But this serves as a good proxy for "Execution Accuracy".
    
    try:
        # Sort by all columns
        df1_sorted = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
        # Rename columns of df2 to match df1 for value comparison (ignoring column alias differences)
        df2.columns = df1.columns 
        df2_sorted = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)
        
        return df1_sorted.equals(df2_sorted)
    except Exception as e:
        # Fallback if sorting fails (e.g. mixed types)
        return False

def run_benchmark(spider_path, output_file, limit=None, strategy="cot"):
    print(f"🚀 Starting Spider Benchmark")
    print(f"📂 Data Path: {spider_path}")
    print(f"🧠 Strategy: {strategy}")
    
    # Load dev.json
    dev_json_path = os.path.join(spider_path, "dev.json")
    if not os.path.exists(dev_json_path):
        print(f"❌ Error: dev.json not found at {dev_json_path}")
        return

    with open(dev_json_path, 'r') as f:
        data = json.load(f)

    if limit:
        data = data[:limit]
        print(f"⚠️ Limiting to first {limit} examples")

    client = setup_openai_client()
    results = []
    correct_count = 0
    
    # Create progress bar
    pbar = tqdm(data, desc="Evaluating")
    
    for item in pbar:
        db_id = item['db_id']
        question = item['question']
        gold_sql = item['query']
        
        # Connect to Spider DB
        db_path = os.path.join(spider_path, "database", db_id, f"{db_id}.sqlite")
        
        if not os.path.exists(db_path):
            error_msg = f"DB not found: {db_path}"
            results.append({
                "question": question,
                "gold_sql": gold_sql,
                "generated_sql": "",
                "status": "error",
                "error": error_msg
            })
            continue

        try:
            # Setup DB specific for this question
            engine = create_engine(f"sqlite:///{db_path}")
            inspector = inspect(engine)
            schema = get_database_schema(inspector)
            
            # Generate SQL
            gen_sql = generate_sql_query(client, schema, question, strategy=strategy)
            
            # Execute Gold (Truth)
            success_gold, res_gold = execute_select_query(engine, gold_sql)
            
            # Execute Generated (Prediction)
            success_gen, res_gen = execute_select_query(engine, gen_sql)
            
            is_correct = False
            error_detail = None
            
            if success_gold and success_gen:
                is_correct = compare_results(res_gold, res_gen)
                status = "correct" if is_correct else "incorrect_result"
            elif not success_gen:
                status = "execution_error"
                error_detail = str(res_gen)
            else:
                status = "gold_error" # Should not happen typically
            
            if is_correct:
                correct_count += 1
                
            results.append({
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "generated_sql": gen_sql,
                "status": status,
                "error": error_detail,
                "feedback": "Match" if is_correct else "Mismatch" 
            })
            
            # Update pbar
            pbar.set_postfix({"acc": f"{correct_count/len(results):.2%}"})
            
        except Exception as e:
            results.append({
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "generated_sql": "",
                "status": "system_error",
                "error": str(e)
            })

    # Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    
    accuracy = correct_count / len(data) if data else 0
    print(f"\n📊 Benchmark Finished!")
    print(f"✅ Accuracy: {accuracy:.2%} ({correct_count}/{len(data)})")
    print(f"💾 Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spider Benchmark")
    parser.add_argument("--spider_path", required=True, help="Path to Spider dataset folder (containing dev.json and database/)")
    parser.add_argument("--output", default="benchmark_results.csv", help="Output CSV file")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--strategy", default="cot", choices=["zeroshot", "fewshot", "cot"], help="Prompting strategy")
    
    args = parser.parse_args()
    
    run_benchmark(args.spider_path, args.output, args.limit, args.strategy)

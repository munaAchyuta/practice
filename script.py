# =============================================
# SchemaGraphSQL MVP for Health Insurance Claims
# File: schema_graph_mvp.py
# =============================================

import json
import networkx as nx
from google import genai
import sqlglot
import os
from typing import List, Dict, Any
import dotenv

dotenv.load_dotenv()

# ============== CONFIG ==============
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

MODEL_COARSE = "gemini-2.5-flash"   # very cheap & fast
MODEL_SQL    = "gemini-3.1-flash-lite-preview"     # powerful model for SQL

JSON_FILE = "sample_data.json"

# =====================================
# 1. BUILD GRAPH FROM JSON (NetworkX)
# =====================================
def build_schema_graph() -> nx.Graph:
    with open(JSON_FILE) as f:
        data = json.load(f)

    G = nx.Graph()

    # Add every table as node + metadata (for later prompt)
    for table in data["tables"]:
        G.add_node(
            table["name"],
            description=table.get("description", ""),
            dimensions=table.get("dimensions", []),
            primary_key=table.get("primary_key", "")
        )

    # Add edges from relationships (exactly as you shared)
    for rel in data.get("relationships", []):
        from_table = rel["from"].split(".")[0]
        to_table   = rel["to"].split(".")[0]
        if from_table in G and to_table in G:
            G.add_edge(from_table, to_table,
                       from_col=rel["from"],
                       to_col=rel["to"])

    print(f"✅ Graph built: {G.number_of_nodes()} tables, {G.number_of_edges()} relationships")
    return G, data  # return graph + raw data for later filtering


# =====================================
# 2. LLM — Coarse Entity Extraction (src / dst tables)
# =====================================
def extract_src_dst(query: str, all_table_names: List[str]) -> Dict[str, List[str]]:
    prompt = f"""You are a senior data engineer for health insurance databases.

Query: {query}

Available tables (use ONLY these names): {', '.join(all_table_names)}

Task (one-shot):
- src = tables that contain filter/condition columns (WHERE, GROUP BY, HAVING, JOIN keys in conditions)
- dst = tables that contain output columns (SELECT)

Return ONLY valid JSON:
{{"src": ["table1", "table2"], "dst": ["table3"]}}

Do not explain. Do not add extra text."""

    response = client.models.generate_content(
        model=MODEL_COARSE,
        contents=prompt,
    )
    try:
        text = response.text.strip()
        if text.startswith("```json"):
            text = text.split("```json")[1].split("```")[0].strip()
        return json.loads(text)
    except:
        return {"src": [], "dst": []}  # fallback


# =====================================
# 3. Graph Pathfinding + Union All Paths
# =====================================
def get_relevant_tables_and_paths(G: nx.Graph, src_list: List[str], dst_list: List[str]):
    relevant_tables = set()
    all_paths = []

    for s in src_list:
        for d in dst_list:
            if nx.has_path(G, s, d):
                for path in nx.all_shortest_paths(G, s, d):
                    all_paths.append(path)
                    relevant_tables.update(path)

    return list(relevant_tables), all_paths


# =====================================
# 4. Post-processing & Column Filtering
# =====================================
def build_filtered_schema(data: Dict, relevant_tables: List[str], paths: List[List[str]]):
    filtered = [t for t in data["tables"] if t["name"] in relevant_tables]

    # Simple string for LLM prompt
    schema_str = "Relevant tables and columns:\n"
    for t in filtered:
        cols = [c["name"] for c in t.get("dimensions", [])]
        schema_str += f"Table: {t['name']} ({t.get('description','')})\nColumns: {cols}\n\n"

    join_str = "Join paths discovered:\n" + "\n".join([" → ".join(p) for p in paths])
    return schema_str + "\n" + join_str


# =====================================
# 5. LLM — Final SQL Generation
# =====================================
def generate_sql(query: str, filtered_schema: str, business_prompt: str = ""):
    prompt = f"""You are an expert SQL developer for US health insurance (claims, benefits, fee schedules, procedure/revenue codes).

User Query: {query}

{filtered_schema}

{business_prompt}

Rules:
- Use only the tables and columns above.
- Use the exact join paths shown.
- Write clean, executable SQL (PostgreSQL / SQL Server dialect).
- Use proper JOIN syntax.
- Add comments if helpful.

Return ONLY the SQL query (no explanation)."""

    response = client.models.generate_content(
        model=MODEL_SQL,
        contents=prompt,
    )
    sql = response.text.strip()
    # Clean markdown if any
    if sql.startswith("```sql"):
        sql = sql.split("```sql")[1].split("```")[0].strip()
    return sql


# =====================================
# 6. sqlglot Parser + Self-Correction
# =====================================
def validate_and_fix_sql(sql: str, max_tries=2):
    for attempt in range(max_tries):
        try:
            sqlglot.parse_one(sql, read="postgres")  # change to "tsql" if SQL Server
            return sql, True
        except Exception as e:
            print(f"❌ Parse error (attempt {attempt+1}): {e}")
            # Very simple self-correction prompt
            fix_prompt = f"Fix this SQL (only return the corrected query):\n{sql}\nError: {str(e)}"
            model = genai.GenerativeModel(MODEL_COARSE)
            response = model.generate_content(
                fix_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0
                )
            )
            sql = response.text.strip()
    return sql, False


# =====================================
# MAIN — End-to-End Test
# =====================================
if __name__ == "__main__":
    G, raw_data = build_schema_graph()

    user_query = "Show me the average claim denial rate for CPT code 99213 by provider across California and New York states last year."

    # 2. Coarse extraction
    table_names = [t["name"] for t in raw_data["tables"]]
    entities = extract_src_dst(user_query, table_names)
    print("🔍 Coarse entities:", entities)

    # 3+4. Graph + filtering
    relevant_tables, paths = get_relevant_tables_and_paths(G, entities["src"], entities["dst"])
    filtered_schema = build_filtered_schema(raw_data, relevant_tables, paths)
    print(f"📊 Relevant tables found: {len(relevant_tables)}")

    # 5. Generate SQL
    business_fewshot = """Business rules:
- Always join on provider ID when provider data is needed.
- Use accumid for benefit-accumulator linking.
- Fee schedules use feeid."""

    sql = generate_sql(user_query, filtered_schema, business_fewshot)
    print("\n📝 Generated SQL:\n", sql)

    # 6. Validate
    final_sql, success = validate_and_fix_sql(sql)
    print("\n✅ Final SQL (valid):" if success else "\n⚠️  Final SQL (after fixes):")
    print(final_sql)
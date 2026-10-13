"""
Text2SQL — Natural Language to SQL (live demo)
Type a question in plain English; the app generates SQL and runs it on a sample
database, showing the SQL and the result. This is a lightweight standalone demo of
my multi-agent Text2SQL system (the full version fine-tunes Mistral-7B with LoRA
and uses a CrewAI multi-agent pipeline: schema reasoning, SQL optimization, and
ambiguity resolution, reaching 80% exact-match accuracy).

Works with zero setup using a built-in schema-aware translator. If an OpenAI key is
added as a Streamlit secret (OPENAI_API_KEY), it uses an LLM for open-ended questions.

Author: Shanmukha Raj  |  github.com/shanmukha009
"""
import re
import sqlite3
import datetime as dt
import random
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Text2SQL — NL to SQL", page_icon="🧠", layout="wide")

# ---------------------- build sample database ----------------------
@st.cache_resource
def build_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT, segment TEXT);
        CREATE TABLE products(product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
        CREATE TABLE orders(order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT);
        CREATE TABLE order_items(order_id INTEGER, product_id INTEGER, quantity INTEGER);
        """
    )
    rng = random.Random(42)
    cities = ["San Francisco", "New York", "Austin", "Seattle", "Chicago"]
    segments = ["Consumer", "Corporate", "Startup"]
    for i in range(1, 51):
        cur.execute("INSERT INTO customers VALUES(?,?,?,?)",
                    (i, f"Customer {i}", rng.choice(cities), rng.choice(segments)))
    cats = {
        "Laptop": "Electronics", "Phone": "Electronics", "Monitor": "Electronics",
        "Desk": "Furniture", "Chair": "Furniture", "Lamp": "Furniture",
        "Notebook": "Office", "Pen Set": "Office", "Backpack": "Office",
    }
    pid = 1
    prices = {"Laptop": 1400, "Phone": 900, "Monitor": 320, "Desk": 480, "Chair": 260,
              "Lamp": 70, "Notebook": 12, "Pen Set": 18, "Backpack": 65}
    for name, cat in cats.items():
        cur.execute("INSERT INTO products VALUES(?,?,?,?)", (pid, name, cat, prices[name]))
        pid += 1
    oid = 1
    start = dt.date(2026, 1, 1)
    for _ in range(600):
        cust = rng.randint(1, 50)
        day = start + dt.timedelta(days=rng.randint(0, 258))  # through mid-Sep 2026
        cur.execute("INSERT INTO orders VALUES(?,?,?)", (oid, cust, day.isoformat()))
        for _ in range(rng.randint(1, 4)):
            cur.execute("INSERT INTO order_items VALUES(?,?,?)",
                        (oid, rng.randint(1, 9), rng.randint(1, 3)))
        oid += 1
    conn.commit()
    return conn

conn = build_db()

SCHEMA = """customers(customer_id, name, city, segment)
products(product_id, name, category, price)
orders(order_id, customer_id, order_date)
order_items(order_id, product_id, quantity)
-- revenue = order_items.quantity * products.price"""

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September",
     "October","November","December"], start=1)}

# ---------------------- built-in schema-aware translator ----------------------
def rule_based_sql(q: str):
    s = q.lower().strip()
    num = re.search(r"\b(top|first)\s+(\d+)", s)
    n = int(num.group(2)) if num else 5

    def rev(expr_alias):
        return f"SUM(oi.quantity * p.price) AS {expr_alias}"

    # count questions
    if re.search(r"how many|number of|count", s):
        if "customer" in s:
            return "SELECT COUNT(*) AS customers FROM customers;"
        if "order" in s:
            return "SELECT COUNT(*) AS orders FROM orders;"
        if "product" in s:
            return "SELECT COUNT(*) AS products FROM products;"

    # average order value
    if "average" in s and "order" in s:
        return ("SELECT ROUND(AVG(order_total),2) AS avg_order_value FROM ("
                "SELECT o.order_id, SUM(oi.quantity*p.price) AS order_total "
                "FROM orders o JOIN order_items oi ON o.order_id=oi.order_id "
                "JOIN products p ON oi.product_id=p.product_id GROUP BY o.order_id);")

    # total revenue / sales
    if ("total" in s or "overall" in s) and ("revenue" in s or "sales" in s):
        return ("SELECT ROUND(SUM(oi.quantity*p.price),2) AS total_revenue "
                "FROM order_items oi JOIN products p ON oi.product_id=p.product_id;")

    # revenue by category
    if ("category" in s) and ("revenue" in s or "sales" in s):
        return ("SELECT p.category, ROUND(SUM(oi.quantity*p.price),2) AS revenue "
                "FROM order_items oi JOIN products p ON oi.product_id=p.product_id "
                "GROUP BY p.category ORDER BY revenue DESC;")

    # revenue by city / segment
    for dim in ("city", "segment"):
        if dim in s and ("revenue" in s or "sales" in s):
            return (f"SELECT c.{dim}, ROUND(SUM(oi.quantity*p.price),2) AS revenue "
                    "FROM orders o JOIN customers c ON o.customer_id=c.customer_id "
                    "JOIN order_items oi ON o.order_id=oi.order_id "
                    "JOIN products p ON oi.product_id=p.product_id "
                    f"GROUP BY c.{dim} ORDER BY revenue DESC;")

    # revenue by month
    if ("month" in s) and ("revenue" in s or "sales" in s):
        return ("SELECT substr(o.order_date,1,7) AS month, "
                "ROUND(SUM(oi.quantity*p.price),2) AS revenue "
                "FROM orders o JOIN order_items oi ON o.order_id=oi.order_id "
                "JOIN products p ON oi.product_id=p.product_id "
                "GROUP BY month ORDER BY month;")

    # top products
    if "product" in s and ("top" in s or "best" in s or "most" in s):
        metric = "orders" if "order" in s else "revenue"
        if metric == "revenue":
            return ("SELECT p.name, ROUND(SUM(oi.quantity*p.price),2) AS revenue "
                    "FROM order_items oi JOIN products p ON oi.product_id=p.product_id "
                    f"GROUP BY p.name ORDER BY revenue DESC LIMIT {n};")
        return ("SELECT p.name, COUNT(DISTINCT oi.order_id) AS orders "
                "FROM order_items oi JOIN products p ON oi.product_id=p.product_id "
                f"GROUP BY p.name ORDER BY orders DESC LIMIT {n};")

    # top customers
    if "customer" in s and ("top" in s or "best" in s or "most" in s):
        return ("SELECT c.name, ROUND(SUM(oi.quantity*p.price),2) AS spend "
                "FROM customers c JOIN orders o ON c.customer_id=o.customer_id "
                "JOIN order_items oi ON o.order_id=oi.order_id "
                "JOIN products p ON oi.product_id=p.product_id "
                f"GROUP BY c.name ORDER BY spend DESC LIMIT {n};")

    # orders in <month>
    for mname, mnum in MONTHS.items():
        if mname in s and "order" in s:
            return ("SELECT COUNT(*) AS orders "
                    f"FROM orders WHERE substr(order_date,6,2)='{mnum:02d}';")

    return None

def llm_sql(q: str):
    """Optional: use OpenAI if a key is configured in Streamlit secrets."""
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        prompt = (f"Given this SQLite schema:\n{SCHEMA}\n\n"
                  f"Write a single SQLite query (no explanation, no markdown) that answers: {q}")
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0)
        sql = r.choices[0].message.content.strip()
        sql = re.sub(r"^```sql|^```|```$", "", sql, flags=re.MULTILINE).strip()
        return sql
    except Exception as e:
        st.warning(f"LLM path unavailable ({e}); using built-in translator.")
        return None

def to_sql(q: str):
    return llm_sql(q) or rule_based_sql(q)

# ---------------------- UI ----------------------
st.title("Text2SQL — Ask your database in plain English")
st.markdown(
    "Type a question in English and this generates the SQL and runs it on a sample "
    "sales database. This is a lightweight standalone demo of my **multi-agent Text2SQL "
    "system** (the full version fine-tunes **Mistral-7B with LoRA** and uses a **CrewAI** "
    "multi-agent pipeline for schema reasoning, SQL optimization, and ambiguity resolution, "
    "reaching **80% exact-match accuracy**)."
)
st.caption("Built by Shanmukha Raj · github.com/shanmukha009")

with st.expander("Database schema"):
    st.code(SCHEMA, language="sql")

examples = [
    "Show the top 5 products by revenue",
    "Who are the top 3 customers by spend?",
    "What is the total revenue?",
    "Revenue by category",
    "Revenue by month",
    "How many orders are there?",
    "What is the average order value?",
    "Revenue by city",
]
st.write("**Try an example:**")
cols = st.columns(4)
if "q" not in st.session_state:
    st.session_state.q = examples[0]
for i, ex in enumerate(examples):
    if cols[i % 4].button(ex, key=f"ex{i}", use_container_width=True):
        st.session_state.q = ex

q = st.text_input("Your question", value=st.session_state.q)

if q:
    sql = to_sql(q)
    if not sql:
        st.error("I couldn't map that one with the built-in translator. Try an example "
                 "above, or add an OpenAI key as a Streamlit secret to handle any question.")
    else:
        st.markdown("**Generated SQL**")
        st.code(sql, language="sql")
        try:
            df = pd.read_sql_query(sql, conn)
            st.markdown("**Result**")
            st.dataframe(df, hide_index=True, use_container_width=True)
            # auto-chart when it looks like a label + number
            if df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]) and 1 < len(df) <= 30:
                st.bar_chart(df.set_index(df.columns[0]))
        except Exception as e:
            st.error(f"Query error: {e}")

st.divider()
st.caption("Full project (fine-tuned Mistral-7B + CrewAI multi-agent): "
           "github.com/shanmukha009/DATA-266-Text2SQL")

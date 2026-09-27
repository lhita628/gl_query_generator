"""
Northbridge Bank — Credit Risk Portfolio Query Engine
======================================================
Streamlit application converted from the Project 3 Jupyter Notebook.

This app lets business users ask natural-language questions about the
commercial lending portfolio. Questions are routed either to a pre-approved
Verified Query Template (VQ1-VQ10) or to fresh, validated, read-only SQL
generation, following the exact same pipeline logic implemented in the
notebook (intent classification -> query construction -> validation gate ->
retry-on-failure -> execution -> narrative response generation).

All SQL templates, prompts, and pipeline logic are preserved unchanged from
the notebook implementation.
"""

import json
import os
import re
import sqlite3

import pandas as pd
import streamlit as st

from langchain_openai import ChatOpenAI 

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Northbridge Bank | Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

# =============================================================================
# CONFIGURATION / CREDENTIALS
# =============================================================================
# Credentials can be supplied in any of the following ways (checked in order):
#   1. Streamlit secrets (.streamlit/secrets.toml) -> [openai] section
#   2. Environment variables OPENAI_API_KEY / OPENAI_API_BASE
#   3. A local config.json file (same format used in the notebook)
#   4. Manual entry in the sidebar at runtime
DB_PATH = "credit_risk_portfolio.db"
TEST_QUERIES_CSV = "test_queries.csv"
CONFIG_FILE = "config.json"


def load_credentials():
    """Load OpenAI-compatible credentials from secrets, env vars, or config.json."""
    api_key = None
    api_base = None

    # 1. Streamlit secrets
    try:
        if "openai" in st.secrets:
            api_key = st.secrets["openai"].get("OPENAI_API_KEY")
            api_base = st.secrets["openai"].get("OPENAI_API_BASE")
        elif "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets.get("OPENAI_API_KEY")
            api_base = st.secrets.get("OPENAI_API_BASE")
    except Exception:
        pass

    # 2. Environment variables
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_base:
        api_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")

    # 3. config.json (same pattern as the notebook)
    if not api_key and os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                api_key = api_key or config.get("OPENAI_API_KEY")
                api_base = api_base or config.get("OPENAI_API_BASE")
        except Exception:
            pass

    # Fix placeholder base URL, same correction as the notebook
    if api_base and api_base.startswith("your"):
        api_base = api_base.replace("your", "", 1)

    return api_key, api_base


# =============================================================================
# DATABASE SCHEMA CONTEXT (verbatim from the notebook)
# =============================================================================
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# =============================================================================
# VERIFIED QUERY TEMPLATE LIBRARY (verbatim from the notebook)
# =============================================================================
verified_query_library = {
    "VQ1": {
        "description": "Calculate the total outstanding exposure and NPA exposure in millions for each sector, sorted by total outstanding exposure in descending order",
        "sample_questions": [
            "Calculate the total outstanding exposure and NPA exposure for each sector",
            "Show sector-wise total outstanding and NPA breakdown in millions",
            "What is the total outstanding and NPA exposure by sector?",
            "Sector wise loan outstanding and non-performing asset summary",
            "How much outstanding loan and NPA exposure does each sector have?",
        ],
        "sql": """SELECT
  sm.sector_name,
  SUM(lm.total_outstanding) / 1000000.0 AS total_outstanding_million,
  SUM(CASE WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN lm.total_outstanding ELSE 0 END) / 1000000.0 AS npa_outstanding_million
FROM
  loan_master lm
JOIN
  sector_master sm ON lm.sector_code = sm.sector_code
GROUP BY
  sm.sector_name
ORDER BY
  total_outstanding_million DESC""",
    },
    "VQ2": {
        "description": "Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)",
        "sample_questions": [
            "Calculate the total outstanding exposure and loan count for each loan category",
            "Show portfolio outstanding by loan category",
            "What is the total outstanding exposure and number of loans per category?",
            "Loan count and total outstanding amount by category in millions",
            "How many loans and how much total outstanding exposure are in each loan category?",
        ],
        "sql": """SELECT
  loan_category,
  SUM(total_outstanding) / 1000000.0 AS total_outstanding_million,
  COUNT(loan_account_number) AS loan_count
FROM
  loan_master
GROUP BY
  loan_category
ORDER BY
  total_outstanding_million DESC""",
    },
    "VQ3": {
        "description": "IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter",
        "sample_questions": [
            "Summarize loan exposure and expected credit loss by IFRS 9 stage for the latest reporting quarter",
            "Show IFRS 9 stage wise ECL summary for September 2025",
            "What is the total EAD, total ECL, and loan count per IFRS 9 stage as of 2025-09-30?",
            "IFRS 9 provision and exposure summary by stage",
            "How much expected credit loss and total EAD are in Stage 1, Stage 2, and Stage 3?",
        ],
        "sql": """SELECT
  ifrs9_stage,
  COUNT(DISTINCT loan_account_number) AS loan_count,
  SUM(ead_amount) / 1000000.0 AS total_ead_million,
  SUM(ecl_amount) / 1000000.0 AS total_ecl_million
FROM
  provisioning
WHERE
  reporting_date = '2025-09-30'
GROUP BY
  ifrs9_stage
ORDER BY
  ifrs9_stage""",
    },
    "VQ4": {
        "description": "Average provision coverage ratio by sector for the latest reporting quarter",
        "sample_questions": [
            "Calculate the average Provision Coverage Ratio for each sector for the latest reporting quarter",
            "Show provision coverage ratio by sector for September 2025",
            "What is the average PCR across sectors as of 2025-09-30?",
            "Sector wise average provision coverage ratio",
            "Which sectors have the highest provision coverage ratio?",
        ],
        "sql": """SELECT
  sm.sector_name,
  AVG(p.provision_coverage_ratio) AS average_provision_coverage_ratio
FROM
  provisioning p
JOIN
  loan_master lm ON p.loan_account_number = lm.loan_account_number
JOIN
  sector_master sm ON lm.sector_code = sm.sector_code
WHERE
  p.reporting_date = '2025-09-30'
GROUP BY
  sm.sector_name
ORDER BY
  average_provision_coverage_ratio DESC""",
    },
    "VQ5": {
        "description": "Top 10 largest loan exposures by outstanding amount at the borrower level",
        "sample_questions": [
            "Identify the 10 individual loans with the highest outstanding exposure",
            "Show the top 10 largest loan exposures across all sectors",
            "Who are the top 10 borrowers by outstanding loan amount?",
            "List the 10 largest loans with their borrower names and asset classification",
            "Top 10 loan accounts by total outstanding in millions",
        ],
        "sql": """SELECT
  lm.borrower_name,
  sm.sector_name,
  lm.total_outstanding / 1000000.0 AS total_outstanding_million,
  lm.asset_classification
FROM
  loan_master lm
JOIN
  sector_master sm ON lm.sector_code = sm.sector_code
ORDER BY
  total_outstanding_million DESC
LIMIT 10""",
    },
    "VQ6": {
        "description": "Top 5 largest exposures aggregated at the business group level",
        "sample_questions": [
            "Identify the five business groups with the highest total outstanding exposure",
            "Show top 5 business groups by total exposure",
            "What are the top 5 largest business group exposures?",
            "Which business groups have the highest total loan outstanding?",
            "Top 5 group names with loan counts and exposure in millions",
        ],
        "sql": """SELECT
  group_name,
  COUNT(loan_account_number) AS loan_count,
  SUM(total_outstanding) / 1000000.0 AS total_outstanding_million
FROM
  loan_master
WHERE
  group_name IS NOT NULL
GROUP BY
  group_name
ORDER BY
  total_outstanding_million DESC
LIMIT 5""",
    },
    "VQ7": {
        "description": "All overdue loan accounts with their days past due and asset classification",
        "sample_questions": [
            "Identify all loans that are currently overdue and show their delinquency information",
            "Show all delinquent loan accounts with days past due greater than zero",
            "List overdue loans with borrower name, sector, DPD, and asset classification",
            "Which loans are past due and what is their DPD breakdown?",
            "All overdue accounts sorted by days past due",
        ],
        "sql": """SELECT
  lm.loan_account_number,
  lm.borrower_name,
  sm.sector_name,
  lm.total_outstanding / 1000000.0 AS total_outstanding_million,
  lm.days_past_due,
  lm.asset_classification
FROM
  loan_master lm
JOIN
  sector_master sm ON lm.sector_code = sm.sector_code
WHERE
  lm.days_past_due > 0
ORDER BY
  lm.days_past_due DESC""",
    },
    "VQ8": {
        "description": "Distribution of loans across days-past-due buckets showing aging profile of the portfolio",
        "sample_questions": [
            "Show how loans and outstanding exposure are distributed across different Days Past Due buckets",
            "What is the distribution of loans across DPD buckets?",
            "Show portfolio aging and DPD breakdown from current to 90+ days",
            "Count of loans and total outstanding amount by delinquency bucket",
            "How much exposure is in 0-Current, 1-30, 31-60, 61-90, and 90+ DPD buckets?",
        ],
        "sql": """SELECT
  CASE
    WHEN days_past_due = 0 THEN '0 (Current)'
    WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
    WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
    WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
    ELSE '90+'
  END AS dpd_bucket,
  COUNT(loan_account_number) AS loan_count,
  SUM(total_outstanding) / 1000000.0 AS total_outstanding_million
FROM
  loan_master
GROUP BY
  dpd_bucket
ORDER BY
  CASE dpd_bucket
    WHEN '0 (Current)' THEN 1
    WHEN '1-30' THEN 2
    WHEN '31-60' THEN 3
    WHEN '61-90' THEN 4
    ELSE 5
  END""",
    },
    "VQ9": {
        "description": "Borrowers whose internal rating was downgraded in the latest rating cycle",
        "sample_questions": [
            "Identify borrowers whose internal credit rating was downgraded in the latest rating cycle",
            "Show latest rating downgrades as of September 30 2025",
            "Which borrowers were downgraded in the latest credit review?",
            "List all downgraded borrowers with previous rating, current rating, and PD estimate",
            "Show recent credit rating downgrades ordered by probability of default",
        ],
        "sql": """SELECT
  borrower_id,
  previous_rating,
  internal_rating,
  pd_estimate
FROM
  borrower_rating
WHERE
  rating_date = '2025-09-30' AND rating_direction = 'Downgraded'
ORDER BY
  pd_estimate DESC""",
    },
    "VQ10": {
        "description": "Expected credit loss trend across all reporting quarters showing provisioning movement over time",
        "sample_questions": [
            "Show how total Expected Credit Loss has changed across reporting quarters",
            "What is the quarterly trend of total ECL?",
            "Expected credit loss amounts grouped by reporting date in millions",
            "Show total ECL evolution over time by reporting quarter",
            "Quarterly trend of total provision amounts",
        ],
        "sql": """SELECT
  reporting_date,
  SUM(ecl_amount) / 1000000.0 AS total_ecl_million
FROM
  provisioning
GROUP BY
  reporting_date
ORDER BY
  reporting_date""",
    },
}


# =============================================================================
# TOOL DEFINITIONS (verbatim pipeline logic from the notebook)
# =============================================================================
def classify_intent(user_question, query_library, llm=None, **kwargs):
    """
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library (VQ1-VQ10).
    - llm (object, optional): The language model instance to invoke.

    Returns:
    - dict: Contains 'route' ('verified' or 'generated'),
                     'query_id' (template ID e.g. 'VQ1' or None),
                     'match_reason' (short explanation of the decision).
    """
    if llm is None:
        raise ValueError("An initialized LLM client instance must be provided to classify_intent.")

    library_entries = []
    for qid, entry in query_library.items():
        sample_q_list = entry.get("sample_questions", [])
        samples_str = "\n    - ".join(sample_q_list) if sample_q_list else "None"

        library_entries.append(
            f"ID: {qid}\n"
            f"Title: {entry.get('title', 'N/A')}\n"
            f"Description: {entry.get('description', '')}\n"
            f"Sample Questions:\n    - {samples_str}"
        )

    formatted_library = "\n\n".join(library_entries)

    classification_prompt = f"""You are an expert intent classification engine for a banking and credit portfolio Text-to-SQL system.
    Your job is to compare the incoming USER QUESTION against a library of pre-verified canonical query templates (VQ1 through VQ10).

### VERIFIED QUERY LIBRARY
{formatted_library}

### USER QUESTION
"{user_question}"

### INSTRUCTIONS & ROUTING RULES
1. Determine if the USER QUESTION matches the semantic intent or core analytical request of ANY template in the library (VQ1 - VQ10).
2. Compare the request against the template title, description, AND sample questions. Allow minor phrasing variations, synonyms, or word reordering.
3. If a match is found:
   - Set "route" to "verified"
   - Set "query_id" to the matching template key (e.g., "VQ1", "VQ2", ..., "VQ10")
4. If NO match is found (e.g., requires custom filters, unavailable metrics, or unrelated schema domain):
   - Set "route" to "generated"
   - Set "query_id" to null

### OUTPUT FORMAT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" | "generated",
  "query_id": "VQ1" | "VQ2" | "VQ3" | "VQ4" | "VQ5" | "VQ6" | "VQ7" | "VQ8" | "VQ9" | "VQ10" | null,
  "match_reason": "One concise sentence explaining why this template was selected or why fresh SQL generation is required."
}}
Do not include any markdown fences or additional conversational text outside the JSON object."""

    cleaned_response = ""
    try:
        raw_response = llm.invoke(classification_prompt)
        response_str = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

        cleaned_response = re.sub(r"```(?:json)?\s*|\s*```", "", response_str.strip()).strip()

        return json.loads(cleaned_response)

    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {
            "route": "generated",
            "query_id": None,
            "match_reason": "Failed to parse classifier output into valid JSON.",
        }
    except Exception as e:
        return {
            "route": "generated",
            "query_id": None,
            "match_reason": f"Execution error during intent classification: {str(e)}",
        }


def generate_query(user_question, schema_context, llm=None, **kwargs):
    """
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    """

    generation_prompt = f"""You are an expert SQLite Data Engineer for a commercial banking analytical database.
Your task is to write a single, syntactically correct SQLite query to answer the user's question based strictly on the schema context provided.

### DATABASE SCHEMA CONTEXT
{schema_context}

### BANKING BUSINESS RULES & SQL GUIDELINES
1. STRICT TABLE LIMITATION: Use ONLY the tables explicitly defined in the schema above (`loan_master`, `sector_master`, `provisioning`, `borrower_rating`). Do NOT invent or reference any external tables (e.g., shipments, accounts, employees).
2. EXPOSURE & MONETARY METRICS:
   - When outputting exposure, outstanding amounts, EAD, or ECL, divide monetary values by 1000000.0 to convert to millions, e.g., `SUM(total_outstanding) / 1000000.0 AS total_outstanding_million`.
3. JOIN RULES:
   - Join `loan_master` to `sector_master` using `sector_code` (`lm.sector_code = sm.sector_code`).
   - Join `provisioning` to `loan_master` using `loan_account_number` (`p.loan_account_number = lm.loan_account_number`).
   - Join `borrower_rating` to `loan_master` using `borrower_id` (`br.borrower_id = lm.borrower_id`).
4. AGGREGATIONS & NULLS:
   - Always alias calculated columns clearly (e.g., `loan_count`, `total_ecl_million`).
   - Handle potential NULL values explicitly using `WHERE column IS NOT NULL` where applicable.
5. NO EXPLANATIONS: Output ONLY raw SQLite executable code. Do NOT include markdown blocks (` ```sql `), commentary, or preambles.

### USER QUESTION
"{user_question}"

### SQL QUERY
"""

    response = llm.invoke(generation_prompt).content.strip()

    sql = re.sub(r"```(?:sql)?\s*", "", response, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```", "", sql).strip()

    if not sql.endswith(";"):
        sql += ";"

    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None, llm=None, evaluator_llm=None, **kwargs):
    """
    Validates a candidate SQL query through five checks before execution:
    1. Read-only shape check
    2. Schema conformance check
    3. Parse & plan dry run (EXPLAIN)
    4. LLM relevance check
    5. Verified template integrity check (verified track only)

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (float, 0.0-1.0).
    """

    result = {
        "passed": False,
        "failed_check": None,
        "details": "",
        "relevance_confidence": None,
    }

    cur = db_connection.cursor()

    # ---------------------------------------------------------------------------
    # Check 1: Read-only shape check
    # ---------------------------------------------------------------------------
    sql_stripped = candidate_sql.strip()
    sql_upper = sql_stripped.upper()

    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        result["failed_check"] = "read_only_shape"
        result["details"] = "Query must start with SELECT or WITH."
        return result

    forbidden_keywords = [
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
        "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "CREATE", "GRANT", "REVOKE",
    ]
    for kw in forbidden_keywords:
        if re.search(r"\b" + kw + r"\b", sql_upper):
            result["failed_check"] = "read_only_shape"
            result["details"] = f"Forbidden destructive keyword detected: {kw}"
            return result

    clean_sql = sql_stripped.rstrip(";").strip()
    if ";" in clean_sql:
        result["failed_check"] = "read_only_shape"
        result["details"] = "Multiple statements separated by semicolon are not allowed."
        return result

    # ---------------------------------------------------------------------------
    # Check 2: Schema conformance check
    # ---------------------------------------------------------------------------
    real_tables = {
        r[0].lower()
        for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())

    sql_keywords = {
        "select", "from", "where", "and", "or", "group", "by", "order", "having",
        "limit", "join", "on", "as", "case", "when", "then", "else", "end", "sum",
        "count", "avg", "min", "max", "round", "desc", "asc", "left", "right",
        "inner", "outer", "distinct", "null", "is", "not", "in", "like", "with",
        "union", "all", "between", "coalesce", "over", "partition",
    }

    referenced_identifiers = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", clean_sql.lower()))

    known_aliases = {
        "lm", "sm", "p", "br", "total_outstanding_million", "npa_outstanding_million",
        "loan_count", "total_ead_million", "total_ecl_million",
        "average_provision_coverage_ratio", "dpd_bucket",
    }

    unknown_identifiers = [
        tok for tok in referenced_identifiers
        if tok not in sql_keywords
        and tok not in real_tables
        and tok not in real_columns
        and tok not in known_aliases
    ]

    if unknown_identifiers:
        # Flagged but not a hard failure here; non-existent tables/columns will
        # fail hard in Check 3 (EXPLAIN).
        pass

    # ---------------------------------------------------------------------------
    # Check 3: Parse & plan dry run using EXPLAIN
    # ---------------------------------------------------------------------------
    try:
        cur.execute(f"EXPLAIN {clean_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result["failed_check"] = "parse_plan_dry_run"
        result["details"] = f"SQLite failed to parse or plan query: {str(e)}"
        return result

    # ---------------------------------------------------------------------------
    # Check 4: LLM relevance check
    # ---------------------------------------------------------------------------
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is from a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may summarize all sectors/categories/stages rather than filtering to "
        "a single row). A separate downstream step highlights specific target metrics. "
        "Do NOT fail this query for lacking a specific WHERE clause narrowing to one entity — "
        "judge whether the underlying metric, tables, and aggregation logic match the question intent."
        if is_verified_track else
        "This SQL was freshly generated for a custom question and must be appropriately "
        "scoped, joined, and filtered to answer the specific question directly."
    )

    relevance_prompt = f"""You are an expert SQL Quality Engineer for a banking database.
Validate whether the candidate SQL query correctly addresses the user's question.

CONTEXT:
{track_context}

USER QUESTION:
"{user_question}"

CANDIDATE SQL:
{candidate_sql}

INSTRUCTIONS:
1. Check if the query selects the appropriate tables, key columns, and aggregations.
2. Check if NPA, DPD, or date parameters align with business intent.
3. Return ONLY a valid JSON dictionary matching this exact schema:
{{
  "verdict": "yes" | "no",
  "confidence": 0.0 to 1.0,
  "reason": "One short sentence explaining your verdict."
}}
Do not include any markdown code fences or commentary."""

    try:
        eval_model = evaluator_llm if evaluator_llm is not None else llm
        relevance_response = eval_model.invoke(relevance_prompt).content.strip()
        cleaned_eval_resp = re.sub(r"```(?:json)?\s*|\s*```", "", relevance_response).strip()

        json_match = re.search(r"\{.*\}", cleaned_eval_resp, re.DOTALL)
        if json_match:
            relevance_json = json.loads(json_match.group())
            confidence = float(relevance_json.get("confidence", 0.0))
            verdict = str(relevance_json.get("verdict", "no")).lower()

            result["relevance_confidence"] = confidence
            if verdict == "no" or confidence < 0.6:
                result["failed_check"] = "llm_relevance"
                result["details"] = f"Relevance check failed: {relevance_json.get('reason', 'Low relevance score.')}"
                return result
        else:
            result["relevance_confidence"] = 0.7
    except Exception:
        result["relevance_confidence"] = 0.7

    # ---------------------------------------------------------------------------
    # Check 5: Template integrity check (verified track only)
    # ---------------------------------------------------------------------------
    if is_verified_track:
        expected_sql = query_library[query_id]["sql"]
        try:
            cur.execute(f"SELECT * FROM ({expected_sql}) LIMIT 0")
            expected_cols = [d[0].lower() for d in cur.description]

            cur.execute(f"SELECT * FROM ({clean_sql}) LIMIT 0")
            actual_cols = [d[0].lower() for d in cur.description]

            if len(expected_cols) != len(actual_cols):
                result["failed_check"] = "template_integrity"
                result["details"] = (
                    f"Template structure mismatch: Expected {len(expected_cols)} output columns "
                    f"({', '.join(expected_cols)}), but got {len(actual_cols)} columns ({', '.join(actual_cols)})."
                )
                return result
        except sqlite3.Error as e:
            result["failed_check"] = "template_integrity"
            result["details"] = f"Template integrity verification error: {str(e)}"
            return result

    result["passed"] = True
    result["details"] = "All 5 validation checks passed successfully."
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context, llm=None, **kwargs):
    """
    Regenerates SQL after a validation failure, feeding the specific error message,
    failed query, user question, and schema context back to the LLM for self-correction.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL query that failed validation.
    - error_message (str): The specific validation check error message.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised candidate SQL as a clean string without markdown wrappers.
    """

    retry_prompt = f"""You are an expert SQLite Data Engineer for a commercial banking analytical database.
A previously generated SQL query failed validation checks. Your task is to analyze the failure reason and generate a revised, fully corrected SQLite query.

### DATABASE SCHEMA CONTEXT
{schema_context}

### BANKING BUSINESS RULES & SQL CONSTRAINTS
1. STRICT TABLE LIMITATION: Use ONLY the tables explicitly defined in the schema above (`loan_master`, `sector_master`, `provisioning`, `borrower_rating`). Do NOT invent or reference any external tables (e.g., shipments, accounts, employees).
2. READ-ONLY REQUIREMENT: The query MUST start with SELECT or WITH. Destructive or data-modifying queries are strictly prohibited.
3. EXPOSURE & MONETARY METRICS: Divide monetary values (total_outstanding, ead_amount, ecl_amount) by 1000000.0 to represent them in millions, e.g., `SUM(total_outstanding) / 1000000.0 AS total_outstanding_million`.
4. JOIN RELATIONSHIPS:
   - `loan_master` JOIN `sector_master` ON `lm.sector_code = sm.sector_code`
   - `provisioning` JOIN `loan_master` ON `p.loan_account_number = lm.loan_account_number`
   - `borrower_rating` JOIN `loan_master` ON `br.borrower_id = lm.borrower_id`

### PREVIOUS FAILURE CONTEXT
User Question: "{user_question}"
Failed SQL:
{failed_sql}

Validation Failure Details:
{error_message}

### INSTRUCTIONS FOR REVISION
- Carefully read the "Validation Failure Details" above.
- If the error indicates missing columns/tables, check table definitions and fix alias prefixes or column names.
- If the error indicates a syntax or parse issue, fix SQLite syntax errors (e.g., missing GROUP BY columns, invalid CASE statements).
- If the error indicates LLM relevance failure, ensure all requested filters, aggregations, and date constraints directly match the user's intent.

### OUTPUT FORMAT
Return ONLY the raw executable SQLite query code. Do NOT wrap in markdown fences (` ```sql `), preambles, or conversational notes.

### REVISED SQL QUERY
"""

    response = llm.invoke(retry_prompt).content.strip()

    revised_sql = re.sub(r"```(?:sql)?\s*", "", response, flags=re.IGNORECASE)
    revised_sql = re.sub(r"\s*```", "", revised_sql).strip()

    if not revised_sql.endswith(";"):
        revised_sql += ";"

    return revised_sql


def execute_query(validated_sql, db_connection):
    """
    Executes a gate-passed SQL query, fetches results safely, and returns
    a formatted pandas DataFrame along with post-execution reasonableness checks.

    Parameters:
    - validated_sql (str): SQL query that has passed all 5 validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame or None),
                    'reasonable' (bool),
                    'warnings' (list of warning strings),
                    'row_count' (int),
                    'error' (str or None).
    """

    result = {
        "dataframe": None,
        "reasonable": True,
        "warnings": [],
        "row_count": 0,
        "error": None,
    }

    try:
        df = pd.read_sql_query(validated_sql, db_connection)
        result["dataframe"] = df
        result["row_count"] = len(df)

        # 1. Check for empty result set
        if df.empty:
            result["warnings"].append("Query returned an empty result set (0 rows).")

        # 2. Inspect numeric columns for unexpected values
        numeric_cols = df.select_dtypes(include=["number", "float64", "int64"]).columns

        for col in numeric_cols:
            col_lower = col.lower()

            is_delta_col = any(k in col_lower for k in ["change", "diff", "variance", "delta", "deviation", "direction"])
            if not is_delta_col and (df[col] < 0).any():
                neg_count = (df[col] < 0).sum()
                result["warnings"].append(
                    f"Column '{col}' contains {neg_count} negative value(s), which may indicate a data quality anomaly."
                )

            null_count = df[col].isnull().sum()
            if len(df) > 0 and null_count > (len(df) * 0.5):
                result["warnings"].append(
                    f"Column '{col}' has {null_count}/{len(df)} missing (NULL) values (>50% rate)."
                )

        # 3. Overall reasonableness threshold
        if len(result["warnings"]) >= 2:
            result["reasonable"] = False

    except (sqlite3.Error, Exception) as exec_err:
        result["reasonable"] = False
        result["error"] = str(exec_err)
        result["warnings"].append(f"Execution failed with error: {str(exec_err)}")

    return result


def generate_response(user_question, dataframe, route, query_id=None, llm=None, **kwargs):
    """
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.
    - llm (object, optional): Language model instance to invoke.

    Returns:
    - str: Natural language response focused on what the user asked.
    """
    if llm is None:
        raise ValueError("An initialized LLM client instance must be provided to generate_response.")

    if dataframe is None or dataframe.empty:
        return (
            "### Executive Summary\n\n"
            "No matching records or data were found in the database for your request."
        )

    data_preview = dataframe.head(100) if len(dataframe) > 100 else dataframe
    data_table_markdown = data_preview.to_markdown(index=False)

    if route == "verified" and query_id:
        track_context = (
            f"Note: Sourced via Verified Query Template `{query_id}`. "
            "Verified templates return complete portfolio summaries. "
            "If the user asked for a specific entity or metric, extract and highlight "
            "the specific row or metric they requested first, then provide the summary context."
        )
    else:
        track_context = (
            "Note: Sourced via custom dynamic query generation. "
            "Focus directly on answering the user's specific question using the data provided."
        )

    response_prompt = f"""You are a Senior Credit Risk & Portfolio Analytics Officer for a commercial bank.
Your job is to transform raw SQL query result data into concise, executive-level natural language insights.

### ROUTE CONTEXT
{track_context}

### USER QUESTION
"{user_question}"

### RETRIEVED DATA ({len(dataframe)} rows returned)
{data_table_markdown}

### INSTRUCTIONS FOR RESPONSE
1. **Direct Answer First**: Begin immediately with a 1-2 sentence direct answer addressing the user's explicit question. Highlight key figures, totals, or top entities.
2. **Key Insights & Portfolio Takeaways**: Provide 2-4 bullet points analyzing key business trends, concentration risks, DPD/NPA spikes, or notable credit metrics.
3. **Data Presentation**: Present the formatted Markdown table shown above. Convert raw numbers into clear financial formats where appropriate (e.g., "$12.4M", "14.5%").
4. **Tone & Constraints**: Maintain an executive, clear, and objective tone. Do NOT mention technical details like SQL tables, pandas DataFrames, or database schemas.

### EXECUTIVE RESPONSE
"""

    raw_response = llm.invoke(response_prompt)
    narrative = raw_response.content.strip() if hasattr(raw_response, "content") else str(raw_response).strip()
    return narrative


# =============================================================================
# PIPELINE ORCHESTRATION ENGINE (verbatim from the notebook)
# =============================================================================
def run_pipeline(
    user_question,
    db_connection,
    query_library,
    schema_context,
    llm,
    evaluator_llm=None,
    verbose=False,
    log_fn=None,
):
    """
    Runs the complete query engine pipeline for a single user question.
    `log_fn`, if provided, receives each status-line string (used to stream
    pipeline progress into the Streamlit UI instead of printing to stdout).
    """

    def _log(msg):
        if verbose:
            print(msg)
        if log_fn:
            log_fn(msg)

    log = {
        "user_question": user_question,
        "route": None,
        "query_id": None,
        "match_reason": None,
        "candidate_sql": None,
        "gate_result": None,
        "retry_used": False,
        "escalated": False,
        "executed_sql": None,
        "row_count": 0,
        "confidence": None,
        "narrative": None,
        "error": None,
    }

    try:
        # Step 1: Intent Classification
        classification = classify_intent(user_question, query_library, llm=llm)
        log["route"] = classification.get("route", "generated")
        log["query_id"] = classification.get("query_id")
        log["match_reason"] = classification.get("match_reason")

        _log(f"**[1] Intent Classification** — Route: `{log['route'].upper()}` | Query ID: `{log['query_id']}`")
        _log(f"Reason: {log['match_reason']}")

        # Step 2: Query Construction
        if log["route"] == "verified" and log["query_id"] in query_library:
            candidate_sql = query_library[log["query_id"]]["sql"]
            _log(f"**[2] Query Construction** — Loaded template `{log['query_id']}` from library.")
        else:
            candidate_sql = generate_query(user_question, schema_context, llm=llm)
            _log("**[2] Query Construction** — Generated candidate SQL via LLM.")

        log["candidate_sql"] = candidate_sql

        # Step 3: Validation Gate
        gate = validate_query(
            user_question=user_question,
            candidate_sql=candidate_sql,
            db_connection=db_connection,
            query_library=query_library,
            query_id=log["query_id"],
            llm=llm,
            evaluator_llm=evaluator_llm,
        )
        log["gate_result"] = gate

        _log(f"**[3] Validation Gate** — Passed: `{gate.get('passed')}` | Confidence: `{gate.get('relevance_confidence')}`")
        if not gate.get("passed"):
            _log(f"Failed Check: `{gate.get('failed_check')}` — {gate.get('details')}")

        # Step 4: Retry Strategy
        if not gate.get("passed") and log["route"] == "generated":
            _log(f"**[-->] Triggering Self-Correction Retry** for: {gate.get('details')}")

            candidate_sql = retry_generation(
                user_question=user_question,
                failed_sql=candidate_sql,
                error_message=f"Failed {gate.get('failed_check')}: {gate.get('details')}",
                schema_context=schema_context,
                llm=llm,
            )
            log["candidate_sql"] = candidate_sql
            log["retry_used"] = True

            gate = validate_query(
                user_question=user_question,
                candidate_sql=candidate_sql,
                db_connection=db_connection,
                query_library=query_library,
                query_id=None,
                llm=llm,
                evaluator_llm=evaluator_llm,
            )
            log["gate_result"] = gate

        # Step 5: Escalation Fallback
        if not gate.get("passed"):
            log["escalated"] = True
            log["confidence"] = "ESCALATED"
            log["narrative"] = (
                f"### Execution Escalated\n\n"
                f"This request could not be reliably resolved within safety and correctness thresholds.\n\n"
                f"**Reason for Escalation:** {gate.get('details')}\n\n"
                f"*The issue has been flagged for human analyst review.*"
            )

            _log(f"**[!] ESCALATED TO HUMAN ANALYST** — {gate.get('details')}")

            return {
                "status": "ESCALATED",
                "narrative": log["narrative"],
                "dataframe": None,
                "sql_executed": None,
                "log": log,
                **log,
            }

        # Step 6: Execution
        log["executed_sql"] = candidate_sql
        exec_result = execute_query(candidate_sql, db_connection)
        df = exec_result.get("dataframe")

        if df is None:
            df = pd.DataFrame()

        log["row_count"] = len(df)

        _log(f"**[4] Execution** — Returned {log['row_count']} row(s)")
        if exec_result.get("warnings"):
            _log(f"Warnings: {exec_result['warnings']}")

        # Step 7: Response Synthesis
        narrative = generate_response(
            user_question=user_question,
            dataframe=df,
            route=log["route"],
            query_id=log["query_id"],
            llm=llm,
        )

        log["narrative"] = narrative
        log["confidence"] = gate.get("relevance_confidence", 1.0)

        _log(f"**[5] Response Generation Complete.** Final Confidence: `{log['confidence']}`")

        return {
            "status": "SUCCESS",
            "narrative": narrative,
            "dataframe": df,
            "sql_executed": candidate_sql,
            "log": log,
            **log,
        }

    except Exception as pipeline_err:
        log["escalated"] = True
        log["error"] = str(pipeline_err)
        log["confidence"] = "ERROR"
        log["narrative"] = (
            f"### Pipeline Execution Error\n\n"
            f"An unexpected error occurred during execution: `{str(pipeline_err)}`"
        )

        _log(f"**[!] PIPELINE EXCEPTION** — {str(pipeline_err)}")

        return {
            "status": "ERROR",
            "narrative": log["narrative"],
            "dataframe": None,
            "sql_executed": log.get("candidate_sql"),
            "log": log,
            **log,
        }


# =============================================================================
# CACHED RESOURCE INITIALIZATION
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_db_connection(db_path):
    """Read-only SQLite connection, cached for the app's lifetime."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    return conn


def get_llms(api_key, api_base, model_name, evaluator_model_name):
    """Build the primary and evaluator LLM clients. Not cached, since API key
    can change at runtime via the sidebar."""
    os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base

    llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key, base_url=api_base or None)
    evaluator_llm = ChatOpenAI(model=evaluator_model_name, temperature=0, api_key=api_key, base_url=api_base or None)
    return llm, evaluator_llm


@st.cache_data(show_spinner=False)
def load_ground_truth(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.title("🏦 Northbridge Bank — Credit Risk Portfolio Query Engine")
    st.caption(
        "Ask routine commercial-lending portfolio questions in plain English. "
        "Every answer shows the SQL used, the raw data returned, and a confidence score."
    )

    # -------------------------------------------------------------------
    # Sidebar: configuration
    # -------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")

        default_key, default_base = load_credentials()

        api_key = st.text_input(
            "OpenAI API Key",
            value=default_key or "",
            type="password",
            help="Loaded automatically from st.secrets, environment variables, or config.json if available.",
        )
        api_base = st.text_input(
            "OpenAI API Base URL (optional)",
            value=default_base or "",
            help="Leave blank to use the default OpenAI endpoint.",
        )

        st.markdown("---")
        model_name = st.selectbox("Primary LLM (routing, generation, response)", ["gpt-4o-mini", "gpt-4o"], index=0)
        evaluator_model_name = st.selectbox("Evaluator LLM (relevance checks)", ["gpt-4o", "gpt-4o-mini"], index=0)

        st.markdown("---")
        db_path_input = st.text_input("Database file", value=DB_PATH)
        show_pipeline_log = st.checkbox("Show pipeline execution trace", value=True)
        show_sql = st.checkbox("Show executed SQL", value=True)
        show_dataframe = st.checkbox("Show raw data table", value=True)

        st.markdown("---")
        st.markdown(
            "**Read-only mode:** the database connection is opened in SQLite "
            "URI read-only mode (`?mode=ro`), and every candidate query passes "
            "a 5-step validation gate before execution."
        )

    if not api_key:
        st.warning("Please provide an OpenAI API key in the sidebar to use the query engine.")
        st.stop()

    if not os.path.exists(db_path_input):
        st.error(
            f"Database file `{db_path_input}` was not found in the app's working directory. "
            "Please upload `credit_risk_portfolio.db` alongside `app.py`."
        )
        st.stop()

    conn = get_db_connection(db_path_input)
    llm, evaluator_llm = get_llms(api_key, api_base, model_name, evaluator_model_name)

    # -------------------------------------------------------------------
    # Tabs: Ask a Question | Verified Query Library | Test Case Evaluation
    # -------------------------------------------------------------------
    tab_ask, tab_library, tab_eval = st.tabs(
        ["💬 Ask a Question", "📚 Verified Query Library", "✅ Test Case Evaluation"]
    )

    # ============================== TAB 1 ==============================
    with tab_ask:
        st.subheader("Ask a portfolio question")

        example_cols = st.columns(3)
        example_questions = [
            "Calculate the total outstanding exposure and NPA exposure for each sector",
            "Show the top 10 largest loan exposures",
            "Which borrowers were downgraded in the latest credit review?",
        ]
        for col, eq in zip(example_cols, example_questions):
            if col.button(eq, use_container_width=True):
                st.session_state["user_question_input"] = eq

        user_question = st.text_area(
            "Your question",
            key="user_question_input",
            placeholder="e.g. What is the total outstanding and NPA exposure by sector?",
            height=80,
        )

        run_clicked = st.button("Run Query", type="primary")

        if run_clicked and user_question.strip():
            log_container = st.container()
            log_lines = []

            def log_fn(msg):
                log_lines.append(msg)

            with st.spinner("Running query engine pipeline..."):
                result = run_pipeline(
                    user_question=user_question.strip(),
                    db_connection=conn,
                    query_library=verified_query_library,
                    schema_context=database_schema,
                    llm=llm,
                    evaluator_llm=evaluator_llm,
                    verbose=False,
                    log_fn=log_fn,
                )

            if show_pipeline_log:
                with log_container.expander("🔍 Pipeline Execution Trace", expanded=False):
                    for line in log_lines:
                        st.markdown(line)

            status = result.get("status")
            route = result.get("route")
            query_id = result.get("query_id")
            confidence = result.get("confidence")

            status_badge = {
                "SUCCESS": "🟢 SUCCESS",
                "ESCALATED": "🟠 ESCALATED",
                "ERROR": "🔴 ERROR",
            }.get(status, status)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Status", status_badge)
            m2.metric("Route", (route or "-").upper())
            m3.metric("Query Template", query_id or "—")
            m4.metric("Confidence", f"{confidence}" if confidence is not None else "—")

            st.markdown("### Executive Response")
            st.markdown(result.get("narrative") or "_No narrative generated._")

            if show_sql and result.get("sql_executed"):
                st.markdown("### SQL Executed")
                st.code(result["sql_executed"], language="sql")
            elif show_sql and result.get("candidate_sql"):
                st.markdown("### Candidate SQL (not executed — escalated)")
                st.code(result["candidate_sql"], language="sql")

            if show_dataframe and result.get("dataframe") is not None:
                st.markdown("### Raw Data Returned")
                st.dataframe(result["dataframe"], use_container_width=True)
                st.caption(f"{result.get('row_count', 0)} row(s) returned.")

            with st.expander("Full audit log (JSON)"):
                audit_log = {k: v for k, v in result.get("log", {}).items() if k != "gate_result"}
                gate_result = result.get("log", {}).get("gate_result")
                st.json(audit_log)
                if gate_result:
                    st.markdown("**Validation gate result:**")
                    st.json(gate_result)
        elif run_clicked:
            st.warning("Please enter a question before running the query engine.")

    # ============================== TAB 2 ==============================
    with tab_library:
        st.subheader("Verified Query Template Library")
        st.caption(
            "These 10 pre-approved SQL templates cover the most common recurring "
            "credit-risk analytics questions. Matching user questions are routed "
            "here automatically instead of generating fresh SQL."
        )
        for qid, entry in verified_query_library.items():
            with st.expander(f"{qid} — {entry['description']}"):
                st.markdown("**Sample questions this template matches:**")
                for sq in entry.get("sample_questions", []):
                    st.markdown(f"- {sq}")
                st.markdown("**SQL:**")
                st.code(entry["sql"], language="sql")
                if st.button(f"Run {qid} directly", key=f"run_{qid}"):
                    with st.spinner(f"Executing {qid}..."):
                        exec_result = execute_query(entry["sql"], conn)
                        df = exec_result.get("dataframe")
                        if df is not None:
                            st.dataframe(df, use_container_width=True)
                            if exec_result.get("warnings"):
                                for w in exec_result["warnings"]:
                                    st.warning(w)
                        else:
                            st.error(exec_result.get("error", "Execution failed."))

    # ============================== TAB 3 ==============================
    with tab_eval:
        st.subheader("Evaluation Against Ground Truth")
        ground_truth = load_ground_truth(TEST_QUERIES_CSV)

        if ground_truth is None:
            st.info(
                f"`{TEST_QUERIES_CSV}` was not found in the app's working directory. "
                "Upload it alongside `app.py` to enable automated evaluation against "
                "the ground-truth test cases."
            )
        else:
            st.dataframe(ground_truth, use_container_width=True)
            run_eval = st.button("Run All Test Cases", type="primary")

            if run_eval:
                evaluation_rows = []
                progress = st.progress(0.0)
                n = len(ground_truth)

                for i, (_, gt) in enumerate(ground_truth.iterrows()):
                    user_q = gt["User Query"]
                    expected_route = gt["Expected Route"] if pd.notna(gt.get("Expected Route")) else None
                    expected_query_id = gt["Expected Query ID"] if pd.notna(gt.get("Expected Query ID")) else None

                    tr = run_pipeline(
                        user_question=user_q,
                        db_connection=conn,
                        query_library=verified_query_library,
                        schema_context=database_schema,
                        llm=llm,
                        evaluator_llm=evaluator_llm,
                        verbose=False,
                    )

                    actual_log = tr.get("log", tr)
                    actual_route = actual_log.get("route")
                    actual_query_id = actual_log.get("query_id")
                    actual_confidence = tr.get("confidence", actual_log.get("confidence"))
                    actual_rows = tr.get("row_count", actual_log.get("row_count", 0))

                    route_match = (
                        (expected_route is None and actual_route is None)
                        or (str(expected_route).strip().lower() == str(actual_route).strip().lower() if expected_route and actual_route else False)
                    )
                    query_id_match = (
                        (expected_query_id is None and actual_query_id is None)
                        or (str(expected_query_id).strip() == str(actual_query_id).strip() if expected_query_id and actual_query_id else False)
                    )

                    evaluation_rows.append({
                        "Test Case": gt.get("Test Case", i + 1),
                        "User Query": user_q,
                        "Expected Route": expected_route,
                        "Actual Route": actual_route,
                        "Route Match": route_match,
                        "Expected Query ID": expected_query_id,
                        "Actual Query ID": actual_query_id,
                        "Query ID Match": query_id_match,
                        "Confidence": actual_confidence,
                        "Rows Returned": actual_rows,
                        "Status": tr.get("status", "UNKNOWN"),
                    })

                    progress.progress((i + 1) / n)

                evaluation_df = pd.DataFrame(evaluation_rows)
                st.markdown("### Evaluation Results")
                st.dataframe(evaluation_df, use_container_width=True)

                path_accuracy = evaluation_df["Route Match"].mean() * 100
                query_accuracy = evaluation_df["Query ID Match"].mean() * 100
                numeric_conf = pd.to_numeric(evaluation_df["Confidence"], errors="coerce")
                avg_confidence = numeric_conf.mean()

                c1, c2, c3 = st.columns(3)
                c1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
                c2.metric("Selected Query Accuracy", f"{query_accuracy:.1f}%")
                c3.metric(
                    "Average Confidence Score",
                    f"{avg_confidence:.2f}" if pd.notna(avg_confidence) else "—",
                )


if __name__ == "__main__":
    main()

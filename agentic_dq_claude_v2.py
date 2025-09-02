import json
import sys
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from langgraph.graph import StateGraph, END, START
import logging
from langchain_core.output_parsers import JsonOutputParser
import networkx as nx
import pandas as pd 
import os 
from requests.exceptions import HTTPError
import requests
from google.cloud import bigquery
import base64
import time
import google.auth
import re
import warnings
import numpy as np
import ast
import configparser
from datetime import datetime
import uuid
from pyvegas.helpers.utils import set_proxy, unset_proxy
from pyvegas.langx import VegasLangchainLLM

set_proxy()

warnings.filterwarnings('ignore')
# Configure logging
# logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config_file_path = os.path.join("/apps/opt/application/dev_smartdq/dev/agentic_dq/config", 'config.ini')
files_read = config.read(config_file_path)

SA_PATH = config['paths']['SA_PATH']
RULES_CSV_PATH = config['paths']['RULES_CSV_PATH']
CONFIG_PATH = config['paths']['CONFIG_PATH']
lineage_graphs = config['paths']['lineage_graphs']
OIDC_TOKEN_PATH = config['paths']['OIDC_TOKEN_PATH']
LOGS_PATH = config['paths']['LOGS_PATH']
VEGAS_API_KEY = config['credentials']['VEGAS_API_KEY']
ENVIRONMENT = config['credentials']['ENVIRONMENT']
USECASE_NAME = config['credentials']['USECASE_NAME']
CONTEXT_NAME = config['credentials']['CONTEXT_NAME']
CLIENT_ID = config['credentials']['CLIENT_ID']
CLIENT_SECRET = config['credentials']['CLIENT_SECRET']
PROJECT_ID = config['credentials']['PROJECT_ID']
GOOGLE_CLOUD_PROJECT = config['credentials']['GOOGLE_CLOUD_PROJECT']

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"
CHATBOT_NAME = "ADQ"

os.environ["VEGAS_API_KEY"] = VEGAS_API_KEY
os.environ["ENVIRONMENT"] = ENVIRONMENT

usecase_name = USECASE_NAME # Add your vegas usecase name here
context_name = CONTEXT_NAME # Add your vegas context name here

llm = VegasLangchainLLM(usecase_name=usecase_name, context_name=context_name, temperature=0)
json_parser = JsonOutputParser()

class RootCauseAnalysisState(TypedDict):
    failed_column : str
    failed_table : str
    analysis_method : Dict[str, Any]
    validation_query : str
    expected_value : float
    expected_std_dev : float
    actual_value : float
    sd_threshold : float
    initial_check_result : Optional[Dict]
    parsed_dq_info : Dict[str, Any]
    trace_data : Optional[Dict[str, Dict[str, List[Dict]]]] # Enriched trace data structure
    paths_to_process: Optional[List[Tuple[int, str, str, str]]]
    mismatched_nodes: Optional[List[Dict[str, Any]]]
    analysis_results: Optional[Dict[str, Any]]
    agent_input: str
    anamoly_node_response: str
    loaded_lineage_graphs : Optional[Dict[str, nx.DiGraph]]

# Emit functions for Node.js communication

def emit_message(message_type: str, content: str, metadata: Dict = None):
    """
    Emit a JSON message to stdout for Node.js to capture and forward to frontend
    """
    message = {
        "type": message_type,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()

def emit_lineage_graph(graph_data: Dict):
    """
    Emit lineage graph structure for dynamic visualization
    """
    message = {
        "type": "lineage_graph",
        "data": graph_data,
        "timestamp": datetime.now().isoformat()
    }
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()

def emit_node_status_update(node_id: str, status: str):
    """
    Emit node status updates for dynamic graph coloring
    """
    message = {
        "type": "node_status_update",
        "data": {
            "nodeId": node_id,
            "status": status
        },
        "timestamp": datetime.now().isoformat()
    }
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()

def emit_step_result(step_number: int, step_type: str, table: str, column: str, sql_query: str, result: str, status: str = "success"):
    """
    Emit structured step results matching the Figma conversation flow
    """
    message = {
        "type": "step_result",
        "data": {
            "step_number": step_number,
            "step_type": step_type,
            "table": table,
            "column": column,
            "sql_query": sql_query,
            "result": result,
            "status": status
        },
        "timestamp": datetime.now().isoformat()
    }
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()

def emit_progress(step_name: str, status: str, details: Dict = None):
    """
    Emit progress updates for the frontend
    """
    progress_message = {
        "type": "progress",
        "step": step_name,
        "status": status,
        "details": details or {},
        "timestamp": datetime.now().isoformat()
    }
    sys.stdout.write(json.dumps(progress_message) + "\n")
    sys.stdout.flush()

def emit_analysis_result(layer: str, table: str, column: str, sql_query: str, reasoning: str, inference: str):
    """
    Emit structured analysis results for display in the chatbot
    """
    result = {
        "type": "analysis_result",
        "layer": layer,
        "table": table,
        "column": column,
        "sql_query": sql_query,
        "reasoning": reasoning,
        "inference": inference,
        "timestamp": datetime.now().isoformat()
    }
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()

def emit_table_data(data: Dict, title: str):
    """
    Emit table data for frontend display
    """
    table_message = {
        "type": "table_data",
        "title": title,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    sys.stdout.write(json.dumps(table_message) + "\n")
    sys.stdout.flush()

def isTokenExpired(path):
    try:
        if(os.path.exists(OIDC_TOKEN_PATH)):
            with open(OIDC_TOKEN_PATH,'r') as f:
                old_access_token = json.load(f)['access_token'].split('.')[1]
                old_access_token += '=' * (-len(old_access_token) % 4)
                old_token_json_decoded = json.loads(base64.b64decode(old_access_token).decode('utf8').replace("'",'"'))
                auth_time = old_token_json_decoded['auth_time']
                expires_in = old_token_json_decoded['expires_in']
                curr_epoch_time = int(time.time())
                if curr_epoch_time - auth_time < expires_in - 120:
                    logger.info("Token is Valid")
                    return False
                else:
                    logger.info("Invalid Token")
        return True
    except Exception as e:
        raise e
    
def exchange_and_save_oidc_token_for_jwt(client_id:str, client_secret:str) -> None:
    logger.info('Retrieving JWT from OIDC provider...')
    url = 'https://ssologinuat.verizon.com/ngauth/oauth2/realms/root/realms/employee/access_token'
    #url = 'https://ssologin.verizon.com/ngauth/oauth2/realms/root/realms/employee/access_token' # production url
    payload = {'grant_type':'client_credentials','client_id':client_id,'client_secret':client_secret,'scope':'read'}

    try:
        response = requests.post(url=url, params=payload)
        response.raise_for_status()
        token = response.json()
        logger.info('Saving token...')
        # Serializing json
        # print(os.path.abspath(OIDC_TOKEN_PATH))
        with open(os.path.join(CONFIG_PATH,'oidc_token.json'),'w') as f:  #dont change the file name
            json.dump(token,f)
    except HTTPError as e:
        raise e

def bigquery_execute(query: str) -> pd.DataFrame:
    client_id = CLIENT_ID
    client_secret = CLIENT_SECRET
    project_id = PROJECT_ID
    # get a jwt
    if isTokenExpired(OIDC_TOKEN_PATH):
        exchange_and_save_oidc_token_for_jwt(client_id=client_id, client_secret=client_secret)
    
    logger.info('Setting environment variable...')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SA_PATH
    os.environ['GOOGLE_CLOUD_PROJECT'] = GOOGLE_CLOUD_PROJECT
    credentials, auth_project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    logger.info(f"project_id={project_id}, credentials={credentials}")
    try:
        client = bigquery.Client(credentials=credentials, project=project_id)
        logger.info(f"Query: {query}")
        df_results = client.query(query).to_dataframe()
        return df_results
    except Exception as e :
        error_message = f"Error executing BigQuery query: {e}"
        logger.info(error_message)
        return error_message

def extract_metric_from_output(df_json, sql_query) -> Optional[float]:
    """Safely extracts the first numeric value from a query output."""
    actual_value_prompt = f"""You are a precise data extraction bot. Your task is to analyze a JSON dataset and extract a single numeric value. 
                            **Dataset (JSON):**\n```json\n{df_json}\n``` 
                            **SQL Query that generated the data:**\n```sql\n{sql_query}\n``` 
                            **Instructions:** 
                            1. **Identify the Metric Column:** From the SQL query, determine the name of the primary metric column (e.g., the result of a SUM, COUNT). 
                            2. **Return the Value:** Return **only the numeric value** from that specific cell. Do not include any other text. **Example Output:**\n1234.56"""
                            
    try:
        response = llm.invoke(actual_value_prompt).content.strip()
        match = re.search(r"[-+]?\d*\d\.?\d+", response)
        if match:
            return float(match.group(0))
        else:
            logger.warning(f"LLM didn't return valid number: {response}")
            return 0.0
    except Exception as e:
        logger.error(f"Failed to extract metric: {e}")
        return 0.0

def get_predecessor_info(node_name: str, lineage_graph: nx.DiGraph) -> List[Dict[str, str]]:
    """
    Finds and returns information about direct predecessors of a given node in the lineage graph.
    A node name is expected in 'table.column' format.
    """
    predecessors_list = []

    try:
        for pred_node in lineage_graph.successors(node_name):
            transformation = lineage_graph.edges[node_name, pred_node].get('transformation', 'N/A')
            predecessors_list.append({
                'prev_table': pred_node.rsplit('.', 1)[0],
                'prev_column': pred_node.rsplit('.', 1)[1],
                'transformation': transformation,
                'source_node_full_name': pred_node
            })
    except Exception as e:
        logger.error(f"Error getting predecessor info for {node_name}: {e}")
    
    return predecessors_list

def build_graph_structure_from_lineage(lineage_graph: nx.DiGraph, l0_key: str) -> Dict:
    """Convert NetworkX lineage graph to frontend-compatible graph structure"""
    try:
        nodes = []
        edges = []
        tables_added = set()
        
        for node in lineage_graph.nodes():
            try:
                if '.' in node:
                    table_name, column_name = node.rsplit('.', 1)
                    table_display_name = table_name.rsplit('.', 1)[-1] if '.' in table_name else table_name
                else:
                    table_display_name = node
                    column_name = node
                
                # Add table node if not already added
                if table_display_name not in tables_added:
                    nodes.append({
                        "id": table_display_name,
                        "label": table_display_name,
                        "type": "table"
                    })
                    tables_added.add(table_display_name)
                
                # Add column node
                nodes.append({
                    "id": node,
                    "label": column_name,
                    "type": "column",
                    "parent": table_display_name
                })
            except Exception as e:
                logger.warning(f"Error processing node {node}: {e}")
                continue
        
        # Add edges
        for source, target in lineage_graph.edges():
            edges.append({"source": source, "target": target})
        
        return {"nodes": nodes, "edges": edges}
        
    except Exception as e:
        logger.error(f"Error building graph structure: {e}")
        return {"nodes": [], "edges": []}

def replace_technical_date_with_business_date(sql_query: str, table_name: str) -> str:
    """Replace technical date columns with business date columns"""
    try:
        logger.info('Updating query technical dates with business dates')
        date_df = pd.read_csv(RULES_CSV_PATH)
        row = date_df[date_df['BQ Table Name'].str.lower() == table_name.lower()]
    
        if row.empty:
            return sql_query
        else:
            business_date = row['Business date'].values[0]
        
            business_date_prompt = f"""You are an expert SQL refactoring assistant. Your task is to analyze an incoming SQL query and replace any technical date-related column names with a specified "business date" column name. You must be very careful to only change the column name and NOT the date values or any other part of the query.

                                **Your Instructions:**
                                1.  **Analyze the SQL Query:** Carefully examine the provided SQL query.
                                2.  **Identify the Target Table:** Your modifications should only apply to columns belonging to the specified `target_table_name`.
                                3.  **Find the Technical Date Column:** Identify the column name in the query that is being used as a date filter or for date-based grouping. These columns often end in `_dt`, `_date`, or similar suffixes (e.g., `activity_dt`, `event_date`, `trans_date`).
                                4.  **Replace the Column Name:** Replace all occurrences of the identified technical date column with the provided `business_date_column_name`.
                                    *   This includes occurrences in the `WHERE` clause, `SELECT` clause, `GROUP BY` clause, and any subqueries.
                                    *   If the column is prefixed with a table alias (e.g., `t1.activity_dt` or `t1.hdp_insert_dt_time`), make sure to preserve the alias (e.g., `t1.business_processing_date`).
                                5.  **DO NOT CHANGE DATE VALUES:** It is critical that you do not alter the actual date literals or timestamp strings in the query (e.g., `'2024-07-15'`, `CAST('2025-01-01' AS DATE)`). Only the column *name* should change.
                           
                                **Context for this Task:**
                                *   **Input SQL Query:**
                                    ```sql
                                    {sql_query}
                                    ```
                                *   **Target Table Name:** `{table_name}`
                                *   **Business Date Column Name to use:** `{business_date}`

                                **Example 1:**
                                *   **Input SQL Query:** `SELECT SUM(sales) FROM my_sales_table WHERE sales_dt = '2024-05-20' GROUP BY product_id;`
                                *   **Target Table Name:** `my_sales_table`
                                *   **Business Date Column Name:** `business_day`
                                *   **Expected Output SQL:** `SELECT SUM(sales) FROM my_sales_table WHERE business_day = '2024-05-20' GROUP BY product_id;`

                                **Example 2 (with alias):**
                                *   **Input SQL Query:** `SELECT t1.customer_id, COUNT(*) FROM fact_sessions t1 WHERE t1.session_start_dt BETWEEN '2024-01-01' AND '2024-01-31' GROUP BY t1.customer_id;`
                                *   **Target Table Name:** `fact_sessions`
                                *   **Business Date Column Name:** `session_business_date`
                                *   **Expected Output SQL:** `SELECT t1.customer_id, COUNT(*) FROM fact_sessions t1 WHERE t1.session_business_date BETWEEN '2024-01-01' AND '2024-01-31' GROUP BY t1.customer_id;`
                   
                                **Example 3 (what to AVOID):**
                                *   **Input SQL Query:** `... WHERE activity_dt = '2023-11-10' ...`
                                *   **Business Date Column Name:** `business_date`
                                *   **WRONG Output:** `... WHERE business_date = 'business_date' ...` (The date value was incorrectly changed)
                                *   **CORRECT Output:** `... WHERE business_date = '2023-11-10' ...`

                                **Your Final Output:**
                                Return only the complete, modified SQL query. Do not add any explanations, markdown formatting, or introductory text."""
        
            raw_sql_query = llm.invoke(business_date_prompt).content
            sql_query = extract_sql_from_text(raw_sql_query)
            return sql_query
    except Exception as e:
        logger.error(f"Error updating query technical dates: {e}")
        return sql_query

def extract_sql_from_text(text: str) -> str:
    """Extracts the SQL query from the LLM response."""
    match = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        sql_query = match.group(1).strip()
        return sql_query
    else:
        match_cte = re.search(r"(WITH\s+[\s\S]+?SELECT[\s\S]*?;)", text, re.IGNORECASE | re.MULTILINE)
        if match_cte:
            query = match_cte.group(1).strip()
            return query
    return text.strip()  # Return original text if no SQL found

def anamoly_identifier_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    user_input = state
    prompt = f"""You are a data validation assistant. Your task is to analyze validation results and generate clear, concise, and professional summaries for data quality issues based on provided metadata.
                Your response must:
                - State the anomaly clearly, identifying the affected table, column, and date.
                - Explain the deviation: "The actual value of [actual_value] fell outside the expected range of [expected_value] +/- [sd_threshold] standard deviations."
                - Use human-readable language suitable for business and data stakeholders.
                - Avoid technical jargon, SQL queries, or recommendations.
                - The summary should be a single, focused paragraph.

                Summarize the following validation failure metadata:
                {user_input}
                        """
    logger.info('Summarizing the inputs recieved.')

    response = llm.invoke(prompt).content
    failed_rule = user_input["validation_query"]
    # st.session_state.messages.append({"role": "bot", "content": response})
    emit_message("bot", summary)
    return {"anamoly_node_response": response, "validation_query": failed_rule}

def analysis_decision_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    emit_progress("analysis_decision", "started", {"step": "Determining analysis approach"})
    
    decider_prompt = f"""As an expert SQL analyst, your task is to determine how columns in a data quality validation query contribute to metrics.
                    Analyze the provided SQL query to identify if:
                    1.  **Each column contributes to a distinct validation metric. In this case, the `path_to_follow` is "Equality".
                    2.  **Multiple columns contribute to a single, combined validation metric** (e.g.,calculation involving several columns for one data quality check). In this case, the `path_to_follow` is "Statistical".
                    SQL Query to Analyze:
                    ```sql
                    {state['validation_query']}
                    ```
                    ```Output JSON Format:
                    {{
                        "path_to_follow": "Equality" | "Statistical"
                    }}
                    ```
                    Respond only with JSON.
                    """

    response_str = llm.invoke(decider_prompt).content
    analysis_method = json_parser.invoke(response_str) # Make sure json_parser is defined
    logger.info(f'Deciding the path. Analysis Type : {analysis_method}')

    if analysis_method['path_to_follow'] == "Equality":
        message = "Based on provided inputs, proceeding with the failed column as the starting point for our root cause analysis."
    else:
        message = "Based on provided inputs, multiple columns are involved in this validation metric, we'll run a statistical check at the L0 layer to pinpoint the problematic column."
    
    emit_message("bot", message)
    emit_progress("analysis_decision", "completed", {"path": analysis_method['path_to_follow']})

    return {'analysis_method': analysis_method['path_to_follow']}

def parse_dq_query_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    emit_progress("parse_dq_query", "started", {"step": "Parsing DQ validation query"})
    
    validation_query = state['validation_query']
    prompt = f"""
            You are a SQL expert. Analyze the following data quality validation SQL query.
            **Extraction Rules:**
             1.  **`target_tables`**:
                 - Identify all tables from the `FROM` and `JOIN` clauses.
                 - For each table, list ONLY the columns on which the validation rule is directly defined.
                 - A validation column is typically found inside an aggregation function (e.g., `SUM(amount)`), a `CASE` statement (e.g., `CASE WHEN email IS NULL...`), or is being checked for a specific property.
                 - **IMPORTANT**: Do NOT include columns that are only used for grouping (`GROUP BY`) or filtering (`WHERE`) in this section.
            
             2.  **`filters`**:
                 - Extract all conditions from the `WHERE` clause.
                 - Each separate condition (usually connected by `AND` or `OR`) should be an element in the list.
                 - If there is no `WHERE` clause, return an empty list `[]`.
            
             3.  **`group_by`**:
                 - Extract all columns or expressions from the `GROUP BY` clause.
                 - **SPECIAL HANDLING**: If the `GROUP BY` clause uses numbers (e.g., `GROUP BY 1, 2`), you MUST map these numbers to the corresponding columns/expressions in the `SELECT` list (1-based index).
                 - If there is no `GROUP BY` clause, return an empty list `[]`.
            
             4.  **`aggregation_function`**:
                 - Identify the primary aggregation function used for the validation check (e.g., `COUNT`, `SUM`, `AVG`, `MAX`).
                 - If the query calculates a sum of flags (e.g., `SUM(CASE WHEN ... THEN 1 ELSE 0 END)`), the function is `SUM`.
                 - If it's a count of records (e.g., `COUNT(*)` or `COUNT(column)`), the function is `COUNT`.
                 - If no aggregation is present, return an empty string `""`.
            
            Don't use any aliases.

            Format the output as a JSON object. 
            Input SQL Query:
            {validation_query}
            Output JSON Format:
            {{
                "target_tables": {{
                        "table_name_1": ["column_name_A", "column_name_B"],
                        "table_name_2": ["column_name_C"]
                }},
                    "filters": ["extracted_where_clause_or_condition"],
                    "group_by": ["extracted_group_by_columns"],
                    "aggregation_function" : "sum/count/mean",
            }}
            Be precise in extracting filters and group by clauses exactly as they appear if possible, or describe them if complex.
            """
    logger.info("Parsing DQ query...")
    parser_chain = llm | json_parser
    parsed_info = parser_chain.invoke(prompt)
    logger.info(f"Parsed DQ Info: {parsed_info}")

    if 'parsed_dq_info' in parsed_info and isinstance(parsed_info['parsed_dq_info'], dict):
        parsed_info = parsed_info['parsed_dq_info']
    
    emit_message("bot", "Based on your inputs, here are the tables, columns, filters, group by, and aggregation functions found.")
    emit_table_data(parsed_info, "Parsed DQ Information")
    emit_progress("parse_dq_query", "completed")
    
    return {"parsed_dq_info": parsed_info}

def initialize_trace_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    Initializes the trace, performs the L0 statistical check, and formats the result
    into the new enriched trace_data structure.
    """
    emit_progress("initialize_trace", "started", {"step": "Performing L0 layer analysis"})
    emit_message("bot", "Identifying columns for root cause analysis and performing initial L0 checks...")

    parsed_info = state['parsed_dq_info']
    trace_data = {}
    paths_to_process = []
    mismatched_nodes = []
    loaded_graphs = {}
    
    if not parsed_info or 'target_tables' not in parsed_info or not parsed_info['target_tables']:
        logger.info("No tables found in parsed DQ info. Cannot initialize trace.")
        return {"trace_data": {}, "paths_to_process": [], "mismatched_nodes": []}

    filters = parsed_info.get('filters', [])
    group_by = parsed_info.get('group_by', [])
    aggregation = parsed_info.get('aggregation_function', '')

    for table, columns in parsed_info['target_tables'].items():
        if table not in trace_data:
            trace_data[table] = {}
        for column in columns:
            l0_sql_prompt = f"""
            Generate a SQL query for table '{table}' that selects the column '{column}' (and potentially other relevant columns like group_by columns or IDs)
            and apply the aggregation function '{aggregation}' based on the following filters and group by clauses extracted from a DQ rule:
            Filters: {', '.join(filters) if filters else 'None'}
            Group By: {', '.join(group_by) if group_by else 'None'}

            Consider the context of tracing data that might violate a DQ rule.
            Get the relational operator and thresholds for the column {column} from the reference validation query and use the same in the SQL query generation.
            The query should return data relevant to the DQ check on '{column}'. 
            Actual Validation query is provided for your reference. Reference Validation Query : {state['validation_query']}.
            Don't add any explanations to the query, just return the executable query only.

            Important constraint:
             - The CASE statement for the {column} column must be used exactly as it appears in the Reference Validation Query. But it should have only column {column} in it.
             - Generated SQL query should only have column {column} for DQ rule validation.
             - If there are any division or multiplier then use the same on individual column.
            """

            l0_query = extract_sql_from_text(llm.invoke(l0_sql_prompt).content)
            l0_query = replace_technical_date_with_business_date(l0_query, table.rsplit('.', 1)[-1])
            l0_output_df = bigquery_execute(l0_query) # Make sure bigquery_execute is defined
            actual_value = extract_metric_from_output(l0_output_df.to_json(orient='records'), l0_query)

            std_dev_prompt = f"""Generate a SQL query to calculate the standard deviation on column '{column}' from the previous date mentioned in the SQL query {l0_query}. 
                                The expected value and standard deviation should be calculated using data from the **last 30 days**, including the previous date. 
                                To calculate standard deviation group by date column and then calculate the standard deviation value.
                                The result from this query should only have the standard deviation value with the column names as expected_value and std_dev.
                                
                                Important Constraint :
                                - Use the case statement as is while getting the data for the previous dates.
                                - Apply the neccasary filters, group by while getting the data for the previous dates.
                                - Respond only with SQL query without any explaination.
                                - Resulting SQL query should be compatible with BigQuery.
                                - The standard deviation and expected value should be calculated using data from the 30 days prior to the date specified in the SQL query.
                                - The new query should calculate the standard deviation and expected value, but exclude the specific date mentioned in the original query. The date range for the calculation should end on the day before the specified date.
                                """

            std_dev_sql = extract_sql_from_text(llm.invoke(std_dev_prompt).content)
            std_dev_df = bigquery_execute(std_dev_sql)

            expected_std_dev = float(std_dev_df['std_dev'].iloc[0]) if not std_dev_df.empty else 0.0
            expected_value = float(std_dev_df['expected_value'].iloc[0]) if not std_dev_df.empty else 0.0

            lower_bound = expected_value - (3 * expected_std_dev)
            upper_bound = expected_value + (3 * expected_std_dev)

            is_out_of_bounds = not (lower_bound <= actual_value <= upper_bound) if actual_value is not None else True

            reasoning = (f"Value {actual_value:.2f} is outside the 3-sigma range [{lower_bound:.2f}, {upper_bound:.2f}]."
                         if is_out_of_bounds else f"Value {actual_value:.2f} is within the expected statistical range.")

            l0_trace_entry = {
                "step_num": 0, "table_name": table, "column_name": column,
                "sql_query": l0_query, "transformation_from_prev": "Initial Anomaly",
                "check_type": "statistical_trend_check", "actual_value": actual_value,
                "historical_expected_value": expected_value, "historical_std_dev": expected_std_dev,
                "lower_bound": lower_bound, "upper_bound": upper_bound,
                "is_mismatch": is_out_of_bounds, "mismatch_reason": reasoning
            }

            trace_data[table][column] = [l0_trace_entry]

            if is_out_of_bounds:
                l0_key = f"{table}.{column}"
                paths_to_process.append((0, table, column, l0_key))
                mismatched_nodes.append({"from_node": "External Alert", "to_node": l0_key, "details": l0_trace_entry})
                lineage_table_name = table.rsplit('.', 1)[-1]
                graph_file_path = f"{lineage_graphs}/{lineage_table_name}.{column}.gexf"
                # graph_file_path = f"lineage_graphs/{l0_key.replace(':','_')}.gexf"
                try:
                    lineage_graph = nx.read_gexf(graph_file_path)
                    loaded_graphs[l0_key] = lineage_graph
                    logger.info(f"Successfully loaded lineage graph for '{l0_key}' from {graph_file_path}")
                except:
                    logger.info(f"Error in loading lineage graph for '{l0_key}' from {graph_file_path}")
                
                inference = f"A significant deviation has been detected in the {column} column's values over the past month. A lineage trace is necessary to uncover the root cause."
            else:
                inference = f"The data in column {column} falls within the expected range of 3 standard deviations, indicating that lineage tracing isn't necessary."

            emit_analysis_result("L0", table, column, l0_query, reasoning, inference)

    emit_progress("initialize_trace", "completed", {"paths_found": len(paths_to_process)})
    logger.info(f"Initialized trace for {len(paths_to_process)} paths.")
    return {"trace_data": trace_data, "paths_to_process": paths_to_process, "mismatched_nodes": mismatched_nodes, "loaded_lineage_graphs" : loaded_graphs}

def databuck_failure_validation(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    Performs the initial check for a single-column failure and formats the result
    into the new enriched trace_data structure.
    """
    # Action 1: Announce validation start
    emit_message("bot", "This appears to be a single-column validation. I am now performing the initial check on layer L0 to validate the alert...")
    
    validation_query = state['validation_query']
    failed_column = state['failed_column']
    failed_table = state['failed_table']

    prompt = f"""Extract a SQL query from the provided "Validation SQL Query." The new query should:
                * Select only the column `{failed_column}` (retaining its original aggregation or case statement) and all other **non-aggregated** columns from the original `SELECT` clause.
                * Preserve all original `WHERE`, `GROUP BY`, and `ORDER BY` clauses.
                Validation SQL Query:
                ```sql
                {validation_query}"""
    response = llm.invoke(prompt).content
    individual_query = extract_sql_from_text(response)
    individual_query = replace_technical_date_with_business_date(individual_query, failed_table.rsplit('.', 1)[-1])
    query_output_df = bigquery_execute(individual_query)
    actual_value = extract_metric_from_output(query_output_df.to_json(orient='records'), individual_query)

    sd_threshold = state['sd_threshold']
    ev = state['expected_value']
    sd = state['expected_std_dev']
    lower_bound = ev - (sd_threshold * sd)
    upper_bound = ev + (sd_threshold * sd)

    is_out_of_bounds = not (lower_bound <= actual_value <= upper_bound) if actual_value is not None else True

    reasoning = (f"The actual value of {actual_value:.2f} is outside the specified threshold [{lower_bound:.2f}, {upper_bound:.2f}] based on historical data."
                 if is_out_of_bounds else f"The actual value of {actual_value:.2f} is within the specified threshold.")

    inference = ("Deviation confirmed. A lineage trace is necessary to uncover the root cause."
                 if is_out_of_bounds else "No deviation found. Lineage tracing is not required.")

    # Action 2: Emit structured analysis result
    emit_message("analysis_result", "L0 Validation Result", {
        "layer": "L0",
        "table": failed_table.rsplit('.', 1)[-1],
        "column": failed_column,
        "sql_query": individual_query,
        "reasoning": reasoning,
        "inference": inference
    })

    trace_data = {}
    paths_to_process = []
    mismatched_nodes = []
    loaded_graphs = {}
    

    if is_out_of_bounds:
        l0_trace_entry = {
            "step_num": 0, "table_name": failed_table, "column_name": failed_column,
            "sql_query": individual_query, "transformation_from_prev": "Initial Anomaly",
            "check_type": "statistical_trend_check", "actual_value": actual_value,
            "historical_expected_value": ev, "historical_std_dev": sd,
            "lower_bound": lower_bound, "upper_bound": upper_bound,
            "is_mismatch": is_out_of_bounds, "mismatch_reason": reasoning
        }

        trace_data[failed_table] = {failed_column: [l0_trace_entry]}
        l0_key = f"{failed_table}.{failed_column}"
        paths_to_process.append((0, failed_table, failed_column, l0_key))
        mismatched_nodes.append({"from_node": "External Alert", "to_node": l0_key, "details": l0_trace_entry})
        lineage_table_name = failed_table.rsplit('.', 1)[-1]
        graph_file_path = f"{lineage_graphs}/{lineage_table_name}.{failed_column}.gexf"
        try:
            lineage_graph = nx.read_gexf(graph_file_path)
            loaded_graphs[l0_key] = lineage_graph
            logger.info(f"Successfully loaded lineage graph for '{l0_key}' from {graph_file_path}")
        except:
            logger.info(f"Error in loading lineage graph for '{l0_key}' from {graph_file_path}")
        
        # Emit detailed message for deviation found case
        emit_message("bot", f"Analyzing the data on L0 layer for {failed_column}")
        emit_message("analysis_detail", f"L0 Analysis - Deviation Detected", {
            "column": failed_column,
            "table": failed_table.rsplit('.', 1)[-1],
            "reasoning": reasoning,
            "sql_query": individual_query,
            "conclusion": "A significant deviation has been detected in the column's values. A lineage trace is necessary to uncover the root cause."
        })
    else:
        # Emit detailed message for no deviation case
        emit_message("bot", f"Analyzing the data on L0 layer for {failed_column}")
        emit_message("analysis_detail", f"L0 Analysis - No Deviation", {
            "column": failed_column,
            "table": failed_table.rsplit('.', 1)[-1],
            "reasoning": reasoning,
            "sql_query": individual_query,
            "conclusion": f"The data in column {failed_column} falls within the expected range, indicating that lineage tracing isn't necessary."
        })

    return {"trace_data": trace_data, "paths_to_process": paths_to_process, "mismatched_nodes": mismatched_nodes, "loaded_lineage_graphs" : loaded_graphs}

def trace_backward_step_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs one step backward, generates queries, executes them, and performs
    value comparisons (Equality vs. Statistical) directly within the step.
    Only mismatched paths are added to the next queue.
    """
    trace_data = state['trace_data'].copy()
    paths_to_process = state['paths_to_process']
    mismatched_nodes = state.get('mismatched_nodes', [])
    loaded_graphs = state.get('loaded_lineage_graphs', {})

    if not paths_to_process:
        logger.info("No paths left to process.")
        return {"paths_to_process": []}

    # Emit step 3 message and lineage graph (only on first run)
    first_run = any(step_num == 0 for step_num, _, _, _ in paths_to_process)
    if first_run:
        emit_message("bot", "Checking for upstream dependencies based on data lineage ...")
        emit_message("bot", "Upstream dependencies based on data lineage")
        
        # Build and emit the initial graph structure
        for _, _, _, l0_key in paths_to_process:
            if l0_key in loaded_graphs:
                graph_structure = build_graph_structure_from_lineage(loaded_graphs[l0_key], l0_key)
                emit_lineage_graph(graph_structure)
                break  # Only emit one graph structure

    next_paths_to_process = []

    for current_step_num, current_table, current_column, l0_key in paths_to_process:
        full_node_name = f"{current_table}.{current_column}"

        lineage_graph = loaded_graphs.get(l0_key)
        if not lineage_graph:
            logger.warning(f"Lineage graph not found for L0 key: {l0_key}. Ending trace for this path.")
            continue

        predecessors = get_predecessor_info(full_node_name, lineage_graph)

        if not predecessors:
            logger.info(f"No predecessors found for {full_node_name}. Ending trace for this path.")
            continue

        l0_table, l0_column = l0_key.rsplit('.', 1)
       # print('trace_data', trace_data)
        current_node_data = next(
            (s for s in trace_data[l0_table][l0_column] if s['step_num'] == current_step_num and s['table_name'] == current_table),
            None
        )

        if not current_node_data:
            logger.error(f"Could not find trace data for {full_node_name} at step {current_step_num}")
            continue

        downstream_actual_value = current_node_data.get('actual_value')
        last_query = current_node_data.get('sql_query')
        is_multi_source = len(predecessors) > 1

        for pred in predecessors:
            pred_table, pred_column = pred['prev_table'], pred['prev_column']
            transformation_logic = pred['transformation']
            pred_table_display = pred_table.rsplit('.', 1)[-1]
            
            # Emit checking status for nodes (Step B from requirements)
            emit_node_status_update(pred_table_display, "checking")
            emit_node_status_update(f"{pred_table}.{pred_column}", "checking")
            
            # Emit conversational message for checking specific column
            emit_message("bot", f"Checking for {pred_column.upper()} Count ...")
            emit_step_result(3, "lineage_check", pred_table, pred_column, "", 
                           f"Identified the {pred_column} column originated from {pred_column} column from {pred_table_display} table.", "success")
            
            emit_message("bot", "I'm running the count checks for the previous date ...")
            
            check_result = {}
			
            if pred_column in ['close_yr_mth', 'bllr_bill_cyc_num']:
                check_result['check_type'] = 'null_check'
                is_mismatch = False
                null_check_prompt = f"""You are an expert SQL query generator specializing in data quality checks. Your task is to create a single, efficient SQL query to calculate the percentage of NULL values for a specific column in a table, based on a date filter from a reference query.
                                        **Your Instructions:**

                                        1.  **Analyze the Reference Query:** First, carefully examine the `reference_sql_query` to identify and extract the complete date filter condition from its `WHERE` clause. This could be a single equality (e.g., `activity_dt = '2024-07-15'`) or a range (e.g., `event_date BETWEEN '...' AND '...'`).                                       
                                        2.  **Construct the New Query:** Build a new SQL query that targets the specified `target_table_name` and `target_column_name`.                                    
                                        3.  **Apply the Date Filter:** In the `WHERE` clause of your new query, you must apply the *exact same* date filter condition you extracted in step 1.                                       
                                        4.  **Calculate Null Percentage:** The `SELECT` clause must calculate the percentage of rows where the `target_column_name` is NULL.
                                            *   The calculation should be: `(COUNT of NULL rows / Total COUNT of rows) * 100`.
                                            *   Use floating-point arithmetic (e.g., multiply by `100.0`) to ensure an accurate decimal result.
                                        5.  **Handle Edge Cases:** Your query MUST handle the case where the filtered result set is empty (`COUNT(*)` is 0). In this situation, you must return `100.0`, as this indicates the column is entirely empty for the given period.
                                        6.  **Format the Output:** The final query must return a single column named exactly `null_pct` with a single row containing the percentage value (from 0.0 to 100.0).

                                        **Context for this Task:**                                   
                                        *   **Target Table Name:** `{pred_table}`
                                        *   **Target Column Name:** `{pred_column}`
                                        *   **Reference SQL Query:**
                                            ```sql
                                            {last_query}
                                            ```

                                        **Example 1: Simple Date Filter**
                                        *   **Target Table Name:** `customer_profiles`
                                        *   **Target Column Name:** `email_address`
                                        *   **Reference SQL Query:** `SELECT status, COUNT(*) FROM daily_logins WHERE login_dt = '2023-11-10' GROUP BY status;`
                                        *   **Expected Output SQL:**
                                            ```sql
                                            SELECT
                                            CASE
                                                WHEN COUNT(*) = 0 THEN 100.0
                                                ELSE (SUM(CASE WHEN email_address IS NULL THEN 1.0 ELSE 0.0 END) * 100.0) / COUNT(*)
                                            END AS null_pct
                                            FROM
                                            customer_profiles
                                            WHERE
                                            login_dt = '2023-11-10';
                                            ```

                                        **Example 2: Date Range Filter**
                                        *   **Target Table Name:** `transactions`
                                        *   **Target Column Name:** `shipping_address_id`
                                        *   **Reference SQL Query:** `SELECT product_id, SUM(amount) FROM sales WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31' GROUP BY product_id;`
                                        *   **Expected Output SQL:**
                                            ```sql
                                            SELECT
                                            CASE
                                                WHEN COUNT(*) = 0 THEN 100.0
                                                ELSE (SUM(CASE WHEN shipping_address_id IS NULL THEN 1.0 ELSE 0.0 END) * 100.0) / COUNT(*)
                                            END AS null_pct
                                            FROM
                                            transactions
                                            WHERE
                                            event_date BETWEEN '2024-01-01' AND '2024-01-31';
                                            ```
                                        **Your Final Output:**

                                        Return only the complete, executable SQL query on Big Query. Do not add any explanations, markdown formatting, or introductory text."""
                backward_query = llm.invoke(null_check_prompt).content
                backward_query = extract_sql_from_text(backward_query)
                backward_query = replace_technical_date_with_business_date(backward_query, pred_table.rsplit('.', 1)[-1])
                null_check_resp = bigquery_execute(backward_query)
                    

                if null_check_resp['null_pct'][0] > 0.02:
                
                    check_result.update({
                        'is_mismatch': True,
                        'mismatch_reason': f"NULL value at this column is more than {null_check_resp['null_pct'][0]}%."
                    })
                else:
                    check_result.update({
                        'is_mismatch': False,
                        'mismatch_reason': f"No issue found for this column."
                    })
            elif not is_multi_source:
                # One-to-one or non-multi-source transformation
                if transformation_logic == 'One-to-One':
                    backward_query = last_query.replace(f"{current_table}.{current_column}", f"{pred_table}.{pred_column}")
                    logger.info(f"One-to-One transformation. Simplified query: {backward_query}")
                else:
                    backward_sql_prompt = f"""Your role is to act as an expert SQL query generator, specifically for data lineage analysis. Given an L{current_step_num} (downstream) SQL query and the transformation logic from L{current_step_num+1} (upstream) to L{current_step_num}, your task is to construct an L{current_step_num+1} SQL query. The critical requirement is that the data produced by the L{current_step_num+1} query, when passed through the given L{current_step_num+1} to L{current_step_num} transformation, must perfectly replicate the results of the L{current_step_num} query.
                                    Output only the resulting SQL query.

                                    **Input:**
                                    * **L{current_step_num} SQL Query:** {last_query}
                                    * **Lineage Information (L{current_step_num+1} to L{current_step_num}):**
                                        * **L{current_step_num} Table:** {current_table}
                                        * **L{current_step_num} Column:** {current_column}
                                        * **L{current_step_num+1} Table:** {pred_table}
                                        * **L{current_step_num+1} Column:** {pred_column}
                                        * **Transformation Logic:** {transformation_logic}

                                    **Important Considerations for the L{current_step_num+1} Query:**
                                    * Always preserve and include `GROUP BY` and `WHERE` clauses from the L{current_step_num} query.
                                    * Avoid direct aggregation on aggregation; use subqueries when needed.
                                    * The transformation logic provided must be used exactly as is, without any modifications to column relationships, filters, or `GROUP BY` statements.
                                    * Match `CASE` conditions and thresholds from the L{current_step_num} query in the L{current_step_num+1} query.
                                    * The ultimate goal is for the L{current_step_num+1} query to produce results that, when transformed, are bit-for-bit identical to the L{current_step_num} query's output.
                                    * Generate aggregated results in the L{current_step_num+1} query to enable comparisons with aggregated historical data."""
									
                    backward_query_response = llm.invoke(backward_sql_prompt).content
                    backward_query = extract_sql_from_text(backward_query_response)
                
                backward_query = replace_technical_date_with_business_date(backward_query, pred_table.rsplit('.', 1)[-1])
                
                backward_output_df = execute_and_correct_query(backward_query)
                upstream_actual_value = extract_metric_from_output(backward_output_df.to_json(orient='records'), backward_query)
                
                # Perform Statistical Check
                check_result = perform_statistical_check(pred_table, pred_column, backward_query, upstream_actual_value)

            else: # is_multi_source
                # Multi-source transformation logic
                pairwise_prompt = f"""
                            You are a SQL formula analyst. A downstream column `{current_column}` is derived from multiple upstream columns.
                            Your task is to isolate the part of the formula that is relevant ONLY to the target upstream column `{pred_column}`.

                            - Full Transformation Logic for `{current_column}`:
                            "{transformation_logic}"

                            - Target Upstream Column to Isolate: `{pred_column}`

                            Instructions:
                            1. Analyze the full transformation logic.
                            2. Extract and return ONLY the expression component that involves `{pred_column}`.
                            3. Preserve any aggregations (SUM, COUNT, etc.) or arithmetic operators (+, -, *, /) directly associated with it.
                            4. If the logic is a simple passthrough (e.g., just the column name), return the column name.


                            Return only the resulting isolated SQL snippet, with no explanation.

                            Example:
                            - Full Logic: "SUM(source.col_b) / NULLIF(SUM(source.col_c), 0)"
                            - Target Column: "col_b"
                            - Your Expected Output: "SUM(source.col_b)"

                            Example 2:
                            - Full Logic: "CASE WHEN source.status = 'A' THEN source.col_x ELSE 0 END - source.col_y"
                            - Target Column: "col_y"
                            - Your Expected Output: "- source.col_y"
                            """
                try:
                    isolated_transformation_logic = llm.invoke(pairwise_prompt).content.strip()
                    logger.info(f"Isolated logic for '{pred_column}': {isolated_transformation_logic}")
                except Exception as e:
                    logger.error(f"Failed to isolate transformation logic with LLM: {e}. Defaulting to full logic.")
                    isolated_transformation_logic = transformation_logic

                backward_sql_prompt = f"""
                                        Your role is to act as an expert SQL query generator for data lineage analysis.
                                        Given a downstream (L{current_step_num}) SQL query and the specific transformation logic from an upstream source (L{current_step_num+1}), construct the upstream SQL query.
                                        - Downstream (L{current_step_num}) SQL Query:
                                        ```sql
                                        {last_query}
                                        ```
                                        - Lineage Information (L{current_step_num+1} to L{current_step_num}):
                                        - Upstream Table: `{pred_table}`
                                        - Upstream Column: `{pred_column}`
                                        - **Isolated Transformation Logic**: "{isolated_transformation_logic}"

                                        Instructions:
                                        - Preserve all `GROUP BY` and `WHERE` clauses from the downstream query.
                                        - Generate an aggregated query for the upstream data using the provided isolated logic.
                                        - Use the same **FILTER DATE** as the date mentioned in the downstream SQL query.
                                        - Analyze the isolated transformation logic. For any table references or aliases found within this logic, check if the referenced table is already included in the main query. If it is, remove the alias or reference from the transformation logic.
                                        - Return only the executable BigQuery SQL.
                                        """
                backward_query_response = llm.invoke(backward_sql_prompt).content
                backward_query = extract_sql_from_text(backward_query_response)
                backward_query = replace_technical_date_with_business_date(backward_query, pred_table.rsplit('.', 1)[-1])

                backward_output_df = execute_and_correct_query(backward_query)
                upstream_actual_value = extract_metric_from_output(backward_output_df.to_json(orient='records'), backward_query)
                
                # Perform Statistical Check
                check_result = perform_statistical_check(pred_table, pred_column, backward_query, upstream_actual_value)

            next_step_num = current_step_num + 1

            new_step_data = {
                "step_num": next_step_num, "table_name": pred_table, "column_name": pred_column,
                "sql_query": backward_query, "transformation_from_prev": transformation_logic,
                **check_result
            }

            trace_data[l0_table][l0_column].append(new_step_data)

            if check_result['is_mismatch']:
                emit_node_status_update(f"{pred_table}.{pred_column}", "mismatch")
                emit_node_status_update(pred_table_display, "mismatch")
                
                # Emit results message matching Figma design
                emit_step_result(3, "count_check_result", pred_table, pred_column, "", 
                               f"Results from count checks for the previous date: The deviation in {pred_column} < 10 originates from the {pred_column} column in the '{pred_table_display}' table, indicating an issue at this earlier stage.", "warning")

                mismatched_nodes.append({
                    "l0_key": l0_key,
                    "from_node": full_node_name,
                    "to_node": f"{pred_table}.{pred_column}",
                    "details": new_step_data
                })
                next_paths_to_process.append((next_step_num, pred_table, pred_column, l0_key))
            else:
                emit_node_status_update(f"{pred_table}.{pred_column}", "match")
                emit_node_status_update(pred_table_display, "match")
                
                emit_step_result(3, "count_check_result", pred_table, pred_column, "", 
                               f"Data consistent at {pred_table_display}.{pred_column} - no issues found.", "success")

    # Check if no further upstream tables are involved
    if not next_paths_to_process and paths_to_process:
        current_columns = [col for _, _, col, _ in paths_to_process]
        emit_message("bot", f"No further upstream tables are involved in the lineage for {' or '.join(current_columns)}. Therefore, the analysis concludes here.")

    return {"trace_data": trace_data, "paths_to_process": next_paths_to_process, "mismatched_nodes": mismatched_nodes}

def execute_and_correct_query(query, retries=3):
    """
    Executes a BigQuery SQL query and attempts to correct it on failure.
    """
    for attempt in range(retries):
        try:
            output_df = bigquery_execute(query)
            if isinstance(output_df, pd.DataFrame):
                return output_df
        except Exception as e:
            logger.warning(f"Query failed on attempt {attempt+1}: {e}")
            sql_correction_prompt = f"""As an expert BigQuery SQL generator, your role is to debug and correct SQL queries based on error messages.
            Examine the BigQuery error output: {e}
            The original errored SQL query is: {query}
            Please generate the fully corrected and runnable BigQuery SQL query."""
            
            try:
                correction_response = llm.invoke(sql_correction_prompt).content
                corrected_query = extract_sql_from_text(correction_response)
                query = corrected_query # Update query for next attempt
            except Exception as llm_e:
                logger.error(f"Failed to get a corrected query from LLM: {llm_e}")
                return pd.DataFrame() # Return an empty DataFrame on LLM failure

    logger.error(f"Failed to execute query after {retries} attempts.")
    return pd.DataFrame()

def perform_statistical_check(table, column, query, actual_value):
    """
    Performs a statistical check on an upstream column.
    """
    check_result = {'check_type': 'statistical_trend_check'}
    std_dev_prompt = f"""Generate a SQL query to calculate the standard deviation on table {table} and column '{column}' from the previous date mentioned in the SQL query {query}.
    The expected value and standard deviation should be calculated using data from the **last 30 days**, including the previous date.
    To calculate standard deviation group by date column and then calculate the standard deviation value.
    The result from this query should only have the standard deviation value with the column names as expected_value and std_dev.

    Important Constraint :
    - Use the case statement as is while getting the data for the previous dates.
    - Apply the neccasary filters, group by while getting the data for the previous dates.
    - Respond only with SQL query without any explaination.
    - Resulting SQL query should be compatible with BigQuery.
    - The standard deviation and expected value should be calculated using data from the 30 days prior to the date specified in the SQL query.
    """
    std_dev_sql = extract_sql_from_text(llm.invoke(std_dev_prompt).content)
    std_dev_df = bigquery_execute(std_dev_sql)

    expected_value = float(std_dev_df['expected_value'].iloc[0]) if not std_dev_df.empty and 'expected_value' in std_dev_df.columns else 0.0
    std_dev = float(std_dev_df['std_dev'].iloc[0]) if not std_dev_df.empty and 'std_dev' in std_dev_df.columns else 0.0

    lower_bound = expected_value - (3 * std_dev)
    upper_bound = expected_value + (3 * std_dev)
    is_mismatch = not (lower_bound <= actual_value <= upper_bound) if actual_value is not None else True

    check_result.update({
        'actual_value': actual_value, 'historical_expected_value': expected_value,
        'historical_std_dev': std_dev, 'is_mismatch': is_mismatch,
        'mismatch_reason': f"Upstream value {actual_value:.2f} is outside its historical 3-sigma range [{lower_bound:.2f}, {upper_bound:.2f}]." if is_mismatch else "Value is within its historical statistical range."
    })
    return check_result

def generate_final_report_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    The final node in the graph. It synthesizes all findings from the entire trace
    into a single, consolidated report that summarizes all issues and root causes.
    """
    emit_message("bot", "The lineage trace is complete. Here is the final root cause summary.")
    
    trace_data = state.get("trace_data", {})
    all_mismatches = state.get("mismatched_nodes", [])

    if not trace_data:
        emit_message("error", "Analysis failed: No trace data was generated.")
        return {"analysis_results": {"error": "No trace data"}}

    # --- 1. GATHER AND CONSOLIDATE ALL DATA FROM THE ENTIRE TRACE ---
    # Consolidate initial issues from all L0 failures
    initial_issues = []
    all_trace_steps = []
    for l0_table, columns_data in trace_data.items():
        for l0_column, trace_steps in columns_data.items():
            initial_issues.append(trace_steps[0])
            all_trace_steps.extend(trace_steps)

    # Get a unique list of all datasets involved in the trace
    impacted_datasets = sorted(list(set(step['table_name'] for step in all_trace_steps)))

    # Build a representative lineage traversal summary table (e.g., from the first path)
    representative_trace_path = next(iter(trace_data.values()), {}).get(next(iter(next(iter(trace_data.values()), {}).keys()), None), [])
    lineage_summary_table = []

    for step in sorted(representative_trace_path, key=lambda x: x['step_num']):
        lineage_summary_table.append({
            "layer_step": f"Layer {step['step_num']}",
            "dataset_table": step['table_name'].rsplit('.',1)[-1],
            "test_performed": step.get('check_type', 'Initial Check')
        })

    # Find ALL deepest failure points across ALL paths
    deepest_failures = []
    if all_mismatches:
        max_step_num = max(m['details']['step_num'] for m in all_mismatches)
        deepest_failures = [m for m in all_mismatches if m['details']['step_num'] == max_step_num]

    # --- 2. CONSTRUCT THE MASTER SYNTHESIS PROMPT ---
    prompt = f"""
    You are a Lead Data Analyst responsible for creating a single, consolidated Root Cause Analysis (RCA) report that summarizes an entire investigation.
    The investigation may have started from multiple initial issues and found multiple root causes. Your job is to synthesize all the provided raw data into one cohesive report using the given JSON template.
    Infer high-level fields like 'Severity', 'Business Impact', and 'RCA Type' from the complete context. If data is missing for a field, use "N/A".

    **RAW CONSOLIDATED DATA FROM THE ENTIRE TRACE:**

    *   **List of All Initial Anomalies (L0) that triggered the trace:**
        ```json
        {json.dumps(initial_issues, indent=2)}
        ```
    
    *   **Consolidated List of All Deepest Failure Point(s) Identified:**
        ```json
        {json.dumps(deepest_failures, indent=2)}
        ```
        
    *   **A Representative Lineage Traversal Path:**
        ```json
        {json.dumps(lineage_summary_table, indent=2)}
        ```

    **INSTRUCTIONS:**
    Fill out every field in the `FINAL_JSON_TEMPLATE` below by synthesizing the provided data.
    - **summary.issue_detected**: **Summarize** all initial anomalies into one.
    - **summary.business_impact**: Infer a single, high-level business impact from all the issues.
    - **root_cause_analysis.cause_identified**: **Synthesize** all deepest failures sentence. Mention only the table and column names caused this issue. If there are multiple tables and columns, add them as contributing factors to the overall problem. 
    - **root_cause_analysis.rca_type**: Infer a single, overarching category for all root causes (e.g., 'Multiple Upstream Data Issues', 'Widespread Transformation Logic Failure').
    - **justification_reasoning**: Create a bulleted list that tells the general story of the investigation, from detection to the isolation of the root cause(s) within 20 words each.

    **FINAL JSON TEMPLATE (Your output MUST be this exact JSON structure):**
    ```json
    {{
      "summary": {{
        "issue_detected": "A short, synthesized description of all initial issues.",
        "severity": "High | Medium | Low",
        "business_impact": "A brief, inferred description of the overall business impact."
      }},
      "lineage_traversal": {json.dumps(lineage_summary_table)},
      "root_cause_analysis": {{
        "origin_node": "The deepest layer where issues were found, e.g., Layer 2 - Transformation Layer.Mention Layer Number and Table name and if you find it as External alert then mention L0 has the issue",
        "cause_identified": "A clear, synthesized description of all identified root causes.",
        "rca_type": "e.g., Multiple Upstream Data Issues",
        "supporting_evidence": "Plausible evidence you infer from the context, e.g., 'Analysis of multiple source files confirmed data corruption prior to ingestion.'"
      }},
      "justification_reasoning": [
        "A bullet point summarizing the initial detection of anomalies across multiple metrics/tables.",
        "A bullet point explaining how lineage tracing was used to investigate the upstream pipeline.",
        "A final bullet point confirming that the root cause(s) were isolated in specific source tables or transformation steps."
      ]
    }}
    ```
    """

    # 3. INVOKE LLM AND RENDER THE SINGLE REPORT
    try:
        report_str = llm.invoke(prompt).content
        #print('report_str', report_str)
        match = re.search(r'```json\n(\{.*?\})\n```', report_str, re.DOTALL)
        if match:
            cleaned_report_str = match.group(1)
            report_data = json.loads(cleaned_report_str)
            
            # Generate RCA ID and execution time
            rca_id = str(uuid.uuid4())[:8]
            execution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            
            # Create final comprehensive report structure
            final_report = {
                "rca_id": rca_id,
                "execution_time": execution_time,
                "report_data": report_data,
                "impacted_datasets": impacted_datasets
            }
            
            # Emit the comprehensive final report
            emit_message("final_report", "Root Cause Analysis Completed", {
                "report": final_report
            })
            
            # Generate friendly summary message for root cause
            if deepest_failures:
                root_cause_details = []
                for failure in deepest_failures:
                    details = failure['details']
                    table_name = details['table_name'].rsplit('.', 1)[-1]
                    column_name = details['column_name']
                    root_cause_details.append(f"'{column_name}' column in the '{table_name}' table")
                
                if len(root_cause_details) == 1:
                    final_summary = f"The {root_cause_details[0]} has deviated from its historical trend. This directly affected downstream columns in the target table."
                else:
                    final_summary = f"The {' and '.join(root_cause_details)} have deviated from their historical trends. This directly affected the downstream analysis."
                
                emit_message("final_summary", final_summary)
            else:
                emit_message("final_summary", "Analysis completed with no definitive root cause identified in the traced lineage.")
            
            # Emit feedback request and action buttons
            emit_message("bot", "Was this helpful?")
            emit_message("action_buttons", "", {
                "buttons": [
                    {"text": "Generate Summary Report", "action": "generate_report"},
                    {"text": "Start New Issue", "action": "start_new"},
                    {"text": "Export Trace", "action": "export_trace"}
                ]
            })
            
            return {"analysis_results": final_report["report_data"]}
        else:
            emit_message("error", "Could not parse the consolidated report from the LLM response.")
            return {"analysis_results": {"error": "LLM response parsing failed."}}
    except Exception as e:
        emit_message("error", f"An error occurred while generating the consolidated report: {e}")
        return {"analysis_results": {"error": str(e)}}
    

def should_continue_tracing(state: RootCauseAnalysisState):
    """Decides whether to continue the backward trace or end."""
    paths = state.get("paths_to_process", [])
    if not paths:
        logger.info("No more paths to process. Ending trace.")
        return "end_trace"
    # Add a max depth check to prevent infinite loops
    current_depth = paths[0][0]
    if current_depth >= 5: # Max 5 layers back
        logger.warning(f"Reached max trace depth of {current_depth}. Ending trace.")
        return "end_trace"

    logger.info(f"Continuing trace. {len(paths)} paths to process at depth {current_depth + 1}.")
    return "continue_trace"

workflow = StateGraph(RootCauseAnalysisState)
# Add nodes
workflow.add_node("issue_summarizer", anamoly_identifier_node)
workflow.add_node("rca_analysis_decision", analysis_decision_node)
workflow.add_node("dq_failure_validation", databuck_failure_validation)
workflow.add_node("parse_dq_query", parse_dq_query_node)
workflow.add_node("initialize_trace", initialize_trace_node)
workflow.add_node("lineage_traversal", trace_backward_step_node)
workflow.add_node("issue_analyser", generate_final_report_node)

# Set up edges
workflow.set_entry_point("issue_summarizer")
workflow.add_edge("issue_summarizer", "rca_analysis_decision")
workflow.add_edge("parse_dq_query", "initialize_trace")

# Branch based on analysis type
workflow.add_conditional_edges(
    "rca_analysis_decision",
    lambda state: "single_column" if state['analysis_method'] == "Equality" else "multi_column",
    {"single_column": "dq_failure_validation", "multi_column": "parse_dq_query"}
)

# After initial checks, decide whether to start the trace loop or end
workflow.add_conditional_edges(
    "dq_failure_validation",
    lambda state: "start_trace" if state.get("paths_to_process") else "end_analysis",
    {"start_trace": "lineage_traversal", "end_analysis": "issue_analyser"} # Go to analyser to give final summary
)

workflow.add_conditional_edges(
    "initialize_trace",
    lambda state: "start_trace" if state.get("paths_to_process") else "end_analysis",
    {"start_trace": "lineage_traversal", "end_analysis": "issue_analyser"} # Go to analyser to give final summary
)

# The main tracing loop
workflow.add_conditional_edges(
    "lineage_traversal",
    should_continue_tracing,
    {
        "continue_trace": "lineage_traversal",  # Loop back to continue tracing
        "end_trace": "issue_analyser"          # Exit loop and summarize results
    }
)

# Final edge to the end
workflow.add_edge("issue_analyser", END)

# Compile the graph
app = workflow.compile()

def main():
    """
    Main execution function that reads input and runs the RCA workflow
    """
    try:
        # Read input from stdin (sent by Node.js)
        input_data = sys.stdin.read().strip()
        
        if not input_data:
            emit_message("error", "No input data received")
            return
        
        # Parse the input JSON
        try:
            user_input = json.loads(input_data)
        except json.JSONDecodeError as e:
            emit_message("error", f"Invalid JSON input: {e}")
            return
        
        emit_message("bot", "Starting Root Cause Analysis process...")
        emit_progress("workflow", "started", {"input_received": True})
        
        # Run the workflow
        final_state = app.invoke(user_input)
        
        emit_progress("workflow", "completed")
        emit_message("bot", "Root Cause Analysis completed successfully!")
        
    except Exception as e:
        error_msg = f"Error in main execution: {str(e)}"
        logger.error(error_msg)
        emit_message("error", error_msg)

if __name__ == "__main__":
    main()
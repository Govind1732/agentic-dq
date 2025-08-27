import json
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
import sys
from datetime import datetime
warnings.filterwarnings('ignore')
# Configure logging
# logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Environment-based configuration
config = configparser.ConfigParser()

# Determine config path based on environment
environment = os.getenv('ADQ_ENVIRONMENT', 'local')
if environment == 'local':
    config_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", 'config.ini')
elif environment == 'dev':
    config_file_path = '/apps/opt/application/dev_smartdq/dev/agentic_dq/config/config.ini'
elif environment == 'prod':
    config_file_path = '/apps/opt/application/prod_smartdq/prod/agentic_dq/config/config.ini'
else:
    # Default fallback
    config_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", 'config.ini')

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

try:
    del os.environ['http_proxy']
    del os.environ['https_proxy']
    del os.environ['no_proxy']
except:
    pass
warnings.filterwarnings('ignore')
# Configure logging
# logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["VEGAS_API_KEY"] = VEGAS_API_KEY
os.environ["ENVIRONMENT"] = ENVIRONMENT

usecase_name = "adq_agent_data_volume" # Add your vegas usecase name here
context_name = "adq_data_volume_context" # Add your vegas context name here

#usecase_name = USECASE_NAME
#context_name = CONTEXT_NAME

# Import pyvegas LLM
try:
    from pyvegas.helpers.utils import set_proxy, unset_proxy
    from pyvegas.langx.llm import VegasChatLLM
    
    # Initialize pyvegas LLM
    set_proxy()
    llm = VegasChatLLM(usecase_name=usecase_name, context_name=context_name, temperature=0)
    PYVEGAS_AVAILABLE = True
    print(f"SUCCESS: pyvegas LLM initialized successfully with usecase: {usecase_name}, context: {context_name}")

except Exception as e:
    print(f"ERROR: pyvegas initialization failed: {e}")
    print("CRITICAL: Cannot proceed without pyvegas LLM")
    # Send error to frontend and exit using unified messaging
    def send_message_fallback(message_type: str, content: str = "", title: str = "", **kwargs):
        """Fallback function for sending messages during initialization"""
        try:
            message = {
                "title": title,
                "type": message_type,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            json_message = json.dumps(message, ensure_ascii=False)
            message_bytes = json_message.encode('utf-8')
            message_length = len(message_bytes)
            
            # Send length prefix followed by newline, then the message
            sys.stdout.buffer.write(f"{message_length}\n".encode('utf-8'))
            sys.stdout.buffer.write(message_bytes)
            sys.stdout.buffer.write(b"\n")
            sys.stdout.buffer.flush()
        except Exception as e:
            # Final fallback to basic print
            print(f"ERROR: Failed to send delimited message: {e}", file=sys.stderr)
            print(json.dumps(message), flush=True)
    
    send_message_fallback("error", f"Failed to initialize pyvegas LLM: {str(e)}", "Initialization Error")
    sys.exit(1)

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
    mismatched_nodes: Optional[List[Dict[str, Any]]] # Stores only nodes with discrepancies
    analysis_results: Optional[Dict[str, Any]]
    agent_input: str
    anamoly_node_response: str
    loaded_lineage_graphs : Optional[Dict[str, nx.DiGraph]]

# Global variable to track if we're running in web mode
WEB_MODE = True  # Set to True for web mode by default

def send_message(message_type: str, content: str = "", title: str = "", data: dict = None, **kwargs):
    """
    Unified message sender for all communication from Python to Node.js
    Sends plain text JSON messages (one per line) for reliable parsing.
    
    Args:
        message_type: Type of message ('update', 'step_result', 'node_status', 'lineage_tree', 'error', 'node_response')
        content: Main message content
        title: Message title (optional)
        data: Additional data payload (optional)
        **kwargs: Additional message-specific parameters
    """
    if not WEB_MODE:
        # Console mode - just print for debugging
        if message_type == "step_result":
            step_number = kwargs.get('step_number', 0)
            print(f"\n{'='*60}")
            print(f"STEP {step_number}: {title}")
            print(f"{'='*60}")
            print(content)
            if data:
                print(f"Data: {data}")
            print(f"{'='*60}\n")
        elif message_type == "node_status":
            node_id = kwargs.get('node_id', '')
            status = kwargs.get('status', '')
            print(f"📊 NODE STATUS: {node_id} -> {status}")
            if content:
                print(f"   Message: {content}")
        elif message_type == "node_response":
            node_name = kwargs.get('node_name', '')
            print(f"🔗 NODE RESPONSE: {node_name}")
            print(f"   {content}")
        else:
            print(f"🤖 {title}: {content}")
        return

    # Web mode - send structured message to Node.js
    message = {
        "timestamp": datetime.now().isoformat()
    }
    
    if message_type == "update":
        message.update({
            "title": title,
            "type": "update", 
            "content": content
        })
        
    elif message_type == "step_result":
        step_number = kwargs.get('step_number', 0)
        message.update({
            "title": f"Step {step_number}: {title}",
            "type": "step_result",
            "content": content,
            "step_number": step_number,
            "data": data or {}
        })
        
    elif message_type == "node_response":
        # Real-time node response for immediate feedback
        node_name = kwargs.get('node_name', title)
        message.update({
            "title": title or f"{node_name} Complete",
            "type": "node_response",
            "content": content,
            "node_name": node_name,
            "data": data or {}
        })
        
    elif message_type == "node_status":
        node_id = kwargs.get('node_id', '')
        status = kwargs.get('status', '')
        # Map internal status to frontend status
        status_mapping = {
            'checking': 'checking',
            'completed_success': 'match', 
            'completed_failure': 'mismatch'
        }
        frontend_status = status_mapping.get(status, status)
        
        message.update({
            "type": "NODE_STATUS",
            "nodeId": node_id,
            "status": frontend_status,
            "message": content
        })
        
    elif message_type == "lineage_tree":
        message.update({
            "type": "LINEAGE_TREE",
            "nodes": kwargs.get('nodes', []),
            "edges": kwargs.get('edges', [])
        })
        
    elif message_type == "error":
        message.update({
            "title": title or "Error",
            "type": "error",
            "content": content
        })
    
    # Send plain text JSON message (one message per line)
    try:
        json_message = json.dumps(message, ensure_ascii=False)
        # print(json_message, flush=True)
    except Exception as e:
        # Fallback error handling
        print(f"ERROR: Failed to send message: {e}", file=sys.stderr)
        print(json.dumps({"type": "error", "content": str(e), "timestamp": datetime.now().isoformat()}), flush=True)

# Legacy compatibility wrappers - all now use unified send_message()
# def send_update(title, update_type, content):
#     """Legacy compatibility wrapper - use send_message() instead"""
#     send_message("update", content, title)

# def send_step_result(step_number, step_title, content, data=None):
#     """Legacy compatibility wrapper - use send_message() instead"""
#     send_message("step_result", content, step_title, data, step_number=step_number)

# def send_node_status_update(table_name, column_name, status, message=""):
#     """Legacy compatibility wrapper - use send_message() instead"""
#     send_message("node_status", message, node_id=f"{table_name}.{column_name}", status=status)

# def send_delimited_message(message_dict):
#     """Legacy compatibility wrapper - use send_message() instead"""
#     msg_type = message_dict.get("type", "update")
#     title = message_dict.get("title", "")
#     content = message_dict.get("content", "")
    
#     if msg_type == "LINEAGE_TREE":
#         send_message("lineage_tree", nodes=message_dict.get("nodes", []), edges=message_dict.get("edges", []))
#     elif msg_type == "NODE_STATUS":
#         send_message("node_status", message_dict.get("message", ""), 
#                     node_id=message_dict.get("nodeId", ""), status=message_dict.get("status", ""))
#     else:
#         send_message(msg_type.lower() if msg_type else "update", content, title)

# def display_message(role, content):
#     """Legacy compatibility wrapper - suppressed in web mode for cleaner 3-step workflow"""
#     # Suppressed in web mode - only key step results are sent
#     if not WEB_MODE:
#         if role == "bot":
#             print(f"{BOT_AVATAR} {CHATBOT_NAME}: {content}")
#         elif role == "user":
#             print(f"{USER_AVATAR} User: {content}")

                
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
    os.environ['http_proxy'] = 'http://proxy.ebiz.verizon.com:80/'
    os.environ['https_proxy'] = 'http://proxy.ebiz.verizon.com:80/'
    os.environ['no_proxy'] = 'http://proxy.ebiz.verizon.com:80/'
    logger.info('Retrieving JWT from OIDC provider...')
    url = 'https://ssologinuat.verizon.com/ngauth/oauth2/realms/root/realms/employee/access_token'
    #url = 'https://ssologin.verizon.com/ngauth/oauth2/realms/root/realms/employee/access_token' # production url
    payload = {'grant_type':'client_credentials','client_id':client_id,'client_secret':client_secret,'scope':'read'}

    try:
        response = requests.post(url=url,params=payload)
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
    #clientid and secret should be vaulted
    client_id = CLIENT_ID
    client_secret = CLIENT_SECRET
    project_id = PROJECT_ID
    # get a jwt
    if isTokenExpired(OIDC_TOKEN_PATH):
        exchange_and_save_oidc_token_for_jwt(client_id=client_id,client_secret=client_secret)
    # set the GOOGLE_APPLICATION_ENVIRONMENT
    logger.info('Setting environment variable...')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SA_PATH
    os.environ['GOOGLE_CLOUD_PROJECT']= GOOGLE_CLOUD_PROJECT
    credentials, auth_project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    logger.info(f"project_id={project_id}, credentials={credentials}")
    try:
        client = bigquery.Client(credentials=credentials, project=project_id)
        logger.info(f"Query: {query}")
        df_results = client.query(query).to_dataframe()
        return df_results
    except Exception as e :
        error_message = f"Error executing BigQuery query: {e}"
        logger.error(error_message)
        # Send error to frontend instead of returning error string
        send_message("error", error_message, "BigQuery Error")
        raise Exception(error_message)

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
            logger.info(f"LLM didnt return the actual value:{response}")
            return 0
    except:
        logger.error("Failed to get actual value")
        return None

def get_predecessor_info(node_name: str, lineage_graph: nx.DiGraph) -> List[Dict[str, str]]:
    """
    Finds and returns information about direct predecessors of a given node in the lineage graph.
    A node name is expected in 'table.column' format.
    """
    predecessors_list = []

    # In a DiGraph from lineage, successors of a node are its upstream sources.
    for pred_node in lineage_graph.successors(node_name):
        # The edge from node_name to pred_node describes the transformation
        transformation = lineage_graph.edges[node_name, pred_node].get('transformation', 'N/A')
        predecessors_list.append({
            'prev_table': pred_node.rsplit('.', 1)[0],
            'prev_column': pred_node.rsplit('.', 1)[1],
            'transformation': transformation,
            'source_node_full_name': pred_node
        })

    return predecessors_list

def replace_technical_date_with_business_date(sql_query : str, table_name : str):
    logger.info('Updating the Query technical dates with business dates')
    
    print("Query before replacing business date : ", sql_query)
    
    date_df = pd.read_csv(RULES_CSV_PATH)
    row = date_df[date_df['BQ Table Name'].str.lower() == table_name.lower()]
    
    if row.empty:
        return sql_query
    else:
        business_date = row['Business date'].values[0]
        print("table and column name",table_name, business_date)
        
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
        print("Query after replacing business date : ", sql_query)
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
    return text # Return original text if no SQL found

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
    
    # Print to console for command-line testing
    if not WEB_MODE:
        print(f"\n{'='*60}")
        print("🔍 ANOMALY IDENTIFIER NODE RESPONSE")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")
    
    # Send real-time node response for immediate frontend feedback
    send_message("node_response", response, "Anomaly Analysis Complete", 
                node_name="anomaly_identifier")
    
    return {"anamoly_node_response": response, "validation_query": failed_rule}

def analysis_decision_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
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
    decision_message = ""
    if analysis_method['path_to_follow'] == "Equality":
        decision_message = f"Based for provided inputs, proceeding with the failed column as the starting point for our root cause analysis."
    else:
        decision_message = f"Based for provided inputs, multiple columns are involved in this validation metric, we'll run a statistical check at the L0 layer to pinpoint the problematic column."
    
    # Print to console for command-line testing
    if not WEB_MODE:
        print(f"\n{'='*60}")
        print("🎯 ANALYSIS DECISION NODE RESPONSE")
        print(f"{'='*60}")
        print(f"Path to follow: {analysis_method['path_to_follow']}")
        print(f"Decision: {decision_message}")
        print(f"{'='*60}\n")
    
    # Send real-time node response for immediate frontend feedback
    send_message("node_response", decision_message, "Analysis Path Determined", 
                node_name="analysis_decision", 
                data={"path_to_follow": analysis_method['path_to_follow']})
    
    return {'analysis_method' : analysis_method['path_to_follow']}

def parse_dq_query_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    Parses the user's DQ SQL query using an LLM to extract key components.
    """
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
    try:
        del os.environ['http_proxy']
        del os.environ['https_proxy']
        del os.environ['no_proxy']
    except:
        pass
    logger.info(f"Parsed DQ Info: {parsed_info}")

    if 'parsed_dq_info' in parsed_info and isinstance(parsed_info['parsed_dq_info'], dict):
        parsed_info = parsed_info['parsed_dq_info']

    # Print to console for command-line testing
    if not WEB_MODE:
        print(f"\n{'='*60}")
        print("📊 PARSE DQ QUERY NODE RESPONSE")
        print(f"{'='*60}")
        print(json.dumps(parsed_info, indent=2))
        print(f"{'='*60}\n")

    # Send real-time node response for immediate frontend feedback
    parsed_summary = f"Successfully parsed validation query and extracted:\n"
    parsed_summary += f"• {len(parsed_info.get('target_tables', {}))} target table(s)\n"
    for table, columns in parsed_info.get('target_tables', {}).items():
        table_short = table.split('.')[-1] if '.' in table else table
        parsed_summary += f"  - {table_short}: {', '.join(columns)}\n"
    parsed_summary += f"• {len(parsed_info.get('filters', []))} filter condition(s)\n"
    parsed_summary += f"• {len(parsed_info.get('group_by', []))} grouping column(s)\n"
    parsed_summary += f"• Aggregation function: {parsed_info.get('aggregation_function', 'none')}"
    
    send_message("node_response", parsed_summary, "Query Parsing Complete", 
                node_name="parse_dq_query", 
                data=parsed_info)

    return {"parsed_dq_info": parsed_info}

def initialize_trace_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    Initializes the trace, performs the L0 statistical check, and formats the result
    into the new enriched trace_data structure.
    """

    # display_message('bot', "Identifying the columns that required root cause analysis...")  # Suppressed in web mode
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
            # Send status update that we are checking this node
            send_message("node_status", f"Initializing trace for {table}.{column}", 
                        node_id=f"{table}.{column}", status="checking")
            
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

            try:
                del os.environ['http_proxy']
                del os.environ['https_proxy']
                del os.environ['no_proxy']
            except:
                pass

            l0_query = extract_sql_from_text(llm.invoke(l0_sql_prompt).content)
            l0_query = replace_technical_date_with_business_date(l0_query, table.rsplit('.', 1)[-1])
            l0_output_df = bigquery_execute(l0_query) # Make sure bigquery_execute is defined
            actual_value = extract_metric_from_output(l0_output_df.to_json(orient='records'), l0_query)
            
            # Send node response for L0 query generation
            send_message("node_response", 
                        f"Generated L0 analysis query for {table.split('.')[-1]}.{column}", 
                        "L0 Query Generated", 
                        node_name="initialize_trace",
                        data={"query": l0_query[:200] + "..." if len(l0_query) > 200 else l0_query})
            

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
                                """

            std_dev_sql = extract_sql_from_text(llm.invoke(std_dev_prompt).content)
            
            # Send node response for statistical analysis
            send_message("node_response", 
                        f"Calculated statistical baseline for {table.split('.')[-1]}.{column} using 30-day historical data", 
                        "Statistical Analysis Complete", 
                        node_name="initialize_trace")
            
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

                # display_message("bot", f""" Analysing the data on L0 layer for <b>{column}</b><br><br>
                #                         <b>Reasoning:</b> {parsed_llm_response.get('reasoning')}<br><br>
                #                         <b>SQL query for actual value:</b><br>
                #                         <pre><code>{l0_query}</code></pre><br>
                #                         <b>SQL query for SD and EV:</b><br>
                #                         <pre><code>{std_dev_sql}</code></pre><br>
                #                         A significant deviation has been detected in the {column} column's values over the past month. A lineage trace is necessary to uncover the root cause.
                #                         """)  # Suppressed in web mode
                
            else:
                # Send completed success status for columns that are within bounds
                send_message("node_status", f"Analysis complete for {table}.{column}", 
                           node_id=f"{table}.{column}", status="completed_success")

                # display_message("bot", f"""Analysing the data on L0 layer for <b>{column}</b><br><br>
                #                         <b>Reasoning:</b> {parsed_llm_response.get('reasoning')}<br><br>
                #                         <b>SQL query for actual value:</b><br>
                #                         <pre><code>{l0_query}</code></pre><br>
                #                         <b>SQL query for SD and EV:</b><br>
                #                         <pre><code>{std_dev_sql}</code></pre><br>
                #                         The data in column <b>{column}</b> falls within the expected range of 3 standard deviations, indicating that lineage tracing isn't necessary.
                #                         """)  # Suppressed in web mode
    
    # Print to console for command-line testing
    if not WEB_MODE:
        print(f"\n{'='*60}")
        print("🔍 INITIALIZE TRACE NODE RESPONSE")
        print(f"{'='*60}")
        print(f"Columns requiring root cause analysis: {len(paths_to_process)}")
        for _, table, column, _ in paths_to_process:
            print(f"  - {table}.{column}")
        print(f"Initial check results:")
        for table, table_results in result.items():
            for column, column_results in table_results.items():
                for check_result in column_results:
                    status = "OUT OF BOUNDS" if check_result["is_out_of_bounds"] else "WITHIN BOUNDS"
                    print(f"  - {table}.{column}: {status}")
                    print(f"    Actual: {check_result['actual_value']}, Range: [{check_result['lower_bound']:.2f}, {check_result['upper_bound']:.2f}]")
        print(f"{'='*60}\n")
                
    logger.info(f"Initialized trace for {len(paths_to_process)} paths.")
    return {"trace_data": trace_data, "paths_to_process": paths_to_process, "mismatched_nodes": mismatched_nodes, "loaded_lineage_graphs" : loaded_graphs}
    
    # Send Step 1 result: Initial Anomaly Validation & Trend Analysis
    if WEB_MODE:
        step1_content = f"✅ Validation rule parsed and statistical analysis completed.\n\n"
        step1_content += f"📊 Found {len(paths_to_process)} column(s) requiring root cause analysis:\n"
        
        for _, table, column, _ in paths_to_process:
            table_short = table.split('.')[-1]
            step1_content += f"• **{table_short}.{column}**\n"
            
        step1_content += f"\n🔍 Statistical trend analysis confirmed significant deviation (>3 standard deviations) from historical baseline."
        
        step1_data = {
            "columns_analyzed": len(paths_to_process),
            "problematic_columns": [{"table": table.split('.')[-1], "column": column} for _, table, column, _ in paths_to_process],
            "validation_rule": state.get('validation_query', ''),
            "threshold_type": "3_standard_deviations"
        }
        
        send_message("step_result", step1_content, "Initial Anomaly Validation & Trend Analysis", 
                    data=step1_data, step_number=1)
    
    return {"initial_check_result": result, "trace_data": trace_data, "paths_to_process": paths_to_process}
	
def databuck_failure_validation(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    Performs the initial check for a single-column failure and formats the result
    into the new enriched trace_data structure.
    """
    # display_message('bot', "Performing initial check on L0 to validate the alert...")
    validation_query = state['validation_query']
    failed_column = state['failed_column']
    failed_table = state['failed_table']
    
    # Send status update that we are checking this node
    send_message("node_status", f"Starting validation for {failed_table}.{failed_column}", 
                node_id=f"{failed_table}.{failed_column}", status="checking")

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

    reasoning = (f"Value {actual_value:.2f} is outside the specified threshold [{lower_bound:.2f}, {upper_bound:.2f}]."
                 if is_out_of_bounds else f"Value {actual_value:.2f} is within the specified threshold.")

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
        # display_message("bot", f""" Analysing the data on L0 layer for <b>{failed_column}</b><br><br>
        #                             <b>Reasoning:</b> {parsed_llm_response.get('reasoning')}<br><br>
        #                             <b>SQL query for actual value:</b><br>
        #                             <pre><code>{individual_query}</code></pre><br>
        #                             A significant deviation has been detected in the {failed_column} column's values over the past month. A lineage trace is necessary to uncover the root cause.
        #                             """)
    else:
        # Send completed success status for columns that are within bounds
        send_message("node_status", f"Validation successful for {failed_table}.{failed_column}", 
                   node_id=f"{failed_table}.{failed_column}", status="completed_success")
        
        # display_message("bot", f""" Analysing the data on L0 layer for <b>{failed_column}</b><br><br>
        #                             <b>Reasoning:</b> {parsed_llm_response.get('reasoning')}<br><br>
        #                             <b>SQL query for actual value:</b><br>
        #                             <pre><code>{individual_query}</code></pre><br>
        #                             The data in column <b>{failed_column}</b> falls within the expected range of 3 standard deviations, indicating that lineage tracing isn't necessary.
        #                             """)
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

    # display_message('bot', "Tracing one layer back in the lineage and comparing data values...")
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
                # display_message("bot", f"""Traversing through lineage for <b>{current_column}</b><br><br>
                #                             <b>Layer Number:</b> {next_step_num}<br><br>
                #                             <b>Table Name:</b> {pred_table}<br><br>
                #                             <b>Column Name:</b> {pred_column}<br><br>
                #                             <b>SQL query:</b><br>
                #                             <pre><code>{backward_query}</code></pre><br><br>
                #                             <b>Inference:</b> DISCREPANCY FOUND at L{next_step_num} ({pred_table}.{pred_column}):<br><b>Reason:</b> {check_result['mismatch_reason']}""")

                mismatched_nodes.append({
                    "l0_key": l0_key,
                    "from_node": full_node_name,
                    "to_node": f"{pred_table}.{pred_column}",
                    "details": new_step_data
                })
                next_paths_to_process.append((next_step_num, pred_table, pred_column, l0_key))
            else:
                pass
                # display_message("bot", f"""Traversing through lineage for <b>{current_column}</b><br><br>
                #                             <b>Layer Number:</b> {next_step_num}<br><br>
                #                             <b>Table Name:</b> {pred_table}<br><br>
                #                             <b>Column Name:</b> {pred_column}<br><br>
                #                             <b>SQL query:</b><br>
                #                             <pre><code>{backward_query}</code></pre><br><br>
                #                             <b>Inference:</b> Data consistent at L{next_step_num} ({pred_table}.{pred_column}):<br><b>Reason:</b> {check_result['mismatch_reason']}""")

    print('next_paths_to_process', next_paths_to_process)
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


def display_report_on_streamlit(report_data: dict, impacted_datasets):
    """
    Takes the final report JSON and renders it in Streamlit
    using the specified template format.
    """

    rca_id = str(uuid.uuid4())[:8]
    execution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    summary_df = pd.DataFrame([
        {"Field": "RCA ID", "Description": rca_id},
        {"Field": "Date/Time", "Description": execution_time},
        {"Field": "Issue Detected", "Description": report_data.get("summary", {}).get("issue_detected", "N/A")},
        {"Field": "Severity", "Description": report_data.get("summary", {}).get("severity", "N/A")},
        {"Field": "Impacted Dataset(s)", "Description": json.dumps(impacted_datasets)},
        {"Field": "Business Impact", "Description": report_data.get("summary", {}).get("business_impact", "N/A")},
    ])

    # st.table(summary_df)
    
    # display_message('bot', f"""<b>1. Summary : </b><br><br>
    #                               {summary_df.to_html(justify='left',index=False)}""")


    traversal_df = pd.DataFrame(report_data.get("lineage_traversal", []))
    if not traversal_df.empty:
        
        # display_message('bot', f"""<b>2. Pipeline Lineage Traversal Summary : </b><br><br>
        #                           {traversal_df.rename(columns={
        #     "layer_step": "Layer / Step",
        #     "dataset_table": "Dataset / Table",
        #     "test_performed": "Test Performed"
        # }).to_html(justify='left',index=False)}""")
    else: 
        pass
        # st.write("N/A")

    rca_df = pd.DataFrame([
        {"Attribute": "Origin Node", "Details": report_data.get("root_cause_analysis", {}).get("origin_node", "N/A")},
        {"Attribute": "Cause Identified", "Details": report_data.get("root_cause_analysis", {}).get("cause_identified", "N/A")},
        {"Attribute": "RCA Type", "Details": report_data.get("root_cause_analysis", {}).get("rca_type", "N/A")},
        {"Attribute": "Supporting Evidence", "Details": report_data.get("root_cause_analysis", {}).get("supporting_evidence", "N/A")},
    ])
    
    # display_message('bot', f"""<b>3. Root Cause Analysis: </b><br><br>
    #                               {rca_df.to_html(justify='left',index=False)}""")

    justification_points = report_data.get("justification_reasoning", [])
    if justification_points:
        justification_text = "<br>".join([f"- {point}" for point in justification_points])
        # st.markdown(justification_text)
        # display_message('bot', f"""<b>4. Justification & Reasoning: </b><br><br>
        #                           {justification_text}""")
    else:
        # st.write("N/A")
        pass

    report_data ={"closure": {
                    "rca_status": "Open",
                    "owner": "Data Engineering Team",
                    "eta_for_fix": "N/A"
                }}
    closure_df = pd.DataFrame([
        {"Field": "RCA Status", "Description": report_data.get("closure", {}).get("rca_status", "N/A")},
        {"Field": "Owner", "Description": report_data.get("closure", {}).get("owner", "N/A")},
        {"Field": "ETA for Fix", "Description": report_data.get("closure", {}).get("eta_for_fix", "N/A")},
    ])
    # display_message('bot', f"""<b>5. Closure: </b><br><br>
    #                               {closure_df.to_html(justify='left',index=False)}""")
    
# --- The New Final Node for the Graph ---
def generate_final_report_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    The final node in the graph. It synthesizes all findings from the entire trace
    into a single, consolidated report that summarizes all issues and root causes.
    """
    # st.header("Consolidated Root Cause Analysis Report")

    trace_data = state.get("trace_data", {})
    all_mismatches = state.get("mismatched_nodes", [])

    if not trace_data:
        st.error("Analysis failed: No trace data was generated.")
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
            
            # Display the single, consolidated report
            display_report_on_streamlit(report_data, impacted_datasets)
            return {"analysis_results": report_data}
        else:
            st.error("Could not parse the consolidated report from the LLM response.")
            return {"analysis_results": {"error": "LLM response parsing failed."}}
    except Exception as e:
        st.error(f"An error occurred while generating the consolidated report: {e}")
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
    """Main function to handle different input modes"""
    global WEB_MODE
    
    try:
        if len(sys.argv) > 1:
            # Command line argument provided
            if sys.argv[1] == '--json':
                # Web mode: Direct JSON from Node.js backend
                WEB_MODE = True
                print("🔍 Processing direct JSON input from backend")
                
                if len(sys.argv) < 3:
                    send_message("error", "JSON data not provided", "Input Error")
                    sys.exit(1)
                
                try:
                    # Parse JSON directly - no encoding/decoding needed
                    json_data = sys.argv[2]
                    user_input = json.loads(json_data)
                    print(f"✅ Successfully parsed JSON input {user_input}")

                except Exception as e:
                    send_message("error", f"Failed to parse JSON input: {e}", "Parse Error")
                    sys.exit(1)
                    
            else:
                # File mode: JSON file path provided
                WEB_MODE = False  # Console output mode for file input
                json_file_path = sys.argv[1]
                print(f"🔍 Reading input from JSON file: {json_file_path}")
                
                try:
                    with open(json_file_path, 'r') as f:
                        user_input = json.load(f)
                    print("✅ Successfully loaded JSON from file")
                except Exception as e:
                    print(f"❌ Error reading JSON file: {e}")
                    sys.exit(1)
        else:
            print("❌ No input provided. Usage:")
            print("  python adq_agents.py <json_file>")
            print("  python adq_agents.py --json '<json_string>'")
            sys.exit(1)

        # Send initial status
        send_message("update", "Initializing Agentic Data Quality Root Cause Analysis...", "RCA Process Started")
        
        # Use the input directly as the initial state (already in correct format from Node.js)
        initial_state = user_input
        
        # Invoke the LangGraph workflow
        app.invoke(initial_state)
        
        # Send final summary
        send_message("update", "Root Cause Analysis completed.", "Final Summary")
        
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        send_message("error", error_message, "Error")
        print(f"❌ {error_message}")
        if not WEB_MODE:
            import traceback
            print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # Re-raise SystemExit to preserve exit codes
        raise
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        if WEB_MODE:
            send_message("Error", "error", error_msg)
        else:
            print(f"❌ ERROR: {error_msg}")
        sys.exit(1)
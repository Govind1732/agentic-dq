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
    # Send error to frontend and exit
    def send_delimited_message_fallback(message_dict):
        """Fallback function for sending delimited messages during initialization"""
        try:
            json_message = json.dumps(message_dict, ensure_ascii=False)
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
            print(json.dumps(message_dict), flush=True)
    
    def send_update(title, update_type, content):
        """Send structured updates to the Node.js server via stdout"""
        update = {
            "title": title,
            "type": update_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        send_delimited_message_fallback(update)
    send_update("Initialization Error", "error", f"Failed to initialize pyvegas LLM: {str(e)}")
    sys.exit(1)

json_parser = JsonOutputParser()

class RootCauseAnalysisState(TypedDict):
    failed_column : str
    failed_table : str
    analysis_method : Dict[str, Any]
    validation_query : str
    expected_value : float
    expected_std_dev : float
    sd_threshold : float
    initial_check_result : Optional[Dict]
    parsed_dq_info : Dict[str, Any]
    trace_data : Optional[Dict] 
    paths_to_process: Optional[List[Tuple[int, str, str, str]]]
    analysis_results: Optional[Dict[str, Any]]
    agent_input: str
    anamoly_node_response: str

# Global variable to track if we're running in web mode
WEB_MODE = True  # Set to True for web mode by default

def send_delimited_message(message_dict):
    """Send a length-prefixed JSON message to Node.js for reliable parsing"""
    try:
        json_message = json.dumps(message_dict, ensure_ascii=False)
        message_bytes = json_message.encode('utf-8')
        message_length = len(message_bytes)
        
        # Send length prefix followed by newline, then the message
        sys.stdout.buffer.write(f"{message_length}\n".encode('utf-8'))
        sys.stdout.buffer.write(message_bytes)
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
    except Exception as e:
        # Fallback to basic print if there's an issue with delimited messaging
        print(f"ERROR: Failed to send delimited message: {e}", file=sys.stderr)
        print(json.dumps(message_dict), flush=True)

def send_update(title, update_type, content):
    """Send structured updates to the Node.js server via stdout"""
    update = {
        "title": title,
        "type": update_type,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    send_delimited_message(update)
    sys.stdout.flush()

def send_node_status_update(table_name: str, column_name: str, status: str, message: str = ""):
    """
    Send a node status update to the frontend for lineage tree visualization.
    
    Args:
        table_name: The name of the table
        column_name: The name of the column
        status: One of 'checking', 'completed_success', 'completed_failure'
        message: Optional status message
    """
    # Use the same node ID format as lineage tree
    node_id = f"{table_name}.{column_name}"
    
    # Map internal status to frontend status for React Flow
    status_mapping = {
        'checking': 'checking',
        'completed_success': 'match', 
        'completed_failure': 'mismatch'
    }
    
    frontend_status = status_mapping.get(status, status)
    
    status_data = {
        "type": "NODE_STATUS",
        "nodeId": node_id,
        "status": frontend_status,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    if WEB_MODE:
        send_delimited_message(status_data)
    else:
        print(f"📊 NODE STATUS: {node_id} -> {frontend_status}")
        if message:
            print(f"   Message: {message}")
    
    logger.info(f"Node status update: {node_id} -> {frontend_status}")

def emit_function_update(function_name, status="active", return_data=None, message=None, agent_type="function"):
    """Emit function call information for web mode"""
    if WEB_MODE:
        flow_data = {
            "stepId": function_name.lower().replace('_', '-'),
            "status": status,
            "data": return_data,
            "message": message,
            "type": agent_type,
            "timestamp": datetime.now().isoformat()
        }
        flow_update_message = {
            "type": "FLOW_UPDATE",
            "data": flow_data
        }
        send_delimited_message(flow_update_message)

def emit_agent_activity(agent_name, action, status="active", data=None):
    """Emit specialized agent activity"""
    emit_function_update(
        f"{agent_name.lower().replace(' ', '-')}-agent",
        status=status,
        return_data=data,
        message=f"{agent_name}: {action}",
        agent_type="agent"
    )

def display_message(role: str, content: str):
    """Display message function that works in both web and console modes"""
    global WEB_MODE
    
    if WEB_MODE:
        # Web mode - send structured conversational message
        if role == "bot":
            send_conversational_message(content, "bot_message")
        elif role == "user":
            send_conversational_message(content, "user_message")
    else:
        # Console mode - just print (fallback)
        if role == "bot":
            print(f"{BOT_AVATAR} {CHATBOT_NAME}: {content}")
        elif role == "user":
            print(f"{USER_AVATAR} User: {content}")

def send_step_message(step_number: int, emoji: str, title: str, content: str, message_type: str = "step"):
    """Send a numbered step message in the conversational flow"""
    message = {
        "step": step_number,
        "emoji": emoji,
        "title": title,
        "content": content,
        "type": message_type,
        "timestamp": datetime.now().isoformat()
    }
    send_delimited_message(message)

def send_conversational_message(content: str, message_type: str = "message", step: int = None, title: str = None):
    """Send a conversational message to the frontend"""
    message = {
        "content": content,
        "type": message_type,
        "timestamp": datetime.now().isoformat()
    }
    if step is not None:
        message["step"] = step
    if title:
        message["title"] = title
    send_delimited_message(message)

def send_initial_bot_introduction():
    """Send initial bot introduction as shown in Figma"""
    intro_message = {
        "type": "bot_introduction",
        "content": "Hello! I'm your Data Quality Assistant. I can help you identify and resolve data quality issues by tracing through your data lineage.",
        "sample_input": {
            "failed_table": "revenue_summary_fact",
            "failed_column": "total_revenue", 
            "validation_query": "SELECT COUNT(*) as validation_result FROM revenue_summary_fact WHERE total_revenue < 0 AND business_date = '2024-01-15'",
            "expected_value": 0,
            "expected_std_dev": 0,
            "sd_threshold": 3
        },
        "timestamp": datetime.now().isoformat()
    }
    send_delimited_message(intro_message)

def send_acknowledgment():
    """Send acknowledgment message as shown in Figma"""
    send_conversational_message(
        "Got it! I'll help you analyze this data quality issue. Let me start by examining your data step by step.",
        "acknowledgment"
    )

def send_step_1_analysis():
    """Send Step 1: Data Quality Issue Analysis"""
    send_step_message(1, "🔍", "Data Quality Issue Analysis", 
                     "I'm analyzing your data quality validation to understand what went wrong.")

def send_step_2_investigation():
    """Send Step 2: Detailed Investigation"""
    send_step_message(2, "🕵️", "Detailed Investigation", 
                     "Now I'm diving deeper to identify which specific data points are causing the issue.")

def send_step_3_lineage():
    """Send Step 3: Upstream Dependencies"""
    send_step_message(3, "🔗", "Upstream Dependencies", 
                     "Tracing the data lineage to find the root cause in upstream systems.")

def send_final_root_cause_summary(root_cause_info):
    """Send Final Root Cause Summary"""
    summary_message = {
        "type": "root_cause_summary",
        "title": "🎯 Root Cause Analysis Complete",
        "content": "Here's what I found:",
        "root_cause": root_cause_info,
        "timestamp": datetime.now().isoformat()
    }
    send_delimited_message(summary_message)

def send_feedback_and_extensions():
    """Send feedback request and extension options"""
    feedback_message = {
        "type": "feedback_and_extensions",
        "content": "Was this analysis helpful?",
        "feedback_options": ["👍 Yes, very helpful", "👎 Needs improvement"],
        "extension_questions": [
            "Can you show me the fix recommendations?",
            "What are the historical trends for this metric?",
            "How can I prevent this issue in the future?"
        ],
        "timestamp": datetime.now().isoformat()
    }
    send_delimited_message(feedback_message)

                
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
    """Execute BigQuery query and return results as DataFrame"""
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
        send_update("BigQuery Error", "error", error_message)
        raise Exception(error_message)


def get_predecessor_info(node_name: str, lineage_graph: nx.DiGraph) -> List[Dict[str, str]]:
    """
    Finds and returns information about direct predecessors of a given node in the lineage graph.
    A node name is expected in 'table.column' format.
    """
    predecessors_list = []
    for pred_node in lineage_graph.successors(node_name):
        # The edge from pred_node to node_name describes the transformation
        transformation = lineage_graph.edges[node_name, pred_node].get('transformation', 'N/A')
        predecessors_list.append({
            'prev_table': pred_node.rsplit('.', 1)[0],
            'prev_column': pred_node.rsplit('.', 1)[1],
            'transformation': transformation,
            'source_node_full_name': pred_node # To help construct next_paths_to_process
        })
    return predecessors_list

def replace_technical_date_with_business_date(sql_query : str, table_name : str):
    logger.info('Updating the Query technical dates with business dates')
    date_df = pd.read_csv(RULES_CSV_PATH)
    prompt = f"""
            Analyze the following SQL query and identify all column names that likely represent date or timestamp values.
            Consider common naming conventions (e.g., 'last_upd_dt',  'event_dt')
            and SQL data types if implied by the column usage.

            Return the result as a Python list of strings. Do not include any explanations.
            If no date columns are found, return an empty list.
            List only columns thata are available as table columns, don't consider aliases as a seperate column names.

            SQL Query:
            ```sql
            {sql_query}
            ```
            Example Output Format:
            ["column_name_1"]
            """

    row = date_df[date_df['BQ Table Name'].str.lower() == table_name.lower()]

    llm_tech_date = llm.invoke(prompt).content

    # Add null-safety check
    if not llm_tech_date:
        logger.warning("LLM returned empty response for date column identification")
        return sql_query

    # print(llm_tech_date)
    match = re.search(r"```python\n(.*?)\n```", llm_tech_date, re.DOTALL | re.IGNORECASE)

    if match:
        llm_tech_date = match.group(1).strip()
    
    try:
        technical_date_cols = ast.literal_eval(llm_tech_date)
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Failed to parse LLM date response: {e}. Using empty list.")
        technical_date_cols = []

    if len(technical_date_cols) == 0 or row.empty:
        return sql_query

    business_date_col = row['Business date'].values[0]

    # print('llm_tech_date', type(technical_date_cols))
    def replacer(match):
        prefix = match.group(1) if match.group(1) else ''
        
        return f"{prefix}{business_date_col}"
    
    for technical_date_col in technical_date_cols:
        pattern = rf'(\b\w+\.)?{technical_date_col}\b'

        sql_query = re.sub(pattern, replacer, sql_query, flags=re.IGNORECASE)
    return sql_query

def extract_sql_from_text(text: str) -> str:
    """Extracts the SQL query from the LLM response."""
    if not text:
        logger.warning("Empty text provided to extract_sql_from_text")
        return ""
        
    match = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        sql_query = match.group(1).strip()
        # print(f"Extracted SQL: {sql_query}")
        return sql_query
    else:
        match_cte = re.search(r"(WITH\s+[\s\S]+?SELECT[\s\S]*?;)", text, re.IGNORECASE | re.MULTILINE)
        if match_cte:
            query = match_cte.group(1).strip()
            return query
    
    # If no SQL found, return empty string
    logger.warning(f"No SQL found in text: {text[:100]}...")
    return ""

def anamoly_identifier_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    # Send acknowledgment and start Step 1
    send_acknowledgment()
    send_step_1_analysis()
    
    user_input = state
    prompt = f"""
                        Analyze the data quality validation results and provide a clear, business-friendly summary.
                        
                        Focus on:
                        - What type of data quality issue occurred
                        - When it happened (date/time if available)
                        - The expected vs actual behavior
                        - Impact in simple business terms
                        
                        Keep the language simple and avoid technical jargon.
                       
                        Validation metadata:
                        {user_input} 
                        """
    logger.info('Analyzing the data quality issue.')
    parser_chain = llm 
    response = parser_chain.invoke(prompt).content
    failed_rule = user_input["validation_query"]
    
    # Send the analysis result as part of Step 1
    send_conversational_message(response, "analysis_result")
    
    # Print to console for command-line testing
    if not WEB_MODE:
        print(f"\n{'='*60}")
        print("🔍 STEP 1: DATA QUALITY ISSUE ANALYSIS")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")
    
    return {"anamoly_node_response": response, "validation_query": failed_rule}
			
def analysis_decision_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    send_step_2_investigation()
    
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
    response_str = llm.invoke(decider_prompt)
    content = response_str.content
    
    # Extract JSON from markdown code blocks if present
    import re
    json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL | re.IGNORECASE)
    if json_match:
        content = json_match.group(1).strip()
    
    try:
        analysis_method = json.loads(content)
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse LLM response for analysis decision: {e}. Response: {content}"
        logger.error(error_msg)
        send_conversational_message(error_msg, "error")
        raise Exception(error_msg)
    
    logger.info('Deciding the path.')
    logger.info('Analysis Type : ', analysis_method)
    
    decision_message = ""
    if analysis_method['path_to_follow'] == "Equality":
        decision_message = f"I can see this is a direct column issue. I'll trace back through the data lineage starting from your failed column to find where the problem originated."
    else:
        decision_message = f"This involves multiple columns working together. I need to run statistical checks to identify which specific column is causing the problem before tracing back."
    
    # Send the investigation result as part of Step 2
    send_conversational_message(decision_message, "investigation_result")
    
    # Print to console for command-line testing
    if not WEB_MODE:
        print(f"\n{'='*60}")
        print("🕵️ STEP 2: DETAILED INVESTIGATION")
        print(f"{'='*60}")
        print(f"Path to follow: {analysis_method['path_to_follow']}")
        print(f"Decision: {decision_message}")
        print(f"{'='*60}\n")
    
    return {'analysis_method' : analysis_method['path_to_follow']}
	
	
def parse_dq_query_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    Parses the user's DQ SQL query using an LLM to extract key components.
    """
    send_step_message(6, "🔍", "Query Analysis", "Now I'm going to dissect your validation query like a surgeon! 🔬 Let me break it down into its key components - tables, columns, filters, and aggregation functions. This will help me understand exactly what data we're working with.")
    
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
    try:
        del os.environ['http_proxy']
        del os.environ['https_proxy']
        del os.environ['no_proxy']
    except:
        pass

    parser_chain = llm
    response = parser_chain.invoke(prompt)
    content = response.content
    
    # Extract JSON from markdown code blocks if present
    import re
    json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL | re.IGNORECASE)
    if json_match:
        content = json_match.group(1).strip()
    
    try:
        parsed_info = json.loads(content)
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse LLM response for DQ query parsing: {e}. Response: {content}"
        logger.error(error_msg)
        send_conversational_message(error_msg, "error")
        raise Exception(error_msg)
        
    logger.info(f"Parsed DQ Info: {parsed_info}")
    
    # Ensure parsed_info is directly the dictionary expected, not nested
    if 'parsed_dq_info' in parsed_info and isinstance(parsed_info['parsed_dq_info'], dict):
        parsed_info = parsed_info['parsed_dq_info']

    # Send the parsed information with step number
    send_step_message(7, "📋", "Query Components Identified", 
                     "Excellent! I've successfully parsed your query and identified all the key components. Here's what I found:")
    
    # Create a human-readable summary
    tables_list = list(parsed_info.get('target_tables', {}).keys())
    columns_count = sum(len(cols) for cols in parsed_info.get('target_tables', {}).values())
    
    summary_message = f"🎯 **Analysis Summary:**\n"
    summary_message += f"• **Tables involved:** {', '.join(tables_list)}\n"
    summary_message += f"• **Columns being validated:** {columns_count} column(s)\n"
    summary_message += f"• **Aggregation function:** {parsed_info.get('aggregation_function', 'None')}\n"
    summary_message += f"• **Filters applied:** {len(parsed_info.get('filters', []))} filter(s)\n"
    summary_message += f"• **Group by clauses:** {len(parsed_info.get('group_by', []))} clause(s)"
    
    send_conversational_message(summary_message, "info")
    
    # Display parsed info as formatted JSON for technical details
    send_conversational_message("Here are the technical details:", "info")
    send_conversational_message(json.dumps(parsed_info, indent=2), "code")

    # Print to console for command-line testing
    if not WEB_MODE:
        print(f"\n{'='*60}")
        print("📊 PARSE DQ QUERY NODE RESPONSE")
        print(f"{'='*60}")
        print(json.dumps(parsed_info, indent=2))
        print(f"{'='*60}\n")

    return {"parsed_dq_info": parsed_info}
	
def initialize_trace_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    Initializes the trace for each target table and column identified.
    Executes the first SQL query (L0 layer) based on the parsed info.
    Sets up the initial paths to process for backward tracing.
    """
    send_step_3_lineage()

    parsed_info = state['parsed_dq_info']
    trace_data = {}
    paths_to_process = []
    result = {}

    if not parsed_info or 'target_tables' not in parsed_info or not parsed_info['target_tables']:
        logger.info("No tables found in parsed DQ info. Cannot initialize trace.")
        return {"trace_data": trace_data, "paths_to_process": paths_to_process}

    filters = parsed_info.get('filters', [])
    group_by = parsed_info.get('group_by', [])
    aggregation = parsed_info.get('aggregation_function', '')
    result = {}
    
    for table, columns in parsed_info['target_tables'].items():
        if table not in trace_data:
            trace_data[table] = {}
            result[table] = {}
            
        for column in columns:
            # Send status update that we are checking this node
            send_node_status_update(table, column, "checking", f"Initializing trace for {table}.{column}")
            
            send_conversational_message(f"🎯 **Now examining column:** `{column}` from table `{table}`", "progress")
            send_conversational_message(f"Let me create a specialized query to check if this column is behaving as expected...", "info")
            
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

            l0_query_response = llm.invoke(l0_sql_prompt).content

            l0_query = extract_sql_from_text(l0_query_response)
            l0_query = replace_technical_date_with_business_date(l0_query, table.rsplit('.')[-1])
            
            send_conversational_message(f"✅ **Query generated successfully!** Here's what I created to analyze `{column}`:", "info")
            send_conversational_message(f"```sql\n{l0_query}\n```", "code")
            
            send_conversational_message(f"⚡ **Executing query...** Let me run this against your data warehouse.", "progress")
            l0_output_df = bigquery_execute(l0_query)
            l0_output = l0_output_df.to_json(orient='records')
            
            send_conversational_message(f"📊 **Current data retrieved!** Now I need to compare this against historical patterns to see if it's abnormal.", "info")

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

            std_dev_response = llm.invoke(std_dev_prompt).content
            std_dev_sql = extract_sql_from_text(std_dev_response)
            
            send_conversational_message(f"📈 **Historical Analysis Query for {column}:**", "info")
            send_conversational_message(f"I'm looking at the last 30 days of data to establish a baseline...", "progress")
            send_conversational_message(f"```sql\n{std_dev_sql}\n```", "code")
            
            std_dev_df = bigquery_execute(std_dev_sql)

            expected_std_dev = float(std_dev_df['std_dev'].values[0])
            expected_value = float(std_dev_df['expected_value'].values[0])

            upper_bound = 3 * expected_std_dev + expected_value
            lower_bound = 3 * expected_std_dev - expected_value
            
            prompt = f"""
                            You are an analytical AI assistant for data quality monitoring. Your task is to analyze the result of a SQL query and determine if it has violated a predefined threshold.
                            Here is the context of the data quality failure:
                            - Failed Metric Column: "{state['failed_column']}"

                            Here is the SQL query that was run to get the current value:
                            ```sql
                            {l0_query}
                            ```
                            Here is the full JSON output from the SQL query:
                            ```json
                            {l0_output}
                            ```
                            Here is the acceptable range (threshold) for the failed metric:
                            - Lower Bound: {lower_bound}
                            - Upper Bound: {upper_bound}

                            **Your Instructions:**
                            1. From the JSON output, identify the specific row that matches the "Context of Failure".
                            2. From that row, extract the numeric value from the "Failed Metric Column"
                            3. Compare this extracted value against the provided threshold.
                            4. Is the value for the failed group within the specified threshold?

                            **Provide your response in the following JSON format ONLY:**
                            {{
                                "comparison_result": "within_bounds" or "out_of_bounds",
                                "identified_value": <the numeric value you extracted>,
                                "reasoning": "A brief explanation of which row you chose, what value you found, and how it compares to the threshold."
                            }}
                    """

            # 4. Call the LLM and parse the structured response
            llm_response_str = llm.invoke(prompt)
            try:
                response_content = llm_response_str.content
                
                # Extract JSON from markdown code blocks if present
                import re
                json_match = re.search(r"```json\n(.*?)\n```", response_content, re.DOTALL | re.IGNORECASE)
                if json_match:
                    response_content = json_match.group(1).strip()
                
                parsed_llm_response = json.loads(response_content)
            except Exception as e:
                error_msg = f"Failed to parse LLM output for initial check: {e}. Output was: {llm_response_str}"
                logger.error(error_msg)
                send_conversational_message(error_msg, "error")
                raise Exception(error_msg)

            # 5. Populate the state with the LLM's findings
            is_out_of_bounds = parsed_llm_response.get('comparison_result') == 'out_of_bounds'
            result[table][column] = [{
                                "is_out_of_bounds": is_out_of_bounds,
                                "actual_value": parsed_llm_response.get('identified_value'),
                                "lower_bound": lower_bound,
                                "upper_bound": upper_bound,
                                "message": parsed_llm_response.get('reasoning'),
                                "l0_validation_query" : l0_query
                            }]

            trace_data[table][column] = [{
                    "step_num": 0,
                    "table_name": table,
                    "column_name": column,
                    "sql_query": l0_query,
                    "sql_output": l0_output,
                    "transformation_from_prev": None
                }]

            if is_out_of_bounds:
                l0_key = f"{table}.{column}"
                paths_to_process.append((0, table, column, l0_key))

                # Send completed failure status for columns that are out of bounds and need tracing
                send_node_status_update(table, column, "completed_failure", f"Anomaly detected in {table}.{column}")
                
                send_conversational_message(f"🚨 **BINGO! Found the culprit!** 🚨", "warning")
                send_conversational_message(f"Column `{column}` is definitely behaving abnormally. Here's what I found:", "warning")
                send_conversational_message(f"📊 **Details:**\n{parsed_llm_response.get('reasoning')}", "info")
                send_conversational_message(f"🔍 **Next Step:** I'll now trace this column backwards through its data lineage to find the root cause. Let's follow the data trail!", "progress")
                
            else:
                # Send completed success status for columns that are within bounds
                send_node_status_update(table, column, "completed_success", f"Analysis complete for {table}.{column}")

                send_conversational_message(f"✅ **{column} within bounds:**\n\n**Reasoning:** {parsed_llm_response.get('reasoning')}\n\n**Result:** No lineage tracing necessary for this column.", "success")
    
    # Build and send the full lineage tree for the paths we need to process
    if paths_to_process:
        # Temporarily set paths_to_process in state for lineage tree building
        temp_state = state.copy()
        temp_state['paths_to_process'] = paths_to_process
        build_and_send_lineage_tree(temp_state)
        
        send_step_message(11, "🔗", "Lineage Analysis Starting", 
                         f"Identified {len(paths_to_process)} column(s) requiring lineage trace. Starting upstream dependency analysis...")

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
                
    logger.info(f"Initialized trace for {len(paths_to_process)} target column paths.")
    return {"initial_check_result": result, "trace_data": trace_data, "paths_to_process": paths_to_process}
	
def databuck_failure_validation(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    Executes the validation query, then uses an LLM to find the relevant row and
    check if the failed value is outside the 3*SD threshold.
    """
    send_step_message(8, "🔬", "L0 Validation", "Performing initial validation check on L0 layer...")

    logger.info("Performing initial check against threshold using LLM...")

    validation_query = state['validation_query']
    failed_column = state['failed_column']
    failed_table = state['failed_table']
    paths_to_process = []
    
    # Send status update that we are checking this node
    send_node_status_update(failed_table, failed_column, "checking", f"Starting validation for {failed_table}.{failed_column}")

    prompt = f"""Extract a SQL query from the provided "Validation SQL Query." The new query should:
                * Select only the column `{failed_column}` (retaining its original aggregation or case statement or thresholds, if any) and all other **non-aggregated** columns from the original `SELECT` clause.
                * Preserve all original `WHERE`, `GROUP BY`, and `ORDER BY` clauses.

                Validation SQL Query:
                ```sql
                {validation_query}"""
    
    response = llm.invoke(prompt).content
    if not response:
        logger.error("LLM returned empty response for initial check query extraction")
        return {"analysis_status": "error", "message": "Failed to get response from LLM"}
    
    individual_query = extract_sql_from_text(response)
    individual_query = replace_technical_date_with_business_date(individual_query, failed_table.rsplit('.')[-1])
    
    send_conversational_message(f"**Generated SQL Query:**\n```sql\n{individual_query}\n```", "code")
    
    query_output = bigquery_execute(individual_query)

    # 2. Calculate threshold
    sd_threshold = state['sd_threshold']
    ev = state['expected_value']
    sd = state['expected_std_dev']
    lower_bound = ev - (sd_threshold * sd)
    upper_bound = ev + (sd_threshold * sd)

    send_conversational_message(f"**Threshold Analysis:**\n- Expected Value: {ev:.2f}\n- Standard Deviation: {sd:.2f}\n- Range: [{lower_bound:.2f}, {upper_bound:.2f}]", "info")

    # 3. Construct a prompt for the LLM to perform the analysis
    query_output_json = query_output.to_json(orient='records')

    prompt = f"""
            You are an analytical AI assistant for data quality monitoring. Your task is to analyze the result of a SQL query and determine if it has violated a predefined threshold.
            Here is the context of the data quality failure:
            - Failed Metric Column: "{state['failed_column']}"

            Here is the SQL query that was run to get the current value:
            ```sql
            {individual_query}
            ```
            Here is the full JSON output from the SQL query:
            ```json
            {query_output_json}
            ```
            Here is the acceptable range (threshold) for the failed metric:
            - Lower Bound: {lower_bound}
            - Upper Bound: {upper_bound}

            **Your Instructions:**
            1. From the JSON output, identify the specific row that matches the "Context of Failure".
            2. From that row, extract the numeric value from the "Failed Metric Column"
            3. Compare this extracted value against the provided threshold.
            4. Is the value for the failed group within the specified threshold?

            **Provide your response in the following JSON format ONLY:**
            {{
                "comparison_result": "within_bounds" or "out_of_bounds",
                "identified_value": <the numeric value you extracted>,
                "reasoning": "A brief explanation of which row you chose, what value you found, and how it compares to the threshold."
            }}
    """

    # 4. Call the LLM and parse the structured response
    llm_response_str = llm.invoke(prompt)
    try:
        response_content = llm_response_str.content
        
        # Extract JSON from markdown code blocks if present
        import re
        json_match = re.search(r"```json\n(.*?)\n```", response_content, re.DOTALL | re.IGNORECASE)
        if json_match:
            response_content = json_match.group(1).strip()
        
        parsed_llm_response = json.loads(response_content)
    except Exception as e:
        error_msg = f"Failed to parse LLM output for databuck validation: {e}. Output was: {llm_response_str}"
        logger.error(error_msg)
        send_conversational_message(error_msg, "error")
        raise Exception(error_msg)

    is_out_of_bounds = parsed_llm_response.get('comparison_result') == 'out_of_bounds'

    result = {
        "is_out_of_bounds": is_out_of_bounds,
        "actual_value": parsed_llm_response.get('identified_value'),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "message": parsed_llm_response.get('reasoning'),
        "l0_validation_query" : individual_query
    }

    trace_data = {}
    l0_key = f"{failed_table}.{failed_column}"
    
    if result["is_out_of_bounds"]:
        trace_data[failed_table] = {}
        trace_data[failed_table][failed_column] = [{
                    "step_num": 0,
                    "table_name": failed_table,
                    "column_name": failed_column,
                    "sql_query": individual_query,
                    "sql_output": query_output_json,
                    "transformation_from_prev": None
                }]
        
        paths_to_process.append((0, failed_table, failed_column, l0_key))
        
        # Send completed failure status for columns that are out of bounds and need tracing
        send_node_status_update(failed_table, failed_column, "completed_failure", f"Anomaly detected in {failed_table}.{failed_column}")
        
        send_step_message(9, "⚠️", "Anomaly Detected", 
                         f"**Analysis Result:** Significant deviation detected in {failed_column}.\n\n**Reasoning:** {parsed_llm_response.get('reasoning')}\n\n**Next Step:** Starting lineage trace to identify root cause.")
    else:
        # Send completed success status for columns that are within bounds
        send_node_status_update(failed_table, failed_column, "completed_success", f"Validation successful for {failed_table}.{failed_column}")
        
        send_step_message(9, "✅", "Validation Passed", 
                         f"**Analysis Result:** The data in column {failed_column} falls within expected range.\n\n**Reasoning:** {parsed_llm_response.get('reasoning')}\n\n**Conclusion:** No lineage tracing necessary.")

    # Build and send the full lineage tree for the paths we need to process
    if paths_to_process:
        # Temporarily set paths_to_process in state for lineage tree building
        temp_state = state.copy()
        temp_state['paths_to_process'] = paths_to_process
        build_and_send_lineage_tree(temp_state)

    logger.info(f"LLM Analysis: {result['message']}")

    return {"initial_check_result": result, "trace_data": trace_data, "paths_to_process" : paths_to_process}
	

def trace_backward_step_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    Performs one step backward in the data lineage for the specified paths.
    Finds predecessors, generates and executes queries for the previous layer,
    and updates the trace data. Handles multiple sources for a single column.
    """

    trace_data = state['trace_data']
    paths_to_process = state['paths_to_process']

    if not paths_to_process:
        logger.info("No paths left to process. Ending backward trace.")
        return {"paths_to_process": []} # Signal to stop looping
    else:
        send_step_message(12, "🔄", "Lineage Tracing", "Tracing back through data lineage to collect information across layers...")
        
    next_paths_to_process = []
    processed_paths_current_step = []

    for current_step_num, current_table, current_column, l0_key in paths_to_process:
        
        full_node_name = f"{current_table}.{current_column}"
        lineage_table = current_table.rsplit('.')[-1]
        
        send_conversational_message(f"🔍 **Analyzing lineage for:** {current_column}", "progress")
        
        # Dummy lineage graph loading - in a real scenario, this would be a lookup or a global graph
        # For demonstration, create a simple graph if not exists for a column
        lineage_graph_path = f"{lineage_graphs}/{lineage_table}.{current_column}.gexf"
        if not os.path.exists("lineage_graphs"):
            os.makedirs("lineage_graphs")
        
        try:
            lineage_graph = nx.read_gexf(lineage_graph_path)
        except Exception as e:
            send_conversational_message(f"⚠️ Error loading lineage graph for {current_column}: {str(e)}", "warning")
            print('Error loading Graph :', e)
            break
        
        predecessors = get_predecessor_info(full_node_name, lineage_graph)
        
        if not predecessors:
            send_conversational_message(f"🔚 **End of lineage:** No upstream dependencies found for {current_column}.", "info")
            logger.info(f"No predecessors found for {full_node_name}. Ending trace for this path.")
            continue # No predecessors, this path ends here
        
        send_conversational_message(f"✅ **Found {len(predecessors)} upstream source(s) for {current_column}**", "success")
        
        l0_table, l0_column = l0_key.rsplit('.', 1)

        last_step_for_this_path = None
        for step_data in trace_data[l0_table][l0_column]:
            if step_data['step_num'] == current_step_num and \
            step_data['table_name'] == current_table and \
            step_data['column_name'] == current_column:
                last_step_for_this_path = step_data
                break
        
        if last_step_for_this_path is None:
            logger.error(f"Could not find last step data for {current_table}.{current_column} at step {current_step_num}.")
            continue

        last_query = last_step_for_this_path['sql_query']
        
        for pred in predecessors:
            pred_table = pred['prev_table']
            pred_column = pred['prev_column']
            transformation_logic = pred['transformation']
            
            # Send status update that we are checking this new node
            send_node_status_update(pred_table, pred_column, "checking", f"Analyzing predecessor {pred_table}.{pred_column}")
            
            send_conversational_message(f"📊 **Checking upstream column:** {pred_column} from {pred_table}", "progress")
            send_conversational_message(f"**Transformation:** {transformation_logic}", "info")
            
            logger.info(f'Generating query for table : {pred_table} and column : {pred_column}')
            backward_query = ""
            if transformation_logic != 'One-to-One':
                
                backward_sql_prompt = f"""Your role is to act as an expert SQL query generator, specifically for data lineage analysis. Given an L0 (downstream) SQL query and the transformation logic from L1 (upstream) to L0, your task is to construct an L1 SQL query. The critical requirement is that the data produced by the L1 query, when passed through the given L1 to L0 transformation, must perfectly replicate the results of the L0 query.
                                Output only the resulting SQL query.

                                **Input:**
                                * **L0 SQL Query:** {last_query}
                                * **Lineage Information (L1 to L0):**
                                    * **L0 Table:** {current_table}
                                    * **L0 Column:** {current_column}
                                    * **L1 Table:** {pred_table}
                                    * **L1 Column:** {pred_column}
                                    * **Transformation Logic:** {transformation_logic}

                                **Important Considerations for the L1 Query:**
                                * Always preserve and include `GROUP BY` and `WHERE` clauses from the L0 query.
                                * Avoid direct aggregation on aggregation; use subqueries when needed.
                                * The transformation logic provided must be used exactly as is, without any modifications to column relationships, filters, or `GROUP BY` statements.
                                * Match `CASE` conditions and thresholds from the L0 query in the L1 query.
                                * The ultimate goal is for the L1 query to produce results that, when transformed, are bit-for-bit identical to the L0 query's output.
                                * Generate aggregated results in the L1 query to enable comparisons with aggregated historical data."""

                backward_query_response = llm.invoke(backward_sql_prompt).content

                if not backward_query_response:
                    logger.warning("LLM returned empty response for backward query generation")
                    continue  # Skip this path
                
                backward_query = extract_sql_from_text(backward_query_response)
            else:
                # If transformation is OneToOne, the L1 query is likely the same as L0, just on the L1 table/column
                backward_query = last_query.replace(f"{current_table}.{current_column}", f"{pred_table}.{pred_column}")
                logger.info(f"One-to-One transformation. Simplified L1 query: {backward_query}")
   
            backward_query = replace_technical_date_with_business_date(backward_query, pred_table.rsplit('.')[-1])
            
            send_conversational_message(f"**Generated upstream query for {pred_column}:**\n```sql\n{backward_query}\n```", "code")
            
            try:
                del os.environ['http_proxy']
                del os.environ['https_proxy']
                del os.environ['no_proxy']
            except:
                pass
            
            # Execute the query
            try:
                backward_output_df = bigquery_execute(backward_query)
                backward_output = backward_output_df.to_json(orient='records')
                send_conversational_message(f"✅ **Successfully retrieved data from {pred_column}**", "success")
            except Exception as initial_error:
                logger.warning(f'Initial query failed: {initial_error}. Attempting to correct SQL query...')
                send_conversational_message(f"⚠️ **Query failed, attempting to fix:** {str(initial_error)}", "warning")
                
                retries = 0
                backward_output = None
                
                while retries < 3 and backward_output is None:
                    try:
                        sql_correction_prompt = f"""As an expert BigQuery SQL generator, your role is to debug and correct SQL queries based on error messages.
                                                    Examine the BigQuery error output:
                                                    {str(initial_error)}

                                                    **Error Resolution Strategy:**
                                                    * **If the error is related to "aggregation of an aggregation":** This is a common BigQuery limitation. To resolve this, break down the query into two stages:
                                                        * **Stage 1 (Inner Query):** Calculate the first level of aggregation (e.g., `SUM(x)`, `COUNT(y)`) in a Common Table Expression (CTE) or a subquery.
                                                        * **Stage 2 (Outer Query):** Apply any subsequent conditional logic or aggregations using the results from Stage 1. For example, if you had `SUM(CASE WHEN ... THEN 1 ELSE 0 END)`, refactor `COUNT(column)` into the inner query first.

                                                    * **For all other error types:** Carefully analyze the error message. Identify the root cause (e.g., syntax error, invalid column name, data type mismatch) and implement the necessary correction.

                                                    The original errored SQL query is:
                                                    {backward_query}
                                                    Please generate the fully corrected and runnable BigQuery SQL query."""
                        
                        backward_query_response = llm.invoke(sql_correction_prompt).content
                        if not backward_query_response:
                            logger.warning("LLM returned empty response for SQL correction")
                            retries += 1
                            continue
                            
                        match = re.search(r"```sql\n(.*?)\n```", backward_query_response, re.DOTALL | re.IGNORECASE)
                        if match:
                            backward_query = match.group(1).strip()
                        else:
                            logger.warning("No SQL found in LLM correction response")
                            retries += 1
                            continue
                            
                        # Try the corrected query
                        corrected_df = bigquery_execute(backward_query)
                        backward_output = corrected_df.to_json(orient='records')
                        send_conversational_message(f"✅ **Query correction successful after {retries + 1} attempt(s)**", "success")
                        logger.info(f"Successfully corrected SQL query after {retries + 1} attempts")
                        
                    except Exception as correction_error:
                        logger.warning(f"SQL correction attempt {retries + 1} failed: {correction_error}")
                        retries += 1
                        
                if backward_output is None:
                    error_msg = f"Failed to execute backward query after 3 correction attempts for {pred_table}.{pred_column}"
                    logger.error(error_msg)
                    send_conversational_message(f"❌ **Failed to retrieve data from {pred_column} after multiple attempts**", "error")
                    continue  # Skip this predecessor and continue with others
                    
            next_step_num = current_step_num + 1

            # Append to the trace for the original L0 path
            trace_data[current_table][current_column].append({
                "step_num": next_step_num,
                "table_name": pred_table,
                "column_name": pred_column,
                "sql_query": backward_query,
                "sql_output": backward_output,
                "transformation_from_prev": transformation_logic
            })
            
            send_conversational_message(f"📝 **Lineage Step {next_step_num} completed:** Successfully traced {current_column} → {pred_column}", "info")
                
    logger.info(f'Lineage traced data : {trace_data}')
    
    # Print to console for command-line testing
    if not WEB_MODE:
        print(f"\n{'='*60}")
        print("🔄 TRACE BACKWARD STEP NODE RESPONSE")
        print(f"{'='*60}")
        paths_processed = len(paths_to_process) - len(next_paths_to_process)
        print(f"Paths processed in this step: {paths_processed}")
        print(f"Remaining paths to process: {len(next_paths_to_process)}")
        
        # Show details of current processing
        for current_step_num, current_table, current_column, l0_key in paths_to_process:
            full_node_name = f"{current_table}.{current_column}"
            lineage_table = current_table.rsplit('.')[-1]
            lineage_graph_path = f"{lineage_graphs}/{lineage_table}.{current_column}.gexf"
            
            if os.path.exists(lineage_graph_path):
                try:
                    lineage_graph = nx.read_gexf(lineage_graph_path)
                    predecessors = get_predecessor_info(full_node_name, lineage_graph)
                    print(f"  Step {current_step_num}: {full_node_name}")
                    if predecessors:
                        print(f"    Found {len(predecessors)} predecessor(s):")
                        for pred in predecessors:
                            print(f"      - {pred['prev_table']}.{pred['prev_column']} (Transform: {pred['transformation']})")
                    else:
                        print(f"    No predecessors found - end of lineage")
                except Exception as e:
                    print(f"    Error loading lineage graph: {e}")
            else:
                print(f"    Lineage graph not found: {lineage_graph_path}")
                
        print(f"{'='*60}\n")
    
    return {"trace_data": trace_data, "paths_to_process": next_paths_to_process}
	
def analyze_results_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    Analyzes the collected trace data using an LLM to infer potential root causes.
    Provides the LLM with context including data samples, queries, and transformations.
    """
    send_step_message(13, "📈", "Root Cause Analysis", "Analyzing collected lineage data to identify the root cause...")

    trace_data = state['trace_data']
    analysis_results = {}

    if not trace_data:
        logger.info("No trace data available for analysis.")
        send_conversational_message("No lineage data was collected for analysis.", "info")
        return {"analysis_results": {"summary": "No trace data generated."}}    
    
    send_conversational_message(f"🔍 **Analyzing {len(trace_data)} table(s) and their lineage paths...**", "progress")

    overall_summary_messages = []

    for l0_table, columns_data in trace_data.items():
        analysis_results[l0_table] = {}
        for l0_column, trace_steps in columns_data.items():
            send_conversational_message(f"🔍 **Analyzing lineage for column:** {l0_column}", "progress")
            
            path_analysis = []
            
            # Sort trace steps by step_num to ensure correct comparison (L0, L1, L2...)
            sorted_trace_steps = sorted(trace_steps, key=lambda x: x['step_num'])

            logger.info(f"Analyzing trace for {l0_table}.{l0_column} with {len(sorted_trace_steps)} steps using LLM.")

            for i in range(len(sorted_trace_steps) - 1):
                if i == 0:
                    send_conversational_message("🔄 **Starting cross-layer analysis to identify root cause...**", "progress")
                current_step = sorted_trace_steps[i]
                prev_step_upstream = sorted_trace_steps[i+1] # This is the upstream step

                current_table_column = f"{current_step['table_name']}.{current_step['column_name']}"
                prev_table_column = f"{prev_step_upstream['table_name']}.{prev_step_upstream['column_name']}"
                
                send_conversational_message(f"🔗 **Comparing:** {prev_table_column} → {current_table_column}", "progress")
                
                # Convert outputs to JSON strings for LLM context
                current_output_json = json.dumps(current_step['sql_output'], indent=2)
                prev_output_json = json.dumps(prev_step_upstream['sql_output'], indent=2)

                # if len(pd.DataFrame(current_output_json)) < 50 or len(pd.DataFrame(current_output_json)) < 50:
                analysis_prompt = f"""
                You are an expert data analyst and root cause investigator.
                You have collected data traces for a data lineage path.
                Your task is to analyze the relationship between two consecutive layers (downstream and upstream)
                based on their SQL queries, the transformation applied, and the resulting data samples.
                Determine if the data at the upstream layer (L{i+1}) correctly transforms into the data at the downstream layer (L{i})
                according to the described transformation logic. Identify any potential discrepancies or mismatches.
                While comparing the results between layers, consider only the rows which has the filtered value mentioned in the SQL queries.
                Context for Analysis:
                ---
                Downstream Layer (L{i}) Information:
                Table.Column: {current_table_column} (Step {current_step['step_num']})
                SQL Query:
                ```sql
                {current_step['sql_query']}
                ```
                Sample Data Output (L{i}):
                ```json
                {current_output_json}
                ```

                ---
                Upstream Layer (L{i+1}) Information:
                Table.Column: {prev_table_column} (Step {prev_step_upstream['step_num']})
                SQL Query:
                ```sql
                {prev_step_upstream['sql_query']}
                ```
                Sample Data Output (L{i+1}):
                ```json
                {prev_output_json}
                ```

                ---
                Transformation from L{i+1} to L{i}:
                "{prev_step_upstream['transformation_from_prev']}"

                ---
                Based on the above information, provide your analysis in a JSON format.
                Focus on:
                1.  **Match Status**: "MATCH", "MISMATCH", "INSUFFICIENT_DATA_FOR_ANALYSIS"
                2.  **Inference**: A simple explanation of why you reached that status within 50 words. If it's a mismatch, try to pinpoint the discripency observed w.r.t to numbers compared.

                Output JSON Format:
                {{
                    "match_status": "MATCH | MISMATCH | INSUFFICIENT_DATA_FOR_ANALYSIS",
                    "inference": "Simple explanation of analysis within 50 words...",
                }}

                Important Constraint :
                * If the absolute difference between L{i} and L{i+1} is less than 1% of L{i}, classify it as a MATCH. Otherwise, it's a MISMATCH.
                """
                
                try:
                    del os.environ['http_proxy']
                    del os.environ['https_proxy']
                    del os.environ['no_proxy']
                except:
                    pass
                logger.info(f"Invoking LLM for analysis between {prev_table_column} and {current_table_column}...")
                llm_analysis_response = llm.invoke(analysis_prompt).content

                if not llm_analysis_response:
                    logger.warning("LLM returned empty response for analysis")
                    analysis_output = {
                        "match_status": "INSUFFICIENT_DATA_FOR_ANALYSIS",
                        "inference": "LLM returned empty response"
                    }
                else:
                    match = re.search(r"```json\n(.*?)\n```", llm_analysis_response, re.DOTALL | re.IGNORECASE)

                    if match:
                        llm_analysis_response = match.group(1).strip()

                    # parser_chain = llm | json_parser
                    # llm_analysis_response = parser_chain.invoke(analysis_prompt)
                    
                    try:
                        # print('Type_JSON', type(llm_analysis_response))
                        analysis_output = json.loads(llm_analysis_response)
                    except json.JSONDecodeError:
                        logger.error(f"LLM returned unparseable JSON: {llm_analysis_response}. Retrying or setting default.")
                        analysis_output = {
                            "match_status": "ERROR_PARSING_LLM_OUTPUT",
                            "inference": f"LLM output could not be parsed: {llm_analysis_response}",
                            "recommendations": "Review LLM prompt or response format."
                        }

                path_analysis.append({
                    "from_step": prev_step_upstream['step_num'],
                    "from_table_column": prev_table_column,
                    "to_step": current_step['step_num'],
                    "to_table_column": current_table_column,
                    "transformation": prev_step_upstream['transformation_from_prev'],
                    **analysis_output # Spread the LLM's analysis output
                })
                
                # Send analysis result with appropriate emoji and styling
                status_emoji = "✅" if analysis_output['match_status'] == "MATCH" else "❌" if analysis_output['match_status'] == "MISMATCH" else "⚠️"
                status_color = "success" if analysis_output['match_status'] == "MATCH" else "error" if analysis_output['match_status'] == "MISMATCH" else "warning"
                
                send_conversational_message(
                    f"{status_emoji} **{prev_table_column} → {current_table_column}**\n\n**Status:** {analysis_output['match_status']}\n\n**Analysis:** {analysis_output['inference']}", 
                    status_color
                )
                
                overall_summary_messages.append(f"Analysis for {prev_table_column} -> {current_table_column}: {analysis_output.get('match_status')}")

            analysis_results[l0_table][l0_column] = path_analysis
            
    overall_summary = "Analysis complete. Review results for discrepancies."
    
    logger.info(f'Analyzed results: {analysis_results}')
    
    # Refine overall summary based on LLM findings
    if any("MISMATCH" in msg or "POSSIBLE_MISMATCH" in msg for msg in overall_summary_messages):
        overall_summary = "Discrepancies or possible discrepancies detected across data lineage. Investigate 'MISMATCH' or 'POSSIBLE_MISMATCH' statuses in details."

    logger.info("Analysis complete.")
    
    # Build final summary based on analysis results
    mismatch_count = sum(1 for msg in overall_summary_messages if "MISMATCH" in msg)
    match_count = sum(1 for msg in overall_summary_messages if "MATCH" in msg)
    
    if mismatch_count > 0:
        final_summary = f"🎯 **Root Cause Identified:** Found {mismatch_count} data transformation issue(s) in the lineage. The root cause appears to be upstream data discrepancies that propagated downstream."
    else:
        final_summary = f"✅ **Analysis Complete:** All {match_count} lineage transformations appear correct. The data quality issue may be caused by factors outside the current lineage scope."
    
    # Send the final root cause summary based on Figma design
    root_cause_info = {
        "mismatch_count": mismatch_count,
        "match_count": match_count,
        "detailed_analysis": analysis_results,
        "summary": final_summary
    }
    
    send_final_root_cause_summary(root_cause_info)
    
    # Send feedback and extension questions
    send_feedback_and_extensions()
    
    # Print to console for command-line testing
    if not WEB_MODE:
        print(f"\n{'='*60}")
        print("📈 ANALYZE RESULTS NODE RESPONSE")
        print(f"{'='*60}")
        print(f"Overall Summary: {overall_summary}")
        print(f"Final Summary: {final_summary}")
        print(f"\nDetailed Analysis Results:")
        
        for l0_table, columns_data in analysis_results.items():
            print(f"\nTable: {l0_table}")
            for l0_column, path_analysis in columns_data.items():
                print(f"  Column: {l0_column}")
                if not path_analysis:
                    print(f"    No analysis data available")
                else:
                    for analysis in path_analysis:
                        print(f"    From: {analysis['from_table_column']} (Step {analysis['from_step']})")
                        print(f"    To: {analysis['to_table_column']} (Step {analysis['to_step']})")
                        print(f"    Transformation: {analysis['transformation']}")
                        print(f"    Match Status: {analysis['match_status']}")
                        print(f"    Inference: {analysis['inference']}")
                        print(f"    ---")
        print(f"{'='*60}\n")
    
    # Send final status updates for all analyzed nodes
    for l0_table, columns_data in analysis_results.items():
        for l0_column, path_analysis in columns_data.items():
            if path_analysis:
                # Check if any analysis shows a mismatch
                has_mismatch = any("MISMATCH" in analysis.get('match_status', '') for analysis in path_analysis)
                
                if has_mismatch:
                    # Send mismatch status for the column
                    send_node_status_update(l0_table, l0_column, "completed_failure", "Root cause identified")
                    
                    # Also send mismatch status for upstream nodes that had mismatches
                    for analysis in path_analysis:
                        if "MISMATCH" in analysis.get('match_status', ''):
                            upstream_table, upstream_column = analysis['from_table_column'].rsplit('.', 1)
                            send_node_status_update(upstream_table, upstream_column, "completed_failure", 
                                                  f"Data mismatch detected: {analysis.get('inference', '')}")
                else:
                    # Send success status
                    send_node_status_update(l0_table, l0_column, "completed_success", "Analysis complete - no issues found")
            else:
                # No analysis data available
                send_node_status_update(l0_table, l0_column, "completed_success", "No upstream dependencies found")
    
    return {
        "analysis_results": {"summary": overall_summary, "details": analysis_results},
        "final_summary": final_summary
    }


def build_and_send_lineage_tree(state: RootCauseAnalysisState):
    """
    Traverses the full lineage for all paths and sends the complete tree structure to the frontend.
    Enhanced for React Flow compatibility with proper node positioning and styling.
    """
    paths_to_process = state.get('paths_to_process', [])
    if not paths_to_process:
        return

    nodes = []
    edges = []
    all_node_ids = set()
    table_positions = {}  # Track table positions for layout
    y_offset = 0

    def get_full_lineage(table, column, level=0):
        node_id = f"{table}.{column}"
        if node_id in all_node_ids:
            return
        
        all_node_ids.add(node_id)
        
        # Determine position based on level in lineage
        x_position = level * 300  # Horizontal spacing between levels
        
        # Get or assign Y position for this table
        if table not in table_positions:
            table_positions[table] = len(table_positions) * 150
        y_position = table_positions[table]
        
        # Add node in React Flow format
        nodes.append({
            'id': node_id,
            'type': 'custom',  # Will use custom node component
            'data': {
                'label': f"{table.split('.')[-1]}.{column}",
                'table': table,
                'column': column,
                'status': 'pending',
                'level': level
            },
            'position': {'x': x_position, 'y': y_position},
            'style': {
                'background': '#f3f4f6',
                'border': '2px solid #d1d5db',
                'borderRadius': '8px',
                'padding': '10px',
                'minWidth': '200px'
            }
        })
        
        lineage_table_name = table.rsplit('.')[-1]
        lineage_graph_path = f"{lineage_graphs}/{lineage_table_name}.{column}.gexf"

        try:
            lineage_graph = nx.read_gexf(lineage_graph_path)
            predecessors = get_predecessor_info(node_id, lineage_graph)
            
            for pred in predecessors:
                pred_node_id = f"{pred['prev_table']}.{pred['prev_column']}"
                
                # Add edge from predecessor to current node (React Flow format)
                edges.append({
                    'id': f"{pred_node_id}->{node_id}",
                    'source': pred_node_id,
                    'target': node_id,
                    'type': 'smoothstep',
                    'animated': False,
                    'data': {
                        'transformation': pred.get('transformation', 'Unknown')
                    },
                    'style': {
                        'stroke': '#6b7280',
                        'strokeWidth': 2
                    }
                })
                
                # Recursively process predecessor at next level
                get_full_lineage(pred['prev_table'], pred['prev_column'], level + 1)

        except FileNotFoundError:
            # This is a source node - no predecessors
            logger.info(f"Source node found: {node_id}")
        except Exception as e:
            logger.error(f"Error building lineage tree for {node_id}: {e}")

    # Build the complete tree for all paths
    for _, table, column, _ in paths_to_process:
        get_full_lineage(table, column, level=0)

    # Send lineage tree to frontend via delimited protocol
    lineage_data = {
        "type": "LINEAGE_TREE",
        "nodes": nodes,
        "edges": edges,
        "timestamp": datetime.now().isoformat()
    }
    
    if WEB_MODE:
        send_delimited_message(lineage_data)
    else:
        print(f"\n{'='*60}")
        print("🌳 LINEAGE TREE STRUCTURE")
        print(f"{'='*60}")
        print(f"Nodes: {len(nodes)}")
        print(f"Edges: {len(edges)}")
        for node in nodes:
            print(f"  - {node['id']}: Level {node['data']['level']}")
        print(f"{'='*60}\n")
    
    logger.info(f"Built lineage tree with {len(nodes)} nodes and {len(edges)} edges")
    

workflow = StateGraph(RootCauseAnalysisState)

# Add nodes
workflow.add_node("issue_summarizer", anamoly_identifier_node)
workflow.add_node("rca_analysis_decision", analysis_decision_node)
workflow.add_node("dq_failure_validation", databuck_failure_validation)
workflow.add_node("parse_dq_query", parse_dq_query_node)
workflow.add_node("initialize_trace", initialize_trace_node)
workflow.add_node("lineage_traversal", trace_backward_step_node)
workflow.add_node("issue_analyser", analyze_results_node)

workflow.add_conditional_edges(
    "rca_analysis_decision",
    lambda state : "single_column" if state['analysis_method'] == "Equality" else "multi_column",
    {"single_column" : "dq_failure_validation", "multi_column" : "parse_dq_query" }
)

# Set up edges
workflow.add_edge("issue_summarizer", "rca_analysis_decision")
workflow.add_edge("parse_dq_query", "initialize_trace")
# workflow.add_edge("dq_failure_validation", "lineage_traversal")
workflow.add_edge("initialize_trace", "lineage_traversal")
workflow.add_edge("lineage_traversal", "issue_analyser")

workflow.set_entry_point("issue_summarizer")

workflow.add_conditional_edges(
    "dq_failure_validation",
    lambda state : "continue" if state["initial_check_result"]["is_out_of_bounds"] else "end",
    {"continue" : "lineage_traversal", "end" : END}
)

workflow.add_conditional_edges(
    "initialize_trace",
    lambda state : "continue" if len(state["paths_to_process"]) > 0 else "end",
    {"continue" : "lineage_traversal", "end" : END}
)


app = workflow.compile()

def handle_text_input_conversion(text_input: str) -> Dict[str, Any]:
    """Convert natural language text input to structured JSON using LLM"""
    
    conversion_prompt = f"""
    You are a data quality assistant. A user has described a data quality issue in natural language.
    Your task is to convert this description into the structured JSON format required for root cause analysis.
    
    User Input: "{text_input}"
    
    Based on the user's description, extract or infer the following information and create a JSON object:
    
    Required JSON Format:
    {{
        "failed_table": "<fully qualified table name if mentioned, otherwise make a reasonable guess>",
        "failed_column": "<column name mentioned or inferred>",
        "db_type": "GCP",
        "validation_query": "<construct a reasonable SQL query based on the issue description>",
        "execution_date": "<date mentioned or use current date>",
        "sd_threshold": 3,
        "expected_std_dev": <estimated standard deviation, default to 100 if not specified>,
        "expected_value": <estimated expected value, default to 1000 if not specified>,
        "actual_value": <actual value if mentioned, otherwise estimate based on issue>
    }}
    
    Guidelines:
    - If table name isn't fully qualified, make it realistic (e.g., "project.dataset.table_name")
    - For the SQL query, create a reasonable aggregation query that would detect the issue mentioned
    - Use reasonable defaults for numeric values if not specified
    - Today's date is {datetime.now().strftime('%Y-%m-%d')}
    
    Respond ONLY with valid JSON.
    """
    
    try:
        llm_response = llm.invoke(conversion_prompt).content
        
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r"```json\n(.*?)\n```", llm_response, re.DOTALL | re.IGNORECASE)
        if json_match:
            llm_response = json_match.group(1).strip()
        
        converted_input = json.loads(llm_response)
        
        send_conversational_message(f"📝 **Converted your description to structured format:**\n```json\n{json.dumps(converted_input, indent=2)}\n```", "info")
        
        return converted_input
        
    except Exception as e:
        error_msg = f"Failed to convert text input to JSON: {str(e)}"
        logger.error(error_msg)
        send_conversational_message(error_msg, "error")
        raise Exception(error_msg)

def validate_and_normalize_input(user_input: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the input JSON to ensure all required fields are present"""
    
    required_fields = {
        "failed_table": "Unknown Table",
        "failed_column": "unknown_column", 
        "validation_query": "SELECT COUNT(*) FROM unknown_table",
        "expected_value": 1000.0,
        "expected_std_dev": 100.0,
        "sd_threshold": 3.0
    }
    
    # Add missing fields with defaults
    for field, default_value in required_fields.items():
        if field not in user_input:
            user_input[field] = default_value
            logger.warning(f"Missing field '{field}', using default: {default_value}")
    
    # Ensure numeric fields are properly typed
    numeric_fields = ["expected_value", "expected_std_dev", "sd_threshold"]
    for field in numeric_fields:
        if field in user_input:
            try:
                user_input[field] = float(user_input[field])
            except (ValueError, TypeError):
                logger.warning(f"Invalid value for {field}, using default")
                user_input[field] = required_fields[field]
    
    return user_input

def main():
    """Main function to handle command line execution"""
    global WEB_MODE
    
    if len(sys.argv) > 1:
        # Check if argument is a file path
        arg = sys.argv[1]
        if arg.endswith('.json') and os.path.exists(arg):
            # Reading from JSON file
            WEB_MODE = False  # Set to console mode for file input
            try:
                with open(arg, 'r') as f:
                    user_input = json.load(f)
                print(f"🔍 Reading input from file: {arg}")
                print(f"Input: {json.dumps(user_input, indent=2)}")
            except Exception as e:
                print(f"ERROR: Failed to read JSON file {arg}: {e}")
                sys.exit(1)
        else:
            # Try to determine if this is JSON or natural language text
            try:
                # First, try to parse as JSON
                user_input = json.loads(arg)
                WEB_MODE = True  # Successfully parsed, assume web mode
                logger.info("Successfully parsed JSON input")
            except json.JSONDecodeError:
                # Not valid JSON, check if it's a special command or natural language
                if arg == "start_introduction":
                    # Special command to send introduction
                    WEB_MODE = True
                    send_initial_bot_introduction()
                    return
                elif any(word in arg.lower() for word in ['see', 'low', 'high', 'issue', 'problem', 'deviation', 'anomaly', 'volumes']):
                    # This looks like natural language text
                    WEB_MODE = True
                    send_conversational_message(f"📝 **Received:** {arg}", "user_message")
                    send_conversational_message(f"🤖 **Processing your description...**", "progress")
                    
                    try:
                        user_input = handle_text_input_conversion(arg)
                    except Exception as e:
                        send_conversational_message(f"❌ **Error:** {str(e)}", "error")
                        sys.exit(1)
                else:
                    # JSON parsing failed - try to fix common PowerShell escaping issues
                    print(f"⚠️  JSON parsing failed for: {repr(arg)}")
                    print(f"🔧 Attempting to fix PowerShell JSON formatting issues...")
                    
                    WEB_MODE = False  # Set to console mode for better error handling
                    
                    # Try multiple fixing strategies
                    fixed_json = arg
                    
                    # Strategy 1: Handle missing quotes around property names
                    try:
                        # Add quotes around unquoted property names
                        fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
                        # Fix any single quotes that might have been introduced
                        fixed_json = fixed_json.replace("'", '"')
                        user_input = json.loads(fixed_json)
                        print("✅ JSON parsing succeeded after fixing property names")
                        
                    except json.JSONDecodeError:
                        # Strategy 2: Try to handle completely mangled JSON
                        try:
                            # Remove any extra backslashes that PowerShell might add
                            fixed_json = arg.replace('\\"', '"').replace("\\'", "'")
                            user_input = json.loads(fixed_json)
                            print("✅ JSON parsing succeeded after removing escape characters")
                            
                        except json.JSONDecodeError:
                            # Strategy 3: Handle the case where outer quotes are stripped
                            try:
                                if not arg.startswith('{'):
                                    fixed_json = '{' + arg + '}'
                                user_input = json.loads(fixed_json)
                                print("✅ JSON parsing succeeded after adding outer braces")
                                
                            except json.JSONDecodeError:
                                print("❌ ERROR: Cannot parse input after multiple attempts.")
                                print("")
                                print("🔍 TROUBLESHOOTING:")
                                print("PowerShell can mangle JSON strings. Try one of these solutions:")
                                print("")
                                print("📁 RECOMMENDED: Use a JSON file instead:")
                                print("   python adq_agents.py sample_input.json")
                                print("")
                                print("💡 OR describe your issue in natural language:")
                                print("   python adq_agents.py \"I see low volumes of netadds on dla_sum_fact\"")
                                print("")
                                print("🐚 OR switch to Command Prompt (cmd) for JSON:")
                                print('   python adq_agents.py "{\\"failed_column\\": \\"port_in_cnt\\", \\"failed_table\\": \\"table_name\\", \\"validation_query\\": \\"SELECT * FROM table\\", \\"expected_value\\": 1000.0, \\"expected_std_dev\\": 50.0, \\"sd_threshold\\": 3.0}"')
                                sys.exit(1)
        
        # Validate and normalize the input
        try:
            user_input = validate_and_normalize_input(user_input)
        except Exception as e:
            if WEB_MODE:
                send_conversational_message(f"❌ **Input validation error:** {str(e)}", "error")
            else:
                print(f"❌ ERROR: {str(e)}")
            sys.exit(1)
        
        if WEB_MODE:
            send_conversational_message("🚀 **Starting Agentic Data Quality Root Cause Analysis...**", "status")
        
        # Run the workflow
        result = app.invoke(user_input)
        
        if WEB_MODE:
            # Send completion message
            send_conversational_message("🎉 **Root Cause Analysis completed successfully!**", "completion")
        else:
            # Console mode final summary
            print(f"\n🎉 ROOT CAUSE ANALYSIS COMPLETED!")
            print(f"{'='*60}")
            if 'final_summary' in result:
                print(f"Final Summary: {result['final_summary']}")
            else:
                print("Final Summary: Root Cause Analysis completed.")
            print(f"{'='*60}")
                
    else:
        # No command line arguments - send initial bot introduction for web mode
        WEB_MODE = True
        send_initial_bot_introduction()
        
        # Wait for user input from Node.js
        # This is handled by the Node.js process calling this script with arguments
        return

if __name__ == "__main__":
    main()
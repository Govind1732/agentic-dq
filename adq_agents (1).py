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
import streamlit as st
from datetime import datetime
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

usecase_name = USECASE_NAME # Add your vegas usecase name here
context_name = CONTEXT_NAME # Add your vegas context name here

#usecase_name = "adq_testing"
#context_name = "text_prompt"

from pyvegas.helpers.utils import set_proxy, unset_proxy
from pyvegas.langx import VegasLangchainLLM

set_proxy()

llm = VegasLangchainLLM(usecase_name =usecase_name, context_name = context_name, temperature=0)

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

def display_message(role: str, content: str):
    if role == "bot":
        st.markdown(
            f'''
            <div style="
                max-width: 70%;
                background-color: #f1f3f4;
                padding: 10px 15px;
                margin: 10px 0;
                border: 1px solid #ccc;
                border-radius: 10px;
                float: left;
                clear: both;">
                <b>{BOT_AVATAR} {CHATBOT_NAME}</b><br>
                <pre style="margin:0;">{content}</pre>
            </div>
            ''', unsafe_allow_html=True
        )
    elif role == "user":
        st.markdown(
            f'''
            <div style="
                max-width: 70%;
                background-color: #dbeafe;
                padding: 10px 15px;
                margin: 10px 0;
                border: 1px solid #ccc;
                border-radius: 10px;
                float: right;
                clear: both;
                text-align: right;">
                <b>{USER_AVATAR}</b>
                <pre style="margin:0;">{content}</pre>
            </div>
            ''', unsafe_allow_html=True
        )

                
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
        logger.info(error_message)
        return error_message


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

    # print(llm_tech_date)
    match = re.search(r"```python\n(.*?)\n```", llm_tech_date, re.DOTALL | re.IGNORECASE)

    if match:
        llm_tech_date = match.group(1).strip()
    
    technical_date_cols = ast.literal_eval(llm_tech_date)

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

def anamoly_identifier_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    user_input = state
    prompt = f"""
                        You are a data validation assistant. Your task is to analyze validation results and generate clear, concise, and professional summaries for data quality issues based on the provided metadata.

                        Your response should:
                        - Clearly identify the type of anomaly and the affected data point
                        - Mention the date or time range if available
                        - Explain how the actual value deviated from expected behavior using statistical context (e.g., more than 3 standard deviations)
                        - Present it in human-readable language for business/data stakeholders
                        - Avoid technical jargon or SQL unless necessary

                        Respond only with a natural language explanation of the validation failure and its context. Be concise but complete. 
                        Don't give any suggestions or recommendations at the end just identify the anomaly.
                       
                        metadata:
                        {user_input} 
                        """
    logger.info('Summarizing the inputs recieved.')
    parser_chain = llm 
    response = parser_chain.invoke(prompt).content
    failed_rule = user_input["validation_query"]
    st.session_state.messages.append({"role": "bot", "content": response})
    display_message("bot", response)
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
    analysis_method = json_parser.invoke(response_str)
    logger.info('Deciding the path.')
    logger.info('Analysis Type : ', analysis_method)
    if analysis_method['path_to_follow'] == "Equality":
        display_message("bot", f"Based for provided inputs, proceeding with the failed column as the starting point for our root cause analysis.")
    else:
        display_message("bot", f"Based for provided inputs, multiple columns are involved in this validation metric, we'll run a statistical check at the L0 layer to pinpoint the problematic column.")
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
    try:
        del os.environ['http_proxy']
        del os.environ['https_proxy']
        del os.environ['no_proxy']
    except:
        pass

    parser_chain = llm | json_parser
    parsed_info = parser_chain.invoke(prompt)
    logger.info(f"Parsed DQ Info: {parsed_info}")
    # parsed_info = {'target_tables': {'vz-it-pr-gk1v-cwlsdo-0.vzw_uda_prd_tbls_rd_v.port_sum_fact': ['port_in_cnt', 'port_out_cnt']}, 'filters': ['activity_dt =2025-07-02'], 'group_by': ['activity_dt', "FORMAT_DATE('%A', cast(activity_dt as date) )", 'activity_cd'], 'aggregation_function': 'sum'}
    # Ensure parsed_info is directly the dictionary expected, not nested
    if 'parsed_dq_info' in parsed_info and isinstance(parsed_info['parsed_dq_info'], dict):
        parsed_info = parsed_info['parsed_dq_info']

    display_message("bot",f"Based on your inputs, here are the tables, columns, filters, group by, and aggregation functions found.")
    st.json(f"{parsed_info}")

    return {"parsed_dq_info": parsed_info}
	
def initialize_trace_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    Initializes the trace for each target table and column identified.
    Executes the first SQL query (L0 layer) based on the parsed info.
    Sets up the initial paths to process for backward tracing.
    """

    display_message('bot', "Identifying the columns that required root cause analysis...")
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
            
            l0_output_df = bigquery_execute(l0_query)
            l0_output = l0_output_df.to_json(orient='records')

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
            std_dev_df = bigquery_execute(std_dev_sql)

            expected_std_dev = float(std_dev_df['std_dev'].values[0])
            expected_value = float(std_dev_df['expected_value'].values[0])

            upper_bound = 3 * expected_std_dev + expected_value
            lower_bound = 3 * expected_std_dev - expected_value
            # print("upper_bound", upper_bound, "lower_bound", lower_bound, "query_output", expected_value)
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
                parsed_llm_response = json_parser.invoke(llm_response_str)
            except Exception as e:
                logger.error(f"Failed to parse LLM output for initial check: {e}. Output was: {llm_response_str}")
                # Default to assuming it's out of bounds to trigger a trace on error
                parsed_llm_response = {
                    "comparison_result": "out_of_bounds",
                    "identified_value": None,
                    "reasoning": "Error: Could not parse the LLM's response. Proceeding with trace as a precaution."
                }

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

                display_message("bot", f""" Analysing the data on L0 layer for <b>{column}</b><br><br>
                                        <b>Reasoning:</b> {parsed_llm_response.get('reasoning')}<br><br>
                                        <b>SQL query for actual value:</b><br>
                                        <pre><code>{l0_query}</code></pre><br>
                                        <b>SQL query for SD and EV:</b><br>
                                        <pre><code>{std_dev_sql}</code></pre><br>
                                        A significant deviation has been detected in the {column} column's values over the past month. A lineage trace is necessary to uncover the root cause.
                                        """)
                
            else:

                display_message("bot", f"""Analysing the data on L0 layer for <b>{column}</b><br><br>
                                        <b>Reasoning:</b> {parsed_llm_response.get('reasoning')}<br><br>
                                        <b>SQL query for actual value:</b><br>
                                        <pre><code>{l0_query}</code></pre><br>
                                        <b>SQL query for SD and EV:</b><br>
                                        <pre><code>{std_dev_sql}</code></pre><br>
                                        The data in column <b>{column}</b> falls within the expected range of 3 standard deviations, indicating that lineage tracing isn't necessary.
                                        """)
                
    logger.info(f"Initialized trace for {len(paths_to_process)} target column paths.")
    return {"initial_check_result": result, "trace_data": trace_data, "paths_to_process": paths_to_process}
	
def databuck_failure_validation(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    Executes the validation query, then uses an LLM to find the relevant row and
    check if the failed value is outside the 3*SD threshold.
    """
    display_message('bot', "Performing the initial check on layer L0 to validate the databuck alert...")

    logger.info("Performing initial check against threshold using LLM...")

    validation_query = state['validation_query']
    failed_column = state['failed_column']
    failed_table = state['failed_table']
    paths_to_process = []

    prompt = f"""Extract a SQL query from the provided "Validation SQL Query." The new query should:
                * Select only the column `{failed_column}` (retaining its original aggregation or case statement or thresholds, if any) and all other **non-aggregated** columns from the original `SELECT` clause.
                * Preserve all original `WHERE`, `GROUP BY`, and `ORDER BY` clauses.

                Validation SQL Query:
                ```sql
                {validation_query}"""
    
    respose = llm.invoke(prompt).content
    individual_query = extract_sql_from_text(respose)
    individual_query = replace_technical_date_with_business_date(individual_query, failed_table.rsplit('.')[-1])
    query_output = bigquery_execute(individual_query)

    # 2. Calculate threshold
    sd_threshold = state['sd_threshold']
    ev = state['expected_value']
    sd = state['expected_std_dev']
    lower_bound = ev - (sd_threshold * sd)
    upper_bound = ev + (sd_threshold * sd)

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
        parsed_llm_response = json_parser.invoke(llm_response_str)
    except Exception as e:
        logger.error(f"Failed to parse LLM output for initial check: {e}. Output was: {llm_response_str}")
        # Default to assuming it's out of bounds to trigger a trace on error
        parsed_llm_response = {
            "comparison_result": "out_of_bounds",
            "identified_value": None,
            "reasoning": "Error: Could not parse the LLM's response. Proceeding with trace as a precaution."
        }

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
        

        display_message("bot", f""" Analysing the data on L0 layer for <b>{failed_column}</b><br><br>
                                    <b>Reasoning:</b> {parsed_llm_response.get('reasoning')}<br><br>
                                    <b>SQL query for actual value:</b><br>
                                    <pre><code>{individual_query}</code></pre><br>
                                    A significant deviation has been detected in the {failed_column} column's values over the past month. A lineage trace is necessary to uncover the root cause.
                                    """)
    else:
        display_message("bot", f""" Analysing the data on L0 layer for <b>{failed_column}</b><br><br>
                                    <b>Reasoning:</b> {parsed_llm_response.get('reasoning')}<br><br>
                                    <b>SQL query for actual value:</b><br>
                                    <pre><code>{individual_query}</code></pre><br>
                                    The data in column <b>{failed_column}</b> falls within the expected range of 3 standard deviations, indicating that lineage tracing isn't necessary.
                                    """)

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
    # max_trace_steps = state.get('max_trace_steps', 3) # Get max steps from state

    if not paths_to_process:
        logger.info("No paths left to process. Ending backward trace.")
        return {"paths_to_process": []} # Signal to stop looping
    else:
        display_message('bot', "Tracing back in the lineage to collect data acorss layers for analysis...")
    next_paths_to_process = []
    processed_paths_current_step = []

    # logger.info(f"Processing {len(paths_to_process)} paths for backward trace. Current step: {paths_to_process[0][0]}")

    for current_step_num, current_table, current_column, l0_key in paths_to_process:
        
        full_node_name = f"{current_table}.{current_column}"
        lineage_table = current_table.rsplit('.')[-1]
        # if lineage_graph.out_degree(full_node_name) > 0:
        # Dummy lineage graph loading - in a real scenario, this would be a lookup or a global graph
        # For demonstration, create a simple graph if not exists for a column
        lineage_graph_path = f"{lineage_graphs}/{lineage_table}.{current_column}.gexf"
        if not os.path.exists("lineage_graphs"):
            os.makedirs("lineage_graphs")
        
        try:
            lineage_graph = nx.read_gexf(lineage_graph_path)
        except Exception as e:
            print('Error loading Graph :', e)
            break
        
        predecessors = get_predecessor_info(full_node_name, lineage_graph)
        
        if not predecessors:
            logger.info(f"No predecessors found for {full_node_name}. Ending trace for this path.")
            continue # No predecessors, this path ends here
        
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

                backward_query = extract_sql_from_text(backward_query_response)
            else:
                # If transformation is OneToOne, the L1 query is likely the same as L0, just on the L1 table/column
                # A more sophisticated approach might substitute table/column names in the query
                # For simplicity, we'll indicate it's OneToOne and might just keep the L0 query structure
                backward_query = last_query.replace(f"{current_table}.{current_column}", f"{pred_table}.{pred_column}")
                logger.info(f"One-to-One transformation. Simplified L1 query: {backward_query}")
   
            backward_query = replace_technical_date_with_business_date(backward_query, pred_table.rsplit('.')[-1])
            try:
                del os.environ['http_proxy']
                del os.environ['https_proxy']
                del os.environ['no_proxy']
            except:
                pass
            
            # Execute the query
            backward_output_df = bigquery_execute(backward_query)

            if isinstance(backward_output_df, pd.DataFrame):
                backward_output = backward_output_df.to_json(orient='records')
            else:
                print('Correcting the SQL Query')
                retries = 0
                while retries < 3:
                    try:
                        
                        sql_correction_prompt = f"""As an expert BigQuery SQL generator, your role is to debug and correct SQL queries based on error messages.
                                                    Examine the BigQuery error output:
                                                    {backward_output_df}

                                                    **Error Resolution Strategy:**
                                                    * **If the error is related to "aggregation of an aggregation":** This is a common BigQuery limitation. To resolve this, break down the query into two stages:
                                                        * **Stage 1 (Inner Query):** Calculate the first level of aggregation (e.g., `SUM(x)`, `COUNT(y)`) in a Common Table Expression (CTE) or a subquery.
                                                        * **Stage 2 (Outer Query):** Apply any subsequent conditional logic or aggregations using the results from Stage 1. For example, if you had `SUM(CASE WHEN COUNT(column) < 10 THEN 1 ELSE 0 END)`, refactor `COUNT(column)` into the inner query first.

                                                    * **For all other error types:** Carefully analyze the error message. Identify the root cause (e.g., syntax error, invalid column name, data type mismatch) and implement the necessary correction.

                                                    The original errored SQL query is:
                                                    {backward_query}
                                                    Please generate the fully corrected and runnable BigQuery SQL query."""
                        
                        backward_query_response = llm.invoke(sql_correction_prompt).content
                        match = re.search(r"```sql\n(.*?)\n```", backward_query_response, re.DOTALL | re.IGNORECASE)
                        if match:
                            backward_query = match.group(1).strip()
                        backward_output_df = bigquery_execute(backward_query)
                        if isinstance(backward_output_df, pd.DataFrame):
                            backward_output = backward_output_df.to_json(orient='records')
                            break
                        else:
                            backward_output = backward_output_df
                            retries+=1
                    except:
                        continue
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

            
            display_message("bot", f""" Traversing through lineage for <b>{current_column}</b><br><br>
                                    <b>Layer Number:</b> {next_step_num}<br><br>
                                    <b>Table Name:</b> {pred_table}<br><br>
                                    <b>Column Name:</b> {pred_column}<br><br>
                                    <b>SQL query:</b><br>
                                    <pre><code>{backward_query}</code></pre><br>
                                    """)
                
    # logger.info(f"Completed processing current batch. {len(next_paths_to_process)} new paths to process.")
    logger.info(f'Lineage traced date : {trace_data}')
    return {"trace_data": trace_data, "paths_to_process": next_paths_to_process}
	
def analyze_results_node(state: RootCauseAnalysisState) -> Dict[str, Any]:
    """
    Analyzes the collected trace data using an LLM to infer potential root causes.
    Provides the LLM with context including data samples, queries, and transformations.
    """

    trace_data = state['trace_data']
    analysis_results = {}

    if not trace_data:
        logger.info("No trace data available for analysis.")
        return {"analysis_results": {"summary": "No trace data generated."}}    
    

    overall_summary_messages = []

    for l0_table, columns_data in trace_data.items():
        analysis_results[l0_table] = {}
        for l0_column, trace_steps in columns_data.items():
            path_analysis = []
            
            # Sort trace steps by step_num to ensure correct comparison (L0, L1, L2...)
            sorted_trace_steps = sorted(trace_steps, key=lambda x: x['step_num'])

            logger.info(f"Analyzing trace for {l0_table}.{l0_column} with {len(sorted_trace_steps)} steps using LLM.")

            for i in range(len(sorted_trace_steps) - 1):
                if i == 0:
                    display_message('bot', "Analyzing the results across layers to identify the Root cause...")
                current_step = sorted_trace_steps[i]
                prev_step_upstream = sorted_trace_steps[i+1] # This is the upstream step

                current_table_column = f"{current_step['table_name']}.{current_step['column_name']}"
                prev_table_column = f"{prev_step_upstream['table_name']}.{prev_step_upstream['column_name']}"
                
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
                overall_summary_messages.append(f"Analysis for {prev_table_column} -> {current_table_column}: {analysis_output.get('match_status')}")

            analysis_results[l0_table][l0_column] = path_analysis
            if len(path_analysis) > 0:
                display_message("bot", f""" Analysis of data transfromation from <b>{prev_table_column}</b> to <b>{current_table_column}</b><br><br>
                                    <b>Data Match Status:</b> {analysis_output['match_status']}<br><br>
                                    <b>Inference:</b> {analysis_output['inference']}<br><br>
                                    """)
            
    overall_summary = "Analysis complete. Review results for discrepancies."
    
    logger.info(f'Analysed results: {path_analysis}')
    # Refine overall summary based on LLM findings
    if any("MISMATCH" in msg or "POSSIBLE_MISMATCH" in msg for msg in overall_summary_messages):
        overall_summary = "Discrepancies or possible discrepancies detected across data lineage. Investigate 'MISMATCH' or 'POSSIBLE_MISMATCH' statuses in details."

    logger.info("Analysis complete.")
    return {"analysis_results": {"summary": overall_summary, "details": analysis_results}}


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
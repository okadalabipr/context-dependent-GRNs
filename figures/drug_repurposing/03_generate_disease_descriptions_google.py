# necessary: google-api-python-client

import os
import math
import time
import datetime
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import json
import numpy as np
import pandas as pd
from getpass import getpass
from openai import OpenAI

from googleapiclient.discovery import build

# google search
def getSearchResponse(keyword, search_count):
    today = datetime.datetime.today().strftime("%Y%m%d")
    timestamp = datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")

    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    
    # Calculate the number of pages needed (10 results per page)
    page_limit = math.ceil(search_count / 10)
    start_index = 1
    response = []
    
    for n_page in range(page_limit):
        try:
            # Pause for 1 second to respect rate limits
            time.sleep(1)
            # Determine the number of results for this page
            # On the last page, adjust if the remainder is less than 10
            if n_page == page_limit - 1 and search_count % 10 != 0:
                num_results = search_count % 10
            else:
                num_results = 10
            
            result = service.cse().list(
                q=keyword,
                cx=CUSTOM_SEARCH_ENGINE_ID,
                lr="lang_en",
                num=num_results,
                start=start_index
            ).execute()
            
            response.append(result)
            
            # Update the start index for the next page if available
            if "nextPage" in result.get("queries", {}):
                start_index = result["queries"]["nextPage"][0]["startIndex"]
            else:
                break
        except Exception as e:
            print(e)
            break

    out = {
        'snapshot_ymd': today,
        'snapshot_timestamp': timestamp,
        'response': response
    }
    jsonstr = json.dumps(out, ensure_ascii=False)
    jsonstr = jsonstr.encode('cp932', 'ignore')
    jsonstr = jsonstr.decode('cp932')
    
    return jsonstr

def search_google_and_get_snippets(keyword, search_count):
    search_results = getSearchResponse(keyword, search_count)
    search_results = json.loads(search_results)
    snippets = []
    for result in search_results["response"]:
        for item in result["items"]:
            snippets.append(item["snippet"])
    return snippets

################
# Logging setting
################
import logging
# Check if no handlers are already set, then configure basicConfig
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,  # Output logs of level INFO and above
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
else:
    # If handlers already exist, set the log level for the existing handlers
    logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)

##########################
# Define the Bing API key and endpoint
##########################
# BING_API_KEY = getpass('Enter your Bing API key: ')
# BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
GOOGLE_API_KEY = getpass(prompt="Enter your Google API KEY: ")
CUSTOM_SEARCH_ENGINE_ID = getpass(prompt="Enter your Custom Search Engine ID: ")
os.environ['OPENAI_API_KEY'] = getpass(prompt="Enter your OpenAI API KEY: ")


################
# Load the previous result
################
prev_df = pd.read_csv("02_disease_descriptions.csv")
prev_df = prev_df.fillna("")
prev_df["success"] = prev_df.apply(lambda row: row["description"].startswith(row["disease"]), axis=1)
logger.info(f"Loaded previous results. Failed: {len(prev_df[~prev_df['success']])}, Succeeded: {len(prev_df[prev_df['success']])}")

disease_names = prev_df[~prev_df["success"]]["disease"].values

################
# Batch requests
################
prompt_template = """
Based on the following web search result snippets, please generate a caoncise one-sentence description regarding the disease "{disease_name}".

### Search Results
{search_results}

### Output
The output must always begin with "{disease_name}, " and consist solely of the disease name and its description on a single line, with no additional information.
"""

req_format = """{{"custom_id": "request_{i}", "method": "POST", "url": "/v1/chat/completions", """\
""""body": {{"model": "o3-mini-2025-01-31", "messages": [{{"role": "user", "content": "{prompt}"}}], "max_completion_tokens": 1000}}}}"""

logger.info(f"Generating requests for {len(prev_df[~prev_df['success']])} diseases")
reqs = []
for idx, row in tqdm(prev_df[~prev_df["success"]].iterrows(), total=len(prev_df[~prev_df["success"]])):
    disease_name = row["disease"].strip()

    # Search for the disease name
    search_results = search_google_and_get_snippets(disease_name, 10)

    # Generate the prompt
    prompt = prompt_template.format(
        disease_name=disease_name,
        search_results="\n".join(search_results)
    ).strip()
    prompt = prompt.replace("\n", "\\n").replace('"', '\\"')
    req = req_format.format(i=idx, prompt=prompt)
    reqs.append(req)
logger.info("Requests generated")

# Save the requests to a file to confirm that they were generated correctly
with open("03_batchinput.jsonl", "w") as file:
    for i, request in enumerate(reqs):
        try:
            request = json.loads(request)
            file.write(json.dumps(request) + "\n")
        except:
            print(i)


client = OpenAI()
batch_input_file = client.files.create(
    file=open("03_batchinput.jsonl", "rb"),
    purpose="batch"
)

########################
# Run the batch request
########################
batch_input_file_id = batch_input_file.id
batch = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "nightly eval job"
    }
)
logger.info("Batch request submitted")


################
# Get the batch completion
################
# Check the status of the batch every 60 seconds
while True:
    batch = client.batches.retrieve(batch.id)
    if batch.status == "completed":
        break
    time.sleep(60)
logger.info("Batch completed")

# Download the batch completion
file_response = client.files.content(batch.output_file_id)
json.dump(file_response.text, open("03_batchoutput.jsonl", "w"))

# Read the batch completion
output_descriptions = []
indices = []
for response in file_response.text.strip().split("\n"):
    response = json.loads(response)
    content = response["response"]["body"]["choices"][0]["message"]["content"]
    idx = int(response["custom_id"].replace("request_", ""))

    output_descriptions.append(content)
    indices.append(idx)
logger.info("Batch completion read")
output_descriptions = np.array(output_descriptions)

output_df = prev_df.copy()
output_df.loc[indices, "description"] = output_descriptions
output_df["success"] = output_df.apply(lambda row: row["description"].startswith(row["disease"]), axis=1)
logger.info(f"Updated results. Failed: {len(output_df[~output_df['success']])}, Succeeded: {len(output_df[output_df['success']])}")
output_df.to_csv("03_disease_descriptions.csv", index=False)


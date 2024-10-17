import json
import os
import tempfile
import time
from code.config import config

import keras
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
from google.oauth2 import service_account
from keras import ops
from pydantic import BaseModel

load_dotenv()
app = FastAPI()


creds = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(creds)
storage_client = storage.Client(credentials=credentials)
bucket = storage_client.get_bucket("params_deployment_template")


def download_and_load_model(model_gcs_path: str) -> keras.Model:
    # Create a temporary file and get its path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_model_file:
        temp_model_path = temp_model_file.name
        print(f"Temporary file {temp_model_path} created.")

    try:
        # Download the model from GCS
        print("Downloading the model...")
        start_download_time = time.time()
        bucket.blob(model_gcs_path).download_to_filename(temp_model_path)
        print(
            f"Model downloaded to {temp_model_path} in {time.time() - start_download_time:.2f} seconds"
        )

        # Load the model from the temporary file
        print("Loading the model...")
        start_load_time = time.time()
        loaded_model = keras.models.load_model(temp_model_path)
        print(
            f"Model loaded successfully in {time.time() - start_load_time:.2f} seconds"
        )
        return loaded_model
    finally:
        # Clean up the temporary file
        os.unlink(temp_model_path)
        print(f"Temporary file {temp_model_path} deleted.")


# track how long it takes to process a request
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-MS"] = str(process_time)
    return response


model = download_and_load_model(model_gcs_path=config.prod_model_gcs_path)

# TODO impl the rest

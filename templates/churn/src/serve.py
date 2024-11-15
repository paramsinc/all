import time
import typing as T
from datetime import datetime
from enum import Enum

import keras
import numpy as np
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

from src.config import config
from src.data import (
    event_type_to_int,
    get_padded_sequences_and_time_diffs_and_labels,
    get_scaler,
    normalize_time_differences,
)

app = FastAPI()


def download_and_load_model(path: str) -> keras.Model:
    return keras.models.load_model(path)


# track how long it takes to process a request
@app.middleware("http")
async def add_process_time_header(request: Request, call_next: T.Callable) -> Response:
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-MS"] = str(process_time)
    return response


model = download_and_load_model(path=config.prod_model_path)
scaler = get_scaler()


class EventType(str, Enum):
    new_membership = "new membership"
    rejoin_membership = "rejoin membership"
    renewed_membership = "renewed membership"
    wash = "wash"
    expired = "expired"
    terminated = "terminated"


class Event(BaseModel):
    event_type: EventType
    timestamp: datetime


# TODO impl the rest
@app.post("/predict_churn", response_model=list[float])
def predict_churn(sequences: list[list[Event]]) -> list[float]:
    x_test_event: list[list[int]] = [
        [event_type_to_int[e.event_type.value] for e in sequence]
        for sequence in sequences
    ]
    x_test_time: list[list[int]] = []
    for sequence in sequences:
        timestamps = [event.timestamp for event in sequence]
        time_diff_sequence: list[int] = []
        for i in range(1, len(timestamps)):
            diff = int((timestamps[i] - timestamps[i - 1]).total_seconds())
            time_diff_sequence.append(diff)
        # For the first event, insert a default time difference (e.g., 0)
        time_diff_sequence.insert(0, 0)
        x_test_time.append(np.array(time_diff_sequence, dtype=np.float32))

    x_test_time = normalize_time_differences(
        time_differences=x_test_time, scaler=scaler
    )

    (
        padded_sequences,
        padded_time_diffs,
        _,
    ) = get_padded_sequences_and_time_diffs_and_labels(
        sequences=x_test_event,
        normalized_time_differences=x_test_time,
        labels=[0 for _ in sequences],
    )

    predictions: list[float] = model.predict([padded_sequences, padded_time_diffs])
    return predictions

"""Separate real Celery consumer with synthetic model subprocesses."""

import os
import sys
from functools import partial

from backend.processing import run_stage
from backend.runtime import resident_models
from backend.settings import Settings
from backend.tasks import worker_main

if __name__ == "__main__":
    settings = Settings(
        detection_python=sys.executable,
        classification_python=sys.executable,
        annotation_python=sys.executable,
    )
    if os.environ.get("MOCK_RESIDENT") == "1":
        with resident_models(settings, module="backend.tests.mock_model") as runtime:
            worker_main(settings, runner=runtime)
    else:
        worker_main(settings, runner=partial(run_stage, module="backend.tests.mock_model"))

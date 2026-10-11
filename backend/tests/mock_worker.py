"""Separate real Celery consumer with synthetic model subprocesses."""

import sys
from functools import partial

from backend.processing import run_stage
from backend.settings import Settings
from backend.tasks import worker_main

if __name__ == "__main__":
    settings = Settings(
        detection_python=sys.executable,
        classification_python=sys.executable,
        annotation_python=sys.executable,
    )
    worker_main(settings, runner=partial(run_stage, module="backend.tests.mock_model"))

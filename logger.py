import logging
from datetime import datetime
from pathlib import Path

# Create logs directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Unique log file per run
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = log_dir / f"experiment_{run_id}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ],
    force=True  # useful if logging was already configured, e.g. notebooks
)

def log(*messages: str) -> None:
    logging.info(" ".join(messages))
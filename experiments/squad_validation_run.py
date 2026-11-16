from experiment_runner.experiment_runner import run_experiment
from corpus.types import Dataset
from models.model import Model
from lexical_side_channel.lsc import LexicalSideChannelBaseline
from experiment_runner.experiment_runner import run_experiment

run_experiment(
    run_id="mistral_squad_validation_run",
    dataset=Dataset.SQUAD_VALIDATION,
    count=3000,
    model_enum=Model.PISCO,
    lexical_side_channel_method_enum=None,
    results_database_path="data/squad_validation/pisco_squad_chunked.db"
)

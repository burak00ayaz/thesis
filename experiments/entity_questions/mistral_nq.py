from experiment_runner.experiment_runner import run_experiment
from corpus.types import Dataset
from models.model import Model
from lexical_side_channel.lsc import LexicalSideChannelBaseline
from experiment_runner.experiment_runner import run_experiment

run_experiment(
    run_id="mistral_nq_entity_questions",
    dataset=Dataset.NQ,
    count=200,
    model_enum=Model.MISTRAL,
    lexical_side_channel_method_enum=None,
    results_database_path="data/mistral_nq.db"
)

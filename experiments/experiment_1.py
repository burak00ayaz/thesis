from experiment_runner.experiment_runner import run_experiment
from questions.types import Dataset
from models.model import Model
from lexical_side_channel.lsc import LexicalSideChannelBaseline
from experiment_runner.experiment_runner import run_experiment

run_experiment(
    run_id="runner_test_pisco_squad_spacy_001",
    dataset=Dataset.SQUAD,
    count=1,
    model_enum=Model.PISCO,
    lexical_side_channel_method_enum=LexicalSideChannelBaseline.SPACY_NER,
    results_database_path="data/runner_test.db"
)
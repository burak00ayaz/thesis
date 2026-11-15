from experiment_runner.experiment_runner import run_experiment
from corpus.types import Dataset
from models.model import Model
from lexical_side_channel.lsc import LexicalSideChannelBaseline
from experiment_runner.experiment_runner import run_experiment

run_experiment(
    run_id="pisco_squad_entity_questions",
    dataset=Dataset.SQUAD,
    count=200,
    model_enum=Model.PISCO,
    lexical_side_channel_method_enum=LexicalSideChannelBaseline.RANKED_NER_LSC,
    results_database_path="data/ranked_ner/pisco_ranked_ner_filtered.db"
)

from experiment_runner.experiment_runner import run_experiment
from questions.types import Dataset
from models.model import Model
from lexical_side_channel.lsc import LexicalSideChannelBaseline
from experiment_runner.experiment_runner import run_experiment

run_experiment(
    run_id="pisco_synthworlds_spacy_001",
    dataset=Dataset.SYNTHWORLDS_RM,
    count=200,
    model_enum=Model.PISCO,
    lexical_side_channel_method_enum=LexicalSideChannelBaseline.SPACY_NER,
    results_database_path="data/experiments.db"
)

run_experiment(
    run_id="pisco_synthworlds_spacy_001",
    dataset=Dataset.SYNTHWORLDS_RM,
    count=200,
    model_enum=Model.PISCO,
    lexical_side_channel_method_enum=None,
    results_database_path="data/experiments.db"
)

run_experiment(
    run_id="pisco_synthworlds_spacy_001",
    dataset=Dataset.SYNTHWORLDS_SM,
    count=200,
    model_enum=Model.PISCO,
    lexical_side_channel_method_enum=LexicalSideChannelBaseline.SPACY_NER,
    results_database_path="data/experiments.db"
)

run_experiment(
    run_id="pisco_synthworlds_spacy_001",
    dataset=Dataset.SYNTHWORLDS_SM,
    count=200,
    model_enum=Model.PISCO,
    lexical_side_channel_method_enum=None,
    results_database_path="data/experiments.db"
)

# PISCO: Synthworlds RM and SM (200 questions)
# Accuracy Real: 0.245
# Accuracy Real with LSC: 0.31
# Accuracy Synth: 0.19
# Accuracy Synth with LSC: 0.255
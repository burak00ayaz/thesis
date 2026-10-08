from experiment_runner.experiment_runner import run_experiment
from questions.types import Dataset
from models.model import Model
from lexical_side_channel.lsc import LexicalSideChannelBaseline
from experiment_runner.experiment_runner import run_experiment

run_experiment(
    run_id="pisco_squad_spacy_001",
    dataset=Dataset.SQUAD,
    count=123,
    model_enum=Model.PISCO,
    lexical_side_channel_method_enum=LexicalSideChannelBaseline.SPACY_LSC,
    results_database_path="data/experiments.db"
)

run_experiment(
    run_id="pisco_squad_spacy_001",
    dataset=Dataset.SQUAD,
    count=123,
    model_enum=Model.PISCO,
    lexical_side_channel_method_enum=None,
    results_database_path="data/experiments.db"
)

run_experiment(
    run_id="pisco_squad_spacy_001",
    dataset=Dataset.SQUAD_MAPPED,
    count=123,
    model_enum=Model.PISCO,
    lexical_side_channel_method_enum=LexicalSideChannelBaseline.SPACY_LSC,
    results_database_path="data/experiments.db"
)

run_experiment(
    run_id="pisco_squad_spacy_001",
    dataset=Dataset.SQUAD_MAPPED,
    count=123,
    model_enum=Model.PISCO,
    lexical_side_channel_method_enum=None,
    results_database_path="data/experiments.db"
)

# PISCO: SQUAD Real and SQUAD Mapped (123 questions)
# Accuracy SQUAD Real: 0.7398373983739838
# Accuracy SQUAD Real with LSC: 0.7967479674796748
# Accuracy SQUAD Synth: 0.6016260162601627
# Accuracy SQUAD Synth with LSC: 0.6910569105691057
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

# PISCO: Synthworlds RM and SM (200 questions)
# Accuracy Real: 0.245
# Accuracy Real with LSC: 0.31
# Accuracy Synth: 0.19
# Accuracy Synth with LSC: 0.255

# PISCO: SQUAD Real and SQUAD Mapped (123 questions)
# Accuracy SQUAD Real: 0.7398373983739838
# Accuracy SQUAD Real with LSC: 0.7967479674796748
# Accuracy SQUAD Synth: 0.6016260162601627
# Accuracy SQUAD Synth with LSC: 0.6910569105691057
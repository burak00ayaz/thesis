from experiment_runner.experiment_runner import run_experiment
from corpus.types import Dataset
from models.model import Model
from lexical_side_channel.lsc import LexicalSideChannelBaseline

run_experiment(
    run_id="mistral_squad_gliner_001",
    dataset=Dataset.SQUAD_MAPPED,
    count=123,
    model_enum=Model.MISTRAL,
    lexical_side_channel_method_enum=LexicalSideChannelBaseline.GLINER_LSC,
    results_database_path="data/experiments_mistral.db"
)

# MISTRAL: SQUAD Real and SQUAD Mapped (Gliner) (123 questions)
# Accuracy SQUAD Real: 0.9105691056910569
# Accuracy SQUAD Real with LSC: 0.8699186991869918
# Accuracy SQUAD Synth: 0.9024390243902439
# Accuracy SQUAD Synth with LSC: 0.8536585365853658
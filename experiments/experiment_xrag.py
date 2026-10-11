from experiment_runner.experiment_batch_runner import run_experiment
from questions.types import Dataset
from models.model import Model
from lexical_side_channel.lsc import LexicalSideChannelBaseline

run_experiment(
    run_id="xrag_squad_test_001",
    dataset=Dataset.SQUAD_MAPPED,
    count=123,
    model_enum=Model.XRAG,
    lexical_side_channel_method_enum=LexicalSideChannelBaseline.GLINER_LSC,
    results_database_path="data/xrag_results.db"
)

# XRAG: SQUAD Real and SQUAD Mapped (123 questions)
# Accuracy SQUAD Real: 0.1951219512195122
# Accuracy SQUAD Real with LSC: 0.5284552845528455
# Accuracy SQUAD Synth: 0.03278688524590164
# Accuracy SQUAD Synth with LSC: 0.4672131147540984


# Accuracy SQUAD with GlinerLarge LSC: 0.6147540983606558
# Accuracy SQUAD Mapped with GlinerLarge LSC: 0.5142857142857142
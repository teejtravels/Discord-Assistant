import os
from precise_runner import PreciseEngine, PreciseRunner

# Path to the Precise engine and model
ENGINE_PATH = '/workspaces/Discord-Assistant/precise-engine'
MODEL_PATH = '/workspaces/Discord-Assistant/mycroft-precise/.venv/lib/python3.8/site-packages/precise_runner/resources/hey-mycroft.pb'

def on_wake():
    print("Wake word detected!")

engine = PreciseEngine(ENGINE_PATH, MODEL_PATH)
runner = PreciseRunner(engine, on_activation=on_wake)
runner.start()
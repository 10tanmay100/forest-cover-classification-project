from forest_cover.pipeline.pipeline import Pipeline
from forest_cover.config.configuration import Configuration
def f():
    p=Pipeline(Configuration)
    print(p.run_pipeline())

f()
# Experiment Registry

Store reproducibility metadata for meaningful experiments in a structured
manifest based on `templates/run_manifest.json`.

For each run, record the code revision, configuration, model and index
versions, environment, metrics, runtime, and artifact provenance where
applicable.

Do not commit large model weights, private data, credentials, or generated
artifacts merely to make an experiment reproducible. Document how required
inputs can be obtained and how outputs can be regenerated.

The initial JSON format can be extended when real experiment requirements
are known.

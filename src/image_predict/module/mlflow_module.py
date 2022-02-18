import mlflow


def start_experiment(tracking_uri: str = None, experiment_name: str = "experiment", description: str = ""):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.set_tag('mlflow.runName', mlflow.active_run().info.run_id)
    mlflow.set_tag('mlflow.note.content', description)
    mlflow.set_tag('description', description)

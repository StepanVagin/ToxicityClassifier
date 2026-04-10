def create_model(config):
    model_type = config["model_type"]
    if model_type == "logistic_regression":
        from models.logistic_regression_model import LogisticRegressionModel

        return LogisticRegressionModel(config)
    if model_type == "albert":
        from models.albert_model import ALBERTModel

        return ALBERTModel(config)
    if model_type == "distilbert":
        from models.distilbert_model import DistilBERTModel

        return DistilBERTModel(config)
    raise ValueError(f"Unknown model type: {model_type}")

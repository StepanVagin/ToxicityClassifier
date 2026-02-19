def create_model(config):
    from models.logistic_regression_model import LogisticRegressionModel
    
    model_type = config['model_type']
    if model_type == 'logistic_regression':
        return LogisticRegressionModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

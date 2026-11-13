"""Named text-scoring model; training and validation live in its estimator."""


class NeuralNetworkModel:
    """Keep the model identity and saved-artifact interface in one place."""

    name = "neural_network"

    def __init__(self, seed: int = 42, **parameters):
        from .neural_network import TorchMLPRegressor
        strategy=parameters.pop('strategy','standard')
        self.strategy=strategy
        if strategy=='additive_ensemble':
            from .ensemble import TextPopularityEnsemble
            self.estimator=TextPopularityEnsemble(seed=seed,**parameters)
        elif strategy=='standard':self.estimator = TorchMLPRegressor(seed=seed, **parameters)
        else:raise ValueError('Unknown neural text strategy')

    def fit(self, features, scores):
        self.estimator.fit(features, scores)
        return self

    def predict(self, features):
        return self.estimator.predict(features)

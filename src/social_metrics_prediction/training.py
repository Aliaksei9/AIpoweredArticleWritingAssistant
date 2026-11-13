"""Cross-fitting the upstream neural network for training without target leakage."""

import numpy as np
from sklearn.model_selection import GroupKFold

from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased


def cross_fitted_scores(features, labels, groups, *, folds=3, seed=42, threads=4,
                        on_progress=None, factory=None, eligible_training_indices=None):
    """Each row's score is predicted by a model that never saw its text group.

    Caller must pass training rows only. Fitted upstream models are discarded;
    the final module-3 artifact is trained separately on all training rows.
    """
    features, labels, groups = np.asarray(features), np.asarray(labels), np.asarray(groups)
    if len(features) != len(labels) or len(groups) != len(labels) or len(set(groups)) < folds:
        raise ValueError("Cross-fitting requires aligned rows and enough distinct text groups")
    predictions = np.full(len(labels), np.nan)
    audit = []
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train, validation) in enumerate(splitter.split(features, labels, groups), 1):
        if eligible_training_indices is not None:
            train=train[np.isin(train,np.asarray(eligible_training_indices,dtype=int))]
            if not len(train):raise ValueError('No canonical text labels in an upstream training fold')
        if on_progress:
            on_progress(f"Text neural network: cross-fit {fold}/{folds}")
        model = factory() if factory else AssesmentConformPPtextBased(seed=seed, num_threads=threads)
        model.train_features(features[train], labels[train])
        predictions[validation] = model.model.predict(features[validation])
        audit.append(dict(train_indices=train.tolist(), validation_indices=validation.tolist()))
    if not np.isfinite(predictions).all():
        raise RuntimeError("Incomplete out-of-fold score predictions")
    return predictions, audit


def train_text_head(encoder, features, scores, *, seed=42, threads=4,strategy='standard'):
    return AssesmentConformPPtextBased(encoder=encoder, seed=seed, num_threads=threads,strategy=strategy).train_features(features, scores)

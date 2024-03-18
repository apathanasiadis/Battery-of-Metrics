import pytest

from sklearn.utils.estimator_checks import check_estimator

from bomtemplate import TemplateEstimator
from bomtemplate import TemplateClassifier
from bomtemplate import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)

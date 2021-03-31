import sys
import os.path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


from src.test_the_test import return_crayon


def test_return_crayon():
    assert return_crayon() == "crayon"
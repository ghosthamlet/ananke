"""
Helper functions that help load datasets in Ananke.
"""

import pandas as pd
import os

MODULE_PATH = os.path.dirname(__file__)


def load_conditionally_ignorable_data():
    """
    Load toy data for the conditionally ignorable model
    where the confounder is Viral Load, T is the treatment
    and the outcome is CD4 counts.

    :return: pandas dataframe.
    """

    path = os.path.join(MODULE_PATH, "simulated/conditionally_ignorable.csv")
    return pd.read_csv(path)


def load_afixable_data():
    """
    Load toy data for an adjustment fixable setting
    where T is the treatment and the outcome is CD4 counts.

    :return: pandas dataframe.
    """

    path = os.path.join(MODULE_PATH, "simulated/a_fixable.csv")
    return pd.read_csv(path)


def load_frontdoor_data():
    """
    Load toy data for frontdoor setting
    where T is the treatment and the outcome is CD4 counts.

    :return: pandas dataframe.
    """

    path = os.path.join(MODULE_PATH, "simulated/frontdoor.csv")
    return pd.read_csv(path)
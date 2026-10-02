import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(ROOT / "sample_funnel.csv")


def make_funnel(stages_users):
    return pd.DataFrame(stages_users, columns=["stage", "users"])

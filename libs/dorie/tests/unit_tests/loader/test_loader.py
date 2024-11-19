from dorie.tests import data
import pandas

def test_example_data():
    df = pandas.read_csv(data.SENTIMATE_CSV)
    assert df.columns.tolist() == ['text', 'label']
    
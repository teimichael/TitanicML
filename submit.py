import pandas as pd


def submit(p):
    test_df = pd.read_csv('./input/test.csv')
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": p
    })
    submission.to_csv('./output/titanic.csv', index=False)

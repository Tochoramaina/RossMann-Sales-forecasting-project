import great_expectations as ge 
import pandas as pd 

def validate_data(df:pd.DataFrame) -> bool:
    ge_df = ge.from_pandas(df)
    ge_df.expect_column_values_to_not_be_null('Store')
    ge_df.expect_column_values_to_be_between("Store", min_value = 1, max_value=1115)
    ge_df.expect_column_values_to_be_between("Sales", min_value =0, max_value =100000)
    ge_df.expect_column_values_to_be_in_set('Promo', [0, 1])
    ge_df.expect_column_values_to_be_in_set("Open", [0, 1])
    ge_df.expect_column_values_to_be_between('CompetitionDistance',  min_value =0, max_value =100000)
    #ge_df.expect_column_values_to_be_of_type("Date", "object")

    results = ge_df.validate()
    return results['success']


if __name__ == '__main__':
    test_df = pd.DataFrame([{'Store': 1, 'Open': 1, 'Promo': 1, 'CompetitionDistance': 1270}])
    if validate_data(test_df):
        print("Data Validation successful")
    else:
        raise ValueError("Data Validation failed")
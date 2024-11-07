import random

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encodeDataAndSplit(filtered_df, categories: list):
    class1, class2 = 1, 0

    # Filter rows by each category and sample 30 rows for training
    random_state = random.randint(30, 100000) % random.randint(30, 200)
    df_category_1 = filtered_df[filtered_df['bird category'] == categories[0]].sample(30, random_state=random_state)
    df_category_2 = filtered_df[filtered_df['bird category'] == categories[1]].sample(30, random_state=random_state)

    # Map categories to class1 and class2 for training data
    df_category_1['category_encoded'] = class2
    df_category_2['category_encoded'] = class1

    # Concatenate the two sampled dataframes for training
    df_train = pd.concat([df_category_1, df_category_2])

    # Get remaining rows for testing
    df_test = filtered_df[
        (filtered_df['bird category'] == categories[0]) | (filtered_df['bird category'] == categories[1])
        ].drop(df_train.index)

    # Map categories to class1 and class2 for testing data
    df_test['category_encoded'] = df_test['bird category'].map({categories[0]: class2, categories[1]: class1})

    # Drop 'bird category' from both DataFrames
    df_train = df_train.drop(columns=['bird category']).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.drop(columns=['bird category']).sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert DataFrames to numpy arrays
    train_output = df_train['category_encoded'].values
    train_input = df_train.drop(columns=["category_encoded"]).values
    test_output = df_test['category_encoded'].values
    test_input = df_test.drop(columns=["category_encoded"]).values

    return train_input, train_output, test_input, test_output


class Data:
    def __init__(self):
        self.df = pd.read_csv("DataSets/birds.csv")
        self.age_encoder = LabelEncoder()
        self.birdsCategories = ["A", "B", "C"]
        self.clean()

    def clean(self):
        # Fill null values with the mode in bird category
        most_frequent_gender = self.df.groupby('bird category')['gender'].apply(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'unknown')
        )

        # Reset the index to align with the main DataFrame
        most_frequent_gender = most_frequent_gender.reset_index(drop=True)
        self.df['gender'] = most_frequent_gender
        self.encode_age()

    def encode_age(self):
        # Assuming there's an 'age' column to encode
        if 'gender' in self.df.columns:
            self.df['gender'] = self.age_encoder.fit_transform(self.df['gender'])

    def decode_age(self, encoded_values):
        # Decode the encoded age values
        return self.age_encoder.inverse_transform(encoded_values)

    def GenerateData(self, categories: list, filtered_df):
        if filtered_df is None:
            filtered_df = self.df
        # Validate input categories
        if not all(cat in self.birdsCategories for cat in categories) or len(categories) != 2:
            raise ValueError("Invalid Categories")

        return encodeDataAndSplit(filtered_df, categories)

    def GenerateDataWithFeatures(self, categories: list, features: list):
        # Validate input features
        if not all(feature in self.df.columns for feature in features):
            raise ValueError("One or more specified features are not in the dataset.")

        # Select only the specified features and 'bird category'
        df_filtered = self.df[features + ['bird category']]
        return self.GenerateData(categories, filtered_df=df_filtered)

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def encode_cat(df, variables):
    onehot = OneHotEncoder(sparse_output=False, drop='first')
    for var in variables:
        name = var + '_encoded'
        # Fit and transform the variable column into one-hot encoded array
        transformed = onehot.fit_transform(df[[var]])
        # Create a DataFrame from the transformed array with appropriate column names
        encoded_df = pd.DataFrame(transformed, columns=[f"{var}_{category}" for category in onehot.categories_[0][1:]])
        # Concatenate the original DataFrame with the new encoded DataFrame
        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        df.drop(columns=[var], inplace=True)
    return df


        
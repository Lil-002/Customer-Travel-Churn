import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json
import os


def prepare_data(file_path='Customertravel.csv', output_dir='prepared_data'):
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")

    # Checking for missing values
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    # Case 1: No missing values at all
    if len(missing_cols) == 0:
        print("✓ No missing values found")

    # Case 2: There are missing values to handle
    else:
        print(f"⚠ Found missing values in columns: {list(missing_cols.index)}")
        print(f" Missing counts:\n{missing_cols}")

        # Process each column with missing values
        for col in missing_cols.index:

            # ---------------------------------------------------------
            # CASE A: Categorical columns (dtype = object)
            # ---------------------------------------------------------
            if df[col].dtype == 'object':
                mode_vals = df[col].mode()

                # Check if the mode exists
                if mode_vals.empty:
                    # No mode → fallback for empty or all-NaN categorical columns
                    fill_value = "Unknown"
                else:
                    # Mode exists → take the first (most frequent) value
                    fill_value = mode_vals.iloc[0]

                # Fill missing categorical values
                df[col] = df[col].fillna(fill_value)

            # ---------------------------------------------------------
            # CASE B: Numeric columns
            # ---------------------------------------------------------
            else:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)

        print("✓ Missing values handled")

    # Can't check for duplicates .. this function will drop 507 rows of customer data
    # Can't check for outliers .. the only column applicable is Age.
    # But we should not apply outlier on Age at all because the Age band from our collected data
    # is very narrow (27-38) and this will reduce the model's ability to predict new
    # market segments.

    # Target variable
    X = df.drop('Target', axis=1)
    y = df['Target']

    # Column types
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"\nNumerical: {numerical_cols}")
    print(f"Categorical: {categorical_cols}")

    #
    # Using 'drop-first' for the oneHotencoding
    #
    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(
            drop='first',  # Standard practice
            sparse_output=False,
            handle_unknown='ignore'
        ))
    ])

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    #
    # Splitting data
    # Must remember to use random_state to 42 throughout pipeline
    # so we get the same result everytime this code runs
    # stratify set to y - to ensure we get the same ratio of churn in the split data
    #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    #
    # Fit and transform
    # Only apply fit_transform only on X_train
    # Coz fit_transform learns parameters like mean, variance and then applies them
    # Don't want information from X_test to get into the processing
    # which will affect the model
    #
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    #
    # Merging numerical and categorical features
    #
    feature_names = numerical_cols.copy()
    encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_features = encoder.get_feature_names_out(categorical_cols)
    feature_names.extend(cat_features)

    #
    # Important - must save the feature list
    # After the onehotencoding - the number, the names and the order of the columns
    # have changed.  We need the feature name list so Shap can use them later
    # for the interpretation of the feature set.
    #
    category_info = {}
    for col in categorical_cols:
        # Get original unique values
        unique_vals = sorted(X_train[col].dropna().unique())
        category_info[col] = {
            'original_values': list(unique_vals),
            'reference_category': unique_vals[0],  # The dropped one
            'encoded_columns': [f for f in cat_features if f.startswith(f"{col}_")]
        }

    print(f"\n{'=' * 60}")
    print("CATEGORY ENCODING DETAILS")
    print(f"{'=' * 60}")
    for col, info in category_info.items():
        print(f"\n{col}:")
        print(f"  All values: {info['original_values']}")
        print(f"  Reference (dropped): '{info['reference_category']}'")
        print(f"  Created features: {info['encoded_columns']}")

    #
    # Saving all the output files
    #

    os.makedirs(output_dir, exist_ok=True)

    #
    # Saving the processed files (scaling and onehotencoding)
    #
    pd.DataFrame(X_train_processed, columns=feature_names).to_csv(
        f'{output_dir}/X_train_processed.csv', index=False
    )
    pd.DataFrame(X_test_processed, columns=feature_names).to_csv(
        f'{output_dir}/X_test_processed.csv', index=False
    )

    #
    # Saving targets as raw as no processing has been applied.
    #
    pd.DataFrame(y_train, columns=['Target']).to_csv(
        f'{output_dir}/y_train_raw.csv', index=False
    )
    pd.DataFrame(y_test, columns=['Target']).to_csv(
        f'{output_dir}/y_test_raw.csv', index=False
    )

    # Save raw features for reference
    X_train.to_csv(f'{output_dir}/X_train_raw.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test_raw.csv', index=False)

    #
    # Using Joblib to save the preprocessor Python arrays
    # over pickle coz joblib handles large arrays better
    # needed for Shap explanations
    # want the exact object back and joblib ensures it loads fast and safely
    #
    joblib.dump(preprocessor, f'{output_dir}/preprocessor.joblib')

    #
    # Save metadata with everything needed (including the encoding choice
    #
    metadata = {
        "feature_names": feature_names,
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
        "category_info": category_info,
        "data_shapes": {
            "X_train_processed": X_train_processed.shape,
            "X_test_processed": X_test_processed.shape,
            "y_train_raw": y_train.shape,
            "y_test_raw": y_test.shape
        },
        "encoding_note": "Used drop='first' for one-hot encoding."
    }

    with open(f'{output_dir}/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print("DATA PREPARATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output: {output_dir}/")
    print(f"Training set: {X_train_processed.shape}")
    print(f"Test set: {X_test_processed.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"\nFiles created:")
    print(f"  X_train_processed.csv, X_test_processed.csv")
    print(f"  y_train_raw.csv, y_test_raw.csv")
    print(f"  X_train_raw.csv, X_test_raw.csv")
    print(f"  preprocessor.joblib")
    print(f"  metadata.json")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, metadata

#
# Main
#
if __name__ == "__main__":
    # Prepare data
    X_train, X_test, y_train, y_test, preprocessor, metadata = prepare_data()
    print('Data prep is completed')

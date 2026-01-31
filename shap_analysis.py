import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load models and test data for Shap analysis
# This includes the Feature_name list saved during the data preprocessing stage
def load_models_and_data(models_dir='models', data_dir='prepared_data'):
    print("Loading models and data for SHAP analysis...")

    # Loading the best model info
    with open(f'{models_dir}/best_model_info.json', 'r', encoding='utf-8') as f:
        best_model_info = json.load(f)

    best_model_name = best_model_info['best_model_name']
    print(f"Best model: {best_model_name}")

    # Load test data
    X_test = pd.read_csv(f'{data_dir}/X_test_processed.csv')
    y_test = pd.read_csv(f'{data_dir}/y_test_raw.csv').iloc[:, 0]

    # Load metadata for feature names and category info
    with open(f'{data_dir}/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    feature_names = metadata['feature_names']
    category_info = metadata.get('category_info', {})

    # Load the best model
    model_dir = f'{models_dir}/{best_model_name.replace(" ", "_").lower()}'

    if best_model_name == 'Artificial Neural Network':
        # If best model is ANN - then need to load model.h5
        # coz keras stores using its own native saving format
        import tensorflow as tf
        model = tf.keras.models.load_model(f'{model_dir}/model.h5')
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        # The rest of the models uses joblib coz they are all scikit-learn models
        model = joblib.load(f'{model_dir}/model.joblib')

    return model, X_test.values, y_test.values, feature_names, best_model_name, category_info


def create_shap_explainer(model, model_name, X_sample):
    # Shap uses different explainer engines based on model type
    print(f"\nCreating SHAP explainer for {model_name}...")

    try:
        if model_name == 'Artificial Neural Network':
            # For ANN - use KernelExplainer with a simple prediction function

            def model_predict(data):
                # Shap needs a 1dimensional vector of model outputs
                # ANN often returns predictions in a 2 column format (n,2)
                # Shap must receive only 1 output per sample
                # Hence model_predict is a wrapper that transforms ANN's prediction output
                # into a proper 1D vector of probabilities for Class 1 (ie. Churn).
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)
                # Predict and return probability of class 1 (churn)
                predictions = model.predict(data, verbose=0)
                # If predictions are 2D with shape (n, 1), flatten
                if predictions.shape[1] == 1:
                    return predictions.flatten()
                # If predictions are 2D with shape (n, 2), return probability of class 1
                elif predictions.shape[1] == 2:
                    return predictions[:, 1]
                else:
                    return predictions

            # Use a background sample for KernelExplainer
            # KernelExplainer is computationally expensive
            # using ony 50 out of 200 records from X_sample
            # is std practice to keep Shap computations manageable

            background = shap.sample(X_sample, min(50, X_sample.shape[0]))
            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(X_sample, nsamples=100, silent=True)

        elif model_name in ['Random Forest', 'Gradient Boosting']:
            # Use TreeEXplainer for Tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # Handle binary classification output
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # Positive class

        elif model_name == 'Logistic Regression':
            # Use LinearExplainer for Linear model
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)

        else:
            # Use KernelExplainer again for anything not matching
            # the above models
            def model_predict(data):
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)
                return model.predict_proba(data)[:, 1]

            background = shap.sample(X_sample, min(50, X_sample.shape[0]))
            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(X_sample, nsamples=100, silent=True)

        print(f"SHAP explainer created successfully")

        # Ensure shap_values is numpy array
        if hasattr(shap_values, 'values'):
            shap_array = shap_values.values
        else:
            shap_array = np.array(shap_values)

        # For ANN KernelExplainer, sometimes we get list of arrays
        if isinstance(shap_array, list):
            shap_array = np.array(shap_array)

        return explainer, shap_array

    except Exception as e:
        print(f"Could not create standard SHAP explainer: {e}")
        print("Creating approximate SHAP values using permutation...")
        print("model name : ", model_name)
        # When the above specialized Shap explainers fail,
        # the Fallback PermutationExplainer is used
        # It works - coz it does not rely on model internals
        # It just estimates feature importance by shuffling features
        # and measuring the change in predictions

        if model_name == 'Artificial Neural Network':
            def model_predict(data):
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)
                predictions = model.predict(data, verbose=0)
                if predictions.shape[1] == 1:
                    return predictions.flatten()
                elif predictions.shape[1] == 2:
                    return predictions[:, 1]
                else:
                    return predictions

            explainer = shap.Explainer(model_predict, X_sample[:50])
        else:
            explainer = shap.Explainer(model.predict_proba, X_sample[:50])

        shap_values = explainer(X_sample[:50])

        if hasattr(shap_values, 'values'):
            shap_array = shap_values.values
        else:
            shap_array = np.array(shap_values)

        return explainer, shap_array


def calculate_shap_importance(shap_values, feature_names):
    # Calculate shap importance from shap values
    print("\nCalculating SHAP-based feature importance...")

    # Shap returns inconsistent types across explainers
    # so have to ensure shap_values is numpy array
    if hasattr(shap_values, 'values'):
        shap_array = shap_values.values
    else:
        shap_array = np.array(shap_values)

    # Shap outputs vary across explainers
    # so have to handle different shapes
    if len(shap_array.shape) == 3:
        # Multi-class: take mean across classes
        shap_array = np.mean(np.abs(shap_array), axis=2)
    elif len(shap_array.shape) == 1:
        # Single sample: reshape
        shap_array = shap_array.reshape(1, -1)

    # Compute global feature importance
    # Shap provides values per sample; to get 'global' importance
    # we take mean(abs(Shap) across all samples.
    # this is the sfd metric used in Shap summary plots
    if len(shap_array.shape) == 2:
        shap_importance = np.abs(shap_array).mean(axis=0)
    else:
        shap_importance = np.abs(shap_array)

    # ensure feature count matches
    # Shap occasionally outputs fewer features due to
    # dropped columns, encoding issues etc.
    n_features = min(len(shap_importance), len(feature_names))

    # Numpy arrays do not store column labels
    # Hence the need to convert to Dataframe to make it easier
    # for business interpretation, plotting and exporting
    feature_importance = pd.DataFrame({
        'feature': feature_names[:n_features],
        'shap_importance': shap_importance[:n_features],
        'shap_importance_normalized': shap_importance[:n_features] / shap_importance[:n_features].sum() * 100
    }).sort_values('shap_importance', ascending=False)

    print(f"Calculated SHAP importance for {len(feature_importance)} features")
    return feature_importance

# Build a clean business friendly table
# compute Shap direction and suggest simple actions
def create_business_friendly_interpretation(
    feature_importance: pd.DataFrame,
    category_info: dict,
    shap_values: np.ndarray = None,
    feature_names: list = None
) -> pd.DataFrame:

    # Copy input and ensure required column exists
    bi = feature_importance.copy().reset_index(drop=True)
    if 'feature' not in bi.columns:
        raise ValueError("feature_importance must contain a 'feature' column")

    # Set up new columns for buisness friendly outputs
    bi['feature_type'] = 'numerical'
    bi['business_interpretation'] = ''
    bi['impact_direction'] = 'neutral'
    bi['mean_shap'] = np.nan
    bi['business_meaning'] = ''
    bi['recommended_action'] = ''
    # Use normalized Shap importance if available ; fallback to percentage share
    bi['confidence_pct'] = bi.get('shap_importance_normalized',
                (bi['shap_importance'] / bi['shap_importance'].sum() * 100)).fillna(0)

    # Convert Shap values to 2D and align to features
    # Coz SHap output can come in several shapes and we need to standardize to
    # a single consistent format before computing per feature averages
    mean_shap_by_feature = None
    if shap_values is not None:
        # normalize shap_values into a 2D array (N, F)
        try:
            if hasattr(shap_values, 'values'):
                arr = shap_values.values
            else:
                arr = np.array(shap_values)

            # If shap returns 3-d (classes), average across classes (signed)
            if arr.ndim == 3:
                # average across classes then across samples
                arr = np.mean(arr, axis=2)

            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            # If arr shape mismatches feature count but feature_names provided, try to align (best-effort)
            if feature_names is not None and arr.shape[1] != len(feature_names):
                # if arr has fewer columns, assume first columns correspond to features in feature_importance order
                # otherwise, if arr has more, truncate
                n = min(arr.shape[1], len(feature_names))
                arr = arr[:, :n]

            mean_shap_by_feature = np.mean(arr, axis=0)  # signed mean SHAP per feature

        except Exception:
            mean_shap_by_feature = None

    # business glossary - readable menaing for known features
    business_glossary = {
        'ServicesOpted': 'Number of services / products the customer uses',
        'Ages': 'Customer age',
        'FrequentFlyer': 'Whether customer is in the frequent-flyer (loyalty) program',
        'AnnualIncomeClass': 'Customer income segment (Low/Middle/High)',
        'AccountSyncedToSocialMedia': 'Customer connected account to social media',
        'BookedHotelOrNot': 'Whether customer booked hotel (engagement indicator)',
        # add more domain-specific mappings here
    }

    # Simple rule based action generator
    def recommend_action(feature, direction, mean_shap, conf):
        # return short suggested action based on feature impact
        f = feature
        d = direction

        if 'ServicesOpted' in f:
            if d == 'increases churn':
                return 'Offer targeted upsell bundle or 30-day free trial to increase engagement'
            elif d == 'reduces churn':
                return 'Highlight existing service benefits in retention messaging'
        if 'FrequentFlyer' in f:
            if d == 'increases churn':
                return 'Target with loyalty incentives and travel benefits'
            else:
                return 'Maintain loyalty perks; consider cross-sell'
        if 'AnnualIncomeClass' in f:
            if 'Low' in f or ('Low Income' in f):
                return 'Offer budget bundles and flexible payment options'
            else:
                return 'Promote premium add-ons and concierge offers'
        if 'AccountSyncedToSocialMedia' in f:
            if d == 'increases churn':
                return 'Prompt account completion and offer referral incentives'
            else:
                return 'Use social channels for retention campaigns'
        if 'BookedHotel' in f or 'BookedHotelOrNot' in f:
            if d == 'increases churn':
                return 'Provide hotel loyalty rewards and special packages'
            else:
                return 'Upsell complementary travel services'
        if 'Age' in f or 'Ages' in f:
            if d == 'increases churn':
                return 'Use segment-specific offers (youth or senior campaigns)'
            else:
                return 'Maintain general retention messaging'
        # fallback
        if d == 'increases churn':
            return 'Investigate root cause; consider targeted retention tests'
        if d == 'reduces churn':
            return 'Leverage this feature in acquisition/retention messaging'
        return ''

    # Build business friendly rows
    for idx, row in bi.iterrows():
        feat = str(row['feature'])
        # determine feature type and readable interpretation
        is_cat = False
        for cat_feature, info in (category_info or {}).items():
            enc_cols = info.get('encoded_columns', []) or []
            if feat in enc_cols:
                # this is an encoded categorical column
                is_cat = True
                actual_value = feat.replace(f"{cat_feature}_", "")
                ref = info.get('reference_category', '')
                bi.at[idx, 'feature_type'] = 'categorical'
                bi.at[idx, 'business_interpretation'] = \
                    f"When {cat_feature} is '{actual_value}' (vs '{ref}')"
                break

        if not is_cat:
            # numerical fallback
            bi.at[idx, 'business_interpretation'] = f"{feat} (numerical value)"

        # business meaning from glossary or fallback
        base_name = feat.split('_')[0]
        bi.at[idx, 'business_meaning'] = business_glossary.get(base_name,
                                                               business_glossary.get(feat, 'Feature related to customer behaviour'))

        # Assign mean SHap and churn direction
        if mean_shap_by_feature is not None and idx < len(mean_shap_by_feature):
            m = float(mean_shap_by_feature[idx])
            bi.at[idx, 'mean_shap'] = m
            # threshold to call neutral
            eps = max(1e-6, np.abs(mean_shap_by_feature).max() * 0.01)
            if m > eps:
                dir_text = 'increases churn'
            elif m < -eps:
                dir_text = 'reduces churn'
            else:
                dir_text = 'neutral'
            bi.at[idx, 'impact_direction'] = dir_text
        else:
            bi.at[idx, 'impact_direction'] = 'neutral'
            bi.at[idx, 'mean_shap'] = np.nan

        # recommended action (rule-based)
        bi.at[idx, 'recommended_action'] = recommend_action(feat, bi.at[idx, 'impact_direction'],
                                                            bi.at[idx, 'mean_shap'], bi.at[idx, 'confidence_pct'])

    # sort by shap_importance descending (preserve original sort if present)
    if 'shap_importance' in bi.columns:
        bi = bi.sort_values('shap_importance', ascending=False).reset_index(drop=True)

    # add rank
    bi.insert(0, 'rank', range(1, len(bi) + 1))

    # final column ordering (extend as needed)
    cols = [
        'rank', 'feature', 'feature_type', 'business_interpretation',
        'business_meaning', 'impact_direction', 'mean_shap',
        'shap_importance', 'shap_importance_normalized',
        'confidence_pct', 'recommended_action'
    ]
    # keep only existing columns in that order
    cols = [c for c in cols if c in bi.columns]
    bi = bi[cols]

    return bi



def generate_shap_plots(explainer, shap_values, X_sample, feature_names,
                        category_info, output_dir='shap_results'):
    # Generate and save Shap Feature Importance (Bar Plot)
    print(f"\nGenerating SHAP Feature Importance (Bar Plot)")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Ensure shapes match
    if shap_values.shape[0] != X_sample.shape[0]:
        print(f"Adjusting SHAP values shape: {shap_values.shape} to match X_sample: {X_sample.shape}")
        # If SHAP values are for a single class in binary classification
        if len(shap_values.shape) == 2 and shap_values.shape[0] != X_sample.shape[0]:
            # Try transposing
            if shap_values.shape[1] == X_sample.shape[0]:
                shap_values = shap_values.T
            # Or take first row if it's representative
            elif shap_values.shape[0] == 1:
                shap_values = np.tile(shap_values, (X_sample.shape[0], 1))

    # Create better feature names with interpretation
    interpretable_feature_names = []
    for feature in feature_names[:X_sample.shape[1]]:
        interpretation_found = False

        # Check if this is an encoded categorical feature
        for cat_feature, info in category_info.items():
            if feature in info['encoded_columns']:
                actual_value = feature.replace(f"{cat_feature}_", "")
                ref_category = info['reference_category']
                interpretable_feature_names.append(
                    f"{cat_feature}: {actual_value} vs {ref_category}"
                )
                interpretation_found = True
                break

        # If not an encoded feature, use original name
        if not interpretation_found:
            interpretable_feature_names.append(feature)

    # Ensure we have the right number of feature names
    interpretable_feature_names = interpretable_feature_names[:X_sample.shape[1]]

    #
    # Summary Bar Plot
    #
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample,
                          feature_names=interpretable_feature_names,
                          plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Bar Plot)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_summary_bar.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("SHAP summary bar plot saved")
    except Exception as e:
        print(f"Could not generate bar plot: {e}")

    return shap_values

def save_comprehensive_results(feature_importance, business_interpretation,
                               category_info, model_name, shap_values,
                               output_dir='shap_results'):
    # Save comprehensive Shap results - enriched + concise and summary files
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving comprehensive SHAP results...")

    # Defensive copy and alignment
    comprehensive = business_interpretation.copy().reset_index(drop=True)

    # Ensure rank exists BEFORE saving enriched file
    # saving rank early - so saved files have a clear consistent feature order
    if 'rank' not in comprehensive.columns:
        comprehensive.insert(0, 'rank', range(1, len(comprehensive) + 1))

    # Save enriched full file (preserve all enrichment columns)
    enriched_path = f'{output_dir}/shap_feature_importance_enriched.csv'
    comprehensive.to_csv(enriched_path, index=False)

    # Create a concise "main" file for simpler consumption (choose columns present)
    # Concise file gives a clean summary for quick use; the full file is too detailed.
    main_cols = [
        'rank', 'feature', 'business_interpretation', 'feature_type',
        'impact_direction', 'mean_shap', 'shap_importance', 'shap_importance_normalized',
        'recommended_action'
    ]
    # keep only existing columns in that order
    main_cols = [c for c in main_cols if c in comprehensive.columns]
    concise_df = comprehensive[main_cols].copy()

    main_path = f'{output_dir}/shap_feature_importance.csv'
    concise_df.to_csv(main_path, index=False)

    # Build reference categories safely
    reference_data = []
    for cat_feature, info in (category_info or {}).items():
        reference_data.append({
            'categorical_feature': cat_feature,
            'reference_category': info.get('reference_category', ''),
            'all_categories': ', '.join(info.get('original_values', [])),
            'encoded_features_in_model': ', '.join(info.get('encoded_columns', []))
        })

    reference_df = pd.DataFrame(reference_data)
    reference_path = f'{output_dir}/reference_categories_info.csv'
    reference_df.to_csv(reference_path, index=False)

    # Build summary (safe shap values handling)
    summary = {
        'model_name': model_name,
        'total_features': len(feature_importance),
        'top_3_features': concise_df.head(3).to_dict('records'),
        'reference_categories': reference_data,
        'visualization_files': [
            'shap_summary_bar.png',
        ],
        'dependence_plots': []
    }

    # Compute shap_importance_vals
    try:
        if hasattr(shap_values, 'values'):
            shap_arr = np.array(shap_values.values)
        else:
            shap_arr = np.array(shap_values)

        # handle 3-d (classes) or 1-d single sample
        if shap_arr.ndim == 3:
            shap_arr = np.mean(shap_arr, axis=2)
        if shap_arr.ndim == 1:
            shap_arr = shap_arr.reshape(1, -1)

        shap_importance_vals = np.abs(shap_arr).mean(axis=0)
    except Exception:
        shap_importance_vals = None

    if shap_importance_vals is not None:
        n_top = min(4, len(shap_importance_vals))
        top_indices = np.argsort(shap_importance_vals)[-n_top:][::-1]
        for idx in top_indices:
            if idx < len(feature_importance):
                feature = feature_importance.iloc[idx]['feature']
                clean_name = str(feature).replace(' ', '_').replace('/', '_').replace(':', '_')
                summary['dependence_plots'].append(f'shap_dependence_{clean_name}.png')

    summary_path = f'{output_dir}/shap_analysis_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Enriched results saved to: {enriched_path}")
    print(f"Concise results saved to: {main_path}")
    print(f"Reference info saved to: {reference_path}")
    print(f"Summary saved to: {summary_path}")

    return comprehensive

def main():
    """Main SHAP analysis pipeline - COMPLETE WITH ALL VISUALIZATIONS"""
    print("=" * 60)
    print("SHAP ANALYSIS PIPELINE")
    print("=" * 60)
    print("Generating comprehensive results with ALL visualizations")
    print("=" * 60)

    try:
        # Step 1: Load models and data
        model, X_test, y_test, feature_names, model_name, category_info = load_models_and_data()

        # Use sample for faster computation
        sample_size = min(200, X_test.shape[0])
        X_sample = X_test[:sample_size]

        print(f"\nData shapes:")
        print(f"  X_sample shape: {X_sample.shape}")
        print(f"  Number of features: {len(feature_names)}")

        # Step 2: Create SHAP explainer
        explainer, shap_values = create_shap_explainer(model, model_name, X_sample)

        print(f"\nSHAP values shape: {shap_values.shape}")

        # Step 3: Calculate importance
        feature_importance = calculate_shap_importance(shap_values, feature_names)

        # Step 4: Generate ALL visualizations
        shap_vals = generate_shap_plots(explainer, shap_values, X_sample,
                                        feature_names, category_info)

        # Step 5: Create business interpretation
        business_interpretation = create_business_friendly_interpretation(
            feature_importance, category_info, shap_values=shap_vals, feature_names=feature_names
        )

        # Step 6: Save comprehensive results
        comprehensive_results = save_comprehensive_results(
            feature_importance, business_interpretation, category_info,
            model_name, shap_vals
        )

        # ============================================================
        # SHAP OUTPUT SUMMARY
        # ============================================================

        print("\n SHAP OUTPUT FILES")
        print("-" * 60)

        print("\n Data Files:")
        print("- shap_feature_importance_enriched.csv")
        print("- shap_feature_importance.csv")
        print("- reference_categories_info.csv")
        print("- shap_analysis_summary.json")

        print("\n Visualization Files:")
        print("- shap_summary_bar.png")

        # Reference categories
        if category_info:
            print(f"\nReference categories (dropped during one-hot encoding):")
            print("-" * 60)
            for cat_feature, info in category_info.items():
                ref = info.get('reference_category', '')
                enc = ', '.join(info.get('encoded_columns', []))
                print(f"{cat_feature}: '{ref}'")
                print(f"Model uses: {enc}")

        print(f"\n All files saved to: 'shap_results/' directory")
        print(" Ready for deployment and presentation")


    except Exception as e:
        print(f"\n SHAP ANALYSIS FAILED DURING SUMMARY OUTPUT: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from datetime import datetime

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# setting deterministic behaviour for ANN training
# to reduce run-to-run variability
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'   # best-effort deterministic ops
import random
random.seed(42)
np.random.seed(42)

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

#
# Set random seed to 42
# so we get the same result everytime this code runs
#
tf.random.set_seed(42)

#
# loading data
#
def load_prepared_data(data_dir='prepared_data'):

    # Loads prepared data
    # returns X_train, X_test, y_train, y_test (arrays) and metadata (dict)

    print("Loading prepared data...")

    # Check if files exist
    required_files = [
        f'{data_dir}/X_train_processed.csv',
        f'{data_dir}/X_test_processed.csv',
        f'{data_dir}/y_train_raw.csv',
        f'{data_dir}/y_test_raw.csv',
        f'{data_dir}/metadata.json'
    ]

    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file not found: {file}")

    # Load processed data
    X_train = pd.read_csv(f'{data_dir}/X_train_processed.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test_processed.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train_raw.csv').iloc[:, 0]
    y_test = pd.read_csv(f'{data_dir}/y_test_raw.csv').iloc[:, 0]

    # Load metadata
    with open(f'{data_dir}/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f" Training set: {X_train.shape}")
    print(f" Test set: {X_test.shape}")
    print(f" Features: {X_train.shape[1]}")
    print(f" Target distribution (train): {pd.Series(y_train).value_counts().to_dict()}")

    return X_train.values, X_test.values, y_train.values, y_test.values, metadata

# ANN - chosen for the Deep Learning model
# ANN - doesnt have built-in mechanism for class-imbalance handling
# must be handled externally
def create_ann_model(input_dim, learning_rate=0.001):

    # Creating and compiling a simple ANN model for binary classification

    model = Sequential([
        # Hidden Layer 1 - learn high level feature interactions
        Dense(128, activation='relu', input_dim=input_dim,
              kernel_initializer='he_normal'),
        BatchNormalization(), # stabilizes and speeds up training
        Dropout(0.3), # prevents overfitting

        # Hidden Layer 2 - deeper pattern extraction
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),

        # Hidden Layer 3 - compress features before output
        Dense(32, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.2),

        # Output Layer: sigmoid to return probability of churn
        Dense(1, activation='sigmoid')
    ])
    # configure optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # compile model for binary classification - loss + key metrics
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    return model

# using logistic regression model as the baseline model
def train_logistic_regression(X_train, y_train, X_test, y_test, feature_names):

    print("\n" + "=" * 40)
    print("TRAINING: Logistic Regression")
    print("=" * 40)

    # Initialize model with balanced class weights
    # to handle class-imbalance
    # liblinear - optimization algo (reliable for small datasets)
    # c=1.0 - normal amount of regularization to keep model stable
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='liblinear',
        C=1.0
    )

    # Train on 100% of training data
    print(f"   Training on {len(X_train)} samples (100% training data)...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Logistic Regression has its built-in mechanism for feature importance
    # During training - it learns coefficients(model.coef) for each feature
    if len(model.coef_.shape) == 2 and model.coef_.shape[0] == 1:
        # Binary classification
        coefficients = np.abs(model.coef_[0])
    elif len(model.coef_.shape) == 2:
        # Multi-class, take average across classes
        coefficients = np.abs(model.coef_).mean(axis=0)
    else:
        coefficients = np.abs(model.coef_)

    feature_importance = pd.DataFrame({
        'feature': feature_names[:len(coefficients)],
        'importance': coefficients,
        'importance_type': 'absolute_coefficient'
    }).sort_values('importance', ascending=False)

    print(f"    Model trained")
    print(f"    Feature importance calculated")

    return model, y_pred, y_pred_proba, feature_importance

# using Random Forest (ensemble) model for benchmark comparison
def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    print("\n" + "=" * 40)
    print("TRAINING: Random Forest")
    print("=" * 40)

    # Random Forest - in-built mechanism for handling class-imbalance
    # it gives more weight to minority class
    # so it doesnt ignore rare cases
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        max_features='sqrt'
    )

    # Train on 100% of training data
    print(f"   Training on {len(X_train)} samples (100% training data)...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Random Forest - also built-in mechanism for feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_,
        'importance_type': 'gini_importance'
    }).sort_values('importance', ascending=False)

    print(f"    Model trained with {model.n_estimators} trees")
    print(f"    Feature importance calculated")

    return model, y_pred, y_pred_proba, feature_importance

# uses gradient boosting as another model for benchmarking comparison
# Gradient Boosting has a built-in mechanism for feature importance
# but none for handling class imbalance; must be handled externally
def train_gradient_boosting(X_train, y_train, X_test, y_test, feature_names):
    print("\n" + "=" * 40)
    print("TRAINING: Gradient Boosting")
    print("=" * 40)

    # Initialize model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        subsample=0.8,
        max_features='sqrt'
    )

    # Train on 100% of training data
    print(f"   Training on {len(X_train)} samples (100% training data)...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Feature importance (split importance)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_,
        'importance_type': 'split_importance'
    }).sort_values('importance', ascending=False)

    print(f"    Model trained with {model.n_estimators} trees")
    print(f"    Feature importance calculated")

    return model, y_pred, y_pred_proba, feature_importance

# Using ANN for the deep learning model - benchmark comaprison
def train_ann(X_train, y_train, X_test, y_test, feature_names):
    print("\n" + "=" * 40)
    print("TRAINING: Artificial Neural Network")
    print("=" * 40)

    # Create model
    model = create_ann_model(X_train.shape[1])

    # Train on 100% of training data (NO validation split)
    print(f"   Training on {len(X_train)} samples (100% training data)...")
    print(f"   NO validation split - Using fixed epochs for fair comparison")

    # using epochs = 50 - a conservative number to avoid overfitting
    epochs = 50  # Conservative number to avoid overfitting

    history = model.fit(
        X_train, y_train,
        epochs=epochs,  # Fixed epochs, no early stopping
        batch_size=32,
        verbose=0
    )

    # Training summary
    n_epochs = len(history.history['loss'])
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]

    print(f"    Training completed in {n_epochs} epochs")
    print(f"    Final training loss: {final_loss:.4f}")
    print(f"    Final training accuracy: {final_accuracy:.4f}")

    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # ANN has no built-in feature importance
    # so permutation importance is used to estimate how much each feature affects predictions
    # this is an external method applied after training
    print("   Calculating permutation importance...")

    # Use lambda function for prediction
    predictor = lambda X: model.predict(X, verbose=0).flatten()

    # Calculate permutation importance on a reasonable subset
    sample_size = min(200, len(X_test))
    try:
        perm_result = permutation_importance(
            predictor,
            X_test[:sample_size],
            y_test[:sample_size],
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
            scoring='neg_mean_squared_error'
        )

        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_result.importances_mean,
            'importance_std': perm_result.importances_std,
            'importance_type': 'permutation_importance'
        }).sort_values('importance', ascending=False)

        print(f"    Permutation importance calculated on {sample_size} samples")

    except Exception as e:
        print(f"   Could not calculate permutation importance: {e}")
        print("   Using uniform importance as placeholder")

        # Create placeholder importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.ones(len(feature_names)) / len(feature_names),
            'importance_std': np.zeros(len(feature_names)),
            'importance_type': 'uniform_placeholder'
        }).sort_values('importance', ascending=False)

    return model, y_pred, y_pred_proba, feature_importance, history

# calculating metrics - for Accuracy, Precision, Recall, F1-score, ROC-AUC
# and Confusion Metrics
def calculate_metrics(y_true, y_pred, y_pred_proba, model_name):

    # Basic metrics
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_pred_proba)) > 1 else 0.5,
        'n_samples': len(y_true)
    }

    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        })

    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    return metrics, report

# evaluating all models - to get to the best performing one on F1-score
def evaluate_and_select_best_model(all_results):
    print("\n" + "=" * 60)
    print("MODEL EVALUATION & COMPARISON")
    print("=" * 60)
    print("All models trained on 100% of training data for fair comparison")

    # Find best model based on F1-score (primary metric)
    best_model_name = None
    best_f1 = 0
    best_metrics = None

    # Also track other metrics for comparison
    all_metrics = {}

    print("\n Model Performance Summary (Test Set):")
    print("-" * 85)
    header = f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}"
    print(header)
    print("-" * 85)

    for name, results in all_results.items():
        metrics = results['metrics']
        all_metrics[name] = metrics

        # Print performance row
        print(f"{name:<25} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['f1']:<10.4f} "
              f"{metrics['roc_auc']:<10.4f}")

        # Update best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_name = name
            best_metrics = metrics

    print("-" * 85)

    # Show best model details
    print(f"\n BEST MODEL: {best_model_name}")
    print(f"   Primary Metric (F1-Score): {best_f1:.4f}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall: {best_metrics['recall']:.4f}")
    print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")

    # Show runner-ups
    print(f"\n Runner-up Models:")
    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_models[1:4], 2):
        print(f"   {i}. {name}: F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")

    return best_model_name, all_results[best_model_name]

# saving models and results
def save_models_and_results(all_results, best_model_name, best_model_results,
                            X_test, y_test, output_dir='models'):
    print(f"\n Saving models and results to '{output_dir}/'...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Track model info for summary
    model_info = {}

    print("\n Model Details:")
    print("-" * 80)

    for name, results in all_results.items():
        # Create model-specific directory
        model_dir = f'{output_dir}/{name.replace(" ", "_").lower()}'
        os.makedirs(model_dir, exist_ok=True)

        print(f"\n{name}:")

        # Special handling for ANN models because it must be saved using Keras native format (h5)
        # unlike the rest of the scikit-learn models
        if name == 'Artificial Neural Network':
            # Save Keras model
            model_path = f'{model_dir}/model.h5'
            results['model'].save(model_path)
            print(f"   Model saved: {model_path}")

            # Save training history - ANN learns over many epochs
            # Useful for checking training behaviour and overfitting
            history_path = f'{model_dir}/training_history.joblib'
            joblib.dump(results['history'], history_path)
            print(f"   Training history saved")

        else:
            # Save scikit-learn model
            model_path = f'{model_dir}/model.joblib'
            joblib.dump(results['model'], model_path)
            print(f"   Model saved: {model_path}")

        # Save metrics
        metrics_path = f'{model_dir}/metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results['metrics'], f, indent=2)
        print(f"   Metrics saved")

        # Save feature importance
        importance_path = f'{model_dir}/feature_importance.csv'
        results['feature_importance'].to_csv(importance_path, index=False)
        print(f"   Feature importance saved")

        # Save classification report
        report_path = f'{model_dir}/classification_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results['report'], f, indent=2)
        print(f"   Classification report saved")

        # Display top features
        top_features = results['feature_importance'].head(3)
        print(f"  Top 3 features:")
        for _, row in top_features.iterrows():
            print(f"    - {row['feature']}: {row['importance']:.4f}")

        # Store in model info
        model_info[name] = {
            'metrics': results['metrics'],
            'feature_importance_top5': results['feature_importance'].head(5).to_dict('records'),
            'model_path': model_path
        }

    # Save consolidated results
    print(f"\n Saving consolidated results...")

    # Save all model results
    all_results_path = f'{output_dir}/all_model_results.json'
    with open(all_results_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2)
    print(f"   All model results saved: {all_results_path}")

    # Save best model info
    best_model_info = {
        'best_model_name': best_model_name,
        'best_model_metrics': best_model_results['metrics'],
        'best_model_top_features': best_model_results['feature_importance'].head(10).to_dict('records'),
        'selection_criteria': 'highest_f1_score',
        'training_note': 'All models trained on 100% of training data for fair comparison',
        'ann_training_note': 'ANN trained with fixed epochs (50), no validation split',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'recommendation': 'For consistent cross-model feature importance comparison, use SHAP analysis'
    }

    best_info_path = f'{output_dir}/best_model_info.json'
    with open(best_info_path, 'w', encoding='utf-8') as f:
        json.dump(best_model_info, f, indent=2)
    print(f"   Best model info saved: {best_info_path}")

    # Save best model predictions
    best_predictions = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': best_model_results['y_pred'],
        'prediction_probability': best_model_results['y_pred_proba']
    })
    predictions_path = f'{output_dir}/best_model_predictions.csv'
    best_predictions.to_csv(predictions_path, index=False)
    print(f"   Best model predictions saved: {predictions_path}")

    # Save confusion matrix for best model
    cm = confusion_matrix(y_test, best_model_results['y_pred'])
    cm_df = pd.DataFrame(cm,
                         index=['Actual Negative', 'Actual Positive'],
                         columns=['Predicted Negative', 'Predicted Positive'])
    cm_path = f'{output_dir}/best_model_confusion_matrix.csv'
    cm_df.to_csv(cm_path)
    print(f"   Confusion matrix saved: {cm_path}")

    print(f"\n All files saved successfully!")

    return model_info, best_model_info


def generate_training_summary(model_info, best_model_info, output_dir='models'):
    """
    Generate a human-readable training summary
    """
    summary_path = f'{output_dir}/training_summary.txt'

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best Model: {best_model_info['best_model_name']}\n")
        f.write(f"Selection Criteria: {best_model_info['selection_criteria']}\n")
        f.write(f"Training Note: {best_model_info['training_note']}\n")
        f.write(f"ANN Training: {best_model_info['ann_training_note']}\n\n")

        f.write("-" * 70 + "\n")
        f.write("FAIR COMPARISON NOTES:\n")
        f.write("-" * 70 + "\n")
        f.write(" All models trained on 100% of training data\n")
        f.write(" NO validation split for ANN (fixed 50 epochs)\n")
        f.write(" Same train/test split for all models\n")
        f.write(" Same evaluation metrics for all models\n\n")

        f.write("-" * 70 + "\n")
        f.write("PERFORMANCE COMPARISON\n")
        f.write("-" * 70 + "\n\n")

        # Create performance table
        f.write(f"{'Model':<25} {'Accuracy':<10} {'F1-Score':<10} {'ROC-AUC':<10}\n")
        f.write("-" * 55 + "\n")

        for name, info in sorted(model_info.items(),
                                 key=lambda x: x[1]['metrics']['f1'],
                                 reverse=True):
            metrics = info['metrics']
            f.write(f"{name:<25} {metrics['accuracy']:<10.4f} "
                    f"{metrics['f1']:<10.4f} {metrics['roc_auc']:<10.4f}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write(f"BEST MODEL DETAILS: {best_model_info['best_model_name']}\n")
        f.write("-" * 70 + "\n\n")

        metrics = best_model_info['best_model_metrics']
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")

        if 'true_positives' in metrics:
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"  True Positives:  {metrics['true_positives']}\n")
            f.write(f"  True Negatives:  {metrics['true_negatives']}\n")
            f.write(f"  False Positives: {metrics['false_positives']}\n")
            f.write(f"  False Negatives: {metrics['false_negatives']}\n")

        f.write(f"\nTop 5 Features:\n")
        for i, feat in enumerate(best_model_info['best_model_top_features'][:5], 1):
            f.write(f"  {i}. {feat['feature']}: {feat['importance']:.4f}\n")

    print(f"Training summary saved: {summary_path}")


def main():
    try:
        # Step 1: Load prepared data
        print("\n STEP 1: Loading prepared data...")
        X_train, X_test, y_train, y_test, metadata = load_prepared_data()
        feature_names = metadata['feature_names']

        # Step 2: Train all models (ALL using 100% training data)
        print("\n STEP 2: Training models...")
        print("   All models using 100% of training data for fair comparison")
        all_results = {}

        # 2.1 Logistic Regression (100% training data)
        lr_model, lr_pred, lr_proba, lr_importance = train_logistic_regression(
            X_train, y_train, X_test, y_test, feature_names
        )
        lr_metrics, lr_report = calculate_metrics(y_test, lr_pred, lr_proba, 'Logistic Regression')

        all_results['Logistic Regression'] = {
            'model': lr_model,
            'y_pred': lr_pred,
            'y_pred_proba': lr_proba,
            'feature_importance': lr_importance,
            'metrics': lr_metrics,
            'report': lr_report
        }

        # 2.2 Random Forest (100% training data)
        rf_model, rf_pred, rf_proba, rf_importance = train_random_forest(
            X_train, y_train, X_test, y_test, feature_names
        )
        rf_metrics, rf_report = calculate_metrics(y_test, rf_pred, rf_proba, 'Random Forest')

        all_results['Random Forest'] = {
            'model': rf_model,
            'y_pred': rf_pred,
            'y_pred_proba': rf_proba,
            'feature_importance': rf_importance,
            'metrics': rf_metrics,
            'report': rf_report
        }

        # 2.3 Gradient Boosting (100% training data)
        gb_model, gb_pred, gb_proba, gb_importance = train_gradient_boosting(
            X_train, y_train, X_test, y_test, feature_names
        )
        gb_metrics, gb_report = calculate_metrics(y_test, gb_pred, gb_proba, 'Gradient Boosting')

        all_results['Gradient Boosting'] = {
            'model': gb_model,
            'y_pred': gb_pred,
            'y_pred_proba': gb_proba,
            'feature_importance': gb_importance,
            'metrics': gb_metrics,
            'report': gb_report
        }

        # 2.4 Artificial Neural Network (100% training data, NO validation split)
        ann_model, ann_pred, ann_proba, ann_importance, ann_history = train_ann(
            X_train, y_train, X_test, y_test, feature_names
        )
        ann_metrics, ann_report = calculate_metrics(y_test, ann_pred, ann_proba, 'Artificial Neural Network')

        all_results['Artificial Neural Network'] = {
            'model': ann_model,
            'history': ann_history,
            'y_pred': ann_pred,
            'y_pred_proba': ann_proba,
            'feature_importance': ann_importance,
            'metrics': ann_metrics,
            'report': ann_report
        }

        # Step 3: Evaluate and select best model
        print("\n STEP 3: Evaluating models...")
        best_model_name, best_model_results = evaluate_and_select_best_model(all_results)

        # Step 4: Save models and results
        print("\n STEP 4: Saving results...")
        model_info, best_model_info = save_models_and_results(
            all_results, best_model_name, best_model_results, X_test, y_test
        )

        # Step 5: Generate summary
        generate_training_summary(model_info, best_model_info)

        return True

    except Exception as e:
        print(f"\n MODEL TRAINING FAILED: {str(e)}")
        print("\nDebugging info:")
        import traceback
        traceback.print_exc()

        # Try to save error log
        try:
            error_log = f"model_training_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(error_log, 'w', encoding='utf-8') as f:
                f.write(f"Error: {str(e)}\n\n")
                traceback.print_exc(file=f)
            print(f"\nError log saved to: {error_log}")
        except:
            pass

        return False

if __name__ == "__main__":
    # Start the training pipeline
    success = main()

    # Exit with appropriate code
    import sys

    sys.exit(0 if success else 1)
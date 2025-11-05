import os
import sys
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlparse

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.main_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object, load_object, load_numpy_array_data, evaluate_models
)
from networksecurity.utils.main_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

import mlflow

# ========== Visualization Utilities ==========
from networksecurity.utils.main_utils.visualization_utils import (
    plot_correlation_matrix,
    plot_feature_importance,
    plot_decision_tree
)



class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        


    def track_mlflow(self, best_model, classification_metric, run_name="Default_Run"):
        """Logs metrics locally using MLflow (safely handles active runs)."""
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run()

            with mlflow.start_run(run_name=run_name):
                mlflow.log_metric("f1_score", classification_metric.f1_score)
                mlflow.log_metric("precision", classification_metric.precision_score)
                mlflow.log_metric("recall", classification_metric.recall_score)
        except Exception as e:
            logging.warning(f"⚠️ MLflow tracking failed: {e}")
            if mlflow.active_run() is not None:
                mlflow.end_run()



    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1, max_iter=1000),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {'criterion': ['gini', 'entropy', 'log_loss']},
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.75, 0.9],
                    'n_estimators': [16, 32, 64, 128, 256]
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    'learning_rate': [.1, .01, .001],
                    'n_estimators': [16, 32, 64, 128, 256]
                }
            }


            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            os.makedirs("Artifacts", exist_ok=True)
            os.makedirs("static/plots", exist_ok=True)


            try:
                if model_report and isinstance(model_report, dict):
                    sorted_models = dict(sorted(model_report.items(), key=lambda x: x[1], reverse=True))
                    plt.figure(figsize=(8, 5))
                    plt.barh(list(sorted_models.keys()), list(sorted_models.values()), color="#007bff")
                    plt.xlabel("F1 Score")
                    plt.ylabel("Model")
                    plt.title("Model Comparison - Based on F1 Score")
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig("Artifacts/model_comparison.png", bbox_inches="tight", dpi=200)
                    plt.close()


                    shutil.copy("Artifacts/model_comparison.png", "static/plots/model_comparison.png")

                    if mlflow.active_run() is None:
                        mlflow.start_run(run_name="model_comparison")
                    mlflow.log_artifact("Artifacts/model_comparison.png")
                    mlflow.end_run()
                else:
                    logging.warning(" Skipped model comparison plot — no valid report.")
            except Exception as e:
                logging.warning(f" Could not plot model comparison: {e}")


            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f" Best Model Selected: {best_model_name} (F1 Score: {best_model_score})")


            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            self.track_mlflow(best_model, classification_train_metric, run_name=f"{best_model_name}_train")

            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            self.track_mlflow(best_model, classification_test_metric, run_name=f"{best_model_name}_test")


            try:
                feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

                # Feature importance
                if hasattr(best_model, "feature_importances_"):
                    plot_feature_importance(best_model, feature_names, "Artifacts/feature_importance.png")
                    shutil.copy("Artifacts/feature_importance.png", "static/plots/feature_importance.png")
                    mlflow.log_artifact("Artifacts/feature_importance.png")

                # Decision Tree Visualization
                if isinstance(best_model, DecisionTreeClassifier):
                    plot_decision_tree(best_model, feature_names, "Artifacts/decision_tree.png")
                    shutil.copy("Artifacts/decision_tree.png", "static/plots/decision_tree.png")
                    mlflow.log_artifact("Artifacts/decision_tree.png")

                elif isinstance(best_model, RandomForestClassifier):
                    estimator = best_model.estimators_[0]
                    plt.figure(figsize=(12, 8))
                    plot_tree(estimator, feature_names=feature_names, filled=True, rounded=True, fontsize=6)
                    plt.title("Sample Tree from Random Forest")
                    plt.savefig("Artifacts/decision_tree.png", bbox_inches='tight', dpi=200)
                    plt.close()
                    shutil.copy("Artifacts/decision_tree.png", "static/plots/decision_tree.png")
                    mlflow.log_artifact("Artifacts/decision_tree.png")

            except Exception as e:
                logging.warning(f" Could not plot feature importance or decision tree: {e}")


            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/model.pkl", best_model)


            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )

            logging.info(f" Model trainer artifact created: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Full training pipeline — loads data, trains model, saves plots and artifacts."""
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )


            try:
                raw_data = pd.read_csv("network_data/phisingData.csv")
                os.makedirs("static/plots", exist_ok=True)
                plot_correlation_matrix(raw_data, "Artifacts/correlation_heatmap.png")
                shutil.copy("Artifacts/correlation_heatmap.png", "static/plots/correlation_heatmap.png")
                mlflow.log_artifact("Artifacts/correlation_heatmap.png")
            except Exception as e:
                logging.warning(f" Correlation heatmap skipped: {e}")


            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

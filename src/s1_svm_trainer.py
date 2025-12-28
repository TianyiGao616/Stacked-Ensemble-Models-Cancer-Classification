import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score

class SVMTrainer:
    def __init__(self, csv_path, kernel, balanced=True, use_stratified_kfold=False):
        self.csv_path = csv_path
        self.kernel = kernel
        self.balanced = balanced
        self.use_stratified_kfold = use_stratified_kfold
        self.model = None
        self.label_encoder = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_and_preprocess_data(self):
        print("Loading dataset...")
        cancer = pd.read_csv(self.csv_path)
        print(f"Dataset shape: {cancer.shape}")
        print(f"Class distribution:\n{cancer['type'].value_counts()}")

        X = cancer.drop(columns=['type', 'samples'], axis=1)
        y = cancer['type']

        self.feature_names = X.columns.tolist()

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.3, stratify=y, random_state=42
        )

        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Number of features: {len(self.feature_names)}")

    def train_svm(self):
        print(f"Training SVM model with kernel = '{self.kernel}', balanced = {self.balanced}, stratified_kfold = {self.use_stratified_kfold}...")

        class_weight = 'balanced' if self.balanced else None
        self.model = SVC(kernel=self.kernel, class_weight=class_weight, random_state=42, probability=True)

        if self.use_stratified_kfold:
            print("Performing Stratified K-Fold cross-validation on training set...")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            acc_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=skf, scoring='accuracy')
            f1_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=skf, scoring='f1_weighted')

            print(f"Cross-validated Accuracy: {acc_scores.mean():.4f}")
            print(f"Cross-validated F1 Score: {f1_scores.mean():.4f}")

        self.model.fit(self.X_train, self.y_train)

        svm_train_pred = self.model.predict(self.X_train)
        svm_test_pred = self.model.predict(self.X_test)

        test_accuracy = accuracy_score(self.y_test, svm_test_pred)
        test_f1 = f1_score(self.y_test, svm_test_pred, average='weighted')

        print(f"SVM Test Accuracy: {test_accuracy:.4f}")
        print(f"SVM Test F1-Score: {test_f1:.4f}")
        print("\nClassification Report:")
        target_names = self.label_encoder.classes_
        print(classification_report(self.y_test, svm_test_pred, target_names=target_names))

        return svm_train_pred, svm_test_pred

    def save_svm_outputs(self, svm_train_pred, svm_test_pred):
        print("Saving SVM outputs...")

        os.makedirs('svm_outputs', exist_ok=True)

        X_train_meta = np.column_stack((self.X_train.values, svm_train_pred))
        X_test_meta = np.column_stack((self.X_test.values, svm_test_pred))

        np.save('svm_outputs/X_train_meta.npy', X_train_meta)
        np.save('svm_outputs/X_test_meta.npy', X_test_meta)

        meta_feature_names = self.feature_names + ['svm_prediction']
        with open('svm_outputs/feature_names.pkl', 'wb') as f:
            pickle.dump(meta_feature_names, f)

        print("SVM outputs saved successfully!")
        print(f"Meta training features shape: {X_train_meta.shape}")
        print(f"Meta test features shape: {X_test_meta.shape}")

        return X_train_meta, X_test_meta

    def run_complete_pipeline(self):
        print("=" * 50)
        print("Starting SVM Training Pipeline")
        print("=" * 50)

        self.load_and_preprocess_data()
        svm_train_pred, svm_test_pred = self.train_svm()
        X_train_meta, X_test_meta = self.save_svm_outputs(svm_train_pred, svm_test_pred)

        print("=" * 50)
        print("SVM Training Pipeline Completed Successfully!")
        print("=" * 50)

        return {
            'model': self.model,
            'X_train_meta': X_train_meta,
            'X_test_meta': X_test_meta,
            'feature_names': self.feature_names + ['svm_prediction']
        }

def main():
    csv_path = 'datasets/Renal_GSE53757.csv'
    kernel_type = 'rbf' # 'linear' or 'rbf'
    balanced = True # Set to False if data is already balanced
    use_stratified_kfold = True # Set to True to perform Stratified K-Fold CV

    trainer = SVMTrainer(csv_path, kernel_type, balanced, use_stratified_kfold)
    results = trainer.run_complete_pipeline()

    print("\nFiles created:")
    print("- svm_outputs/X_train_meta.npy")
    print("- svm_outputs/X_test_meta.npy")

if __name__ == "__main__":
    main()

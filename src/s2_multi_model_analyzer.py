import os
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report

class MultiModelAnalyzer:
    def __init__(self, svm_outputs_dir='svm_outputs'):
        self.svm_outputs_dir = svm_outputs_dir
        self.models = {}
        self.feature_names = None
        self.X_train_meta = None
        self.X_test_meta = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        
        # Define model configurations
        self.model_configs = {
            'decision_tree': {
                'model': DecisionTreeClassifier(criterion='gini', max_depth=20, min_samples_leaf=2, random_state=42),
                'name': 'Decision Tree'
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    class_weight="balanced",
                    random_state=42
                ),
                'name': 'Random Forest'
            },
            'logistic_regression': {
                'model': LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                ),
                'name': 'Logistic Regression'
            }
        }

    def load_svm_outputs(self):
        print("Loading SVM outputs...")

        meta_train = np.load(f'{self.svm_outputs_dir}/X_train_meta.npy')
        meta_test = np.load(f'{self.svm_outputs_dir}/X_test_meta.npy')

        self.X_train_meta = meta_train[:, :-1]
        self.y_train = meta_train[:, -1].astype(int)
        self.X_test_meta = meta_test[:, :-1]
        self.y_test = meta_test[:, -1].astype(int)

        # Load original feature names from pickle
        with open(f'{self.svm_outputs_dir}/feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)[:-1]  # Exclude svm_prediction column

        print(f"Loaded training features shape: {self.X_train_meta.shape}")
        print(f"Loaded test features shape: {self.X_test_meta.shape}")
        print(f"Total features: {len(self.feature_names)}")

    def optimize_feature_selection_for_model(self, model_key, model_config):
        """Only used for Decision Tree - uses RFE to find optimal number of features"""
        print(f"Optimizing feature selection for {model_config['name']}...")

        n_feature_list = [25, 50, 75, 100, 125, 150, 175, 200]
        # Filter out values larger than available features
        n_feature_list = [k for k in n_feature_list if k <= len(self.feature_names)]
        
        f1_scores = []
        cv_scores = []

        for k in n_feature_list:
            print(f"  Testing {k} features...")
            model = model_config['model']
            rfe = RFE(estimator=model, n_features_to_select=k, step=0.1)
            rfe.fit(self.X_train_meta, self.y_train)

            sel_X_train = rfe.transform(self.X_train_meta)
            sel_X_test = rfe.transform(self.X_test_meta)

            model.fit(sel_X_train, self.y_train)
            preds = model.predict(sel_X_test)

            f1 = f1_score(self.y_test, preds, average='weighted')
            f1_scores.append(f1)

            cv = cross_val_score(model, sel_X_train, self.y_train, cv=5, scoring='f1_weighted')
            cv_scores.append(cv.mean())

        if f1_scores:
            best_index = f1_scores.index(max(f1_scores))
            best_k = n_feature_list[best_index]
            print(f"  Best F1 score: {f1_scores[best_index]:.4f} with {best_k} features")
            return best_k
        else:
            return min(50, len(self.feature_names))

    def train_decision_tree_with_rfe(self, n_features):
        """Train Decision Tree with RFE feature selection"""
        print(f"Training Decision Tree with RFE using {n_features} features...")
        
        model = DecisionTreeClassifier(criterion='gini', max_depth=20, min_samples_leaf=2, random_state=42)
        rfe = RFE(estimator=model, n_features_to_select=n_features, step=50)
        rfe.fit(self.X_train_meta, self.y_train)

        selected_mask = rfe.get_support()
        selected_indices = np.where(selected_mask)[0]
        selected_features = [self.feature_names[i] for i in selected_indices]

        sel_X_train = rfe.transform(self.X_train_meta)
        sel_X_test = rfe.transform(self.X_test_meta)

        model.fit(sel_X_train, self.y_train)

        preds = model.predict(sel_X_test)
        acc = accuracy_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds, average='weighted')
        
        print(f"  Decision Tree Accuracy: {acc:.4f}")
        print(f"  Decision Tree F1-Score: {f1:.4f}")

        cv = cross_val_score(model, sel_X_train, self.y_train, cv=5, scoring='f1_weighted')
        print(f"  Cross-validation F1-Score: {cv.mean():.4f} (+/- {cv.std() * 2:.4f})")
        
        return model, selected_features

    def train_random_forest(self):
        """Train Random Forest with RFE, then keep top 10 important features"""
        print("Training Random Forest...")
        
        # Step 1: Use RFE to find optimal number of features
        best_k = self.optimize_feature_selection_for_model('random_forest', self.model_configs['random_forest'])
        
        # Step 2: Apply RFE with optimal number of features
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42
        )
        
        rfe = RFE(estimator=rf_model, n_features_to_select=best_k, step=50)
        rfe.fit(self.X_train_meta, self.y_train)
        
        # Get RFE-selected features
        selected_mask = rfe.get_support()
        selected_indices = np.where(selected_mask)[0]
        rfe_selected_features = [self.feature_names[i] for i in selected_indices]
        
        # Step 3: Train final model on RFE-selected features
        sel_X_train = rfe.transform(self.X_train_meta)
        sel_X_test = rfe.transform(self.X_test_meta)
        
        # Create fresh model instance for final training
        final_rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42
        )
        
        final_rf_model.fit(sel_X_train, self.y_train)
        rf_pred = final_rf_model.predict(sel_X_test)
        
        print("  f1_score:", f1_score(self.y_test, rf_pred))
        
        cv_scores_rf = cross_val_score(final_rf_model, sel_X_train, self.y_train, cv=5)
        print("  Cross val scores:", cv_scores_rf)
        print("  Mean cross val score:", cv_scores_rf.mean())
        
        # Step 4: From RFE-selected features, keep only top 10 by importance
        importances = final_rf_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature_Name': rfe_selected_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Keep only top 10 features
        top_10_features = importance_df.head(10)['Feature_Name'].tolist()
        top_10_importances = importance_df.head(10)['Importance'].tolist()
        
        print(f"  RFE selected {best_k} features, keeping top 10 by importance")
        
        return final_rf_model, top_10_features, top_10_importances

    def train_logistic_regression(self):
        """Train Logistic Regression with RFE, then keep top 10 important features"""
        print("Training Logistic Regression...")
        
        # Step 1: Use RFE to find optimal number of features
        best_k = self.optimize_feature_selection_for_model('logistic_regression', self.model_configs['logistic_regression'])
        
        # Step 2: Apply RFE with optimal number of features
        lr_model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
        
        rfe = RFE(estimator=lr_model, n_features_to_select=best_k, step=50)
        rfe.fit(self.X_train_meta, self.y_train)
        
        # Get RFE-selected features
        selected_mask = rfe.get_support()
        selected_indices = np.where(selected_mask)[0]
        rfe_selected_features = [self.feature_names[i] for i in selected_indices]
        
        # Step 3: Train final model on RFE-selected features
        sel_X_train = rfe.transform(self.X_train_meta)
        sel_X_test = rfe.transform(self.X_test_meta)
        
        # Create fresh model instance for final training
        final_lr_model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
        
        final_lr_model.fit(sel_X_train, self.y_train)
        lr_pred = final_lr_model.predict(sel_X_test)
        
        print("  f1_score:", f1_score(self.y_test, lr_pred))
        
        cv_scores_lr = cross_val_score(final_lr_model, sel_X_train, self.y_train, cv=5)
        print("  Cross val scores:", cv_scores_lr)
        print("  Mean cross val score:", cv_scores_lr.mean())
        
        # Step 4: From RFE-selected features, keep only top 10 by importance
        coefficients = np.abs(final_lr_model.coef_[0]) if len(final_lr_model.coef_.shape) > 1 else np.abs(final_lr_model.coef_)
        importance_df = pd.DataFrame({
            'Feature_Name': rfe_selected_features,
            'Importance': coefficients
        }).sort_values(by='Importance', ascending=False)
        
        # Keep only top 10 features
        top_10_features = importance_df.head(10)['Feature_Name'].tolist()
        top_10_importances = importance_df.head(10)['Importance'].tolist()
        
        print(f"  RFE selected {best_k} features, keeping top 10 by importance")
        
        return final_lr_model, top_10_features, top_10_importances

    def extract_feature_importance(self, model_key, model, selected_features):
        print(f"Extracting feature importance for {model_key}...")
        
        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For logistic regression, use absolute values of coefficients
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            print(f"Warning: Cannot extract feature importance for {model_key}")
            return None
        
        # For Random Forest and Logistic Regression, selected_features already contains top 10
        # The importances correspond to these selected features
        if len(selected_features) != len(importances):
            print(f"Warning: Feature count mismatch for {model_key}")
            print(f"  Selected features: {len(selected_features)}")
            print(f"  Importance scores: {len(importances)}")
            return None
            
        protein_importance = pd.DataFrame({
            'Feature_Name': selected_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        protein_importance['Rank'] = range(1, len(protein_importance) + 1)

        print(f"  Total features analyzed: {len(protein_importance)}")
        print(f"  Features with importance > 0: {len(protein_importance[protein_importance['Importance'] > 0])}")
        print(f"  Top 5 most important features:")
        print(protein_importance.head(5)[['Rank', 'Feature_Name', 'Importance']])
        
        return protein_importance

    def save_model_results(self, model_key, protein_importance, model, selected_features):
        print(f"Saving {model_key} results...")
        
        output_dir = f'protein_outputs_{model_key}'
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed importance
        protein_importance.to_csv(f'{output_dir}/protein_importance_detailed.csv', index=False)
        
        # Save only important proteins (importance > 0)
        important = protein_importance[protein_importance['Importance'] > 0].copy()
        important.to_csv(f'{output_dir}/important_proteins.csv', index=False)

        # Save protein list
        with open(f'{output_dir}/protein_list.txt', 'w') as f:
            for p in important['Feature_Name']:
                f.write(f"{p}\n")

        # Save model-specific information
        if model_key == 'decision_tree':
            rules = export_text(model, feature_names=selected_features)
            with open(f'{output_dir}/decision_tree_rules.txt', 'w') as f:
                f.write(rules)
        elif model_key == 'random_forest':
            with open(f'{output_dir}/random_forest_info.txt', 'w') as f:
                f.write(f"Random Forest Model Information\n")
                f.write(f"Number of trees: {model.n_estimators}\n")
                f.write(f"Max depth: {model.max_depth}\n")
                f.write(f"Class weight: {model.class_weight}\n")
                f.write(f"Top 10 features selected\n")
        elif model_key == 'logistic_regression':
            with open(f'{output_dir}/logistic_regression_info.txt', 'w') as f:
                f.write(f"Logistic Regression Model Information\n")
                f.write(f"Penalty: {model.penalty}\n")
                f.write(f"Solver: {model.solver}\n")
                f.write(f"Class weight: {model.class_weight}\n")
                f.write(f"Top 10 features selected\n")

        # Save analysis summary
        summary = {
            'model_type': model_key,
            'total_proteins_analyzed': len(protein_importance),
            'important_proteins': len(important),
            'top_protein': important.iloc[0]['Feature_Name'] if len(important) > 0 else 'None',
            'max_importance': important.iloc[0]['Importance'] if len(important) > 0 else 0
        }
        
        with open(f'{output_dir}/analysis_summary.txt', 'w') as f:
            f.write(f"Protein Importance Analysis Summary - {model_key}\n")
            f.write("=" * 50 + "\n")
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")
                
        print(f"  Results saved to {output_dir}/")
        print(f"  Important proteins (importance > 0): {len(important)}")
        
        return important

    def analyze_single_model(self, model_key, model_config):
        """Analyze a single model type"""
        print(f"\n{'='*50}")
        print(f"Analyzing {model_config['name'].upper()}")
        print(f"{'='*50}")
        
        try:
            if model_key == 'decision_tree':
                # Decision Tree: Use RFE for feature selection, keep all selected features
                best_k = self.optimize_feature_selection_for_model(model_key, model_config)
                final_model, selected_features = self.train_decision_tree_with_rfe(best_k)
                
            elif model_key == 'random_forest':
                # Random Forest: Use RFE for feature selection, then keep top 10 by importance
                final_model, selected_features, selected_importances = self.train_random_forest()
                
            elif model_key == 'logistic_regression':
                # Logistic Regression: Use RFE for feature selection, then keep top 10 by importance
                final_model, selected_features, selected_importances = self.train_logistic_regression()
            
            # Extract feature importance
            if model_key == 'decision_tree':
                protein_importance = self.extract_feature_importance(model_key, final_model, selected_features)
            else:
                # For RF and LR, we already have the top 10 features and their importances
                protein_importance = pd.DataFrame({
                    'Feature_Name': selected_features,
                    'Importance': selected_importances
                }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
                protein_importance['Rank'] = range(1, len(protein_importance) + 1)
                
                print(f"  Total features analyzed: {len(protein_importance)}")
                print(f"  Features with importance > 0: {len(protein_importance[protein_importance['Importance'] > 0])}")
                print(f"  Top 5 most important features:")
                print(protein_importance.head(5)[['Rank', 'Feature_Name', 'Importance']])
            
            if protein_importance is not None:
                # Save results
                important_proteins = self.save_model_results(
                    model_key, protein_importance, final_model, selected_features
                )
                
                return {
                    'protein_importance': protein_importance,
                    'important_proteins': important_proteins,
                    'model': final_model,
                    'selected_features': selected_features
                }
            else:
                print(f"❌ Could not extract feature importance for {model_key}")
                return None
                
        except Exception as e:
            print(f"❌ Error analyzing {model_key}: {str(e)}")
            return None

    def train_decision_tree_with_rfe(self, n_features):
        """Train Decision Tree with RFE feature selection"""
        print(f"Training Decision Tree with RFE using {n_features} features...")
        
        model = DecisionTreeClassifier(criterion='gini', max_depth=20, min_samples_leaf=2, random_state=42)
        rfe = RFE(estimator=model, n_features_to_select=n_features, step=50)
        rfe.fit(self.X_train_meta, self.y_train)

        selected_mask = rfe.get_support()
        selected_indices = np.where(selected_mask)[0]
        selected_features = [self.feature_names[i] for i in selected_indices]

        sel_X_train = rfe.transform(self.X_train_meta)
        sel_X_test = rfe.transform(self.X_test_meta)

        model.fit(sel_X_train, self.y_train)

        preds = model.predict(sel_X_test)
        acc = accuracy_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds, average='weighted')
        
        print(f"  Decision Tree Accuracy: {acc:.4f}")
        print(f"  Decision Tree F1-Score: {f1:.4f}")

        cv = cross_val_score(model, sel_X_train, self.y_train, cv=5, scoring='f1_weighted')
        print(f"  Cross-validation F1-Score: {cv.mean():.4f} (+/- {cv.std() * 2:.4f})")
        
        return model, selected_features

    def generate_comparison_report(self):
        """Generate a comparison report across all models"""
        print("\n" + "=" * 60)
        print("GENERATING COMPARISON REPORT ACROSS ALL MODELS")
        print("=" * 60)
        
        comparison_data = []
        
        for model_key, result in self.results.items():
            if 'important_proteins' in result and len(result['important_proteins']) > 0:
                important_df = result['important_proteins']
                comparison_data.append({
                    'Model': model_key.replace('_', ' ').title(),
                    'Total_Features_Used': len(result['selected_features']),
                    'Important_Proteins': len(important_df),
                    'Top_Protein': important_df.iloc[0]['Feature_Name'],
                    'Max_Importance': round(important_df.iloc[0]['Importance'], 6)
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Save comparison report
            os.makedirs('comparison_outputs', exist_ok=True)
            comparison_df.to_csv('comparison_outputs/model_comparison.csv', index=False)
            
            print("\nModel Performance Summary:")
            print(comparison_df.to_string(index=False))
            
            # Find model with most important proteins
            best_model_row = comparison_df.loc[comparison_df['Important_Proteins'].idxmax()]
            print(f"\n🏆 Model with most important proteins: {best_model_row['Model']}")
            print(f"   Features used: {best_model_row['Total_Features_Used']}")
            print(f"   Important proteins identified: {best_model_row['Important_Proteins']}")
            print(f"   Top protein: {best_model_row['Top_Protein']}")
            
            print(f"\n📊 Comparison report saved to: comparison_outputs/model_comparison.csv")

    def run_complete_analysis(self):
        print("=" * 70)
        print("Starting Multi-Model Analysis Pipeline")
        print("(Using SVM outputs as base features)")
        print("=" * 70)

        # Load SVM outputs
        self.load_svm_outputs()
        
        # Analyze each model type
        successful_models = []
        for model_key, model_config in self.model_configs.items():
            result = self.analyze_single_model(model_key, model_config)
            if result is not None:
                self.results[model_key] = result
                successful_models.append(model_key)
                print(f"✅ {model_config['name']} analysis completed successfully!")
            else:
                print(f"❌ {model_config['name']} analysis failed!")
        
        # Generate comparison report
        if successful_models:
            self.generate_comparison_report()
        
        print("=" * 70)
        print(f"Multi-Model Analysis Pipeline Completed!")
        print(f"Successfully analyzed {len(successful_models)} models")
        print("=" * 70)
        
        print("\nOutputs created:")
        for model_key in successful_models:
            print(f"- protein_outputs_{model_key}/")
        if successful_models:
            print("- comparison_outputs/")
        
        return self.results

def main():
    analyzer = MultiModelAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print(f"\n🎉 Analysis completed for {len(results)} models!")
    for model_type in results.keys():
        print(f"   - {model_type}")

if __name__ == "__main__":
    main()
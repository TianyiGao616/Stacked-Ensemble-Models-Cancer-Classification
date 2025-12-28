import os
import pandas as pd
import openai
from time import sleep
import numpy as np

class MultiModelGeneDescriptionGenerator:
    def __init__(self, api_key, cancer_type="Renal cancer"):
        """
        Initialize the multi-model gene description generator.
        
        Args:
            api_key (str): OpenAI API key
            cancer_type (str): Type of cancer to focus descriptions on
        """
        openai.api_key = api_key
        self.cancer_type = cancer_type
        self.model_types = ["decision_tree", "random_forest", "logistic_regression"]
        self.results = {}
        
    def describe_gene_and_cancer(self, gene):
        """Generate description for a single gene"""
        prompt = (
            f"Provide a concise biological function description of the human gene '{gene}', "
            f"followed by a paragraph explaining how this gene may relate to or impact {self.cancer_type}. "
            "Use scientific language suitable for a research report."
        )
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {e}"
    
    def process_single_model(self, model_type):
        """Process gene descriptions for a single model"""
        print(f"\n{'='*60}")
        print(f"Processing gene descriptions for {model_type.upper()}")
        if model_type == "decision_tree":
            print("(Comprehensive analysis - RFE selected features)")
        else:
            print("(Focused analysis - Top 10 features)")
        print(f"{'='*60}")
        
        # Define input path
        input_dir = f"gene_outputs_{model_type}"
        input_file = os.path.join(input_dir, "important_proteins_with_genes.csv")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} does not exist. Skipping {model_type}...")
            return None
            
        # Load the annotated gene data
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} annotated proteins for {model_type}")
        
        # Extract unique, non-null gene symbols
        gene_symbols = df['SYMBOL'].dropna().unique().tolist()
        print(f"Found {len(gene_symbols)} unique gene symbols to describe")
        
        if not gene_symbols:
            print(f"No valid gene symbols found for {model_type}")
            return None
        
        # Show expected counts based on model type
        if model_type == "decision_tree":
            print(f"  Expected: Large number of genes (comprehensive analysis)")
        else:
            print(f"  Expected: Up to 10 genes (focused analysis)")
        
        # Generate descriptions with progress tracking
        results = []
        total_genes = len(gene_symbols)
        successful_descriptions = 0
        failed_descriptions = 0
        
        for i, gene in enumerate(gene_symbols, 1):
            try:
                print(f"  Querying gene {i}/{total_genes}: {gene}", end="")
                desc = self.describe_gene_and_cancer(gene)
                if not desc.startswith('Error:'):
                    successful_descriptions += 1
                    print(" ✓")
                else:
                    failed_descriptions += 1
                    print(" ✗")
                results.append({"Gene": gene, "Description": desc})
                sleep(1.2)  # Prevent rate limit
            except Exception as e:
                failed_descriptions += 1
                print(f" ✗ (Error: {str(e)[:50]}...)")
                results.append({"Gene": gene, "Description": f"Error: {e}"})
        
        # Print summary for this model
        print(f"\n📊 {model_type.upper()} Summary:")
        print(f"   Total genes processed: {total_genes}")
        print(f"   Successful descriptions: {successful_descriptions}")
        print(f"   Failed descriptions: {failed_descriptions}")
        print(f"   Success rate: {(successful_descriptions/total_genes)*100:.1f}%")
        
        # Save results
        output_dir = f"description_outputs_{model_type}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"gene_descriptions_{self.cancer_type.lower().replace(' ', '_')}.csv")
        
        # Create comprehensive results DataFrame with additional metadata
        results_df = pd.DataFrame(results)
        results_df['Model_Type'] = model_type
        results_df['Success'] = ~results_df['Description'].str.startswith('Error:')
        results_df['Description_Length'] = results_df['Description'].str.len()
        
        results_df.to_csv(output_file, index=False)
        print(f"✅ Saved results to: {output_file}")
        
        return {
            'model_type': model_type,
            'gene_count': len(gene_symbols),
            'successful_descriptions': successful_descriptions,
            'failed_descriptions': failed_descriptions,
            'descriptions': results,
            'output_file': output_file,
            'success_rate': (successful_descriptions/total_genes)*100 if total_genes > 0 else 0
        }
    
    def process_all_models(self):
        """Process gene descriptions for all models"""
        print("🧬 Starting Multi-Model Gene Description Generation")
        print("Models: Decision Tree (comprehensive), Random Forest & Logistic Regression (top 10)")
        print("=" * 80)
        
        all_results = {}
        successful_models = []
        
        for model_type in self.model_types:
            result = self.process_single_model(model_type)
            if result is not None:
                all_results[model_type] = result
                successful_models.append(model_type)
        
        # Generate comprehensive reports
        if successful_models:
            self.generate_summary_report(all_results, successful_models)
            self.analyze_gene_overlap(all_results, successful_models)
            self.generate_model_comparison(all_results, successful_models)
        
        print("=" * 80)
        print(f"Multi-Model Gene Description Generation Completed!")
        print(f"Successfully processed {len(successful_models)} models")
        print("=" * 80)
        
        return all_results
    
    def generate_summary_report(self, all_results, successful_models):
        """Generate a comprehensive summary report across all models"""
        print("\n📊 COMPREHENSIVE GENE DESCRIPTION SUMMARY REPORT")
        print("=" * 70)
        
        summary_data = []
        total_unique_genes = set()
        
        for model_type in successful_models:
            result = all_results[model_type]
            
            # Collect successful genes
            successful_genes = set()
            for desc in result['descriptions']:
                if not desc['Description'].startswith('Error:'):
                    successful_genes.add(desc['Gene'])
                    total_unique_genes.add(desc['Gene'])
            
            model_description = "Comprehensive (RFE)" if model_type == "decision_tree" else "Focused (Top 10)"
            
            summary_data.append({
                'Model': model_type.replace('_', ' ').title(),
                'Model_Type': model_description,
                'Total_Genes': result['gene_count'],
                'Successful_Descriptions': result['successful_descriptions'],
                'Failed_Descriptions': result['failed_descriptions'],
                'Success_Rate': round(result['success_rate'], 2),
                'Unique_Genes': len(successful_genes)
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Save summary report
            os.makedirs('description_summary', exist_ok=True)
            summary_df.to_csv('description_summary/comprehensive_gene_description_summary.csv', index=False)
            
            print("\nModel Summary:")
            print(summary_df.to_string(index=False))
            
            print(f"\n📈 Overall Statistics:")
            print(f"   Total unique genes described across all models: {len(total_unique_genes)}")
            print(f"   Average success rate: {summary_df['Success_Rate'].mean():.2f}%")
            print(f"   Total descriptions generated: {summary_df['Successful_Descriptions'].sum()}")
            print(f"   Total failed attempts: {summary_df['Failed_Descriptions'].sum()}")
            
            # Model-specific insights
            dt_row = summary_df[summary_df['Model'] == 'Decision Tree']
            rf_row = summary_df[summary_df['Model'] == 'Random Forest']
            lr_row = summary_df[summary_df['Model'] == 'Logistic Regression']
            
            print(f"\n🔍 Model-Specific Insights:")
            if len(dt_row) > 0:
                print(f"   Decision Tree: Comprehensive analysis of {dt_row.iloc[0]['Total_Genes']} genes")
            if len(rf_row) > 0:
                print(f"   Random Forest: Focused analysis of top {rf_row.iloc[0]['Total_Genes']} genes")
            if len(lr_row) > 0:
                print(f"   Logistic Regression: Focused analysis of top {lr_row.iloc[0]['Total_Genes']} genes")
            
            # Find best performing model
            best_model = summary_df.loc[summary_df['Success_Rate'].idxmax()]
            print(f"🏆 Highest success rate: {best_model['Model']} ({best_model['Success_Rate']}%)")
            
            print(f"\n📁 Summary saved to: description_summary/comprehensive_gene_description_summary.csv")

    def analyze_gene_overlap(self, all_results, successful_models):
        """Analyze overlap of important genes across models"""
        print("\n🔍 ANALYZING GENE OVERLAP ACROSS MODELS")
        print("=" * 60)
        
        model_genes = {}
        
        # Collect successfully described genes from each model
        for model_type in successful_models:
            result = all_results[model_type]
            genes = set()
            for desc in result['descriptions']:
                if not desc['Description'].startswith('Error:'):
                    genes.add(desc['Gene'])
            model_genes[model_type] = genes
            
            model_name = model_type.replace('_', ' ').title()
            print(f"{model_name}: {len(genes)} successfully described genes")
        
        if len(model_genes) < 2:
            print("Need at least 2 models for overlap analysis.")
            return
        
        # Pairwise overlap analysis
        print(f"\n🔄 Pairwise Overlap Analysis:")
        overlap_data = []
        
        model_list = list(model_genes.keys())
        for i in range(len(model_list)):
            for j in range(i+1, len(model_list)):
                model1, model2 = model_list[i], model_list[j]
                genes1, genes2 = model_genes[model1], model_genes[model2]
                
                overlap = genes1.intersection(genes2)
                union = genes1.union(genes2)
                jaccard = len(overlap) / len(union) if len(union) > 0 else 0
                
                print(f"   {model1} ∩ {model2}: {len(overlap)} genes (Jaccard: {jaccard:.3f})")
                
                overlap_data.append({
                    'Model1': model1,
                    'Model2': model2,
                    'Model1_Genes': len(genes1),
                    'Model2_Genes': len(genes2),
                    'Overlap_Count': len(overlap),
                    'Jaccard_Index': round(jaccard, 3),
                    'Overlapping_Genes': '; '.join(sorted(list(overlap))) if overlap else 'None'
                })
        
        # Find consensus genes (genes appearing in multiple models)
        all_genes = list(set().union(*model_genes.values()))
        gene_frequency = {}
        for gene in all_genes:
            count = sum(1 for genes in model_genes.values() if gene in genes)
            gene_frequency[gene] = count
        
        # Genes appearing in multiple models
        consensus_genes = {gene: count for gene, count in gene_frequency.items() if count > 1}
        
        if consensus_genes:
            print(f"\n🎯 Consensus Genes (appearing in multiple models):")
            sorted_consensus = sorted(consensus_genes.items(), key=lambda x: x[1], reverse=True)
            for gene, count in sorted_consensus:
                print(f"   {gene}: appears in {count}/{len(successful_models)} models")
        
        # Save overlap analysis
        os.makedirs('gene_overlap_analysis', exist_ok=True)
        
        if overlap_data:
            overlap_df = pd.DataFrame(overlap_data)
            overlap_df.to_csv('gene_overlap_analysis/pairwise_gene_overlap.csv', index=False)
        
        if consensus_genes:
            consensus_df = pd.DataFrame([
                {'Gene': gene, 'Model_Count': count, 'Frequency': f"{count}/{len(successful_models)}"}
                for gene, count in sorted_consensus
            ])
            consensus_df.to_csv('gene_overlap_analysis/consensus_genes.csv', index=False)
        
        print(f"\n📁 Overlap analysis saved to: gene_overlap_analysis/")

    def generate_model_comparison(self, all_results, successful_models):
        """Generate detailed comparison between focused and comprehensive approaches"""
        print("\n🔬 COMPREHENSIVE VS FOCUSED ANALYSIS COMPARISON")
        print("=" * 60)
        
        dt_results = all_results.get('decision_tree')
        focused_models = ['random_forest', 'logistic_regression']
        
        if dt_results:
            print(f"Decision Tree (Comprehensive):")
            print(f"   Genes analyzed: {dt_results['gene_count']}")
            print(f"   Success rate: {dt_results['success_rate']:.1f}%")
            
            dt_genes = set()
            for desc in dt_results['descriptions']:
                if not desc['Description'].startswith('Error:'):
                    dt_genes.add(desc['Gene'])
        
        print(f"\nFocused Models (Top 10):")
        focused_genes_combined = set()
        
        for model_type in focused_models:
            if model_type in all_results:
                result = all_results[model_type]
                model_genes = set()
                for desc in result['descriptions']:
                    if not desc['Description'].startswith('Error:'):
                        model_genes.add(desc['Gene'])
                        focused_genes_combined.add(desc['Gene'])
                
                print(f"   {model_type.replace('_', ' ').title()}: {len(model_genes)} genes, {result['success_rate']:.1f}% success")
        
        # Compare comprehensive vs focused
        if dt_results and focused_genes_combined:
            overlap_with_dt = dt_genes.intersection(focused_genes_combined)
            unique_to_dt = dt_genes - focused_genes_combined
            unique_to_focused = focused_genes_combined - dt_genes
            
            print(f"\n📊 Comprehensive vs Focused Comparison:")
            print(f"   Genes found by Decision Tree only: {len(unique_to_dt)}")
            print(f"   Genes found by Focused models only: {len(unique_to_focused)}")
            print(f"   Genes found by both approaches: {len(overlap_with_dt)}")
            
            if overlap_with_dt:
                print(f"   Consensus genes: {', '.join(sorted(list(overlap_with_dt)))}")
            
            # Save comparison
            comparison_data = {
                'Comprehensive_Only': list(unique_to_dt),
                'Focused_Only': list(unique_to_focused),
                'Both_Approaches': list(overlap_with_dt)
            }
            
            # Convert to DataFrame for saving
            max_len = max(len(v) for v in comparison_data.values())
            for key in comparison_data:
                comparison_data[key].extend([''] * (max_len - len(comparison_data[key])))
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv('gene_overlap_analysis/comprehensive_vs_focused_comparison.csv', index=False)
            
            print(f"📁 Comparison saved to: gene_overlap_analysis/comprehensive_vs_focused_comparison.csv")

def main():
    # Set your OpenAI API key here
    api_key = ""
    
    generator = MultiModelGeneDescriptionGenerator(
        api_key=api_key,
        cancer_type="Renal cancer"
    )
    
    results = generator.process_all_models()
    
    print(f"\n🎉 Gene description generation completed!")
    print("Generated directories:")
    for model_type in generator.model_types:
        output_dir = f"description_outputs_{model_type}"
        if os.path.exists(output_dir):
            print(f"   - {output_dir}/")
    
    if results:
        print("   - description_summary/")
        print("   - gene_overlap_analysis/")
        
        # Final recommendations
        print(f"\n💡 Analysis Recommendations:")
        print("   1. Check consensus genes for the most reliable biomarkers")
        print("   2. Decision Tree provides comprehensive gene discovery")
        print("   3. Random Forest & Logistic Regression focus on top candidates")
        print("   4. Review overlap analysis for validation priorities")

if __name__ == "__main__":
    main()

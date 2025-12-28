# Install required packages if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if (!requireNamespace("AnnotationDbi", quietly = TRUE)) BiocManager::install("AnnotationDbi")
if (!requireNamespace("hgu133plus2.db", quietly = TRUE)) BiocManager::install("hgu133plus2.db")

# Load necessary libraries
library(AnnotationDbi)
library(hgu133plus2.db)

# Define the model types (updated for 3 models from step 2)
model_types <- c("decision_tree", "random_forest", "logistic_regression")

# Function to process each model's results
process_model_gene_annotation <- function(model_type) {
  cat("=" %&% "=" %&% "=" %&% "=" %&% "=" %&% "\n")
  cat("Processing", model_type, "gene annotations...\n")
  cat("=" %&% "=" %&% "=" %&% "=" %&% "=" %&% "\n")
  
  # Define input and output paths
  input_dir <- paste0("protein_outputs_", model_type)
  input_file <- file.path(input_dir, "important_proteins.csv")
  output_dir <- paste0("gene_outputs_", model_type)
  
  # Check if input file exists
  if (!file.exists(input_file)) {
    cat("Warning: File", input_file, "does not exist. Skipping", model_type, "...\n")
    return(NULL)
  }
  
  # Read the important proteins CSV
  protein_df <- read.csv(input_file)
  cat("Loaded", nrow(protein_df), "important proteins for", model_type, "\n")
  
  # Show feature count expectation
  if (model_type == "decision_tree") {
    cat("  Expected: 50-200 proteins (RFE-selected)\n")
  } else {
    cat("  Expected: Exactly 10 proteins (RFE + Top 10)\n")
  }
  
  # Extract unique probe IDs
  probe_ids <- unique(protein_df$Feature_Name)
  cat("Processing", length(probe_ids), "unique probe IDs...\n")
  
  # Map probe IDs to gene SYMBOL and GENENAME
  tryCatch({
    gene_info <- AnnotationDbi::select(hgu133plus2.db,
                                       keys = probe_ids,
                                       columns = c("SYMBOL", "GENENAME"),
                                       keytype = "PROBEID")
    
    # Merge with original data
    merged <- merge(protein_df, gene_info, by.x = "Feature_Name", by.y = "PROBEID", all.x = TRUE)
    
    # Sort by importance rank to maintain order
    merged <- merged[order(merged$Rank), ]
    
    # Create output directory
    if (!dir.exists(output_dir)) dir.create(output_dir)
    
    # Save annotated results
    output_file <- file.path(output_dir, "important_proteins_with_genes.csv")
    write.csv(merged, file = output_file, row.names = FALSE)
    
    # Print summary with model-specific insights
    mapped_genes <- sum(!is.na(merged$SYMBOL))
    unmapped_proteins <- nrow(merged) - mapped_genes
    
    cat("✅ Successfully mapped", mapped_genes, "out of", nrow(merged), "proteins to gene symbols\n")
    if (unmapped_proteins > 0) {
      cat("⚠️ ", unmapped_proteins, "proteins could not be mapped to gene symbols\n")
    }
    
    # Show top mapped genes
    if (mapped_genes > 0) {
      top_genes <- merged[!is.na(merged$SYMBOL), ][1:min(5, mapped_genes), ]
      cat("🔬 Top mapped genes for", model_type, ":\n")
      for (i in 1:nrow(top_genes)) {
        cat("   ", i, ".", top_genes$SYMBOL[i], "(Importance:", 
            round(top_genes$Importance[i], 4), ")\n")
      }
    }
    
    cat("📁 Results saved to:", output_file, "\n")
    
    return(merged)
    
  }, error = function(e) {
    cat("❌ Error processing", model_type, ":", e$message, "\n")
    return(NULL)
  })
}

# Define custom string concatenation operator for cleaner output
"%&%" <- function(a, b) paste0(a, b)

# Process all models
cat("🧬 Starting Multi-Model Gene Annotation Pipeline\n")
cat("Processing 3 models: Decision Tree (comprehensive), Random Forest & Logistic Regression (top 10)\n")
cat("=" %&% "=" %&% "=" %&% "=" %&% "=" %&% "=" %&% "=" %&% "\n\n")

results <- list()
successful_models <- c()

for (model_type in model_types) {
  result <- process_model_gene_annotation(model_type)
  if (!is.null(result)) {
    results[[model_type]] <- result
    successful_models <- c(successful_models, model_type)
  }
  cat("\n")
}

# Generate detailed summary report
if (length(successful_models) > 0) {
  cat("📊 DETAILED GENE ANNOTATION SUMMARY REPORT\n")
  cat("=" %&% "=" %&% "=" %&% "=" %&% "=" %&% "=" %&% "=" %&% "\n")
  
  summary_data <- data.frame(
    Model = character(),
    Model_Type = character(),
    Total_Proteins = numeric(),
    Mapped_Genes = numeric(),
    Unmapped_Proteins = numeric(),
    Mapping_Rate = numeric(),
    Top_Gene = character(),
    stringsAsFactors = FALSE
  )
  
  for (model_type in successful_models) {
    result <- results[[model_type]]
    total_proteins <- nrow(result)
    mapped_genes <- sum(!is.na(result$SYMBOL))
    unmapped_proteins <- total_proteins - mapped_genes
    mapping_rate <- round((mapped_genes / total_proteins) * 100, 2)
    
    # Get top gene (first mapped gene)
    top_gene <- ifelse(mapped_genes > 0, 
                      result[!is.na(result$SYMBOL), ]$SYMBOL[1], 
                      "None")
    
    # Determine model type description
    model_description <- ifelse(model_type == "decision_tree", 
                               "Comprehensive (RFE)", 
                               "Focused (Top 10)")
    
    summary_data <- rbind(summary_data, data.frame(
      Model = model_type,
      Model_Type = model_description,
      Total_Proteins = total_proteins,
      Mapped_Genes = mapped_genes,
      Unmapped_Proteins = unmapped_proteins,
      Mapping_Rate = mapping_rate,
      Top_Gene = top_gene
    ))
  }
  
  print(summary_data)
  
  # Save comprehensive summary report
  summary_dir <- "gene_annotation_summary"
  if (!dir.exists(summary_dir)) dir.create(summary_dir)
  write.csv(summary_data, file = file.path(summary_dir, "detailed_gene_annotation_summary.csv"), row.names = FALSE)
  
  cat("\n📈 Summary statistics:\n")
  cat("   Average mapping rate across all models:", round(mean(summary_data$Mapping_Rate), 2), "%\n")
  cat("   Total proteins processed:", sum(summary_data$Total_Proteins), "\n")
  cat("   Total genes successfully mapped:", sum(summary_data$Mapped_Genes), "\n")
  cat("   Total unmapped proteins:", sum(summary_data$Unmapped_Proteins), "\n")
  
  # Model-specific insights
  dt_data <- summary_data[summary_data$Model == "decision_tree", ]
  rf_data <- summary_data[summary_data$Model == "random_forest", ]
  lr_data <- summary_data[summary_data$Model == "logistic_regression", ]
  
  cat("\n🔍 Model-specific insights:\n")
  if (nrow(dt_data) > 0) {
    cat("   Decision Tree: Comprehensive analysis with", dt_data$Total_Proteins, "proteins\n")
  }
  if (nrow(rf_data) > 0) {
    cat("   Random Forest: Focused on", rf_data$Total_Proteins, "most important proteins\n")
  }
  if (nrow(lr_data) > 0) {
    cat("   Logistic Regression: Focused on", lr_data$Total_Proteins, "most important proteins\n")
  }
  
  # Find model with highest mapping rate
  best_model <- summary_data[which.max(summary_data$Mapping_Rate), ]
  cat("🏆 Best mapping rate:", best_model$Model, "(", best_model$Mapping_Rate, "%)\n")
  
  # Find most frequently appearing top gene across models
  top_genes <- summary_data$Top_Gene[summary_data$Top_Gene != "None"]
  if (length(top_genes) > 0) {
    gene_counts <- table(top_genes)
    if (max(gene_counts) > 1) {
      consensus_gene <- names(gene_counts)[which.max(gene_counts)]
      cat("🎯 Consensus top gene:", consensus_gene, "(appears in", max(gene_counts), "models)\n")
    }
  }
  
  cat("\n📁 Detailed summary report saved to:", file.path(summary_dir, "detailed_gene_annotation_summary.csv"), "\n")
  
  # Generate gene overlap analysis
  cat("\n🔄 Analyzing gene overlap between models...\n")
  gene_sets <- list()
  for (model_type in successful_models) {
    mapped_genes <- results[[model_type]][!is.na(results[[model_type]]$SYMBOL), ]$SYMBOL
    gene_sets[[model_type]] <- unique(mapped_genes)
    cat("   ", model_type, ":", length(gene_sets[[model_type]]), "unique genes\n")
  }
  
  # Find common genes across all models
  if (length(gene_sets) > 1) {
    common_genes <- Reduce(intersect, gene_sets)
    cat("   Common genes across ALL models:", length(common_genes), "\n")
    if (length(common_genes) > 0) {
      cat("   Common genes:", paste(common_genes, collapse = ", "), "\n")
      
      # Save common genes
      write.csv(data.frame(Common_Genes = common_genes), 
                file = file.path(summary_dir, "common_genes_all_models.csv"), 
                row.names = FALSE)
    }
  }
  
} else {
  cat("❌ No models were successfully processed. Please check that the protein output files exist.\n")
}

cat("\n🎉 Multi-Model Gene Annotation Pipeline Completed!\n")
cat("Generated directories:\n")
for (model_type in successful_models) {
  cat("   - gene_outputs_", model_type, "/\n", sep = "")
}
if (length(successful_models) > 0) {
  cat("   - gene_annotation_summary/\n")
}
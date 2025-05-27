import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import time
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns
from collections import defaultdict

# Function to load and process a transcript
def load_labeled_transcript(file_path):
    """
    Load a transcript file with the provided format and prepare it for evaluation.
    
    Args:
        file_path: Path to the transcript JSON file
        
    Returns:
        A dictionary with prepared data for evaluation
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract the results
        utterances = data.get('results', [])
        
        # Create a list of utterances for model input
        conversation = []
        for utterance in utterances:
            conversation.append({
                "role": utterance["Role"],
                "content": utterance["Content"]
            })
        
        # Extract ground truth EndOfTurn labels
        ground_truth = [int(utterance["EndOfTurn"]) for utterance in utterances]
        
        # Create context windows for each utterance
        contexts = []
        for i in range(len(conversation)):
            contexts.append(conversation[:i+1])
        
        return {
            "conversation": conversation,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "utterances": utterances,  # Keep original format for reference
            "file_path": file_path,
            "file_name": os.path.basename(file_path)
        }
    except Exception as e:
        print(f"Error loading transcript {file_path}: {str(e)}")
        return None

# Function to evaluate a single transcript
def evaluate_single_transcript(model, tokenizer, data, device="cuda"):
    """
    Evaluate the model on a single transcript.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer for preprocessing
        data: Processed transcript data
        device: The device to use for inference
        
    Returns:
        Dictionary of evaluation results
    """
    # Make predictions
    predictions = []
    latencies = []
    
    for i, context in enumerate(data['contexts']):
        # Format messages
        start_time = time.time()
        text = tokenizer.apply_chat_template(
            context,
            add_generation_prompt=False,
            add_special_tokens=False,
            tokenize=False
        )
        
        # Remove the EOU token from current utterance
        ix = text.rfind("<|im_end|>")
        if ix >= 0:
            text = text[:ix]
        
        # Tokenize
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            eou_probability = probabilities[0, 1].item()
        
        predictions.append(eou_probability)
        latencies.append(time.time() - start_time)
    
    # Create evaluation results
    results = {}
    results["file_path"] = data["file_path"]
    results["file_name"] = data["file_name"]
    
    # Detailed utterance analysis
    utterance_analysis = []
    for i, (utterance, prob) in enumerate(zip(data['utterances'], predictions)):
        binary_pred = 1 if prob >= 0.5 else 0  # Using 0.5 as default threshold
        is_correct = binary_pred == data['ground_truth'][i]
        
        utterance_analysis.append({
            "index": i,
            "content": utterance["Content"],
            "role": utterance["Role"],
            "ground_truth": data['ground_truth'][i],
            "predicted_probability": prob,
            "predicted_label": binary_pred,
            "is_correct": is_correct
        })
    
    results["utterance_analysis"] = utterance_analysis
    
    # Compute F1 scores at different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    f1_scores = {}
    
    for threshold in thresholds:
        binary_preds = [1 if p >= threshold else 0 for p in predictions]
        if len(np.unique(data['ground_truth'])) > 1:  # Check if we have both positive and negative examples
            f1 = f1_score(data['ground_truth'], binary_preds)
        else:
            f1 = 0.0  # If we only have one class, F1 is not defined
        f1_scores[str(threshold)] = f1
    
    results["f1_scores"] = f1_scores
    
    # Find optimal threshold based on F1 score
    if any(f1_scores.values()):  # If any F1 score is non-zero
        optimal_threshold = max(f1_scores.items(), key=lambda x: x[1])[0]
        results["optimal_threshold"] = float(optimal_threshold)
    else:
        results["optimal_threshold"] = 0.5  # Default if no good threshold found
    
    # Compute metrics with optimal threshold
    binary_preds = [1 if p >= float(results["optimal_threshold"]) else 0 for p in predictions]
    
    # Calculate ROC and PR curves if we have both positive and negative examples
    if len(np.unique(data['ground_truth'])) > 1:
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(data['ground_truth'], predictions)
        roc_auc = auc(fpr, tpr)
        
        results["roc"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
            "auc": roc_auc
        }
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(data['ground_truth'], predictions)
        
        results["pr"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        }
    else:
        # Placeholder for ROC and PR if we don't have both classes
        results["roc"] = {"auc": 0.5}
        results["pr"] = {}
    
    # Confusion matrix
    cm = confusion_matrix(data['ground_truth'], binary_preds)
    results["confusion_matrix"] = cm.tolist()
    
    # Overall metrics
    results["metrics"] = {
        "accuracy": accuracy_score(data['ground_truth'], binary_preds),
        "precision": precision_score(data['ground_truth'], binary_preds, zero_division=0),
        "recall": recall_score(data['ground_truth'], binary_preds, zero_division=0),
        "f1": f1_score(data['ground_truth'], binary_preds, zero_division=0),
        "total_utterances": len(data['ground_truth']),
        "eot_ratio": sum(data['ground_truth']) / len(data['ground_truth'])
    }
    
    # Latency statistics
    results["latency"] = {
        "mean": np.mean(latencies),
        "median": np.median(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies),
        "p95": np.percentile(latencies, 95)
    }
    
    # Role-based analysis
    role_analysis = {}
    
    roles = set(u["Role"] for u in data['utterances'])
    for role in roles:
        role_indices = [i for i, u in enumerate(data['utterances']) if u["Role"] == role]
        if not role_indices:
            continue
            
        role_gt = [data['ground_truth'][i] for i in role_indices]
        role_preds = [predictions[i] for i in role_indices]
        role_binary_preds = [1 if p >= float(results["optimal_threshold"]) else 0 for p in role_preds]
        
        role_analysis[role] = {
            "count": len(role_indices),
            "accuracy": accuracy_score(role_gt, role_binary_preds),
            "precision": precision_score(role_gt, role_binary_preds, zero_division=0),
            "recall": recall_score(role_gt, role_binary_preds, zero_division=0),
            "f1": f1_score(role_gt, role_binary_preds, zero_division=0),
            "eot_ratio": sum(role_gt) / len(role_gt) if len(role_gt) > 0 else 0
        }
    
    results["role_analysis"] = role_analysis
    
    # Error analysis
    error_summary = {
        "false_positives": 0,
        "false_negatives": 0
    }
    
    for i, (gt, pred) in enumerate(zip(data['ground_truth'], binary_preds)):
        if gt == 1 and pred == 0:
            error_summary["false_negatives"] += 1
        elif gt == 0 and pred == 1:
            error_summary["false_positives"] += 1
    
    results["error_summary"] = error_summary
    
    return results

# Function to evaluate a batch of transcripts
def evaluate_transcript_batch(transcript_dir, model_name="livekit/turn-detector", output_dir="results", 
                             batch_size=None, num_workers=4):
    """
    Evaluate the model on multiple transcripts in a directory.
    
    Args:
        transcript_dir: Directory containing transcript files
        model_name: Name or path of the model to evaluate
        output_dir: Directory to save results
        batch_size: Number of transcripts to process (None for all)
        num_workers: Number of parallel workers for processing
        
    Returns:
        Dictionary of aggregated evaluation results
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "transcripts"), exist_ok=True)
    
    # Get transcript files
    transcript_files = glob.glob(os.path.join(transcript_dir, "*.txt")) + glob.glob(os.path.join(transcript_dir, "*.json"))
    
    if batch_size is not None:
        transcript_files = transcript_files[:batch_size]
    
    print(f"Found {len(transcript_files)} transcript files")
    
    # Load model and tokenizer
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load transcripts
    print("Loading transcripts...")
    transcript_data = []
    for file_path in tqdm(transcript_files):
        data = load_labeled_transcript(file_path)
        if data is not None:
            transcript_data.append(data)
    
    print(f"Successfully loaded {len(transcript_data)} transcripts")
    
    # Process transcripts
    print("Processing transcripts...")
    transcript_results = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(transcript_data), 10)):
        batch = transcript_data[i:i+10]
        for data in batch:
            result = evaluate_single_transcript(model, tokenizer, data, device)
            transcript_results.append(result)
            
            # Save individual transcript results
            with open(os.path.join(output_dir, "transcripts", f"{os.path.basename(data['file_path'])}.results.json"), "w") as f:
                json.dump(result, f, indent=2)
    
    # Aggregate results
    print("Aggregating results...")
    aggregate_results = aggregate_transcript_results(transcript_results)
    
    # Save aggregated results
    with open(os.path.join(output_dir, "aggregate_results.json"), "w") as f:
        json.dump(aggregate_results, f, indent=2)
    
    # Generate visualizations
    print("Generating visualizations...")
    generate_aggregate_visualizations(aggregate_results, transcript_results, output_dir)
    
    return aggregate_results

# Function to aggregate results from multiple transcripts
def aggregate_transcript_results(transcript_results):
    """
    Aggregate results from multiple transcripts.
    
    Args:
        transcript_results: List of transcript evaluation results
        
    Returns:
        Dictionary of aggregated results
    """
    aggregate = {}
    
    # Overall metrics
    all_metrics = [r["metrics"] for r in transcript_results]
    
    aggregate["metrics"] = {
        "mean_accuracy": np.mean([m["accuracy"] for m in all_metrics]),
        "mean_precision": np.mean([m["precision"] for m in all_metrics]),
        "mean_recall": np.mean([m["recall"] for m in all_metrics]),
        "mean_f1": np.mean([m["f1"] for m in all_metrics]),
        "median_accuracy": np.median([m["accuracy"] for m in all_metrics]),
        "median_precision": np.median([m["precision"] for m in all_metrics]),
        "median_recall": np.median([m["recall"] for m in all_metrics]),
        "median_f1": np.median([m["f1"] for m in all_metrics]),
        "min_accuracy": np.min([m["accuracy"] for m in all_metrics]),
        "min_precision": np.min([m["precision"] for m in all_metrics]),
        "min_recall": np.min([m["recall"] for m in all_metrics]),
        "min_f1": np.min([m["f1"] for m in all_metrics]),
        "max_accuracy": np.max([m["accuracy"] for m in all_metrics]),
        "max_precision": np.max([m["precision"] for m in all_metrics]),
        "max_recall": np.max([m["recall"] for m in all_metrics]),
        "max_f1": np.max([m["f1"] for m in all_metrics]),
        "std_accuracy": np.std([m["accuracy"] for m in all_metrics]),
        "std_precision": np.std([m["precision"] for m in all_metrics]),
        "std_recall": np.std([m["recall"] for m in all_metrics]),
        "std_f1": np.std([m["f1"] for m in all_metrics]),
        "total_transcripts": len(transcript_results),
        "total_utterances": sum(m["total_utterances"] for m in all_metrics),
        "mean_eot_ratio": np.mean([m["eot_ratio"] for m in all_metrics]),
    }
    
    # Per-threshold F1 scores
    f1_by_threshold = defaultdict(list)
    for result in transcript_results:
        for threshold, f1 in result["f1_scores"].items():
            f1_by_threshold[threshold].append(f1)
    
    aggregate["f1_by_threshold"] = {
        threshold: {
            "mean": np.mean(scores),
            "median": np.median(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "std": np.std(scores)
        }
        for threshold, scores in f1_by_threshold.items()
    }
    
    # Find overall optimal threshold
    optimal_threshold = max(
        aggregate["f1_by_threshold"].items(), 
        key=lambda x: x[1]["mean"]
    )[0]
    
    aggregate["optimal_threshold"] = optimal_threshold
    
    # Aggregate AUC scores
    auc_scores = [r["roc"]["auc"] for r in transcript_results if "auc" in r["roc"]]
    
    aggregate["roc"] = {
        "mean_auc": np.mean(auc_scores) if auc_scores else 0,
        "median_auc": np.median(auc_scores) if auc_scores else 0,
        "min_auc": np.min(auc_scores) if auc_scores else 0,
        "max_auc": np.max(auc_scores) if auc_scores else 0,
        "std_auc": np.std(auc_scores) if auc_scores else 0
    }
    
    # Aggregate latency statistics
    all_latencies = [r["latency"] for r in transcript_results]
    
    aggregate["latency"] = {
        "mean_mean": np.mean([l["mean"] for l in all_latencies]),
        "mean_median": np.mean([l["median"] for l in all_latencies]),
        "mean_min": np.mean([l["min"] for l in all_latencies]),
        "mean_max": np.mean([l["max"] for l in all_latencies]),
        "mean_p95": np.mean([l["p95"] for l in all_latencies]),
        "overall_mean": np.mean([l["mean"] for l in all_latencies]),
        "overall_median": np.median([l["median"] for l in all_latencies])
    }
    
    # Aggregate role-based metrics
    role_metrics = defaultdict(lambda: defaultdict(list))
    
    for result in transcript_results:
        for role, metrics in result["role_analysis"].items():
            for metric, value in metrics.items():
                if metric != "count":  # Skip count, we'll sum it separately
                    role_metrics[role][metric].append(value)
    
    aggregate["role_analysis"] = {}
    
    for role, metrics in role_metrics.items():
        aggregate["role_analysis"][role] = {
            metric: {
                "mean": np.mean(values),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "std": np.std(values)
            }
            for metric, values in metrics.items()
        }
        
        # Add total count for each role
        aggregate["role_analysis"][role]["total_count"] = sum(
            result["role_analysis"].get(role, {}).get("count", 0)
            for result in transcript_results
        )
    
    # Aggregate error summary
    all_errors = [r["error_summary"] for r in transcript_results]
    
    aggregate["error_summary"] = {
        "total_false_positives": sum(e["false_positives"] for e in all_errors),
        "total_false_negatives": sum(e["false_negatives"] for e in all_errors),
        "mean_false_positives": np.mean([e["false_positives"] for e in all_errors]),
        "mean_false_negatives": np.mean([e["false_negatives"] for e in all_errors])
    }
    
    # Add transcript-level metrics
    aggregate["transcript_metrics"] = [
        {
            "file_name": result["file_name"],
            "accuracy": result["metrics"]["accuracy"],
            "precision": result["metrics"]["precision"],
            "recall": result["metrics"]["recall"],
            "f1": result["metrics"]["f1"],
            "total_utterances": result["metrics"]["total_utterances"],
            "eot_ratio": result["metrics"]["eot_ratio"],
            "auc": result["roc"].get("auc", 0),
            "optimal_threshold": result["optimal_threshold"],
            "false_positives": result["error_summary"]["false_positives"],
            "false_negatives": result["error_summary"]["false_negatives"]
        }
        for result in transcript_results
    ]
    
    return aggregate

# Function to generate visualizations for aggregated results
def generate_aggregate_visualizations(aggregate_results, transcript_results, output_dir="results"):
    """
    Generate visualizations for aggregated transcript results.
    
    Args:
        aggregate_results: Aggregated results
        transcript_results: Individual transcript results
        output_dir: Directory to save visualizations
    """
    # 1. Distribution of metric scores across transcripts
    plt.figure(figsize=(12, 8))
    
    metrics = ["accuracy", "precision", "recall", "f1"]
    transcript_metrics = pd.DataFrame(aggregate_results["transcript_metrics"])
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        sns.histplot(transcript_metrics[metric], kde=True)
        plt.xlabel(metric.capitalize())
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {metric.capitalize()} Across Transcripts")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "metric_distributions.png"), dpi=300)
    
    # 2. F1 scores by threshold
    plt.figure(figsize=(10, 8))
    
    thresholds = sorted(aggregate_results["f1_by_threshold"].keys(), key=float)
    means = [aggregate_results["f1_by_threshold"][t]["mean"] for t in thresholds]
    stds = [aggregate_results["f1_by_threshold"][t]["std"] for t in thresholds]
    
    plt.errorbar(
        [float(t) for t in thresholds],
        means,
        yerr=stds,
        fmt='o-',
        capsize=5,
        label="Mean F1 Score ± Std Dev"
    )
    
    # Mark optimal threshold
    optimal_idx = thresholds.index(aggregate_results["optimal_threshold"])
    plt.scatter(
        [float(aggregate_results["optimal_threshold"])],
        [means[optimal_idx]],
        marker='*',
        s=200,
        c='red',
        label=f"Optimal threshold ({float(aggregate_results['optimal_threshold']):.2f})"
    )
    
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('Mean F1 Scores Across Different Thresholds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "plots", "f1_by_threshold.png"), dpi=300)
    
    # 3. Performance by role
    plt.figure(figsize=(14, 10))
    
    # Only include roles that appear in multiple transcripts
    roles = [role for role in aggregate_results["role_analysis"].keys() 
             if aggregate_results["role_analysis"][role]["total_count"] > 10]
    
    metrics = ["accuracy", "precision", "recall", "f1", "eot_ratio"]
    width = 0.15
    x = np.arange(len(roles))
    
    for i, metric in enumerate(metrics):
        values = [aggregate_results["role_analysis"][role][metric]["mean"] for role in roles]
        plt.bar(x + (i - 2) * width, values, width, label=metric.capitalize())
    
    plt.xlabel('Role')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Role')
    plt.xticks(x, roles)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "role_performance.png"), dpi=300)
    
    # 4. Error analysis
    plt.figure(figsize=(10, 6))
    
    error_types = ["false_positives", "false_negatives"]
    error_means = [aggregate_results["error_summary"][f"mean_{et}"] for et in error_types]
    
    plt.bar(error_types, error_means)
    plt.xlabel('Error Type')
    plt.ylabel('Mean Count per Transcript')
    plt.title('Mean Error Counts per Transcript')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "plots", "error_analysis.png"), dpi=300)
    
    # 5. Utterance count vs. performance scatterplot
    plt.figure(figsize=(10, 8))
    
    plt.scatter(
        transcript_metrics["total_utterances"],
        transcript_metrics["f1"],
        alpha=0.7
    )
    
    plt.xlabel('Number of Utterances')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Transcript Length')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "plots", "performance_vs_length.png"), dpi=300)
    
    # 6. EOT ratio vs. performance scatterplot
    plt.figure(figsize=(10, 8))
    
    plt.scatter(
        transcript_metrics["eot_ratio"],
        transcript_metrics["f1"],
        alpha=0.7
    )
    
    plt.xlabel('EOT Ratio (Proportion of End-of-Turn Utterances)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. EOT Ratio')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "plots", "performance_vs_eot_ratio.png"), dpi=300)
    
    # 7. Performance heatmap for transcripts
    if len(transcript_metrics) <= 50:  # Only create heatmap if not too many transcripts
        plt.figure(figsize=(14, 10))
        
        # Select metrics for heatmap
        heatmap_data = transcript_metrics[["file_name", "accuracy", "precision", "recall", "f1", "auc"]].copy()
        heatmap_data.set_index("file_name", inplace=True)
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5)
        plt.title('Performance Metrics Across Transcripts')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "transcript_heatmap.png"), dpi=300)

# Function to create aggregate CSV report
def create_aggregate_report(aggregate_results, output_dir="results"):
    """
    Create a CSV report from aggregated results.
    
    Args:
        aggregate_results: Aggregated evaluation results
        output_dir: Directory to save the report
    """
    # 1. Create transcript metrics table
    transcript_df = pd.DataFrame(aggregate_results["transcript_metrics"])
    transcript_df.to_csv(os.path.join(output_dir, "transcript_metrics.csv"), index=False)
    
    # 2. Create role metrics table
    role_data = []
    
    for role, metrics in aggregate_results["role_analysis"].items():
        row = {"role": role, "total_count": metrics["total_count"]}
        
        for metric in ["accuracy", "precision", "recall", "f1", "eot_ratio"]:
            for stat in ["mean", "median", "min", "max", "std"]:
                row[f"{metric}_{stat}"] = metrics[metric][stat]
        
        role_data.append(row)
    
    role_df = pd.DataFrame(role_data)
    role_df.to_csv(os.path.join(output_dir, "role_metrics.csv"), index=False)
    
    # 3. Create overall metrics table
    overall_data = []
    
    for metric in ["accuracy", "precision", "recall", "f1"]:
        row = {"metric": metric}
        
        for stat in ["mean", "median", "min", "max", "std"]:
            row[stat] = aggregate_results["metrics"][f"{stat}_{metric}"]
        
        overall_data.append(row)
    
    overall_df = pd.DataFrame(overall_data)
    overall_df.to_csv(os.path.join(output_dir, "overall_metrics.csv"), index=False)
    
    # 4. Create threshold analysis table
    threshold_data = []
    
    for threshold, metrics in aggregate_results["f1_by_threshold"].items():
        row = {"threshold": threshold}
        row.update(metrics)
        threshold_data.append(row)
    
    threshold_df = pd.DataFrame(threshold_data)
    threshold_df.to_csv(os.path.join(output_dir, "threshold_analysis.csv"), index=False)
    
    # 5. Create summary report
    with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
        f.write("END-OF-TURN DETECTION EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total transcripts evaluated: {aggregate_results['metrics']['total_transcripts']}\n")
        f.write(f"Total utterances: {aggregate_results['metrics']['total_utterances']}\n")
        f.write(f"Average EOT ratio: {aggregate_results['metrics']['mean_eot_ratio']:.4f}\n\n")
        
        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean Accuracy: {aggregate_results['metrics']['mean_accuracy']:.4f}\n")
        f.write(f"Mean Precision: {aggregate_results['metrics']['mean_precision']:.4f}\n")
        f.write(f"Mean Recall: {aggregate_results['metrics']['mean_recall']:.4f}\n")
        f.write(f"Mean F1 Score: {aggregate_results['metrics']['mean_f1']:.4f}\n")
        f.write(f"Mean AUC: {aggregate_results['roc']['mean_auc']:.4f}\n\n")
        
        f.write("OPTIMAL THRESHOLD\n")
        f.write("-" * 20 + "\n")
        f.write(f"Best threshold: {aggregate_results['optimal_threshold']}\n")
        f.write(f"Mean F1 at best threshold: {aggregate_results['f1_by_threshold'][aggregate_results['optimal_threshold']]['mean']:.4f}\n\n")
        
        f.write("ROLE ANALYSIS\n")
        f.write("-" * 20 + "\n")
        for role, metrics in aggregate_results["role_analysis"].items():
            f.write(f"Role: {role}\n")
            f.write(f"  Total utterances: {metrics['total_count']}\n")
            f.write(f"  Mean F1 Score: {metrics['f1']['mean']:.4f}\n")
            f.write(f"  Mean EOT Ratio: {metrics['eot_ratio']['mean']:.4f}\n\n")
        
        f.write("ERROR ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total False Positives: {aggregate_results['error_summary']['total_false_positives']}\n")
        f.write(f"Total False Negatives: {aggregate_results['error_summary']['total_false_negatives']}\n")
        f.write(f"Mean False Positives per transcript: {aggregate_results['error_summary']['mean_false_positives']:.2f}\n")
        f.write(f"Mean False Negatives per transcript: {aggregate_results['error_summary']['mean_false_negatives']:.2f}\n")

# Example usage
if __name__ == "__main__":
    # Set paths
    transcript_dir = "transcripts"  # Directory containing transcript files
    model_name = "livekit/turn-detector"
    output_dir = "batch_results"
    
    # Run batch evaluation
    results = evaluate_transcript_batch(
        transcript_dir=transcript_dir,
        model_name=model_name,
        output_dir=output_dir,
        batch_size=None,  # Process all transcripts
        num_workers=4
    )
    
    # Create aggregate report
    create_aggregate_report(results, output_dir)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total transcripts: {results['metrics']['total_transcripts']}")
    print(f"Total utterances: {results['metrics']['total_utterances']}")
    print(f"Mean Accuracy: {results['metrics']['mean_accuracy']:.4f}")
    print(f"Mean F1 Score: {results['metrics']['mean_f1']:.4f}")
    print(f"Optimal Threshold: {results['optimal_threshold']}")
    print(f"Mean AUC: {results['roc']['mean_auc']:.4f}")
    
    print("\nResults saved to", output_dir)

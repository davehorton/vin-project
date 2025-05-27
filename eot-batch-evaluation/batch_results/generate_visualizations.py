import os
import json
import sys
from datetime import datetime

def generate_visualization(result_data, output_path):
    """
    Generate an HTML visualization for a transcript evaluation result.
    
    Args:
        result_data (dict): The transcript evaluation result data
        output_path (str): Path to save the HTML file
    """
    # Extract the filename from the file_path
    file_name = result_data.get("file_name", "transcript")
    
    # Get metrics
    metrics = result_data.get("metrics", {})
    accuracy = metrics.get("accuracy", 0) * 100
    precision = metrics.get("precision", 0) * 100
    recall = metrics.get("recall", 0) * 100
    f1 = metrics.get("f1", 0) * 100
    eot_ratio = metrics.get("eot_ratio", 0) * 100
    
    # Get ROC AUC
    roc = result_data.get("roc", {})
    auc = roc.get("auc", 0)
    
    # Get error summary
    error_summary = result_data.get("error_summary", {})
    false_positives = error_summary.get("false_positives", 0)
    false_negatives = error_summary.get("false_negatives", 0)
    total_errors = false_positives + false_negatives
    
    # Get role analysis
    role_analysis = result_data.get("role_analysis", {})
    
    # Get optimal threshold and F1 scores
    optimal_threshold = result_data.get("optimal_threshold", 0.5)
    f1_scores = result_data.get("f1_scores", {})
    
    # Get utterance analysis
    utterance_analysis = result_data.get("utterance_analysis", [])
    
    # Create HTML content
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EOT Detection Evaluation - {file_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }}
        h2 {{
            margin-top: 30px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        .metrics-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            flex: 1 1 200px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
            margin-top: 5px;
        }}
        .conversation {{
            margin-top: 30px;
            border: 1px solid #eee;
            border-radius: 8px;
        }}
        .utterance {{
            padding: 12px 15px;
            margin: 8px 0;
            border-radius: 6px;
            position: relative;
        }}
        .assistant {{
            background-color: #e3f2fd;
            margin-right: 10%;
            border-left: 5px solid #2196F3;
        }}
        .caller {{
            background-color: #f1f8e9;
            margin-left: 10%;
            border-left: 5px solid #8bc34a;
        }}
        .prediction-marker {{
            position: absolute;
            right: 10px;
            top: 10px;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        .eot-true {{
            border-right: 5px solid #4caf50;
        }}
        .eot-false {{
            border-right: 5px solid #ff9800;
        }}
        .pred-correct {{
            background-color: #e8f5e9;
            color: #2e7d32;
        }}
        .pred-incorrect {{
            background-color: #ffebee;
            color: #c62828;
        }}
        .truth-footer {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 14px;
        }}
        .truth-indicator {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .indicator {{
            display: inline-block;
            width: 15px;
            height: 15px;
            border-radius: 3px;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f5f5f5;
            border-radius: 6px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border-radius: 4px;
        }}
        .error-analysis {{
            margin-top: 30px;
        }}
        .error-chart {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 15px;
        }}
        .error-bar {{
            height: 30px;
            margin: 10px 0;
            background-color: #ff5252;
            color: white;
            display: flex;
            align-items: center;
            padding-left: 10px;
            border-radius: 4px;
        }}
        .role-metrics {{
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }}
        .role-card {{
            flex: 1;
            padding: 15px;
            border-radius: 8px;
            background-color: #f8f9fa;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        .role-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 18px;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }}
        .threshold-analysis {{
            margin-top: 30px;
        }}
        .threshold-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .threshold-table th,
        .threshold-table td {{
            padding: 10px;
            text-align: center;
            border: 1px solid #ddd;
        }}
        .threshold-table th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .threshold-table tr.optimal {{
            background-color: #e3f2fd;
            font-weight: bold;
        }}
        .file-info {{
            margin-top: 20px;
            font-size: 12px;
            color: #666;
            text-align: right;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>End-of-Turn Detection Evaluation</h1>
        
        <h2>Performance Overview</h2>
        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-name">Accuracy</div>
                <div class="metric-value">{accuracy:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">Precision</div>
                <div class="metric-value">{precision:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">Recall</div>
                <div class="metric-value">{recall:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">F1 Score</div>
                <div class="metric-value">{f1:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">AUC</div>
                <div class="metric-value">{auc:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-name">EOT Ratio</div>
                <div class="metric-value">{eot_ratio:.1f}%</div>
            </div>
        </div>

        <div class="legend">
            <h3>Legend</h3>
            <div class="legend-item">
                <div class="legend-color" style="border-right: 5px solid #4caf50;"></div>
                <div>Ground Truth: End-of-Turn = True</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="border-right: 5px solid #ff9800;"></div>
                <div>Ground Truth: End-of-Turn = False</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #e8f5e9;"></div>
                <div>Correct Prediction</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ffebee;"></div>
                <div>Incorrect Prediction</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #e3f2fd; border-left: 5px solid #2196F3;"></div>
                <div>Assistant</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #f1f8e9; border-left: 5px solid #8bc34a;"></div>
                <div>Caller</div>
            </div>
        </div>

        <h2>Conversation Transcript with Predictions</h2>
        <div class="conversation">
"""

    # Add each utterance
    for utterance in utterance_analysis:
        role = utterance.get("role", "").lower()
        content = utterance.get("content", "")
        ground_truth = utterance.get("ground_truth", 0)
        prediction = utterance.get("predicted_probability", 0)
        is_correct = utterance.get("is_correct", False)
        
        eot_class = "eot-true" if ground_truth == 1 else "eot-false"
        pred_class = "pred-correct" if is_correct else "pred-incorrect"
        
        html += f"""
            <div class="utterance {role} {eot_class}">
                <div class="prediction-marker {pred_class}">Prediction: {prediction:.3f}</div>
                <strong>{utterance.get("role", "")}:</strong> {content}
            </div>
            """
    
    # Calculate error percentages if there are errors
    fp_percent = 0
    fn_percent = 0
    if total_errors > 0:
        fp_percent = (false_positives / total_errors) * 100
        fn_percent = (false_negatives / total_errors) * 100
    
    html += f"""
        </div>

        <h2>Error Analysis</h2>
        <div class="error-analysis">
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-name">False Positives</div>
                    <div class="metric-value">{false_positives}</div>
                    <div>Model incorrectly predicted EOT</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">False Negatives</div>
                    <div class="metric-value">{false_negatives}</div>
                    <div>Model missed actual EOT</div>
                </div>
            </div>
            
            <div class="error-chart">
                <div>Error Distribution</div>
                <div class="error-bar" style="width: {fp_percent:.1f}%;">
                    False Positives: {false_positives} ({fp_percent:.1f}%)
                </div>
                <div class="error-bar" style="width: {fn_percent:.1f}%; background-color: #ff9800;">
                    False Negatives: {false_negatives} ({fn_percent:.1f}%)
                </div>
            </div>
        </div>

        <h2>Role-Based Performance</h2>
        <div class="role-metrics">
"""

    # Add role metrics
    for role, metrics in role_analysis.items():
        count = metrics.get("count", 0)
        role_accuracy = metrics.get("accuracy", 0) * 100
        role_precision = metrics.get("precision", 0) * 100
        role_recall = metrics.get("recall", 0) * 100
        role_f1 = metrics.get("f1", 0) * 100
        role_eot_ratio = metrics.get("eot_ratio", 0) * 100
        
        html += f"""
            <div class="role-card">
                <div class="role-title">{role} (Count: {count})</div>
                <div class="metric-row">
                    <div>Accuracy:</div>
                    <div>{role_accuracy:.1f}%</div>
                </div>
                <div class="metric-row">
                    <div>Precision:</div>
                    <div>{role_precision:.1f}%</div>
                </div>
                <div class="metric-row">
                    <div>Recall:</div>
                    <div>{role_recall:.1f}%</div>
                </div>
                <div class="metric-row">
                    <div>F1 Score:</div>
                    <div>{role_f1:.1f}%</div>
                </div>
                <div class="metric-row">
                    <div>EOT Ratio:</div>
                    <div>{role_eot_ratio:.1f}%</div>
                </div>
            </div>
            """
    
    html += f"""
        </div>

        <h2>Threshold Analysis</h2>
        <div class="threshold-analysis">
            <p>The optimal threshold for this transcript was determined to be <strong>{optimal_threshold}</strong>, which yielded the highest F1 score.</p>
            
            <table class="threshold-table">
                <thead>
                    <tr>
                        <th>Threshold</th>
                        <th>F1 Score</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Add threshold data
    for threshold, score in sorted(f1_scores.items(), key=lambda x: float(x[0])):
        is_optimal = threshold == str(optimal_threshold)
        optimal_class = "optimal" if is_optimal else ""
        
        html += f"""
                    <tr class="{optimal_class}">
                        <td>{threshold}</td>
                        <td>{score:.3f}</td>
                    </tr>
        """
    
    html += f"""
                </tbody>
            </table>
        </div>

        <h2>Summary</h2>
        <div>
            <p>This evaluation shows that the model {'struggled with' if f1 < 50 else 'performed reasonably on'} this particular transcript, achieving an accuracy of {accuracy:.1f}% and an F1 score of {f1:.1f}%. The model particularly had {'difficulty with false negatives' if false_negatives > false_positives else 'issues with false positives'} ({false_negatives if false_negatives > false_positives else false_positives}), meaning it {'frequently missed actual end-of-turn points' if false_negatives > false_positives else 'frequently predicted end-of-turn points incorrectly'}.</p>
"""

    # Add role comparison if there are multiple roles
    if len(role_analysis) > 1:
        roles = list(role_analysis.keys())
        if len(roles) >= 2:
            role1, role2 = roles[0], roles[1]
            f1_role1 = role_analysis[role1].get("f1", 0) * 100
            f1_role2 = role_analysis[role2].get("f1", 0) * 100
            
            better_role = role1 if f1_role1 > f1_role2 else role2
            worse_role = role2 if f1_role1 > f1_role2 else role1
            
            html += f"""
            <p>The model performed better on {better_role} utterances (F1: {max(f1_role1, f1_role2):.1f}%) than on {worse_role} utterances (F1: {min(f1_role1, f1_role2):.1f}%). The optimal threshold for this transcript was determined to be {optimal_threshold}, {'which is quite low compared to the default of 0.5' if float(optimal_threshold) < 0.4 else 'which is close to the default of 0.5' if 0.4 <= float(optimal_threshold) <= 0.6 else 'which is quite high compared to the default of 0.5'}.</p>
            """
    
    html += f"""
            <p>The {'low' if auc < 0.7 else 'moderate' if 0.7 <= auc < 0.8 else 'good'} AUC score ({auc:.2f}) indicates that the model's ability to distinguish between end-of-turn and non-end-of-turn utterances in this transcript is {'poor' if auc < 0.7 else 'moderate' if 0.7 <= auc < 0.8 else 'good'}. This may be due to the conversational style, the acoustic features not captured in the transcript, or the specific patterns of turn-taking in this conversation.</p>
        </div>
        
        <div class="file-info">
            Transcript: {file_name}<br>
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    # Save the HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

def process_directory(input_dir, output_dir):
    """
    Process all transcript evaluation files in a directory and generate HTML visualizations.
    
    Args:
        input_dir (str): Directory containing transcript evaluation files
        output_dir (str): Directory to save HTML visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    # Process each file
    for i, json_file in enumerate(json_files):
        input_path = os.path.join(input_dir, json_file)
        
        try:
            # Load the JSON data
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Generate a filename for the HTML file
            base_name = os.path.splitext(json_file)[0]
            output_filename = f"{base_name}.html"
            output_path = os.path.join(output_dir, output_filename)
            
            # Generate the visualization
            generate_visualization(data, output_path)
            
            print(f"[{i+1}/{len(json_files)}] Generated visualization for {json_file} -> {output_filename}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    print(f"Completed generating {len(json_files)} visualizations in {output_dir}")

def generate_index_page(output_dir):
    """
    Generate an index HTML page with links to all visualization files.
    
    Args:
        output_dir (str): Directory containing HTML visualizations
    """
    # Get all HTML files in the output directory
    html_files = [f for f in os.listdir(output_dir) if f.endswith('.html') and f != "index.html"]
    html_files.sort()
    
    # Create index HTML content
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EOT Detection Evaluations</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f9f9f9;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
            color: #2c3e50;
        }
        .file-list {
            list-style-type: none;
            padding: 0;
        }
        .file-item {
            padding: 12px 15px;
            margin: 8px 0;
            background-color: #f8f9fa;
            border-radius: 6px;
            border-left: 5px solid #3498db;
            transition: all 0.2s ease;
        }
        .file-item:hover {
            background-color: #e3f2fd;
            transform: translateX(5px);
        }
        .file-link {
            text-decoration: none;
            color: #2c3e50;
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>End-of-Turn Detection Evaluation Results</h1>
        <p>This page contains links to visualization pages for each transcript evaluation.</p>
        
        <ul class="file-list">
"""
    
    # Add links to each HTML file
    for html_file in html_files:
        index_html += f"""
            <li class="file-item">
                <a href="{html_file}" class="file-link">{html_file}</a>
            </li>
        """
    
    index_html += """
        </ul>
        
        <div style="margin-top: 30px; text-align: center; font-size: 12px; color: #666;">
            Generated on {date_time}
        </div>
    </div>
</body>
</html>
""".format(date_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Save the index HTML file
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)
    
    print(f"Generated index page with links to {len(html_files)} visualizations at {index_path}")

if __name__ == "__main__":
    # Get command line arguments
    if len(sys.argv) != 3:
        print("Usage: python generate_visualizations.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Process all files in the input directory
    process_directory(input_dir, output_dir)
    
    # Generate an index page
    generate_index_page(output_dir)
    
    print(f"\nVisualization generation complete!")
    print(f"Open {os.path.join(output_dir, 'index.html')} in a web browser to view the results.")

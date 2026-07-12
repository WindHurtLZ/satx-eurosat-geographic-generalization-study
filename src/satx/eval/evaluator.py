import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

from satx.data import CLASS_NAMES
from satx.utils.paths import resolve_project_path

def generate_evaluation_artifact(y_true, y_pred, output_path: str | Path):
    """
    Calculate evaluation metrics and generate standardized JSON artifact.

    Args:
        y_true (list or array): Ground truth labels.
        y_pred (list or array): Predicted labels.
        output_path (str): File path to save the evaluation JSON artifact.
        
    Returns:
        dict: The generated artifact containing metrics and confusion matrix.
    """
    output_path = resolve_project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate core metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    # Construct the artifact dictionary
    artifact = {
        "metadata": {
            "description": "EuroSAT Geographic Generalization Evaluation Results",
            "classes": CLASS_NAMES
        },
        "overall_metrics": {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1)
        },
        "per_class_metrics": {
            class_name: float(f1) for class_name, f1 in zip(CLASS_NAMES, per_class_f1)
        },
        "confusion_matrix": cm.tolist()
    }

    # Export artifact to JSON
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(artifact, f, indent=4)
    
    print(f"Evaluation artifact successfully saved to: {output_path}")
    return artifact


def print_evaluation_results(artifact):
    """
    Print the evaluation results nicely to the console.
    """
    print("\n" + "="*55)
    print("EVALUATION RESULTS SUMMARY")
    print("="*55)
    print(f"Overall Accuracy : {artifact['overall_metrics']['accuracy']:.4f}")
    print(f"Macro F1 Score   : {artifact['overall_metrics']['macro_f1']:.4f}")
    print("-" * 55)
    print("Per-Class F1 Scores:")
    for cls, f1 in artifact['per_class_metrics'].items():
        print(f"  - {cls:<25}: {f1:.4f}")
    print("="*55 + "\n")


def plot_evaluation_results(artifact, output_dir: str | Path):
    """
    Generate and save visualization plots for the evaluation results.
    """
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = artifact['metadata']['classes']
    cm = np.array(artifact['confusion_matrix'])

    # 1. Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, colorbar=True) # 明确启用 colorbar
    plt.title('Confusion Matrix', pad=20)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. Plot Per-Class F1 Scores (Bar Chart)
    f1_scores = list(artifact['per_class_metrics'].values())
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, f1_scores, color='skyblue')
    plt.title('Per-Class F1 Score')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    f1_path = output_dir / "per_class_f1.png"
    plt.savefig(f1_path, dpi=300)
    plt.close()

    print(f"Visualizations successfully saved to: {output_dir}")


if __name__ == "__main__":
    mock_output_dir = resolve_project_path("outputs/evaluation_mock")
    mock_output_dir.mkdir(parents=True, exist_ok=True)

    # Run a local test with mock data
    mock_y_true = np.random.randint(0, len(CLASS_NAMES), 100)
    mock_y_pred = mock_y_true.copy()
    
    # Introduce noise to simulate prediction errors
    noise_indices = np.random.choice(100, 20, replace=False)
    mock_y_pred[noise_indices] = np.random.randint(0, len(CLASS_NAMES), 20)

    # Define output paths
    output_file_path = mock_output_dir / "mock_results.json"

    # Generate artifact
    result = generate_evaluation_artifact(mock_y_true, mock_y_pred, output_file_path)
    
    # Print results to terminal
    print_evaluation_results(result)
    
    # Generate visualization plots
    plot_evaluation_results(result, mock_output_dir)

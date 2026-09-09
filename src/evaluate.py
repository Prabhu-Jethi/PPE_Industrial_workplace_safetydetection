import os
import json
from ultralytics import YOLO


def evaluate_model():
    ## Load the trained model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best.pt"))
    model = YOLO(model_path)

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "data.yaml"))

    ## Evaluate the model on the validation dataset
    metrics = model.val(data=data_path)

    ## Save the evaluation metrics to a JSON file
    precision = metrics.results_dict["metrics/precision(B)"]
    recall = metrics.results_dict["metrics/recall(B)"]
    map50 = metrics.results_dict["metrics/mAP50(B)"]

    print("\nEvaluation Results...")
    print(f"Precision: {precision:.4f} {precision*100:.1f}")
    print(f"Recall: {recall:.4f} {recall*100:.1f}")
    print(f"mAP@0.5: {map50:.4f}")

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    metrics_file = os.path.join(results_dir, "metrics.json")

    output_data = {
        "precision": precision,
        "recall": recall,
        "mAP50": map50
    }

    with open(metrics_file, "w") as f:
        json.dump(output_data, f, indent=4)
        print(f"\nMetrics successfully saved to {metrics_file}")



if __name__ == "__main__":
    evaluate_model()
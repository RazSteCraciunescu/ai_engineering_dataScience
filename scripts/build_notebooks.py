from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(parents=True, exist_ok=True)


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def make(title: str, cells: list) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [markdown(f"# {title}\n\nExecuted learning notebook for the Public Feedback Intelligence Platform.")] + cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return notebook


def execute(name: str, notebook: nbf.NotebookNode) -> None:
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    target = NOTEBOOKS / name
    nbf.write(notebook, target)
    print(target.relative_to(ROOT))


def main() -> None:
    execute(
        "01_vectors_tensors_and_gradient_descent.ipynb",
        make(
            "Vectors, tensors, masks and gradient descent",
            [
                markdown(
                    "A scalar has no axis, a vector has one, a matrix has two, and a tensor is the general n-dimensional case. "
                    "Embeddings are learned vectors; masks select which values participate in a calculation."
                ),
                code(
                    """
import numpy as np
from feedback_intelligence.statistics import euclidean, minmax, standardize, gradient_descent

scalar = np.array(4.2)
vector = np.array([0.2, 0.8, -0.1])
matrix = np.arange(12).reshape(3, 4)
tensor = np.arange(24).reshape(2, 3, 4)
mask = matrix % 2 == 0

print("shapes:", scalar.shape, vector.shape, matrix.shape, tensor.shape)
print("masked values:", matrix[mask].tolist())
print("euclidean distance:", round(euclidean([1, 2, 3], [2, 4, 4]), 3))
print("min-max:", minmax([10, 15, 30]).round(3).tolist())
print("standardized:", standardize([10, 15, 30]).round(3).tolist())
"""
                ),
                markdown(
                    "Gradient descent updates parameters opposite the loss gradient. The learning rate controls step size; "
                    "steps that are too large can overshoot, while tiny steps converge slowly."
                ),
                code(
                    """
rng = np.random.default_rng(42)
x = np.linspace(-2, 2, 120)
y = 2.6 * x + 1.3 + rng.normal(0, 0.25, x.size)
fit = gradient_descent(x, y, learning_rate=0.04, iterations=1600)
print({"weight": round(fit["weight"], 3), "bias": round(fit["bias"], 3), "final_loss": round(fit["loss_history"][-1], 5)})
"""
                ),
            ],
        ),
    )

    execute(
        "02_eda_distributions_and_outliers.ipynb",
        make(
            "EDA, distributions and outliers",
            [
                code(
                    """
from pathlib import Path
import pandas as pd
from feedback_intelligence.statistics import summarize

ROOT = Path.cwd()
frame = pd.read_csv(ROOT / "data" / "processed" / "feedback.csv")
print(frame.shape)
frame[["record_id", "category", "priority", "resolution_hours", "is_outlier"]].head()
"""
                ),
                markdown(
                    "The mean uses every magnitude and therefore moves toward extreme values. The median depends on order and is usually more robust for skewed operational durations."
                ),
                code(
                    """
summary = summarize(frame["resolution_hours"].dropna())
{k: round(v, 3) for k, v in summary.items()}
"""
                ),
                code(
                    """
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(frame["resolution_hours"].dropna(), bins=32, kde=True, color=sns.color_palette()[0], ax=ax)
ax.axvline(frame["resolution_hours"].median(), linestyle="--", label="median")
ax.axvline(frame["resolution_hours"].mean(), linestyle=":", label="mean")
ax.set(title="Right-skewed resolution-time distribution", xlabel="hours")
ax.legend()
plt.show()
"""
                ),
                markdown(
                    "IQR flags are evidence for investigation, not automatic deletion instructions. The raw duration is preserved while a capped training-only feature limits extreme leverage."
                ),
                code(
                    """
frame.groupby("is_outlier")["resolution_hours"].agg(["count", "mean", "median", "max"]).round(2)
"""
                ),
            ],
        ),
    )

    execute(
        "03_machine_learning_evaluation.ipynb",
        make(
            "Machine-learning evaluation",
            [
                markdown(
                    "Accuracy answers how often the model was correct overall. Precision asks how many predicted members of a class were correct; recall asks how many true members were found. "
                    "Macro-F1 gives each class equal weight and is useful when classes differ in size."
                ),
                code(
                    """
import json
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
metrics = json.loads((ROOT / "artifacts" / "reports" / "model-metrics.json").read_text())
classifier = metrics["priority_classifier"]
regressor = metrics["resolution_regressor"]
pd.DataFrame({
    "metric": ["accuracy", "macro precision", "macro recall", "macro F1", "MAE (hours)", "R squared"],
    "value": [classifier["accuracy"], classifier["macro_precision"], classifier["macro_recall"], classifier["macro_f1"], regressor["mae_hours"], regressor["r2"]],
}).round(3)
"""
                ),
                code(
                    """
import matplotlib.pyplot as plt
import seaborn as sns

labels = classifier["labels"]
matrix = classifier["confusion_matrix"]
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
ax.set(xlabel="predicted", ylabel="actual", title="Held-out priority confusion matrix")
plt.show()
"""
                ),
                markdown(
                    "Preprocessing is fitted only on the training split. Resolution outcomes and satisfaction are excluded from priority prediction, preventing obvious target leakage."
                ),
                code(
                    """
from feedback_intelligence.modeling import predict_priority, predict_resolution

sample = {
    "title": "Workspace reloads during editing",
    "message": "Users lose unsaved work when the editor refreshes unexpectedly.",
    "category": "reliability",
    "channel": "web",
    "product": "workspace",
    "status": "open",
    "affected_users": 95,
    "attachment_count": 2,
    "priority": "high",
}
print(predict_priority(sample))
print(predict_resolution(sample))
"""
                ),
            ],
        ),
    )

    execute(
        "04_embeddings_vector_search_and_rag.ipynb",
        make(
            "Embeddings, vector search and local RAG",
            [
                markdown(
                    "The local index converts documents into TF-IDF vectors and then projects them into a compact dense space with truncated SVD. Cosine similarity compares direction; Euclidean distance compares straight-line separation."
                ),
                code(
                    """
from feedback_intelligence.embeddings import load_index, search

index = load_index()
print("documents:", len(index["documents"]))
print("embedding matrix shape:", index["vectors"].shape)
results = search("Why can accuracy hide poor minority-class performance?", limit=3, index=index)
[(item["title"], round(item["cosine_similarity"], 3), round(item["euclidean_distance"], 3)) for item in results]
"""
                ),
                markdown(
                    "Retrieval-augmented generation separates evidence selection from response generation. This repository's default answerer is deterministic so it works without a model download; an Ollama adapter is included separately."
                ),
                code(
                    """
from feedback_intelligence.rag import answer

response = answer("Why should outliers be investigated before removal?", limit=3)
print(response["answer"])
response["evidence"]
"""
                ),
                markdown(
                    "Chroma and Qdrant examples show how the same pattern can move to a dedicated vector database. Production retrieval also needs authorization, versioned embeddings, relevance evaluation, observability and source retention rules."
                ),
            ],
        ),
    )

    execute(
        "05_opencv_image_quality.ipynb",
        make(
            "OpenCV image-quality analysis",
            [
                markdown(
                    "Computer vision is not limited to neural networks. Classical OpenCV measurements can cheaply flag exposure, contrast, blur and noise before an expensive model is considered."
                ),
                code(
                    """
from pathlib import Path
import pandas as pd
from feedback_intelligence.cv_quality import analyze_directory

results = analyze_directory()
quality = pd.DataFrame(results)
quality[["file", "brightness", "contrast", "laplacian_variance", "noise_score", "blurred"]]
"""
                ),
                code(
                    """
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
quality.sort_values("laplacian_variance").plot.barh(x="file", y="laplacian_variance", legend=False, ax=ax)
ax.set(title="Sharpness proxy: variance of the Laplacian", xlabel="larger generally means sharper", ylabel="")
plt.show()
"""
                ),
                markdown(
                    "Stable Diffusion serves a different purpose: it generates images from latent representations. A local, opt-in example is included, but model weights and generated images are not treated as dataset evidence."
                ),
            ],
        ),
    )


if __name__ == "__main__":
    main()

# Machine-learning methodology

The project trains a priority classifier and resolution-time regressor, plus K-means clusters and Isolation Forest anomalies. The split occurs before preprocessing is fitted. Text vectorization, categorical encoding and scaling live inside each pipeline.

Classification reports accuracy, macro precision, macro recall, macro-F1, a confusion matrix and per-class results. Regression reports MAE, RMSE and R-squared. Accuracy alone can be misleading on an imbalanced target because a model may ignore a rare class.

Simple baselines are deliberate: they are faster to reproduce and easier to inspect than a large neural network. The optional examples demonstrate Keras, TensorFlow, PyTorch and XGBoost without pretending complexity automatically improves quality.

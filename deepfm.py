import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.preprocessing import StandardScaler
import pandas as pd
import math

def round_up_to_next_digit(num):
    # Find the next power of 10
    return 10 ** math.ceil(math.log10(num))

# Helper function to detect numerical and categorical columns
def detect_column_types(X, threshold = 2):
    categorical_cols = []
    numerical_cols = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            if X[col].nunique() <= threshold:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        elif pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    return categorical_cols, numerical_cols

# DeepFM Model for categorical features
class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim, deep_layers):
        super(DeepFM, self).__init__()
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim

        # Embedding layers for FM and deep component
        self.embeddings = nn.ModuleList(
            [nn.Embedding(field_dim, embed_dim) for field_dim in field_dims]
        )
        self.linear = nn.ModuleList(
            [nn.Embedding(field_dim, 1) for field_dim in field_dims]
        )

        # Deep component
        input_dim = self.num_fields * embed_dim
        layers = []
        for layer_size in deep_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ReLU())
            input_dim = layer_size
        self.deep_layers = nn.Sequential(*layers)

        # Final layer
        self.final_layer = nn.Linear(2 + deep_layers[-1], 1)

    def forward(self, x):
        linear_out = sum(emb(x[:, i]) for i, emb in enumerate(self.linear))

        embeds = torch.stack(
            [emb(x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )
        sum_of_embeds = torch.sum(embeds, dim=1)
        square_of_sum = sum_of_embeds ** 2
        sum_of_square = torch.sum(embeds ** 2, dim=1)
        second_order_out = 0.5 * (square_of_sum - sum_of_square).sum(
            dim=1, keepdim=True
        )

        deep_input = embeds.view(embeds.size(0), -1)
        deep_out = self.deep_layers(deep_input)

        concat_out = torch.cat([linear_out, second_order_out, deep_out], dim=1)
        output = self.final_layer(concat_out)
        return torch.sigmoid(output)

# DNN Model for numerical features
class DNN(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32]):
        super(DNN, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dnn(x)

# Combined model: DeepFM for categorical features and DNN for numerical features
class DeepFM_DNN_Model(nn.Module):
    def __init__(self, categorical_field_dims, numerical_input_dim, embed_dim, deep_layers, dnn_hidden_layers):
        super(DeepFM_DNN_Model, self).__init__()
        
        # DeepFM component for categorical features
        self.deepfm = DeepFM(categorical_field_dims, embed_dim, deep_layers)
        
        # DNN component for numerical features
        self.dnn = DNN(numerical_input_dim, hidden_layers=dnn_hidden_layers)
        
        # Final layer to combine both outputs
        total_input_dim = dnn_hidden_layers[-1] + 1  # +1 for the linear part
        self.final_layer = nn.Linear(total_input_dim, 1)
    
    def forward(self, categorical_data, numerical_data):
        # Process categorical data through DeepFM
        cat_out = self.deepfm(categorical_data)
        
        # Process numerical data through DNN
        numerical_out = self.dnn(numerical_data)

        #print(cat_out.shape)
        #print(numerical_out.shape)
        
        # Combine all outputs
        combined_out = torch.cat([numerical_out, cat_out], dim = 1)
        
        # Final output
        output = self.final_layer(combined_out)
        return torch.sigmoid(output).squeeze(1)

# Custom Sklearn-like classifier
class DeepFMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        embed_dim = 128,
        deep_layers=(200, 200, 200),
        dnn_layers=(200, 200, 200),
        lr = 0.001,
        epochs = 50,
        categorical_field_dims = None,
    ):
        self.embed_dim = embed_dim
        self.deep_layers = deep_layers
        self.dnn_layers = dnn_layers
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.categorical_field_dims = categorical_field_dims
        self.categorical_cols = None
        self.numerical_cols = None

    def fit(self, X, y):
        # Detect categorical and numerical columns and store them
        self.categorical_cols, self.numerical_cols = detect_column_types(X)

        # Label encode categorical features
        X = X.copy()
        for col in self.categorical_cols:
            X[col] = pd.factorize(X[col])[0]

        # Prepare categorical and numerical columns
        if self.categorical_field_dims is None:
            self.categorical_field_dims = [
                #round_up_to_next_digit
                #round_up_to_next_digit((len(np.unique(X[col])) + 1) * 2) for col in self.categorical_cols
                (len(np.unique(X[col])) + 1) * 2 for col in self.categorical_cols
            ]
        X_categorical = X[self.categorical_cols]
        X_numerical = X[self.numerical_cols]

        # Convert numerical data to numpy
        X_numerical = X_numerical.to_numpy()

        # Initialize the model
        self.model = DeepFM_DNN_Model(
            self.categorical_field_dims,
            X_numerical.shape[1],
            self.embed_dim,
            self.deep_layers,
            self.dnn_layers,
        )
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Convert to tensors
        X_cat_tensor = torch.tensor(X_categorical.values, dtype=torch.long)
        X_num_tensor = torch.tensor(X_numerical, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_cat_tensor, X_num_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def _prepare_data(self, X):
        # Ensure categorical and numerical columns have been stored
        if self.categorical_cols is None or self.numerical_cols is None:
            raise ValueError("The model must be fitted before calling predict or predict_proba.")

        # Label encode categorical features
        X = X.copy()
        for col in self.categorical_cols:
            X[col] = pd.factorize(X[col])[0]

        # Prepare categorical and numerical columns
        X_categorical = X[self.categorical_cols]
        X_numerical = X[self.numerical_cols]
        X_numerical = X_numerical.to_numpy()

        # Convert to tensors
        X_cat_tensor = torch.tensor(X_categorical.values, dtype=torch.long)
        X_num_tensor = torch.tensor(X_numerical, dtype=torch.float32)

        return X_cat_tensor, X_num_tensor

    def predict_proba(self, X):
        self.model.eval()

        # Prepare data
        X_cat_tensor, X_num_tensor = self._prepare_data(X)

        with torch.no_grad():
            proba = self.model(X_cat_tensor, X_num_tensor).numpy()
        return np.stack([1 - proba, proba], axis=1)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        # Calculate accuracy
        y_pred = self.predict(X)
        return (y_pred == y).mean()



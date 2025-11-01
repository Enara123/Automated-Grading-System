import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Dataset Class
class CodeDataset(Dataset):
    def __init__(self, features, targets, difficulty):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
        self.difficulty = torch.tensor(difficulty, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.difficulty[idx]


# Model Definition
class GradingNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # output in 0-1
        )

    def forward(self, x):
        return self.net(x)

# Weighted MSE Loss
def weighted_mse_loss(preds, targets, difficulty):
    return ((preds - targets) ** 2 * difficulty).mean()

# Training Function
def train_model(csv_path, epochs=20, batch_size=32, lr=1e-3):
    # Load data
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["submission_id", "grade"]).values
    y = df["grade"].values
    difficulty = df["problem_difficulty"].values

    # Train-test split
    X_train, X_test, y_train, y_test, diff_train, diff_test = train_test_split(
        X, y, difficulty, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Scale grades 0-1
    y_train_scaled = y_train / 100.0
    y_test_scaled = y_test / 100.0

    # Create datasets and loaders
    train_dataset = CodeDataset(X_train, y_train_scaled, diff_train)
    test_dataset = CodeDataset(X_test, y_test_scaled, diff_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = GradingNN(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets, batch_difficulty in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            batch_difficulty = batch_difficulty.to(device)

            optimizer.zero_grad()
            preds = model(batch_features)
            loss = weighted_mse_loss(preds, batch_targets, batch_difficulty)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_features.size(0)
        train_loss /= len(train_loader.dataset)

        # Test loss
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets, batch_difficulty in test_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                batch_difficulty = batch_difficulty.to(device)

                preds = model(batch_features)
                loss = weighted_mse_loss(preds, batch_targets, batch_difficulty)
                test_loss += loss.item() * batch_features.size(0)
        test_loss /= len(test_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        

    # Evaluation metrics
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch_features, batch_targets, _ in test_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            preds = model(batch_features) * 100  # back to 0-100
            y_pred.extend(preds.cpu().numpy().flatten())
            y_true.extend(batch_targets.cpu().numpy().flatten() * 100)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # Scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([0, 100], [0, 100], 'r--')
    plt.xlabel("True Grades")
    plt.ylabel("Predicted Grades")
    plt.title("Predicted vs True Grades")
    plt.show()

    # Save model & scaler
    torch.save(model.state_dict(), "../model/grading_model.pth")
    joblib.dump(scaler, "../model/scaler.pkl")
    print("Model and scaler saved!")

    return model, scaler, device


# Prediction Function
# def prediction(features, trained_model, scaler, device):
#     all_features = scaler.transform([features])

#     input_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
#     with torch.no_grad():
#         pred_scaled = trained_model(input_tensor).cpu().numpy()[0][0]

#     return pred_scaled * 100

def prediction(features, trained_model, scaler, device):

    # 4️⃣ Check feature size against model input
    expected_dim = trained_model.net[0].in_features
    if features.shape[0] != expected_dim:
        raise ValueError(
            f"Feature size mismatch! Expected {expected_dim}, got {features.shape[0]}"
        )

    print("Combined features shape:", features.shape)
    print("Combined features values (first 10):", features[:10])

    # 5️⃣ Scale features
    combined_scaled = scaler.transform([features])
    print("Scaled features (first 10):", combined_scaled[0][:10])

    # 6️⃣ Convert to tensor
    input_tensor = torch.tensor(combined_scaled, dtype=torch.float32).to(device)
    print("Input tensor shape:", input_tensor.shape)

    # 7️⃣ Predict with model
    trained_model.eval()
    with torch.no_grad():
        pred_scaled = trained_model(input_tensor).cpu().numpy()[0][0]

    print("Raw model output (0-1):", pred_scaled)

    # 8️⃣ Scale to 0–100
    final_grade = round(pred_scaled * 100)
    print("Final predicted grade (0-100):", final_grade)

    return final_grade
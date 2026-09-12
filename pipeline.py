import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# ---- Preprocessing ----

df = pd.read_csv('health_lifestyle_with_target.csv')
df['gender_num'] = (df['gender'] == 'Male').astype(int)

feature_cols = ['age', 'gender_num', 'bmi', 'daily_steps', 'sleep_hours', 'smoker', 'alcohol']
X = df[feature_cols]
y = df['expected_death_age']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Scaled row example:", X_train_scaled[0])

# ---- Tensors ----

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# ---- DataLoaders ----

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for batch_X, batch_y in train_loader:
    print("Batch X shape:", batch_X.shape)
    print("Batch y shape:", batch_y.shape)
    break

# ---- Model architecture ----

class MortalityNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = MortalityNet(input_size=7)
print(model)

# ---- Loss function and optimizer ----

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---- Training loop ----

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    train_loss /= len(train_dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item() * batch_X.size(0)
    val_loss /= len(val_dataset)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {train_loss:.3f} — Val Loss: {val_loss:.3f}")

torch.save(model.state_dict(), 'mortality_model.pt')
print("Model saved: mortality_model.pt")
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- Evaluar la red neuronal en el set de TEST (nunca visto) ----

model.eval()
with torch.no_grad():
    nn_test_preds = model(X_test_tensor).numpy().flatten()

y_test_np = y_test_tensor.numpy().flatten()
nn_mae = mean_absolute_error(y_test_np, nn_test_preds)
nn_rmse = np.sqrt(mean_squared_error(y_test_np, nn_test_preds))

# ---- Baseline: Regresión Lineal (sklearn, sin PyTorch) ----

lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
lin_test_preds = lin_model.predict(X_test_scaled)

lin_mae = mean_absolute_error(y_test, lin_test_preds)
lin_rmse = np.sqrt(mean_squared_error(y_test, lin_test_preds))

# ---- Comparación ----

print("\n--- Resultados en TEST set ---")
print(f"Red Neuronal   — MAE: {nn_mae:.3f} años — RMSE: {nn_rmse:.3f} años")
print(f"Regresión Lineal — MAE: {lin_mae:.3f} años — RMSE: {lin_rmse:.3f} años")
# ---- Paso 8: Clasificación de riesgo ----

qx_table = pd.read_csv('vbt_2015_ultimate_qx.csv', index_col=0)
qx_table.index.name = 'age'
age_end = qx_table.index.max()

def baseline_expected_age(age, gender, smoker):
    """Esperanza de vida SOLO por edad/sexo/fumador, sin ajustes de estilo de vida (multiplicador=1)."""
    col = f"{'male' if gender=='Male' else 'female'}_{'smoker' if smoker==1 else 'nonsmoker'}"
    age = min(int(age), age_end)
    survival_prob = 1.0
    e_x = 0.0
    for future_age in range(age, age_end):
        qx = qx_table.loc[future_age, col]
        px = 1 - qx
        e_x += survival_prob
        survival_prob *= px
        if survival_prob < 0.0001:
            break
    return age + e_x

# Recuperar las filas originales correspondientes al test set
test_df = df.loc[X_test.index].copy()
test_df['baseline_expected_age'] = test_df.apply(
    lambda row: baseline_expected_age(row['age'], row['gender'], row['smoker']), axis=1
)
test_df['nn_predicted_age'] = nn_test_preds
test_df['diff_years'] = test_df['nn_predicted_age'] - test_df['baseline_expected_age']

def classify_risk(diff):
    if diff >= 2:
        return 'Preferred'
    elif diff <= -2:
        return 'Substandard'
    else:
        return 'Standard'

test_df['risk_class'] = test_df['diff_years'].apply(classify_risk)

print("\n--- Risk classification (test set) ---")
print(test_df['risk_class'].value_counts())
print("\nSample:")
print(test_df[['age','gender','smoker','bmi','daily_steps','sleep_hours','alcohol',
               'baseline_expected_age','nn_predicted_age','diff_years','risk_class']].sample(8, random_state=1).to_string())
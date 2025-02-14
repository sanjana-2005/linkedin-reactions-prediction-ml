# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib
import os

# Specify the directory to save the models
save_dir = 'C:/Users/shalu/Downloads/'

# Load dataset with correct encoding
file_path = 'C:/Users/shalu/Downloads/archive (6)/company_data.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Handling missing values and data type conversion for 'followers'
data['followers'] = data['followers'].str.replace(',', '', regex=False)
data['followers'] = pd.to_numeric(data['followers'], errors='coerce')
data['followers'].fillna(data['followers'].median(), inplace=True)

# Impute categorical columns with mode
for col in ["headline", "location", "content", "media_type"]:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Drop columns with high percentages of missing values
data.drop(columns=["views", "votes"], inplace=True)

# Handle 'connections' column if it exists
if 'connections' in data.columns:
    data['connections'] = data['connections'].str.replace(',', '').astype(float)

# Outlier Detection and Removal using IQR
def whisker(col):
    q1, q3 = np.percentile(col, [25, 75])
    iqr = q3 - q1
    lw = q1 - (1.5 * iqr)
    uw = q3 + (1.5 * iqr)
    return lw, uw

for col in ['num_hashtags', 'reactions', 'comments', 'followers']:
    lw, uw = whisker(data[col])
    data[col] = np.clip(data[col], lw, uw)

# Feature Engineering: Create new interaction terms and transformations
data['followers_connections_interaction'] = data['followers'] * data['connections']
data['comments_hashtags_interaction'] = data['comments'] * data['num_hashtags']
data['followers_sq'] = data['followers'] ** 2
data['log_comments'] = np.log1p(data['comments'])
data['interaction_advanced'] = data['followers'] * data['num_hashtags'] * data['comments']
data['log_followers'] = np.log1p(data['followers'])
data['log_reactions'] = np.log1p(data['reactions'])

# Define features (X) and target (y) with new features
X = data[['headline', 'location', 'log_followers', 'connections', 'media_type', 
           'num_hashtags', 'comments', 'followers_connections_interaction', 
           'comments_hashtags_interaction', 'followers_sq', 'log_comments', 'interaction_advanced']]
y = data['log_reactions']  # Log-transformed target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode categorical features and save encoders
label_encoders = {}
for col in ['headline', 'location', 'media_type']:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le
    # Save each encoder in the specified directory
    encoder_path = os.path.join(save_dir, f'{col}_encoder.pkl')
    joblib.dump(le, encoder_path)
    print(f"Encoder for {col} saved at {encoder_path}")

# Preprocessing steps
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[('num', RobustScaler(), numeric_features)]
)

# Define SVR Model Pipeline
svr_model = SVR(kernel='rbf')
pipeline_svr = Pipeline(steps=[('preprocessor', preprocessor), 
                               ('pca', PCA(n_components=8)), 
                               ('svr', svr_model)])

# Train the model and save
pipeline_svr.fit(X_train, y_train)

# Save the model
model_path = os.path.join(save_dir, 'svr_pipeline_model1.pkl')
joblib.dump(pipeline_svr, model_path)
print(f"Model saved at {model_path}")

# Save the RobustScaler
scaler = pipeline_svr.named_steps['preprocessor'].transformers_[0][1]
scaler_path = os.path.join(save_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved at {scaler_path}")

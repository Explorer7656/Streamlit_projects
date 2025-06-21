
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# --- Load the data ---
df = pd.read_csv('personality_dataset.csv')
df.dropna(inplace=True)

# --- Label Encoding for categorical features ---
le = LabelEncoder()
df['Drained_after_socializing']=le.fit_transform(df['Drained_after_socializing'])
df['Stage_fear']=le.fit_transform(df['Stage_fear'])
df['Personality']=le.fit_transform(df['Personality'])
y=df['Personality']
x=df[['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside',
      'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']]

# --- Split the data into training and testing sets ---
train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)

# --- Define the preprocessor ---
# We'll use a ColumnTransformer to handle categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), x.select_dtypes(include=['float64', 'int64']).columns),  # Scale numerical features
        ('cat', 'passthrough', ['Drained_after_socializing', 'Stage_fear'])  # Pass through categorical features as-is
    ])

# --- Define the pipeline ---
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=0))
])

# --- Hyperparameter Grid Search ---
param_grid = {
    'classifier__n_estimators': [10, 100, 250, 500],
    'classifier__max_features': [5, 10, 100, 250],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# --- Grid Search for hyperparameter tuning ---
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# --- Fit the model ---
grid_search.fit(train_X, train_y)

# --- Best Model and Hyperparameters ---
best_model = grid_search.best_estimator_
print(f"Best hyperparameters: {grid_search.best_params_}")

# --- Evaluate the model ---
prediction = best_model.predict(test_X)
error = accuracy_score(test_y, prediction)
print(f"Accuracy of model: {error}")

# --- Serialize the best model ---
filename = 'introvert_extrovert.pkl'
with open(filename, 'wb') as file:
    pickle.dump(best_model, file)

print(f'Model saved as {filename}')
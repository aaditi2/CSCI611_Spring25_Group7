import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

def create_keras_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

openml.config.cache_directory = './openml_cache'
datasets_df = openml.datasets.list_datasets(output_format='dataframe')
filtered = datasets_df[
    (datasets_df['NumberOfClasses'] > 1) &
    (datasets_df['NumberOfInstances'] < 10000) &
    (datasets_df['NumberOfFeatures'] < 100)
]

results = []
clean_count = 0
max_needed = 50 

for i, row in filtered.iterrows():
    if clean_count >= max_needed:
        break

    try:
        dataset = openml.datasets.get_dataset(row['did'])
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

        X = X.select_dtypes(include=[np.number]).fillna(0)
        y = y[:len(X)]

        if len(X) < 50 or len(np.unique(y)) < 2:
            continue

        # Meta-feature labeling
        num_instances = X.shape[0]
        num_features = X.shape[1]
        num_classes = len(np.unique(y))
        class_counts = np.bincount(pd.factorize(y)[0])
        imbalance_ratio = max(class_counts) / min(class_counts) if min(class_counts) > 0 else 1
        feature_instance_ratio = num_features / num_instances

        size_label = 'small' if num_instances < 500 else 'medium' if num_instances < 2000 else 'large'
        balance_label = 'imbalanced' if imbalance_ratio > 1.5 else 'balanced'
        density_label = 'high-dimensional' if feature_instance_ratio > 0.1 else 'low-dimensional'
        meta_label = f"{size_label}-{balance_label}-{density_label}"

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)
        y_train_enc, uniques = pd.factorize(y_train)
        y_test_enc = np.array([np.where(uniques == val)[0][0] for val in y_test])

        # TabNet
        tabnet = TabNetClassifier(verbose=0)
        tabnet.fit(
            X_train, y_train_enc,
            eval_set=[(X_test, y_test_enc)],
            eval_metric=['accuracy'],
            max_epochs=100, patience=10,
        )
        y_pred_tabnet = tabnet.predict(X_test)
        acc_tabnet = accuracy_score(y_test_enc, y_pred_tabnet)
        results.append({
            'Dataset': row['name'],
            'Model': 'TabNet',
            'Accuracy': round(acc_tabnet, 4),
            'Meta-Label': meta_label
        })

        # SVM & MLP
        models = {
            "SVM": SVC(),
            "MLP": MLPClassifier(max_iter=300)
        }

        for name, model in models.items():
            model.fit(X_train, y_train_enc)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test_enc, y_pred)
            results.append({
                'Dataset': row['name'],
                'Model': name,
                'Accuracy': round(acc, 4),
                'Meta-Label': meta_label
            })

        # Keras NN
        num_classes = len(np.unique(y_train_enc))
        y_cat_train = to_categorical(y_train_enc, num_classes=num_classes)
        y_cat_test = to_categorical(y_test_enc, num_classes=num_classes)

        keras_model = create_keras_model(X_train.shape[1], num_classes)
        keras_model.fit(
            X_train, y_cat_train,
            validation_split=0.1,
            epochs=15,
            batch_size=32,
            verbose=0,
            callbacks=[EarlyStopping(patience=3)]
        )
        _, keras_acc = keras_model.evaluate(X_test, y_cat_test, verbose=0)
        results.append({
            'Dataset': row['name'],
            'Model': 'KerasNN',
            'Accuracy': round(keras_acc, 4),
            'Meta-Label': meta_label
        })

        clean_count += 1

    except Exception as e:
        continue

# --- Final Display ---
df = pd.DataFrame(results)
print(f"\n✅ Completed {clean_count} datasets × 4 models = {len(df)} results\n")
print(df.head(20).to_string(index=False))

# Best model per dataset
best_models = df.loc[df.groupby("Dataset")["Accuracy"].idxmax()]
print("\n🏆 Best Model per Dataset:")
print(best_models[['Dataset', 'Model', 'Accuracy']].to_string(index=False))

# Grouped accuracy
grouped = df.groupby(['Meta-Label', 'Model'])['Accuracy'].mean().reset_index()
pivoted = grouped.pivot(index='Meta-Label', columns='Model', values='Accuracy').round(4)

print("\n📊 Average Accuracy by Meta-Group and Model:\n")
print(pivoted.fillna("-"))
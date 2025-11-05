""" 
Actividad 2 — Árbol de Decisión
Autor: Fabricio Bermúdez

Script en Python que entrena y evalúa un modelo de Árbol de Decisión 
para la detección de anomalías en logs de HDFS.

Dataset:
- dataset/preprocessed/Event_occurrence_matrix.csv
- dataset/preprocessed/anomaly_label.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def cargar_datos():
    rel_base = os.path.join('dataset', 'preprocessed')
    alt_base = '/mnt/data/dataset_extracted/dataset/preprocessed'

    def try_read(base_path):
        occ = pd.read_csv(os.path.join(base_path, 'Event_occurrence_matrix.csv'))
        lab = pd.read_csv(os.path.join(base_path, 'anomaly_label.csv'))
        return occ, lab

    try:
        df_occ, df_lab = try_read(rel_base)
        base_used = rel_base
    except Exception:
        df_occ, df_lab = try_read(alt_base)
        base_used = alt_base

    print(f"Datos cargados desde: {base_used}")
    return df_occ, df_lab

def preparar_datos(df_occ, df_lab):
    df = df_occ.merge(df_lab, on='BlockId', suffixes=('', '_true'))
    cols_excluir = ['BlockId', 'Label', 'Label_true', 'Type']
    feature_cols = [c for c in df.columns if c not in cols_excluir]
    X = df[feature_cols]
    y = df['Label_true'] if 'Label_true' in df.columns else df['Label']
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y), feature_cols

def entrenar_arbol(X_train, X_test, y_train, y_test, feature_cols):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.6f}\n")
    print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(values_format='d')
    plt.title('Matriz de Confusión — Árbol de Decisión')
    plt.show()

    # Importancia de variables
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 características más importantes:")
    print(importances.head(10))
    importances.head(10).plot(kind='bar')
    plt.title('Top 10 Importancias — Árbol de Decisión')
    plt.show()

    # Árbol reducido
    small_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    small_tree.fit(X_train, y_train)
    print("\nReglas del árbol reducido (max_depth=4):")
    print(export_text(small_tree, feature_names=feature_cols))
    plt.figure(figsize=(14, 8))
    plot_tree(small_tree, feature_names=feature_cols, class_names=small_tree.classes_, filled=False)
    plt.title("Árbol de Decisión (max_depth=4)")
    plt.show()

if __name__ == "__main__":
    df_occ, df_lab = cargar_datos()
    (X_train, X_test, y_train, y_test), feature_cols = preparar_datos(df_occ, df_lab)
    entrenar_arbol(X_train, X_test, y_train, y_test, feature_cols)

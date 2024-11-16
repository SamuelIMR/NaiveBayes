import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict


def naive_bayes_train(data, target_col):
    
    class_probs = data[target_col].value_counts(normalize=True).to_dict()

    cond_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for _, row in data.iterrows():
        target_value = row[target_col]
        for col in data.columns:
            if col != target_col:
                cond_probs[target_value][col][row[col]] += 1
    

    for target_value in cond_probs:
        for col in cond_probs[target_value]:
            total_count = sum(cond_probs[target_value][col].values()) + len(cond_probs[target_value][col])  
            for val in cond_probs[target_value][col]:
                cond_probs[target_value][col][val] /= total_count
    
    return class_probs, cond_probs


def naive_bayes_predict(instance, class_probs, cond_probs, target_col):
    max_prob = -float('inf')
    best_class = None
    
    for class_value in class_probs:
        prob = np.log(class_probs[class_value])  
        for col, value in instance.items():
            if col != target_col:
                
                prob += np.log(cond_probs[class_value][col].get(value, 1e-6))
        
        if prob > max_prob:
            max_prob = prob
            best_class = class_value
    
    return best_class

def get_class_probabilities(instance, class_probs, cond_probs, target_col):
    prob_per_class = {}
    
    for class_value in class_probs:
        prob = np.log(class_probs[class_value])  
        for col, value in instance.items():
            if col != target_col:
                prob += np.log(cond_probs[class_value][col].get(value, 1e-6))
        
        prob_per_class[class_value] = prob
    
    return prob_per_class

def print_conditional_probabilities(cond_probs):
    for class_value in cond_probs:
        print(f"\nProbabilidades condicionales para la clase '{class_value}':")
        for attribute in cond_probs[class_value]:
            print(f"  Atributo '{attribute}':")
            for value, prob in cond_probs[class_value][attribute].items():
                print(f"    Valor '{value}': {prob:.4f}")


df = pd.read_csv('Clima.csv')

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

target_column = "Play"

class_probs, cond_probs = naive_bayes_train(train_df, target_column)

# Mostrar probabilidades calculadas para el 70% del entrenamiento
print("Probabilidades de clase (70% entrenamiento):")
for class_value, prob in class_probs.items():
    print(f"  Clase '{class_value}': {prob:.4f}")


print_conditional_probabilities(cond_probs)


new_instance = {
    "Outlook": "Sunny", 
    "Temp": "Hot", 
    "Humidity": "High", 
    "Windy": "False"
}

# Ahora repite el proceso con el 30% de prueba
test_class_probs, test_cond_probs = naive_bayes_train(test_df, target_column)

print("\nProbabilidades de clase (30% prueba):")
for class_value, prob in test_class_probs.items():
    print(f"  Clase '{class_value}': {prob:.4f}")


print_conditional_probabilities(test_cond_probs)

test_predicted_class = naive_bayes_predict(new_instance, test_class_probs, test_cond_probs, target_column)
print("\nClase predicha para la nueva instancia (datos de prueba):", test_predicted_class)

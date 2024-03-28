import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

# Define a vocabulary of ingredients and recipes
ingredients_vocab = ['chicken', 'tomatoes', 'basil', 'garlic', 'salt', 'pepper', 'olive oil']
recipes_vocab = ['Chicken with Tomato Sauce', 'Tomato Basil Soup', 'Garlic Chicken', 'Pepper Chicken']

# Convert ingredients and recipes to numerical representation
ingredient_to_int = {ingredient: i for i, ingredient in enumerate(ingredients_vocab)}
recipe_to_int = {recipe: i for i, recipe in enumerate(recipes_vocab)}

# Define the maximum sequence length for ingredients and recipes
max_ingredients_length = max([len(recipe.split()) for recipe in recipes_vocab])

# Create training data
X = []
y = []

for recipe in recipes_vocab:
    recipe_int = [recipe_to_int[recipe]]
    X.append(recipe_int * max_ingredients_length)  # Pad with recipe index for each ingredient
    y.append(recipe_int)

X = np.array(X)
y = np.array(y)

# Define the model architecture
model = Sequential([
    Embedding(len(ingredients_vocab), 10),
    LSTM(50),
    Dense(len(recipes_vocab), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(X, y, epochs=100, batch_size=1)

# Function to get user input
def get_user_input():
    user_input = input("Enter ingredients (comma-separated): ")
    return [ingredient.strip() for ingredient in user_input.split(",")]

# Generate a recipe using user input
# Generate a recipe using user input
def generate_recipe_from_user_input():
    ingredients = get_user_input()
    ingredients_int = [ingredient_to_int[ingredient] for ingredient in ingredients if ingredient in ingredient_to_int]
    ingredients_int += [0] * (max_ingredients_length - len(ingredients_int))
    ingredients_int = np.array([ingredients_int])
    predicted_probabilities = model.predict(ingredients_int)
    recipe_int = np.argmax(predicted_probabilities, axis=1)
    recipe = recipes_vocab[recipe_int[0]]
    return recipe


# Test the recipe generator
print("Welcome to Recipe Generator!")
while True:
    recipe = generate_recipe_from_user_input()
    print("Generated Recipe:", recipe)
    choice = input("Do you want to generate another recipe? (yes/no): ")
    if choice.lower() != 'yes':
        break

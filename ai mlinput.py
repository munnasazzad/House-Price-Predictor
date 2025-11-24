import random

# Sample dataset (prices in Indian Rupees)
houses = [
    {'size': 1200, 'location': 'suburban', 'age': 10, 'price': 5000000},
    {'size': 1500, 'location': 'urban', 'age': 5, 'price': 7500000},
    {'size': 1800, 'location': 'urban', 'age': 2, 'price': 9000000},
    {'size': 2000, 'location': 'suburban', 'age': 15, 'price': 6000000},
    {'size': 2500, 'location': 'urban', 'age': 3, 'price': 12000000},
    {'size': 1000, 'location': 'rural', 'age': 20, 'price': 3500000},
    {'size': 1600, 'location': 'suburban', 'age': 8, 'price': 6500000},
    {'size': 2200, 'location': 'urban', 'age': 1, 'price': 11000000},
    {'size': 1400, 'location': 'rural', 'age': 25, 'price': 4000000},
    {'size': 1900, 'location': 'suburban', 'age': 12, 'price': 7000000},
    {'size': 2300, 'location': 'urban', 'age': 4, 'price': 10500000},
    {'size': 1100, 'location': 'rural', 'age': 18, 'price': 3800000},
    {'size': 1700, 'location': 'suburban', 'age': 7, 'price': 7200000},
    {'size': 2100, 'location': 'urban', 'age': 6, 'price': 10000000},
    {'size': 2400, 'location': 'urban', 'age': 2, 'price': 11500000},
    {'size': 1300, 'location': 'rural', 'age': 22, 'price': 4200000},
    {'size': 1950, 'location': 'suburban', 'age': 11, 'price': 7500000},
    {'size': 2600, 'location': 'urban', 'age': 3, 'price': 13000000},
    {'size': 1450, 'location': 'rural', 'age': 16, 'price': 4500000},
    {'size': 2050, 'location': 'suburban', 'age': 9, 'price': 8000000},
]

# Encode locations as numbers
location_map = {'rural': 0, 'suburban': 1, 'urban': 2}

def encode_location(location):
    return location_map.get(location, 1)

def decode_location(code):
    for loc, val in location_map.items():
        if val == code:
            return loc
    return 'suburban'

# Prepare training data
def prepare_data(houses):
    X = []
    y = []
    for house in houses:
        features = [
            house['size'],
            encode_location(house['location']),
            house['age']
        ]
        X.append(features)
        y.append(house['price'])
    return X, y

# Calculate mean
def mean(values):
    return sum(values) / len(values)

# Simple linear regression using gradient descent
def train_model(X, y, learning_rate=0.0001, iterations=1000):
    n_samples = len(X)
    n_features = len(X[0])
    
    # Initialize weights and bias
    weights = [0.0] * n_features
    bias = 0.0
    
    # Normalize features for better training
    X_means = [mean([x[i] for x in X]) for i in range(n_features)]
    X_stds = [max((sum([(x[i] - X_means[i])**2 for x in X]) / n_samples)**0.5, 0.001) for i in range(n_features)]
    
    X_normalized = [[(x[i] - X_means[i]) / X_stds[i] for i in range(n_features)] for x in X]
    
    # Gradient descent
    for iteration in range(iterations):
        # Calculate predictions
        predictions = [sum(w * x[i] for i, w in enumerate(weights)) + bias for x in X_normalized]
        
        # Calculate gradients
        errors = [pred - actual for pred, actual in zip(predictions, y)]
        
        # Update weights
        for i in range(n_features):
            gradient = sum(errors[j] * X_normalized[j][i] for j in range(n_samples)) / n_samples
            weights[i] -= learning_rate * gradient
        
        # Update bias
        bias_gradient = sum(errors) / n_samples
        bias -= learning_rate * bias_gradient
    
    return weights, bias, X_means, X_stds

# Predict house price
def predict(size, location, age, weights, bias, X_means, X_stds):
    location_code = encode_location(location)
    features = [size, location_code, age]
    
    # Normalize features
    normalized = [(features[i] - X_means[i]) / X_stds[i] for i in range(len(features))]
    
    # Calculate prediction
    prediction = sum(w * f for w, f in zip(weights, normalized)) + bias
    return prediction

# Train the model
print("=" * 50)
print("TRAINING MODEL...")
print("=" * 50)

X_train, y_train = prepare_data(houses)
weights, bias, X_means, X_stds = train_model(X_train, y_train, learning_rate=0.0001, iterations=2000)

print("Training complete!")
print()

# Calculate model accuracy
predictions = []
for house in houses:
    pred = predict(house['size'], house['location'], house['age'], weights, bias, X_means, X_stds)
    predictions.append(pred)

errors = [abs(pred - house['price']) for pred, house in zip(predictions, houses)]
avg_error = mean(errors)

print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Average Prediction Error: ₹{avg_error:,.2f}")
print()

print("=" * 50)
print("FEATURE WEIGHTS")
print("=" * 50)
print(f"Size weight: {weights[0]:.4f}")
print(f"Location weight: {weights[1]:.4f}")
print(f"Age weight: {weights[2]:.4f}")
print(f"Bias: {bias:.2f}")
print()

# Example predictions
print("=" * 50)
print("EXAMPLE PREDICTIONS")
print("=" * 50)

test_cases = [
    (1800, 'urban', 5),
    (1500, 'suburban', 10),
    (2200, 'rural', 3),
    (2000, 'urban', 7),
]

for size, location, age in test_cases:
    predicted_price = predict(size, location, age, weights, bias, X_means, X_stds)
    print(f"{size} sqft, {location:10s}, {age:2d} years → ₹{predicted_price:,.2f}")

print()
print("=" * 50)
print("PREDICT YOUR HOUSE PRICE")
print("=" * 50)

while True:
    try:
        # Get user input
        print("\nEnter house details (or type 'quit' to exit):")
        
        size_input = input("Size (square feet): ").strip()
        if size_input.lower() == 'quit':
            break
        size = int(size_input)
        
        location = input("Location (urban/suburban/rural): ").strip().lower()
        if location == 'quit':
            break
        if location not in ['urban', 'suburban', 'rural']:
            print("Invalid location! Please use: urban, suburban, or rural")
            continue
    
        age_input = input("Age (years): ").strip()
        if age_input.lower() == 'quit':
            break
        age = int(age_input)
        
        # Make prediction
        predicted_price = predict(size, location, age, weights, bias, X_means, X_stds)
        
        print("\n" + "=" * 50)
        print(f"PREDICTED PRICE: ₹{predicted_price:,.2f}")
        print("=" * 50)
        
    except ValueError:
        print("Invalid input! Please enter numbers for size and age.")
    except KeyboardInterrupt:
        print("\n\nExiting...")
        break
print("\nThank you for using the house price predictor!")

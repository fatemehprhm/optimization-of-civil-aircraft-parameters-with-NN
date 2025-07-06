# Aircraft Parameter Optimization Using Neural Networks

## Overview

This project focuses on optimizing initial civil aircraft design parameters in the early stages of aircraft design using neural networks. The system predicts various aircraft characteristics based on passenger capacity and range, providing more accurate estimates than traditional statistical methods.

## Problem Statement

In the early stages of aircraft design, knowledge about the design is very limited and many details are unknown. Traditional statistical models like multiple regression analysis often assume independence between input variables, while in reality, there are significant correlations between these parameters. This project addresses this limitation by using neural networks to capture complex relationships between aircraft parameters.

## Dataset

The project uses a database containing information from **100 passenger aircraft** with takeoff weight above 30 tons, including:
- Tupolev Tu-124
- Aerospatiale SE 210 Caravelle 12
- Fokker F-70
- CRJ 900
- Airbus A320
- And more...

Data sourced from Jane's aircraft books from 1960 onwards.

### Parameters Included

**Input Parameters:**
- Aircraft Range (km)
- Passenger Capacity

**Output Parameters:**
- Takeoff Weight (tons)
- Empty Weight (tons)
- Wing Span (m)
- Overall Length (m)
- Overall Height (m)
- Thrust Force (lbs)
- Wing Area (m²)

## Methodology

### Neural Network Architecture
- **Layers**: 4 layers (3 hidden + 1 output)
- **Neurons**: 10, 25, 10, and 7 neurons respectively
- **Activation Function**: Hyperbolic tangent (tanh)
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: 0.01
- **Loss Function**: Mean Squared Error (MSE)

### Training Configuration
- **Data Split**: 85% training, 15% testing
- **Batch Size**: 20
- **Maximum Epochs**: 500
- **Early Stopping**: Patience of 5 epochs
- **Convergence**: Achieved after ~430 iterations

## Key Features

1. **Data Normalization**: Standardization using z-score normalization
2. **Correlation Analysis**: Heat map visualization of parameter relationships
3. **Performance Comparison**: Neural network vs. SVD (Singular Value Decomposition)
4. **Validation**: Real aircraft data validation using Airbus A320

## Results

### A320 Validation Results
| Parameter | Real Value | Predicted | Error (%) |
|-----------|------------|-----------|-----------|
| Takeoff Weight (tons) | 73.5 | 77.73 | 5.76% |
| Empty Weight (tons) | 42.1 | 42.79 | 1.64% |
| Wing Area (m²) | 122.4 | 146.44 | 19.64% |
| Thrust (lbs) | 154000 | 153797.4 | 0.38% |
| Length (m) | 37.57 | 41.24 | 9.77% |
| Height (m) | 11.76 | 11.04 | 6.08% |
| Span (m) | 34.09 | 33.51 | 1.70% |

### Performance Comparison
The neural network approach showed superior performance compared to SVD method in most cases, providing more accurate predictions for aircraft design parameters.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Scikit-learn
- NumPy
- Pandas
- Seaborn
- Matplotlib
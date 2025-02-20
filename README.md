# ml-dataset
Various ml model and dataset

## **1. Reinforcement Learning (RL)**
Reinforcement Learning is an ML technique where an **agent** learns by interacting with an **environment** to maximize rewards.  
- **Key Elements**: Agent, Environment, Actions, Rewards, Policy  
- **Applications**: Robotics, Game AI (e.g., AlphaGo), Autonomous Vehicles  

---

## **2. Applications of AI**
AI is used in various domains, including:  
### **Lifestyle**:  
- Virtual assistants (Alexa, Siri, Google Assistant)  
- Smart home automation  
- AI-based personal finance tools  

### **Robotics**:  
- Self-driving cars (Tesla, Waymo)  
- Industrial automation  
- Healthcare robots (surgical robots, AI-driven prosthetics)  

### **Social Media**:  
- AI-powered content recommendation (YouTube, TikTok, Instagram)  
- Fake news detection  
- Chatbots for customer service  

### **Gaming**:  
- AI opponents (e.g., DeepMind's AlphaGo)  
- Procedural content generation  
- AI-based NPCs (non-playable characters)  

---

## **3. Iris Dataset Import & Splitting**
The **Iris dataset** is commonly used for ML classification.  
### **Code to Load and Split Data**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- `test_size=0.2` → 20% of the data is used for testing, 80% for training.  

---

## **4. Data Scaling Techniques**
ML models perform better when input features are **scaled**.  

### **(a) StandardScaler (Standardization)**
- **Formula**:  
  \[
  X_{\text{scaled}} = \frac{X - \mu}{\sigma}
  \]
- **Effect**: Mean = 0, Std Dev = 1  
- **Best for**: Normally distributed data  
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use only transform on test data!
```

### **(b) MinMaxScaler (Normalization)**
- **Formula**:  
  \[
  X_{\text{scaled}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
  \]
- **Effect**: Scales values between **0 and 1**  
- **Best for**: Data without outliers  
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## **5. One-Hot Encoding (OHE)**
One-hot encoding converts categorical values into numerical form.  

### **Example: Encoding "Color" Feature**
| Color | Red | Blue | Green |
|-------|-----|------|-------|
| Red   | 1   | 0    | 0     |
| Blue  | 0   | 1    | 0     |
| Green | 0   | 0    | 1     |

### **Code Implementation**
```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Categorical data
colors = np.array([['Red'], ['Blue'], ['Green'], ['Red']])

# Apply OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded_colors = encoder.fit_transform(colors)

print(encoded_colors)
```

---

## **Summary**
| Concept         | Purpose |
|----------------|---------|
| Reinforcement Learning | Learning through rewards and penalties |
| AI Applications | AI in lifestyle, robotics, social media, gaming |
| Data Splitting | `train_test_split()` for ML datasets |
| StandardScaler | Standardization (mean = 0, std = 1) |
| MinMaxScaler | Normalization (scales between 0 and 1) |
| One-Hot Encoding | Converts categorical data into numerical form |

# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="757" height="917" alt="image" src="https://github.com/user-attachments/assets/11d519f7-1343-4500-9ecc-913582e5b88d" />


## DESIGN STEPS
### STEP 1: 
Load the existing market customer data with features and their corresponding segment labels (A, B, C, D).

### STEP 2: 
Clean the dataset by handling missing values, encoding categorical variables, and normalizing numerical features.


### STEP 3: 
Divide the data into training and testing sets to evaluate the model’s performance.



### STEP 4: 
Initialize the neural network by defining input layer, hidden layers, output layer, activation functions, and loss function.

### STEP 5: 
Train the neural network using the training data by adjusting weights through backpropagation to minimize classification error.


### STEP 6: 
Evaluate the model using the test dataset and measure accuracy, precision, or loss.

### STEP 7:
Use the trained model to classify customers in the new market into segments A, B, C, or D.


## PROGRAM

### Name: R Raihaan Ahmed 

### Register Number: 212224040260

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)
        
    def forward(self, x):
       x=F.relu(self.fc1(x))
       x=F.relu(self.fc2(x))
       x=F.relu(self.fc3(x))
       x=self.fc4(x)
       return x



        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
   model.train()
  for epoch in range(epochs):
    for inputs,labels in train_loader:
      optimizer.zero_grad()
      outputs=model(inputs)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
# Initialize model
model =PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

```
### Dataset Information
<img width="1168" height="291" alt="image" src="https://github.com/user-attachments/assets/500d3687-0e66-495a-be72-a6c394ef28e7" />


### OUTPUT

## Confusion Matrix
<img width="1086" height="609" alt="image" src="https://github.com/user-attachments/assets/003c7956-f238-4c88-9744-1ed987829bca" />

## Classification Report
<img width="492" height="351" alt="image" src="https://github.com/user-attachments/assets/01f2eff4-bba1-4ad1-a6b5-914483193f2e" />


### New Sample Data Prediction
Include your sample input and output here
<img width="278" height="77" alt="image" src="https://github.com/user-attachments/assets/05b09339-8ad5-4580-a730-1cbab9f4bd82" />

## RESULT
The neural network classification model was executed successfully and customer segments (A, B, C, and D) were predicted accurately for the given dataset.

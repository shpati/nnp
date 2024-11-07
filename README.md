# nnp

Neural Network Program (with file input and weight initialization and storage)

  Key Functions:
  - Forward propagation
  - Backpropagation
  - Training with error calculation
  - Weight initialization
  - Fast

  The program will:
  1. Read the parameters for the configuration of the Neural Network from the parameters file. 
     1.1. If the parameters file is not found it asks whether the users wants to input the parameters manually or load the default values. 
     1.2. If the parameters file is found and it has weights included it asks if those weights should be loaded. 
  2. Read the training data from the train file
  3. Train the neural network using that data, if no saved weights are found and loaded.
  4. Show progress during training
  5. Test the network with the test samples from test file and show results. 
     If the test file doesn't exist, falls back to testing with the train file.
  6. Allows user to do predictions / forecasting with the trained neural network.
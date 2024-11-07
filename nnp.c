/////////////////////////////////////////////////////////////////////////////////////////////
//  Neural Network Program (with file input and weight initialization and storage) - Shpati Koleka, MIT Licensed.
//
//  Key Functions:
//  - Forward propagation
//  - Backpropagation
//  - Training with error calculation
//  - Weight initialization
//  - Fast
//
//  The program will:
//  1. Read the parameters for the configuration of the Neural Network from the parameters file. 
//     1.1. If the parameters file is not found it asks whether the users wants to input the parameters manually or load the default values. 
//     1.2. If the parameters file is found and it has weights included it asks if those weights should be loaded. 
//  2. Read the training data from the train file
//  3. Train the neural network using that data, if no saved weights are found and loaded.
//  4. Show progress during training
//  5. Test the network with the test samples from test file and show results. 
//     If the test file doesn't exist, falls back to testing with the train file.
//  6. Allows user to do predictions / forecasting with the trained neural network.
//
/////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

const char* parameters_file = "parameters.txt";
const char* train_file = "train.txt";
const char* test_file = "test.txt";
const char* version = "0.1";

// Default parameters (used if parameters_file is not found)
int INPUT_NODES = 256;
int HIDDEN_NODES = 16;
int OUTPUT_NODES = 10;
double LEARNING_RATE = 0.1;
int MAX_EPOCHS = 10000;
double ERROR_THRESHOLD = 0.0001;
int MAX_SAMPLES = 10000;

typedef struct {
    double** hidden_weights;
    double** output_weights;
    double* hidden_bias;
    double* output_bias;
    double* hidden_activation;
    double* output_activation;
} NeuralNetwork;

// Function prototypes
void read_parameters(const char* filename);
NeuralNetwork* create_network();
void free_network(NeuralNetwork* network);
int file_exists(const char* filename);
int read_data(const char* filename, double** inputs, double** outputs, const char* purpose);
double sigmoid(double x);
double sigmoid_derivative(double x);
void initialize_network(NeuralNetwork* network);
void forward_propagate(NeuralNetwork* network, double* input);
void backpropagate(NeuralNetwork* network, double* input, double* target);
double train(NeuralNetwork* network, double** training_inputs, double** training_outputs, int num_samples);
void test_network(NeuralNetwork* network, double** test_inputs, double** test_outputs, int num_samples);
void free_data(double** data, int num_samples);

// Function to write default parameters to file
void write_default_parameters(const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not create the parameters file\n");
        return;
    }

    fprintf(file, "INPUT_NODES %d\n", INPUT_NODES);
    fprintf(file, "HIDDEN_NODES %d\n", HIDDEN_NODES);
    fprintf(file, "OUTPUT_NODES %d\n", OUTPUT_NODES);
    fprintf(file, "LEARNING_RATE %.2f\n", LEARNING_RATE);
    fprintf(file, "MAX_EPOCHS %d\n", MAX_EPOCHS);
    fprintf(file, "ERROR_THRESHOLD %.6f\n", ERROR_THRESHOLD);
    fprintf(file, "MAX_SAMPLES %d\n\n", MAX_SAMPLES);
    fclose(file);
    printf("\nSaved the default parameters to the %s file.\n", parameters_file);
}

// Function to get user input and write parameters to file
void write_custom_parameters(const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not create the parameters file\n");
        return;
    }

    printf("\nEnter the following parameters (press Enter to use the default value):\n");

    //while (getchar() != '\n');  // Clear the input buffer

    // INPUT_NODES
    printf("INPUT_NODES [Default=%d]: ", INPUT_NODES);
    int input_nodes = INPUT_NODES;
    char buffer[100]; // To read the input
    if (fgets(buffer, sizeof(buffer), stdin)) {
        if (buffer[0] != '\n' && sscanf(buffer, "%d", &input_nodes) != 1) {
            // If the user enters an invalid value, default is used
            input_nodes = INPUT_NODES;
        }
    }
    fprintf(file, "INPUT_NODES %d\n", input_nodes);

    // HIDDEN_NODES
    printf("HIDDEN_NODES [Default=%d]: ", HIDDEN_NODES);
    int hidden_nodes = HIDDEN_NODES;
    if (fgets(buffer, sizeof(buffer), stdin)) {
        if (buffer[0] != '\n' && sscanf(buffer, "%d", &hidden_nodes) != 1) {
            hidden_nodes = HIDDEN_NODES;
        }
    }
    fprintf(file, "HIDDEN_NODES %d\n", hidden_nodes);

    // OUTPUT_NODES
    printf("OUTPUT_NODES [Default=%d]: ", OUTPUT_NODES);
    int output_nodes = OUTPUT_NODES;
    if (fgets(buffer, sizeof(buffer), stdin)) {
        if (buffer[0] != '\n' && sscanf(buffer, "%d", &output_nodes) != 1) {
            output_nodes = OUTPUT_NODES;
        }
    }
    fprintf(file, "OUTPUT_NODES %d\n", output_nodes);

    // LEARNING_RATE
    printf("LEARNING_RATE [Default=%.2f]: ", LEARNING_RATE);
    double learning_rate = LEARNING_RATE;
    if (fgets(buffer, sizeof(buffer), stdin)) {
        if (buffer[0] != '\n' && sscanf(buffer, "%lf", &learning_rate) != 1) {
            learning_rate = LEARNING_RATE;
        }
    }
    fprintf(file, "LEARNING_RATE %.2f\n", learning_rate);

    // MAX_EPOCHS
    printf("MAX_EPOCHS [Default=%d]: ", MAX_EPOCHS);
    int max_epochs = MAX_EPOCHS;
    if (fgets(buffer, sizeof(buffer), stdin)) {
        if (buffer[0] != '\n' && sscanf(buffer, "%d", &max_epochs) != 1) {
            max_epochs = MAX_EPOCHS;
        }
    }
    fprintf(file, "MAX_EPOCHS %d\n", max_epochs);

    // ERROR_THRESHOLD
    printf("ERROR_THRESHOLD [Default=%.6f]: ", ERROR_THRESHOLD);
    double error_threshold = ERROR_THRESHOLD;
    if (fgets(buffer, sizeof(buffer), stdin)) {
        if (buffer[0] != '\n' && sscanf(buffer, "%lf", &error_threshold) != 1) {
            error_threshold = ERROR_THRESHOLD;
        }
    }
    fprintf(file, "ERROR_THRESHOLD %.6f\n", error_threshold);

    // MAX_SAMPLES
    printf("MAX_SAMPLES [Default=%d]: ", MAX_SAMPLES);
    int max_samples = MAX_SAMPLES;
    if (fgets(buffer, sizeof(buffer), stdin)) {
        if (buffer[0] != '\n' && sscanf(buffer, "%d", &max_samples) != 1) {
            max_samples = MAX_SAMPLES;
        }
    }
    fprintf(file, "MAX_SAMPLES %d\n\n", max_samples);

    fclose(file);
    printf("\nParameters file %s created using the custom parameters.\n", parameters_file);
}

// Function to get yes/no input from user
char get_yes_no_input(const char* prompt, char def) {
    char answer[256];
    answer[0] = def;
    printf("%s [Default=%c]: ", prompt, def); 
    fgets(answer, sizeof(answer), stdin); 
    if (answer[0] == 'y' || answer[0] == 'Y') return 'y';
    if (answer[0] == 'n' || answer[0] == 'N') return 'n';
    return def;
}

// Function to read parameters from file
void read_parameters(const char* filename) {
    if (!file_exists(filename)) {
        printf("\nNOTE: The %s file was not found.\n", parameters_file);
        
        if (get_yes_no_input("Do you want to enter parameters manually (y/n)", 'n') == 'y') {
            write_custom_parameters(filename);
        } else {
            write_default_parameters(filename);
        }
        //while (getchar() != '\n');  // Clear the input buffer
    }

    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening %s file. Using default values.\n", parameters_file);
        return;
    }

    char line[256];
    char name[256];
    double value;
    int parameters_found = 0;

    while (fgets(line, sizeof(line), file)) {
        // Skip comments, empty lines, and weights section
        if (line[0] == '#' || line[0] == '\n' || strstr(line, "[TRAINED_WEIGHTS]")) {
            if (strstr(line, "[TRAINED_WEIGHTS]")) break; // Stop at weights section
            continue;
        }
        
        // Remove newline if present
        line[strcspn(line, "\n")] = 0;
        
        if (sscanf(line, "%s %lf", name, &value) == 2) {

            if (strcmp(name, "INPUT_NODES") == 0) {
                INPUT_NODES = (int)value;
                parameters_found += 1;
            }
            else if (strcmp(name, "HIDDEN_NODES") == 0) {
                HIDDEN_NODES = (int)value;
                parameters_found += 10;
            }
            else if (strcmp(name, "OUTPUT_NODES") == 0) {
                OUTPUT_NODES = (int)value;
                parameters_found += 100;
            }
            else if (strcmp(name, "LEARNING_RATE") == 0) {
                LEARNING_RATE = value;
                parameters_found += 1000;
            }
            else if (strcmp(name, "MAX_EPOCHS") == 0) {
                MAX_EPOCHS = (int)value;
                parameters_found += 10000;
            }
            else if (strcmp(name, "ERROR_THRESHOLD") == 0) {
                ERROR_THRESHOLD = value;
                parameters_found += 100000;
            }
            else if (strcmp(name, "MAX_SAMPLES") == 0) {
                MAX_SAMPLES = (int)value;
                parameters_found +=1000000;
            }
        }
    }

    fclose(file);

    // If no parameters were found, write the default ones
    if (parameters_found != 1111111) {
        printf("\nNOTE: The %s file was missing one or more parameters.\n", parameters_file);
        write_default_parameters(filename);
    }
    
    printf("\nParameters loaded:\n");
    printf("INPUT_NODES: %d\n", INPUT_NODES);
    printf("HIDDEN_NODES: %d\n", HIDDEN_NODES);
    printf("OUTPUT_NODES: %d\n", OUTPUT_NODES);
    printf("LEARNING_RATE: %f\n", LEARNING_RATE);
    printf("MAX_EPOCHS: %d\n", MAX_EPOCHS);
    printf("ERROR_THRESHOLD: %f\n", ERROR_THRESHOLD);
    printf("MAX_SAMPLES: %d\n\n", MAX_SAMPLES);
}

// Function to save weights to parameters file
void save_weights(NeuralNetwork* network, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Could not open the parameters file [%s] for reading\n", parameters_file);
        return;
    }

    // Read existing parameters into temporary file
    FILE* temp = tmpfile();
    char line[256];
    int in_weights_section = 0;
    
    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "[TRAINED_WEIGHTS]") != NULL) {
            in_weights_section = 1;
            continue;
        } else if (in_weights_section && line[0] == '[') {
            in_weights_section = 0;
        }
        
        if (!in_weights_section) {
            fputs(line, temp);
        }
    }
    fclose(file);

    // Open file for writing
    file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not open the parameters file [%s] for writing\n", parameters_file);
        return;
    }

    // Copy parameters back
    rewind(temp);
    while (fgets(line, sizeof(line), temp)) {
        fputs(line, file);
    }

    // Write weights section
    fprintf(file, "[TRAINED_WEIGHTS]\n");
    
    // Hidden weights
    fprintf(file, "# Hidden weights\n");
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            fprintf(file, "hidden_weight_%d_%d %.10f\n", i, j, network->hidden_weights[i][j]);
        }
    }
    
    // Hidden bias
    fprintf(file, "# Hidden bias\n");
    for (int i = 0; i < HIDDEN_NODES; i++) {
        fprintf(file, "hidden_bias_%d %.10f\n", i, network->hidden_bias[i]);
    }
    
    // Output weights
    fprintf(file, "# Output weights\n");
    for (int i = 0; i < HIDDEN_NODES; i++) {
        for (int j = 0; j < OUTPUT_NODES; j++) {
            fprintf(file, "output_weight_%d_%d %.10f\n", i, j, network->output_weights[i][j]);
        }
    }
    
    // Output bias
    fprintf(file, "# Output bias\n");
    for (int i = 0; i < OUTPUT_NODES; i++) {
        fprintf(file, "output_bias_%d %.10f\n", i, network->output_bias[i]);
    }

    fclose(file);
    printf("Weights saved to the parameters file %s\n", parameters_file);
}

// Function to check if weights exist in parameters file
int weights_exist(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) return 0;

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "[TRAINED_WEIGHTS]") != NULL) {
            fclose(file);
            return 1;
        }
    }
    
    fclose(file);
    return 0;
}

// Function to load weights from parameters file
int load_weights(NeuralNetwork* network, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) return 0;

    char line[256];
    int in_weights_section = 0;
    int weights_loaded = 0;
    
    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "[TRAINED_WEIGHTS]") != NULL) {
            in_weights_section = 1;
            continue;
        }
        
        if (in_weights_section && line[0] != '#') {
            char name[256];
            double value;
            if (sscanf(line, "%s %lf", name, &value) == 2) {
                if (strncmp(name, "hidden_weight_", 13) == 0) {
                    int i, j;
                    sscanf(name, "hidden_weight_%d_%d", &i, &j);
                    if (i < INPUT_NODES && j < HIDDEN_NODES) {
                        network->hidden_weights[i][j] = value;
                        weights_loaded = 1;
                    }
                }
                else if (strncmp(name, "hidden_bias_", 11) == 0) {
                    int i;
                    sscanf(name, "hidden_bias_%d", &i);
                    if (i < HIDDEN_NODES) {
                        network->hidden_bias[i] = value;
                    }
                }
                else if (strncmp(name, "output_weight_", 13) == 0) {
                    int i, j;
                    sscanf(name, "output_weight_%d_%d", &i, &j);
                    if (i < HIDDEN_NODES && j < OUTPUT_NODES) {
                        network->output_weights[i][j] = value;
                    }
                }
                else if (strncmp(name, "output_bias_", 11) == 0) {
                    int i;
                    sscanf(name, "output_bias_%d", &i);
                    if (i < OUTPUT_NODES) {
                        network->output_bias[i] = value;
                    }
                }
            }
        }
    }
    
    fclose(file);
    return weights_loaded;
}

// Function to allocate memory for the network
NeuralNetwork* create_network() {
    NeuralNetwork* network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Allocate memory for weights and biases
    network->hidden_weights = (double**)malloc(INPUT_NODES * sizeof(double*));
    for (int i = 0; i < INPUT_NODES; i++) {
        network->hidden_weights[i] = (double*)malloc(HIDDEN_NODES * sizeof(double));
    }
    
    network->output_weights = (double**)malloc(HIDDEN_NODES * sizeof(double*));
    for (int i = 0; i < HIDDEN_NODES; i++) {
        network->output_weights[i] = (double*)malloc(OUTPUT_NODES * sizeof(double));
    }
    
    network->hidden_bias = (double*)malloc(HIDDEN_NODES * sizeof(double));
    network->output_bias = (double*)malloc(OUTPUT_NODES * sizeof(double));
    network->hidden_activation = (double*)malloc(HIDDEN_NODES * sizeof(double));
    network->output_activation = (double*)malloc(OUTPUT_NODES * sizeof(double));
    
    return network;
}

// Function to free network memory
void free_network(NeuralNetwork* network) {
    for (int i = 0; i < INPUT_NODES; i++) {
        free(network->hidden_weights[i]);
    }
    free(network->hidden_weights);
    
    for (int i = 0; i < HIDDEN_NODES; i++) {
        free(network->output_weights[i]);
    }
    free(network->output_weights);
    
    free(network->hidden_bias);
    free(network->output_bias);
    free(network->hidden_activation);
    free(network->output_activation);
    free(network);
}

int file_exists(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file != NULL) {
        fclose(file);
        return 1;
    }
    return 0;
}

int read_data(const char* filename, 
             double** inputs,
             double** outputs,
             const char* purpose) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Could not open %s file %s\n", purpose, filename);
        exit(1);
    }

    int num_samples = 0;
    double* input_row = (double*)malloc(INPUT_NODES * sizeof(double));
    double* output_row = (double*)malloc(OUTPUT_NODES * sizeof(double));
    
    while (num_samples < MAX_SAMPLES) {
        int valid_read = 1;
        // Read inputs
        for (int i = 0; i < INPUT_NODES && valid_read; i++) {
            if (fscanf(file, "%lf", &input_row[i]) != 1) {
                valid_read = 0;
            }
        }
        // Read outputs
        for (int i = 0; i < OUTPUT_NODES && valid_read; i++) {
            if (fscanf(file, "%lf", &output_row[i]) != 1) {
                valid_read = 0;
            }
        }
        
        if (!valid_read) break;
        
        // Allocate and copy data
        inputs[num_samples] = (double*)malloc(INPUT_NODES * sizeof(double));
        outputs[num_samples] = (double*)malloc(OUTPUT_NODES * sizeof(double));
        memcpy(inputs[num_samples], input_row, INPUT_NODES * sizeof(double));
        memcpy(outputs[num_samples], output_row, OUTPUT_NODES * sizeof(double));
        
        num_samples++;
    }

    free(input_row);
    free(output_row);
    fclose(file);
    printf("\nRead %d samples from %s (%s data)\n", num_samples, filename, purpose);
    return num_samples;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

void initialize_network(NeuralNetwork* network) {
    srand(time(NULL));
    
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            network->hidden_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    
    for (int i = 0; i < HIDDEN_NODES; i++) {
        network->hidden_bias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            network->output_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    
    for (int i = 0; i < OUTPUT_NODES; i++) {
        network->output_bias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

void forward_propagate(NeuralNetwork* network, double* input) {
    for (int i = 0; i < HIDDEN_NODES; i++) {
        double activation = network->hidden_bias[i];
        for (int j = 0; j < INPUT_NODES; j++) {
            activation += input[j] * network->hidden_weights[j][i];
        }
        network->hidden_activation[i] = sigmoid(activation);
    }
    
    for (int i = 0; i < OUTPUT_NODES; i++) {
        double activation = network->output_bias[i];
        for (int j = 0; j < HIDDEN_NODES; j++) {
            activation += network->hidden_activation[j] * network->output_weights[j][i];
        }
        network->output_activation[i] = sigmoid(activation);
    }
}

void backpropagate(NeuralNetwork* network, double* input, double* target) {
    // Allocate space for the output layer deltas
    double* output_delta = (double*)malloc(OUTPUT_NODES * sizeof(double));
    double* hidden_delta = (double*)malloc(HIDDEN_NODES * sizeof(double));

    // Calculate output layer delta
    for (int i = 0; i < OUTPUT_NODES; i++) {
        double error = target[i] - network->output_activation[i];
        output_delta[i] = error * sigmoid_derivative(network->output_activation[i]);
    }

    // Calculate hidden layer delta
    for (int j = 0; j < HIDDEN_NODES; j++) {
        double error = 0.0;
        for (int k = 0; k < OUTPUT_NODES; k++) {
            error += output_delta[k] * network->output_weights[j][k];
        }
        hidden_delta[j] = error * sigmoid_derivative(network->hidden_activation[j]);
    }

    // Update weights between hidden layer and output layer
    for (int j = 0; j < HIDDEN_NODES; j++) {
        for (int k = 0; k < OUTPUT_NODES; k++) {
            network->output_weights[j][k] += LEARNING_RATE * output_delta[k] * network->hidden_activation[j];
        }
    }

    // Update weights between input layer and hidden layer
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            network->hidden_weights[i][j] += LEARNING_RATE * hidden_delta[j] * input[i];
        }
    }

    // Free allocated memory for deltas
    free(output_delta);
    free(hidden_delta);
}

// Function to make a single prediction
void make_prediction(NeuralNetwork* network) {
    double* input = (double*)malloc(INPUT_NODES * sizeof(double));
    
    printf("\nEnter %d input values:\n", INPUT_NODES);
    for (int i = 0; i < INPUT_NODES; i++) {
        printf("Input %d: ", i + 1);
        scanf("%lf", &input[i]);
    }
    
    forward_propagate(network, input);
    
    printf("\nPrediction:\n");
    for (int i = 0; i < OUTPUT_NODES; i++) {
        printf("Output %d: %f\n", i + 1, network->output_activation[i]);
    }
    
    free(input);
}

double train(NeuralNetwork* network, double** training_inputs, double** training_outputs, int num_samples) {
    double total_error = 0.0;
    
    for (int sample = 0; sample < num_samples; sample++) {
        // Forward propagate to get predictions
        forward_propagate(network, training_inputs[sample]);
        
        // Calculate error and accumulate it for reporting
        double sample_error = 0.0;
        for (int i = 0; i < OUTPUT_NODES; i++) {
            double error = training_outputs[sample][i] - network->output_activation[i];
            sample_error += error * error; // Squared error
        }
        total_error += sample_error / OUTPUT_NODES;

        // Backpropagate to adjust weights
        backpropagate(network, training_inputs[sample], training_outputs[sample]);
    }
    
    // Return average error for this epoch
    return total_error / num_samples;
}

void test_network(NeuralNetwork* network, double** test_inputs, double** test_outputs, int num_samples) {
    int correct = 0;
    for (int sample = 0; sample < num_samples; sample++) {
        forward_propagate(network, test_inputs[sample]);
        
        // Calculate error or success based on your criteria (e.g., threshold or specific output range)
        int is_correct = 1;
        for (int i = 0; i < OUTPUT_NODES; i++) {
            double predicted = network->output_activation[i];
            double expected = test_outputs[sample][i];
            if (fabs(predicted - expected) > 0.5) { // Assuming binary classification for simplicity
                is_correct = 0;
                break;
            }
        }
        
        if (is_correct) correct++;
    }
    
    printf("Test Accuracy: %.2f%%\n", (double)correct / num_samples * 100);
}

void free_data(double** data, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        free(data[i]);
    }
    free(data);
}

// Modified main function
int main(int argc, char *argv[]) {
    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            printf("\n'nnp' stands for Neural Network Program. It is written in the C language by Shpati Koleka.\n\n");
            printf("PROGRAM DESCRIPTION:\n");
            printf(" - The program uses the %s file to configure and train a neural network.\n", parameters_file);
            printf(" - If the %s file is not found it will be created using default values.\n", parameters_file);
            printf(" - The parameters within the %s file can be modified using any text editor.\n", parameters_file); 
            printf(" - The program uses the %s and %s files for loading the respective sample data.\n", train_file, test_file);
            printf(" - After each training the obtained weights are saved for reuse in the %s file.\n", parameters_file);
            printf(" - The user is given an option to make predictions using the trained neural network.\n\n");
            printf("NOTES:\n");
            printf(" - The parameters and sample files mentioned above should be placed in the same folder as nnp.\n");
            printf(" - Each sample is a separate line of text within the %s and %s files.\n", train_file, test_file);
            printf(" - For each sample include first the input values and then the output values.\n");
            printf(" - Separate the values constituting the sample using one space character between each.\n");
            printf(" - Example of one sample (separate line) with two inputs and one output: 0.1 0.2 0.15\n");
            printf(" - Forward propagation and backpropagation are implemented.\n");
            printf(" - The sigmoid function is the activation function used.\n");
            return 0;
        }
        if (strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0) {
            printf("\nVersion %s\n", version);
            return 0;
        }

    }

    // Variable declarations
    double** training_inputs;
    double** training_outputs;
    double** test_inputs;
    double** test_outputs;
    int num_training;
    int num_test;
    char answer[256];

    // Read parameters from file
    read_parameters(parameters_file);
    
    // Create network
    NeuralNetwork* network = create_network();
    
    // Check if weights exist and ask user what to do
    if (weights_exist(parameters_file)) {
        if (get_yes_no_input("Saved weights found within the parameters file. \nDo you want to use them? (y/n)", 'y') == 'y') {
            if (load_weights(network, parameters_file)) {
                printf("\nThe saved weights found in %s were successfully loaded!\n", parameters_file);
                goto prediction;
            } else {
                printf("Error loading saved weights. Will proceed with training.\n");
            }
        }
    } else {
        printf("No saved weights found. Will proceed with training.\n");
    }

    // Training phase
    training_inputs = (double**)malloc(MAX_SAMPLES * sizeof(double*));
    training_outputs = (double**)malloc(MAX_SAMPLES * sizeof(double*));
    
    num_training = read_data(train_file, training_inputs, training_outputs, "training");
    
    if (num_training == 0) {
        printf("No training data found in %s\n", train_file);
        free(training_inputs);
        free(training_outputs);
        free_network(network);
        return 1;
    }
    
    initialize_network(network);
    
    printf("\nTraining neural network with %d samples...\n", num_training);
    for (int epoch = 0; epoch <= MAX_EPOCHS; epoch++) {
        double error = train(network, training_inputs, training_outputs, num_training);
        
        if (epoch % 1000 == 0) {
            printf("Epoch %d: Error = %f\n", epoch, error);
        }
        
        if (error < ERROR_THRESHOLD) {
            printf("Training completed at epoch %d\n\n", epoch);
            break;
        }
    }
    
    // Save weights after training
    save_weights(network, parameters_file);
    
    // Testing phase
    if (file_exists(test_file)) {
        test_inputs = (double**)malloc(MAX_SAMPLES * sizeof(double*));
        test_outputs = (double**)malloc(MAX_SAMPLES * sizeof(double*));
        num_test = read_data(test_file, test_inputs, test_outputs, "test");
        test_network(network, test_inputs, test_outputs, num_test);
        free_data(test_inputs, num_test);
        free_data(test_outputs, num_test);
    } else {
        printf("\nNo %s found, using training data for testing...\n", test_file);
        test_network(network, training_inputs, training_outputs, num_training);
    }
    
    free_data(training_inputs, num_training);
    free_data(training_outputs, num_training);

prediction:
    if (get_yes_no_input("\nDo you want to make a prediction / forecast? (y/n)", 'n') == 'y') {
        make_prediction(network);
        while (getchar() != '\n');  // Clear the input buffer
        goto prediction;
    }
    
    // Cleanup
    free_network(network);
    
    return 0;
}
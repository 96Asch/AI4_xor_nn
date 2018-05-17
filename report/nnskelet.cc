
//
// C++-programma voor neuraal netwerk (NN) met \'e\'en output-knoop
// Zie www.liacs.leidenuniv.nl/~kosterswa/AI/nnhelp.pdf
// 14 mei 2018
// Compileren: make
// Gebruik:    ./AI4 [args...]
// Voor een lijst met argumenten, gebruik: ./AI4 -h

// Voorbeeld:  ./AI4 -a 0.05 -e 500000 -i 3 -o 2 -l 1 -r 20

#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <string>
#include <getopt.h>

using namespace std;

const unsigned MAX = 200;
const double BETA = 1.0;

enum Op {
	XOR,
	OR,
	AND
};

// Input variables (with defaults)
double alpha = 0.1;
unsigned i_epochs = 500000;
unsigned i_hidden_amount = 4;
unsigned i_inputs_amount = 2;
unsigned i_runs = 1;
double i_accepted_error = 0.1;
bool i_relu = false;
Op i_op = XOR;

// Struct to store the result acquired by running the neural network
struct Result {
	// Variables
	double input[MAX];
	double output = 0;

	// Functions
	Result() {
		for (unsigned i = 0; i < i_inputs_amount+1; i++)
			input[i] = 0;
	}

	Result(double a[MAX], double b) {
		for (unsigned i = 1; i < i_inputs_amount+1; i++)
			input[i] = a[i];
		output = b;
	}

	friend std::ostream& operator<< (std::ostream &os, const Result& o) {
		for (unsigned i = 1; i < i_inputs_amount+1; i++)
			os << o.input[i] << ", ";
		os << " : " << o.output;
		return os;
	}
};



// g-function (sigmoid)
double g (double x) {
	return 1 / (1 + exp(-BETA * x ));
}//g

// Prime of g(x)
double gprime (double x) {
	return BETA * g(x) * (1 - g(x));
}//gprime

// Rectified linear unit
double relu(double x) {
	return max(0.0, x);
}

double reluprime(double x) {
	return ((x <= 0) ? 0 : 1);
}

double activationfunc(double x) {
	return ((i_relu) ? relu(x) : g(x));
}

double activationprime(double x) {
	return ((i_relu) ? reluprime(x) : gprime(x));
}

// Compute random value in domain [a,b] (can be negative)
double rand_gen(double a, double b) {
    return (rand()/(double)(RAND_MAX))*abs(a-b)+a;
}

//*********************************************************************
//Operation-dependent expected value-functions
static inline double xoroperation(double input[MAX]) {
	double x = 0;
	for (unsigned i = 1; i < i_inputs_amount+1; i++)
		x += std::round(input[i]);
	return ((int)(x) % i_inputs_amount == 1);
}

static inline double oroperation(double input[MAX]) {
	double x = 0;
	for (unsigned i = 1; i < i_inputs_amount+1; i++)
		x += std::round(input[i]);

	return ((int)(x) >= 1);
}

static inline double andoperation(double input[MAX]) {
	double x = 0;
	for (unsigned i = 1; i < i_inputs_amount+1; i++)
		x += std::round(input[i]);

	return ((unsigned)(x) >= i_inputs_amount);
}

//*********************************************************************
// Execute globally set operation and return expected result
double op(double input[MAX]) {
	switch (i_op) {
		case XOR: return xoroperation(input);
		case OR:  return oroperation(input);
		case AND: return andoperation(input);
		default: throw std::runtime_error("No operation specified");
	}
}

// Determine if success is achieved by the network for given inputs a, b and output c
bool op_success(double input[MAX], double c) {
	return (op(input) >= c-i_accepted_error && op(input) <= c+i_accepted_error);
}

//*********************************************************************

// Set the hidden layer of the neural network by doint a matrix multiplication
// of the inputs from the previous layer with the matrix of weights.
void setHiddenLayer(double weights[MAX][MAX], double activation[MAX], double inhidden[MAX], double output[MAX]) {
	for (unsigned i = 0; i < i_hidden_amount+1; i++) {
		output[i] = 0;
		for (unsigned j = 0; j < i_inputs_amount+1; j++)
			output[i] += activation[j]*weights[i][j];
		inhidden[i] = output[i];
		output[i] = activationfunc(output[i]);
	}
}

// Set the final output layer of the neural network by doing a vector multiplication
// of the inputs from the previous layer with the vector of weights.
void setOutputLayer(double weights[MAX], double activation[MAX], double & inoutput, double & netoutput) {
	double output = 0;
	for(unsigned i = 0; i < i_hidden_amount+1; i++)
		output += weights[i] * activation[i];
	inoutput = output;
	netoutput = activationfunc(output);
}

//*********************************************************************

// Main neural network function. Constructed as advised by the assignment
Result fire(char* filename) {
	double input[MAX], inputtohidden[MAX][MAX], hiddentooutput[MAX];
	double inhidden[MAX], acthidden[MAX], inoutput, netoutput, target;
	double error, delta, deltahidden[MAX];


	input[0] = -1;                  // invoer bias-knoop: altijd -1
	acthidden[0] = -1;              // verborgen bias-knoop: altijd -1
	ifstream in;

	if(filename) {
		in.open(filename);
		if(!in.is_open())
			throw std::runtime_error("Error opening file");
		string temp;
	}

	unsigned i, j, k;
	for (i = 0; i < MAX; i++) {
		hiddentooutput[i] = rand_gen(-1, 1);
		for (j = 0; j < MAX; j++)
			inputtohidden[i][j] = rand_gen(-1, 1);
	}

	for ( i = 0; i < i_epochs; i++ ) {
		if (filename) {
			std::string line;
			if (getline(in, line)) {
				for (unsigned i = 0; i < i_inputs_amount; i++)
					input[i+1] = line[i*3] -'0' ;
				target = line[i_inputs_amount * 3] - '0';
			} else {
				in.clear();
				in.seekg(0, ios::beg);
			}
		} else {

			for (unsigned i = 0; i < i_inputs_amount; i++)
				input[i+1] = rand() % 2;
			target = op(input);
		}

		setHiddenLayer(inputtohidden, input, inhidden, acthidden);
		setOutputLayer(hiddentooutput, acthidden, inoutput, netoutput);

		error = target - netoutput;
		delta = error * activationprime(inoutput);
		for (j = 0; j < i_hidden_amount+1; j++)
			deltahidden[j] = activationprime(inhidden[j]) * hiddentooutput[j] * delta;

		for (j = 0; j < i_hidden_amount+1; j++)
			hiddentooutput[j] = hiddentooutput[j] + alpha * acthidden[j] * delta;
		
		for (k = 0; k < i_inputs_amount+1; k++)
			for (j = 1; j < i_hidden_amount+1; j++)
				inputtohidden[k][j] = inputtohidden[k][j] + alpha * input[k] * deltahidden[j];
	}//for
	in.close();
	return Result(input, netoutput);
}

bool correct_result(Result r) {
	return (op_success(r.input, r.output));
}

// Function to show help when user args contain -h and on invalid input args
static void showHelp(const char *progName) {
	std::cerr << progName
	<< "[-l learning rate] [-a error-value] [-v hidden-amount] [-e epochs] [-g useReLU] " <<
	"[-i inputs] [-o operation] [-r runs] [filename]" << std::endl;
	std::cerr << R"HERE(
    -l learning rate	 The learning rate of the network
    -a error-value       accepted error value
    -v hidden-amount     Amount of hidden nodes
    -e epochs            Configure amount of epochs (training amount)
    -i inputs            Amount of inputs
    -g activationtype    0 = sigmoid, 1 = ReLu
    -o operation         Operation type. 0 = XOR, 1 = OR, 2 = AND
    -r runs              Amount of runs
)HERE";
}


int main (int argc, char* argv[]) {
	char c;
	const char *progName = argv[0];
	try {
		while ((c = getopt(argc, argv, "a:e:g:i:l:o:r:v:h")) != -1){
			int x = ((c != 'h' && c != 'a') ? stoi(optarg) : -1);
			double d = ((c == 'a' || c == 'l') ? atof(optarg) : -1);
			switch (c) {
				case 'l':
					if (d >= -1 && d <= 1)
						alpha = ((d > 0) ? d : d * -1);
					break;
				case 'a':
					if (d >= -1 && d <= 1)
						i_accepted_error = ((d > 0) ? d : d * -1);
					else {
						showHelp(progName);
						exit(-1);
					}
					break;
				case 'v':
					i_hidden_amount = ((x >= 0) ? x : x * -1);
					break;
				case 'e':
					i_epochs = ((x >= 0) ? x : x * -1);
					break;
				case 'i':
					i_inputs_amount = ((x >= 0) ? x : x * -1);
					break;
				case 'g':
					if (x != 0 && x != 1) {
						showHelp(progName);
						exit(-1);
					}
					i_relu = (x > 0) ? true : false;
					break;
				case 'o':
					switch(x) {
						case 0:
							i_op = XOR;
							break;
						case 1:
							i_op = OR;
							break;
						case 2:
							i_op = AND;
							break;
						default:
							showHelp(progName);
							exit(-1);
					}
					break;
				case 'r': 
					i_runs = ((x >= 0) ? x : x * -1);
					break;
				case 'h':
				default:
					showHelp(progName);
					exit(-1);
			}
		}//while
		argc -= optind;
		argv += optind;

		unsigned corrects=0, total=0;
		try {
			unsigned randx, counter = 0;
			srand(time(NULL));
			Result r;
			for (unsigned i = 0; i < i_runs; i++) {
				total++;
				randx = rand();
				if (argc == 1)
					r = fire(argv[0]);
				else
					r = fire(NULL);
				cout << ++counter << "\t| " << r << endl;
				if (correct_result(r))
					corrects++;
				srand(randx);
			}
		} catch (std::runtime_error &error) {
			std::cerr << "Runtime error: " << error.what() << std::endl;
			return -1;
		}
		cout <<
		"----------------------------------------------------------" << endl <<
		"Operation:      " << ((i_op==0) ? "XOR" : ((i_op==1) ? "OR" : "AND")) 
		<< endl << endl <<
		"Act. function:  " << (i_relu ? "ReLU" : "Sigmoid") << endl <<
		"Learning rate:  " << alpha << endl <<
		"Accepted error: " << i_accepted_error << endl << endl <<
		"Corrects:       " << corrects << endl <<
		"Totals:         " << total << endl <<
		"Percentage:     " << ((double) corrects/(double) total)*100<<'%'<<endl;
		return 0;
	} catch (...) {
		return -1;
	}
}//main


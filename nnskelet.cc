
//
// C++-programma voor neuraal netwerk (NN) met \'e\'en output-knoop
// Zie www.liacs.leidenuniv.nl/~kosterswa/AI/nnhelp.pdf
// 19 april 2018
// Compileren: g++ -Wall -O2 -o nn nn.cc
// Gebruik:    ./nn <inputs> <hiddens> <epochs>
// Voorbeeld:  ./nn 2 3 100000
//

#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <string>
#include <getopt.h>

using namespace std;

const unsigned MAX = 200;
const double ALPHA = 0.1;
const double BETA = 1.0;

enum Op {
	XOR,
	OR,
	AND
};

// Input variables
unsigned i_epochs = 500000;
unsigned i_hidden_amount = 4;
unsigned i_inputs_amount = 2;
unsigned i_runs=1;
double i_accepted_error = 0.1;
bool i_relu = true;
Op i_op = XOR;

// Struct to store the result acquired by running the neural network
struct Result {
	Result(){
		for (unsigned i = 0; i < i_inputs_amount+1; i++)
			input[i] = 0;
	}

	Result(double a[MAX], double b) {
		for (unsigned i = 1; i < i_inputs_amount+1; i++) {
			input[i] = a[i];
		}
		output = b;
	}
	double input[MAX];
	double output = 0;

	friend std::ostream& operator<< (std::ostream &os, const Result& o) {
		for (unsigned i = 1; i < i_inputs_amount+1; i++)
			os << o.input[i] << ", ";
		os << " : " << o.output;
		return os;
	}
};



// g-function (sigmoid)
double g (double x) {
	return 1 / ( 1 + exp ( - BETA * x ) );
}//g

// Prime of g(x)
double gprime (double x) {
	return BETA * g (x) * ( 1 - g (x) );
}//gprime

// rectified linear unit
double relu(double x) {
	return max(0.0,x);
}

double squishifier(double x) {
	if(i_relu)
		return relu(x);
	return g(x);
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

void setHiddenLayer(double weights[MAX][MAX], double activation[MAX], double inhidden[MAX], double output[MAX]) {
	for (unsigned i = 0; i < i_hidden_amount+1; i++) {
		output[i] = 0;
		for (unsigned j = 0; j < i_inputs_amount+1; j++)
			output[i] += activation[j]*weights[i][j];
		inhidden[i] = output[i];
		output[i] = squishifier(output[i]);
	}
}

void setOutputLayer(double weights[MAX], double activation[MAX], double & inoutput, double & netoutput) {
	double output = 0;
	for(unsigned i = 0; i < i_hidden_amount+1; i++)
		output += weights[i] * activation[i];
	inoutput = output;
	netoutput = squishifier(output);
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
				throw std::runtime_error("EOF reached while requesting more training data");
			}
		} else {

			for (unsigned i = 0; i < i_inputs_amount; i++)
				input[i+1] = rand() % 2;
			target = op(input);
		}

		setHiddenLayer(inputtohidden, input, inhidden, acthidden);
		setOutputLayer(hiddentooutput, acthidden, inoutput, netoutput);

		error = target - netoutput;
		delta = error * gprime(inoutput);
		for (j = 0; j < i_hidden_amount+1; j++)
			deltahidden[j] = gprime(inhidden[j]) * hiddentooutput[j] * delta;

		for (j = 0; j < i_hidden_amount+1; j++)
			hiddentooutput[j] = hiddentooutput[j] + ALPHA * acthidden[j] * delta;
		
		for (k = 0; k < i_inputs_amount+1; k++)
			for (j = 1; j < i_hidden_amount+1; j++)
				inputtohidden[k][j] = inputtohidden[k][j] + ALPHA * input[k] * deltahidden[j];
	}//for
	in.close();
	return Result(input, netoutput);
}

static void showHelp(const char *progName) {
	std::cerr << progName
	<< " [-a error-value] [-d hidden-amount] [-e epochs] [-i inputs] " <<
	"[-o operation] [-r runs] [-l relu] [filename]" << std::endl;
	std::cerr << R"HERE(
    -a error-value       accepted error value
    -d hidden-amount     Amount of hidden nodes
    -e epochs            Configure amount of epochs (training amount)
    -i inputs            Amount of inputs
    -o operation         Operation type. 0 = XOR, 1 = OR, 2 = AND
    -r runs              Amount of runs
    -l 0 or 1			 Use ReLu
)HERE";
}

bool correct_result(Result r) {
	return (op_success(r.input, r.output));
}


int main (int argc, char* argv[]) {
	char c;
	const char *progName = argv[0];
	try {
		while ((c = getopt(argc, argv, "a:d:e:i:o:r:l:h")) != -1){
			int x = ((c != 'h' && c != 'a') ? stoi(optarg) : -1);
			double d = ((c == 'a') ? atof(optarg) : -1);
			switch (c) {
				case 'a':
					if (d >= -1 && d <= 1)
						i_accepted_error = ((d > 0) ? d : d * -1);
					else {
						showHelp(progName);
						exit(-1);
					}
				case 'd':
					i_hidden_amount = ((x >= 0) ? x : x * -1);
					break;
				case 'e':
					i_epochs = ((x >= 0) ? x : x * -1);
					break;
				case 'i':
					i_inputs_amount = ((x >= 0) ? x : x * -1);
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
				case 'l':
					i_relu = (x > 0) ? true : false;
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
			unsigned randx;
			srand(time(NULL));
			Result r;
			for (unsigned i = 0; i < i_runs; i++) {
				total++;
				randx = rand();
				if (argc == 1)
					r = fire(argv[0]);
				else
					r = fire(NULL);
				cout << r << endl;
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
		"Corrects:       " << corrects << endl <<
		"Totals:         " << total << endl <<
		"Percentage:     " << ((double) corrects/(double) total)*100<<'%'<<endl;
		return 0;
	} catch (...) {
		return -1;
	}
}//main


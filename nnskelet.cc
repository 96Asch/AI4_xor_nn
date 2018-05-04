
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

const unsigned MAX = 2000;
const double ALPHA = 0.1;
const double BETA = 1.0;
const unsigned Max_filesize = 10;
enum Op {
	XOR,
	OR,
	AND
};

// Input variables
unsigned i_epochs = 4000;
unsigned i_hidden_amount = 4;
unsigned i_inputs_amount = 2;
unsigned i_runs=1;
Op i_op = XOR;

// Struct to store the result acquired by running the neural network
struct Result {
	Result(){}
	Result(double a, double b, double c) : input_one(a), input_two(b), output(c) {}
	double input_one = 0, input_two = 0, output = 0;

	friend std::ostream& operator<< (std::ostream &os, const Result& o) {
		os << o.input_one << ", " << o.input_two << " : " << o.output;
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

// Compute random value in domain [a,b] (can be negative)
double rand_gen(double a, double b) {
    return (rand()/(double)(RAND_MAX))*abs(a-b)+a;
}

//*********************************************************************
//Operation-dependent expected value-functions
static inline double xoroperation(double a, double b) {
	return ((int)(std::round(a) + std::round(b)) % 2 == 1);
}

static inline double oroperation(double a, double b) {
	return ((int)(std::round(a) + std::round(b)) >= 1);
}

static inline double andoperation(double a, double b) {
	return ((int)(std::round(a) + std::round(b)) >= 2);
}

//*********************************************************************
// Execute globally set operation and return expected result
double op(double a, double b) {
	switch (i_op) {
		case XOR: return xoroperation(a, b);
		case OR:  return oroperation(a, b);
		case AND: return andoperation(a, b);
		default: throw std::runtime_error("No operation specified");
	}
}

// Determine if success is achieved by the network for given inputs a, b and output c
bool op_success(double a, double b, double c) {
	return (op(a, b) >= c-ALPHA && op(a, b) <= c+ALPHA);
}

//*********************************************************************

void setHiddenLayer(double weights[MAX][MAX], double activation[MAX], double inhidden[MAX], double output[MAX]) {
	for (unsigned i = 0; i < i_hidden_amount+1; i++) {
		output[i] = 0;
		for (unsigned j = 0; j < i_inputs_amount+1; j++)
			output[i] += activation[j]*weights[i][j];
		inhidden[i] = output[i];
		output[i] = g(output[i]);
	}
}

void setOutputLayer(double weights[MAX], double activation[MAX], double & inoutput, double & netoutput) {
	double output = 0;
	for(unsigned i = 0; i < i_hidden_amount+1; i++)
		output += weights[i] * activation[i];
	inoutput = output;
	netoutput = g(output);
}

//*********************************************************************

// Main neural network function. Constructed as advised by the assignment
Result fire(string filename) {
	double input[MAX], inputtohidden[MAX][MAX], hiddentooutput[MAX];
	double inhidden[MAX], acthidden[MAX], inoutput, netoutput, target;
	double error, delta, deltahidden[MAX];


	input[0] = -1;                  // invoer bias-knoop: altijd -1
	acthidden[0] = -1;              // verborgen bias-knoop: altijd -1
	ifstream in;

	if(filename.size() != 0) {
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
		if (filename.size() != 0) {
			std::string line;
			if (getline(in, line)) {
				input[1] = line[0] - '0';
				input[2] = line[3] - '0';
				target = line[6] - '0';
			} else {
				throw std::runtime_error("EOF reached while requesting more training data");
			}
		} else {
			input[1] = rand() % 2;
			input[2] = rand() % 2;
			target = op(input[1], input[2]);
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
	return Result(input[1], input[2], netoutput);
}

static void showHelp(const char *progName) {
	std::cerr << progName
	<< " [-d hidden-amount] [-e epochs] [-f filename] [-h hidden-amount] [-i inputs] \n" <<
	"[-o operation] [-r runs] \t[filename]" << std::endl;
	std::cerr << R"HERE(
    -d hidden-amount     Amount of hidden nodes
    -e epochs            Configure amount of epochs (training amount)
    -i inputs            Amount of inputs
    -o operation         Operation type. 0 = XOR, 1 = OR, 2 = AND
    -r runs              Amount of runs
)HERE";
}

bool correct_result(Result r) {
	return (op_success(r.input_one, r.input_two, r.output));
}


int main (int argc, char* argv[]) {
	char c;
	const char *progName = argv[0];
	try {
		while ((c = getopt(argc, argv, "d:e:i:o:r:h")) != -1){
			int x = ((c != 'h') ? std::stoi(optarg) : -1);
			switch (c) {
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
				if (argc == 1) {
					r = fire(argv[0]);
				} else
					r = fire(string(""));
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


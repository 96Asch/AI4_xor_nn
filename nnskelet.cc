
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

using namespace std;

const unsigned MAX = 20;
const double ALPHA = 0.1;
const double BETA = 1.0;

enum Op {
	XOR,
	OR,
	AND
};


const Op _op = XOR;

// g-functie (sigmoid)
double g (double x) {
	return 1 / ( 1 + exp ( - BETA * x ) );
}//g

// afgeleide van g
double gprime (double x) {
	return BETA * g (x) * ( 1 - g (x) );
}//gprime

// bepaal random waarde tussen a en b
double randf(double a, double b) {
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

double op(double a, double b) {
	switch (_op) {
		case XOR: return xoroperation(a, b);
		case OR:  return oroperation(a, b);
		case AND: return andoperation(a, b);
		default: throw std::runtime_error("No operation specified");
	}
}


bool op_success(double a, double b, bool c) {
	return (op(a, b) == c);
}

//*********************************************************************

void copyArray(double src[MAX], double* dest) {
	for(int i = 1; i < MAX; i++)
		dest[i] = src[i];
}

void setHiddenLayer(size_t size, double bias, double weights[MAX][MAX], double activation[MAX], double* output) {
	for (unsigned k = 1; k < size; k++) {
		for (unsigned j = 0; j < size; j++) {
			output[k] += weights[k][j] * activation[j];
		}
	}
	for(unsigned i = 1; i < size; i++)
		output[i] = g(output[i]+ bias);
}


int main (int argc, char* argv[ ]) {
	if ( argc != 4 && argc != 5) {
		cout << "Gebruik: " << argv[0] << " <inputs> <hiddens> <epochs> <optional: filename>" << endl;
		return 1;
	}//if
//{
	int inputs, hiddens;            // aantal invoer- en verborgen knopen
	double input[MAX];              // de invoer is input[1]...input[inputs]
	double inputtohidden[MAX][MAX]; // gewichten van invoerknopen 0..inputs
	                                // naar verborgen knopen 1..hiddens
	double hiddentooutput[MAX];     // gewichten van verborgen knopen 0..hiddens
	                                // naar de ene uitvoerknoop
	double inhidden[MAX];           // invoer voor de verborgen knopen 1..hiddens
	double acthidden[MAX];          // en de uitvoer daarvan
	double inoutput;                // invoer voor de ene uitvoerknoop
	double netoutput;               // en de uitvoer daarvan: de net-uitvoer
	double target;                  // gewenste uitvoer
	double error;                   // verschil tussen gewenste en 
	                                // geproduceerde uitvoer
	double delta;                   // de delta voor de uitvoerknoop
	double deltahidden[MAX];        // de delta's voor de verborgen 
	                                // knopen 1..hiddens
	int epochs;                     // aantal trainingsvoorbeelden
	char* fName = NULL;
	inputs = atoi (argv[1]);
	hiddens = atoi (argv[2]);
	epochs = atoi (argv[3]);
	input[0] = -1;                  // invoer bias-knoop: altijd -1
	acthidden[0] = -1;              // verborgen bias-knoop: altijd -1
	srand (time(NULL));
	ifstream in;
//}
	if(argc == 5) {
		fName = argv[4];
		in.open(fName);
		if(!in.is_open())
			throw std::runtime_error("Error opening file");
	}

	//int seed = 1234;              // eventueel voor random-generator

	//TODO-1 initialiseer de gewichten random tussen -1 en 1: 
	// inputtohidden en hiddentooutput
	// rand ( ) levert geheel randomgetal tussen 0 en RAND_MAX; denk aan casten
	unsigned i, j, k;
	for (i = 0; i < MAX; i++) {
		hiddentooutput[i] = randf(-1, 1);
		for (j = 0; j < MAX; j++) {
			inputtohidden[i][j] = randf(-1, 1);;
		}
	}

	for ( i = 0; i < epochs; i++ ) {
		//TODO-2 lees een voorbeeld in naar input en target, of genereer dat ter plekke:
		// als voorbeeld: de XOR-functie, waarvoor geldt dat inputs = 2
		// int x = rand ( ) % 2; int y = rand ( ) % 2; int dexor = ( x + y ) % 2;
		// input[1] = x; input[2] = y; target = dexor;
		//input lijkt op: 0, 1, 1
		if (fName) {
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
		}
		
		//TODO-3 stuur het voorbeeld door het netwerk
		// reken inhidden's uit, acthidden's, inoutput en netoutput
		copyArray(input, inhidden); //inputs zijn de invoer van de eerste laag van de hidden knopen.
		setHiddenLayer(hiddens, input[0], inputtohidden, inhidden, acthidden); // bereken de output van de hidden laag.

		//Set inoutput by calculating acthidden.
		//Set netoutput to 0 or 1 based on inoutput.


		

		//TODO-4 bereken error, delta, en deltahidden
		error = target - netoutput;
		delta = error * gprime(inoutput);
		for (j = 0; j < MAX; j++)
			deltahidden[j] = gprime(inhidden[j]) * hiddentooutput[j] * delta;

		//TODO-5 update gewichten hiddentooutput en inputtohidden
		for (j = 0; j < MAX; j++)
			hiddentooutput[j] = hiddentooutput[j] + ALPHA * acthidden[j] * delta;
		
		for (k = 0; k < MAX; k++)
			for (j = 0; j < MAX; j++)
				inputtohidden[k][j] = inputtohidden[k][j] + ALPHA * input[k] * deltahidden[j];
	}//for

	//TODO-6 beoordeel het netwerk en rapporteer

	in.close();
	return 0;
}//main


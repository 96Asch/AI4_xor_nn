
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

const int MAX = 20;
const double ALPHA = 0.1;
const double BETA = 1.0;

enum Op {
	XOR,
	OR,
	AND
};

const Op _op = OR;

// g-functie (sigmoid)
double g (double x) {
	return 1 / ( 1 + exp ( - BETA * x ) );
}//g

// afgeleide van g
double gprime (double x) {
	return BETA * g (x) * ( 1 - g (x) );
}//gprime

// bepaal random waarde tussen a en b
double randf(double a, double b){
    return (rand()/(double)(RAND_MAX))*abs(a-b)+a;
}

double xoroperation(double a, double b) {
//	return (a + b) % 2.0d;
}

double oroperation(double a, double b) {
	return ((a + b) == 1.0d);
}

double andoperation(double a, double b) {
	return ((a + b) == 2.0d);
}


double op(double a, double b) {
	switch (_op) {
		case XOR: return xoroperation(a, b);
		case OR:  return oroperation(a, b);
		case AND: return andoperation(a, b);
		default: throw std::runtime_error("No operation specified");
	}

}


void readFile(const char* filename, double input[]) {
//	ifstream in(filename);
//	if(! in.is_open())
//		throw std::runtime_error("Error opening file");
//	std::string line;
//
//	for (int i = 0; i < MAX && getline(in, line) && line.size() == 6; i++) {
//		input[i] = line[0];
//		input[i+1] = line[3];
//	}

}

int main (int argc, char* argv[ ]) {

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
	int i, j, k;                    // tellertjes
	//int seed = 1234;              // eventueel voor random-generator

	if ( argc != 4 ) {
		cout << "Gebruik: " << argv[0] << " <inputs> <hiddens> <epochs>" << endl;
		return 1;
	}//if
	inputs = atoi (argv[1]);
	hiddens = atoi (argv[2]);
	epochs = atoi (argv[3]);
	input[0] = -1;                  // invoer bias-knoop: altijd -1
	acthidden[0] = -1;              // verborgen bias-knoop: altijd -1
	srand (time(NULL));

	//TODO-1 initialiseer de gewichten random tussen -1 en 1: 
	// inputtohidden en hiddentooutput
	// rand ( ) levert geheel randomgetal tussen 0 en RAND_MAX; denk aan casten
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
		for(j = 0; j < inputs; j++)
			input[j] = rand() % 2;

		//TODO-3 stuur het voorbeeld door het netwerk
		// reken inhidden's uit, acthidden's, inoutput en netoutput

		//TODO-4 bereken error, delta, en deltahidden

		//TODO-5 update gewichten hiddentooutput en inputtohidden

	}//for

	//TODO-6 beoordeel het netwerk en rapporteer

	return 0;
}//main


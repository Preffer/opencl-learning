#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char *argv[]) {
	// Create the two input vectors and one result matrix
	if (argc < 2) {
		cout << "Usage: ./matrix (R)" << endl;
		return 2;
	}

	const int R = stoi(argv[1]);

	float* A = new float[R * R];
	float* B = new float[R * R];
	float* C = new float[R * R];

	for (int i = 0; i < R * R; i++) {
		A[i] = rand() % 100;
		B[i] = rand() % 100;
	}

	#pragma omp parallel for
	for (int row = 0; row < R; row++) {
		for(int col = 0; col < R; col++) {
			float sum = 0;
			for (int i = 0; i < R; i++) {
				sum += A[row * R + i] * B[i * R + col];
			}
			C[row * R + col] = sum;
		}
	}

	if(argc == 2){
		ofstream outa("a.txt");
		ofstream outb("b.txt");
		ofstream outc("c.txt");
		for (int row = 0; row < R; row++) {
			for (int col = 0; col < R; col ++) {
				outa << A[row * R + col] << " ";
				outb << B[row * R + col] << " ";
				outc << C[row * R + col] << " ";
			}
			outa << endl;
			outb << endl;
			outc << endl;
		}
	}
}

#include <utility>
#include <iostream>
#include <fstream>
#include <string>

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
		ofstream fout;
		fout.open("a.txt");
		for (int row = 0; row < R; row++) {
			for (int col = 0; col < R; col ++) {
				fout << A[row*R + col] << " ";
			}
			fout << endl;
		}
		fout.close();

		fout.open("b.txt");
		for (int row = 0; row < R; row++) {
			for (int col = 0; col < R; col ++) {
				fout << B[row*R + col] << " ";
			}
			fout << endl;
		}
		fout.close();

		fout.open("c.txt");
		for (int row = 0; row < R; row++) {
			for (int col = 0; col < R; col ++) {
				fout << C[row*R + col] << " ";
			}
			fout << endl;
		}
		fout.close();
	}
}
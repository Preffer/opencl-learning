#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char* argv[]) {
	if (argc < 2) {
		cout << "Usage: ./matrix R [save](default: no)" << endl;
		return 2;
	}

	const int R = stoi(argv[1]);

	float** A = new float*[R];
	float** B = new float*[R];
	float** C = new float*[R];

	for (int row = 0; row < R; row++) {
		A[row] = new float[R];
		B[row] = new float[R];
		C[row] = new float[R];
		float* a = A[row];
		float* b = B[row];
		for (int col = 0; col < R; col++) {
			a[col] = rand() % 100;
			b[col] = rand() % 100;
		}
	}

	#pragma omp parallel for
	for (int row = 0; row < R; row++) {
		float* a = A[row];
		for (int col = 0; col < R; col++) {
			float* b = B[col];
			float sum = 0;
			for (int i = 0; i < R; i++) {
				sum += a[i] * b[i];
			}
			C[row][col] = sum;
		}
	}

	if (argc > 2) {
		ofstream outa("a.txt");
		ofstream outb("b.txt");
		ofstream outc("c.txt");
		for (int row = 0; row < R; row++) {
			for (int col = 0; col < R; col++) {
				outa << A[row][col] << " ";
				outb << B[row][col] << " ";
				outc << C[row][col] << " ";
			}
			outa << endl;
			outb << endl;
			outc << endl;
		}
	}
}

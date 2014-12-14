#define __CL_ENABLE_EXCEPTIONS
#include <iostream>
#include <string>
#include <fstream>
#include <CL/cl.hpp>

using namespace cl;
using namespace std;

/*
 * Always use row-major
 */

int main(int argc, char *argv[]) {
	if (argc <= 1) {
		cout << "Usage: ./matrix (R)" << endl;
		return 2;
	}

	const int RANK = stoi(argv[1]);
	const int SIZE = RANK / 2;

	float* A1 = new float[SIZE * SIZE];
	float* A2 = new float[SIZE * SIZE];
	float* A3 = new float[SIZE * SIZE];
	float* A4 = new float[SIZE * SIZE];

	float* B1 = new float[SIZE * SIZE];
	float* B2 = new float[SIZE * SIZE];
	float* B3 = new float[SIZE * SIZE];
	float* B4 = new float[SIZE * SIZE];

	float* C1;
	float* C2;
	float* C3;
	float* C4;

	for (int i = 0; i < SIZE * SIZE; i++) {
		A1[i] = rand() % 100;
		A2[i] = rand() % 100;
		A3[i] = rand() % 100;
		A4[i] = rand() % 100;

		B1[i] = rand() % 100;
		B2[i] = rand() % 100;
		B3[i] = rand() % 100;
		B4[i] = rand() % 100;
	}

	try {
		std::vector<Platform> platforms;
		Platform::get(&platforms);

		cl_context_properties cps[3] = {
			CL_CONTEXT_PLATFORM,
			(cl_context_properties) (platforms[0]) (),
			0
		};
		Context context(CL_DEVICE_TYPE_GPU, cps);

		std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

		CommandQueue queue = CommandQueue(context, devices[0]);

		ifstream sourceFile("cl_prod.c");
		std::string sourceCode(istreambuf_iterator<char>(sourceFile), (istreambuf_iterator<char>()));
		Program::Sources source(1, make_pair(sourceCode.c_str(), sourceCode.length() + 1));

		Program program = Program(context, source);
		program.build(devices);
		Kernel kernel(program, "prod");

		ImageFormat format(CL_R, CL_FLOAT);

		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;
		
		cl::size_t<3> region;
		region[0] = SIZE;
		region[1] = SIZE;
		region[2] = 1;

		Image2D matrixA1 = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, format, SIZE, SIZE, 0, A1);
		Image2D matrixA2 = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, format, SIZE, SIZE, 0, A2);
		Image2D matrixA3 = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, format, SIZE, SIZE, 0, A3);
		Image2D matrixA4 = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, format, SIZE, SIZE, 0, A4);

		Image2D matrixB1 = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, format, SIZE, SIZE, 0, B1);
		Image2D matrixB2 = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, format, SIZE, SIZE, 0, B2);
		Image2D matrixB3 = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, format, SIZE, SIZE, 0, B3);
		Image2D matrixB4 = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, format, SIZE, SIZE, 0, B4);

		Image2D matrixC1 = Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, format, SIZE, SIZE);
		Image2D matrixC2 = Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, format, SIZE, SIZE);
		Image2D matrixC3 = Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, format, SIZE, SIZE);
		Image2D matrixC4 = Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, format, SIZE, SIZE);

		kernel.setArg(0, matrixA1);
		kernel.setArg(1, matrixA2);
		kernel.setArg(2, matrixA3);
		kernel.setArg(3, matrixA4);
		kernel.setArg(4, matrixB1);
		kernel.setArg(5, matrixB2);
		kernel.setArg(6, matrixB3);
		kernel.setArg(7, matrixB4);
		kernel.setArg(8, matrixC1);
		kernel.setArg(9, matrixC2);
		kernel.setArg(10, matrixC3);
		kernel.setArg(11, matrixC4);
		kernel.setArg(12, SIZE);

		queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(SIZE, SIZE, 4), NullRange);

		queue.finish();

		C1 = (float*) queue.enqueueMapImage(matrixC1, CL_TRUE, CL_MAP_READ, origin, region, new ::size_t(SIZE * sizeof(float)), NULL);
		C2 = (float*) queue.enqueueMapImage(matrixC2, CL_TRUE, CL_MAP_READ, origin, region, new ::size_t(SIZE * sizeof(float)), NULL);
		C3 = (float*) queue.enqueueMapImage(matrixC3, CL_TRUE, CL_MAP_READ, origin, region, new ::size_t(SIZE * sizeof(float)), NULL);
		C4 = (float*) queue.enqueueMapImage(matrixC4, CL_TRUE, CL_MAP_READ, origin, region, new ::size_t(SIZE * sizeof(float)), NULL);

		if(argc == 2){
			ofstream outa("a.txt");
			ofstream outb("b.txt");
			ofstream outc("c.txt");

			for (int row = 0; row < SIZE; row++) {
				for (int col = 0; col < SIZE; col ++) {
					outa << A1[row * SIZE + col] << " ";
					outb << B1[row * SIZE + col] << " ";
					outc << C1[row * SIZE + col] << " ";
				}

				for (int col = 0; col < SIZE; col ++) {
					outa << A2[row * SIZE + col] << " ";
					outb << B2[row * SIZE + col] << " ";
					outc << C2[row * SIZE + col] << " ";
				}
				outa << endl;
				outb << endl;
				outc << endl;
			}

			for (int row = 0; row < SIZE; row++) {
				for (int col = 0; col < SIZE; col ++) {
					outa << A3[row * SIZE + col] << " ";
					outb << B3[row * SIZE + col] << " ";
					outc << C3[row * SIZE + col] << " ";
				}

				for (int col = 0; col < SIZE; col ++) {
					outa << A4[row * SIZE + col] << " ";
					outb << B4[row * SIZE + col] << " ";
					outc << C4[row * SIZE + col] << " ";
				}
				outa << endl;
				outb << endl;
				outc << endl;
			}
		}

	} catch(Error error) {
		cout << error.what() << "(" << error.err() << ")" << endl;
	}

	return 0;
}

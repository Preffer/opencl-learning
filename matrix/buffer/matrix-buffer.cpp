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

	try {
		// Get available platforms
		std::vector <Platform> platforms;
		Platform::get(&platforms);

		// Select the default platform and create a context using this platform and the GPU
		cl_context_properties cps[3] = {
			CL_CONTEXT_PLATFORM,
			(cl_context_properties) (platforms[0]) (),
			0
		};
		Context context(CL_DEVICE_TYPE_GPU, cps);

		// Get a list of devices on this platform
		std::vector < Device > devices = context.getInfo < CL_CONTEXT_DEVICES > ();

		// Create a command queue and use the first device
		CommandQueue queue = CommandQueue(context, devices[0]);

		// Read source file
		ifstream sourceFile("cl_prod.c");
		std::string sourceCode(istreambuf_iterator <char>(sourceFile), (istreambuf_iterator <char>()));
		Program::Sources source(1, make_pair(sourceCode.c_str(), sourceCode.length() + 1));

		// Make program of the source code in the context
		Program program = Program(context, source);

		// Build program for these specific devices
		program.build(devices);

		// Make kernel
		Kernel kernel(program, "prod");

		// Create matrix
		Buffer bufferA = Buffer(context, CL_MEM_READ_ONLY, R * R * sizeof(float));
		Buffer bufferB = Buffer(context, CL_MEM_READ_ONLY, R * R * sizeof(float));
		Buffer bufferC = Buffer(context, CL_MEM_WRITE_ONLY, R * R * sizeof(float));

		// Copy matrix A and B to the memory buffers
		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, R * R * sizeof(float), A);
		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, R * R * sizeof(float), B);

		// Set arguments to kernel
		kernel.setArg(0, bufferA);
		kernel.setArg(1, bufferB);
		kernel.setArg(2, bufferC);
		kernel.setArg(3, R);

		// Run the kernel on specific ND range
		queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(R * R), NullRange);

		// Read buffer C into a local list
		queue.finish();
		queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, R * R * sizeof(float), C);

	} catch(Error error) {
		cout << error.what() << "(" << error.err() << ")" << endl;
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

	return 0;
}

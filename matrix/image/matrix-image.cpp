#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
using namespace cl;
using namespace std;

/*
 group by row, in each group have col's operation
*/

int main(int argc, char *argv[]) {
	// Create the two input vectors and one result matrix
	if (argc < 2) {
		cout << "Usage: ./matrix (R)" << endl;
		return 2;
	}

	const int R = stoi(argv[1]);

	cl_float* A = new cl_float[R * R];
	cl_float* B = new cl_float[R * R];
	cl_float* C = new cl_float[R * R];

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

		// Define format
		ImageFormat format(CL_R, CL_FLOAT);

		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;
		
		cl::size_t<3> region;
		region[0] = R;
		region[1] = R;
		region[2] = 1;

		// Create matrix
		Image2D matrixA = Image2D(context, CL_MEM_READ_ONLY, format, R, R);
		Image2D matrixB = Image2D(context, CL_MEM_READ_ONLY, format, R, R);
		Image2D matrixC = Image2D(context, CL_MEM_WRITE_ONLY, format, R, R);

		// Copy matrix A and B to the memory buffers
		queue.enqueueWriteImage(matrixA, CL_TRUE, origin, region, 0, 0, A);
		queue.enqueueWriteImage(matrixB, CL_TRUE, origin, region, 0, 0, B);

		// Set arguments to kernel
		kernel.setArg(0, matrixA);
		kernel.setArg(1, matrixB);
		kernel.setArg(2, matrixC);
		kernel.setArg(3, R);

		// Run the kernel on specific ND range
		queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(R, R), NullRange);

		// Read buffer C into a local list
		queue.finish();
		queue.enqueueReadImage(matrixC, CL_TRUE, origin, region, 0, 0, C);

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

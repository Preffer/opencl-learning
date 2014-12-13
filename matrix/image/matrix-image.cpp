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
	float* C;

	for (int i = 0; i < R * R; i++) {
		A[i] = rand() % 100;
		B[i] = rand() % 100;
	}

	try {
		// Get available platforms
		std::vector<Platform> platforms;
		Platform::get(&platforms);

		// Select the default platform and create a context using this platform and the GPU
		cl_context_properties cps[3] = {
			CL_CONTEXT_PLATFORM,
			(cl_context_properties) (platforms[0]) (),
			0
		};
		Context context(CL_DEVICE_TYPE_GPU, cps);

		// Get a list of devices on this platform
		std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

		// Create a command queue and use the first device
		CommandQueue queue = CommandQueue(context, devices[0]);

		// Read source file
		ifstream sourceFile("cl_prod.c");
		std::string sourceCode(istreambuf_iterator<char>(sourceFile), (istreambuf_iterator<char>()));
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

		Image2D matrixA = Image2D(context, CL_MEM_READ_ONLY, format, R, R);
		Image2D matrixB = Image2D(context, CL_MEM_READ_ONLY, format, R, R);
		Image2D matrixC = Image2D(context, CL_MEM_WRITE_ONLY, format, R, R);

		queue.enqueueWriteImage(matrixA, CL_TRUE, origin, region, 0, 0, A);
		queue.enqueueWriteImage(matrixB, CL_TRUE, origin, region, 0, 0, B);

		kernel.setArg(0, matrixA);
		kernel.setArg(1, matrixB);
		kernel.setArg(2, matrixC);
		kernel.setArg(3, R);

		queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(R, R), NullRange);

		queue.finish();
		C = (float*) queue.enqueueMapImage(matrixC, CL_TRUE, CL_MAP_READ, origin, region, new ::size_t(R * sizeof(float)), NULL);

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

	} catch(Error error) {
		cout << error.what() << "(" << error.err() << ")" << endl;
	}

	return 0;
}

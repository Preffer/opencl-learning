#define __CL_ENABLE_EXCEPTIONS
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <CL/cl.hpp>
#include <sys/time.h>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

using namespace cl;
using namespace std;
using namespace boost;
namespace po = boost::program_options;

/*
 * Always use row-major
 */

struct timeval tpstart, tpend, tpcalc;

void logTime(std::string message);
void showFlops(int rank);

int main(int argc, char *argv[]) {

	gettimeofday(&tpstart, NULL);

	po::options_description desc("Options");
	desc.add_options()
		("rank,r", po::value<int>()->default_value(1024), "rank of each matrix, should be divisible by 4*slice")
		("slice,s", po::value<int>()->default_value(1), "slices of each matrix")
		("output,o", po::value<std::string>()->default_value("no"), "output of the result {no|file|console}")
		("help,h", "show this help info");

	po::variables_map vm;

	try{
		po::store(po::parse_command_line(argc, argv, desc), vm);
		if (vm.count("help")) {
			cout << desc << endl;
			return 1;
		}
		std::string output = vm["output"].as<std::string>();
		if((output != "no") && (output != "file" ) && (output != "console")){
			throw po::validation_error(po::validation_error::invalid_option_value, "output");
		}
		po::notify(vm);
	} catch(po::error& e) {
		cerr << "Error: " << e.what() << endl << endl;
		cout << desc << endl;
		return -1;
	}

	const int RANK = vm["rank"].as<int>();
	const int SLICE = vm["slice"].as<int>();
	const int SIZE = RANK / SLICE;
	const int PITCH = SIZE / 4;

	if(RANK != SLICE * PITCH * 4){
		cerr << format("Error: %1%*%1% matrix can't be slice into 4 * %2%*%2% parts.") % RANK % SLICE << endl;
		cout << desc << endl;
		return -2;
	}

	logTime((format("Generating %1%*%1% Matrix with %2% slice(s)") % RANK % SLICE).str());
	float** A = new float*[SLICE * SLICE];
	float** B = new float*[SLICE * SLICE];
	float** C = new float*[SLICE * SLICE];

	for(int i = 0; i < SLICE * SLICE; i++){
		A[i] = new float[SIZE * SIZE];
		B[i] = new float[SIZE * SIZE];
		C[i] = NULL;
		for(int j = 0; j < SIZE * SIZE; j++){
			A[i][j] = rand() % 100;
			B[i][j] = rand() % 100;
		}
	}

	try {
		logTime("Initializing OpenCL...");
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

		std::string code;
		code += "__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;";
		code += (format("__constant int PITCH = %1%;") % PITCH).str();
		code +=	"__kernel void prod(";

		for(int i = 0; i < SLICE; i++){
			code += (format("__read_only image2d_t A%1%, __read_only image2d_t B%1%, ") % i).str();
		}

		code += "__write_only image2d_t C) { \
					const int col = get_global_id(0); \
					const int row = get_global_id(1); \
					float4 sum = (float4)(0, 0, 0, 0);";

		for(int j = 0; j < SLICE; j++){
			code += (format(
					"for (int i = 0; i < PITCH; i++) { \
						float4 dataA = read_imagef(A%1%, sampler, (int2)(i, row)); \
						sum += (float4)( \
							dot(dataA, read_imagef(B%1%, sampler, (int2)(i, col))), \
							dot(dataA, read_imagef(B%1%, sampler, (int2)(i, col + 1))), \
							dot(dataA, read_imagef(B%1%, sampler, (int2)(i, col + 2))), \
							dot(dataA, read_imagef(B%1%, sampler, (int2)(i, col + 3))) \
						); \
					}") % j).str();
		}
		code += 	"write_imagef(C, (int2)(col, row), sum); \
				}";

		Program::Sources source(1, make_pair(code.c_str(), code.length() + 1));

		Program program = Program(context, source);
		program.build(devices);
		Kernel kernel(program, "prod");

		logTime("Finish Initialize OpenCL");

		ImageFormat format(CL_RGBA, CL_FLOAT);

		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;
		
		cl::size_t<3> region;
		region[0] = PITCH;
		region[1] = SIZE;
		region[2] = 1;

		::size_t* mapSize = new ::size_t(SIZE * sizeof(float));

		Image2D** matrixA = new Image2D*[SLICE];
		Image2D** matrixB = new Image2D*[SLICE];
		Image2D* matrixC = NULL;

		gettimeofday(&tpcalc, NULL);

		for(int row = 0; row < SLICE; row++){
			for(int i = 0; i < SLICE; i++){
				matrixA[i] = new Image2D(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, format, PITCH, SIZE, 0, A[row * SLICE + i]);
				kernel.setArg(2 * i, *(matrixA[i]));
			}
			for(int col = 0; col < SLICE; col++){
				for(int i = 0; i < SLICE; i++){
					matrixB[i] = new Image2D(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, format, PITCH, SIZE, 0, B[i * SLICE + col]);
					kernel.setArg(2 * i + 1, *(matrixB[i]));
				}
				matrixC = new Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, format, PITCH, SIZE);
				kernel.setArg(2 * SLICE, *matrixC);

				logTime((boost::format("Start Computation Block (%1%, %2%)...") % row % col).str());
				queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(PITCH, SIZE), NullRange);
				queue.finish();
				logTime((boost::format("Finish Computation Block (%1%, %2%)") % row % col).str());
				
				C[row * SLICE + col] = (float*) queue.enqueueMapImage(*matrixC, CL_TRUE, CL_MAP_READ, origin, region, mapSize, NULL);

				for(int i = 0; i < SLICE; i++){
					delete matrixB[i];
				}
				delete matrixC;
			}
			for(int i = 0; i < SLICE; i++){
				delete matrixA[i];
			}
		}

		delete mapSize;

		showFlops(RANK);

		if(vm["output"].as<std::string>() != "no"){
			ostringstream outa;
			ostringstream outb;
			ostringstream outc;

			for(int blockRow = 0; blockRow < SLICE; blockRow++){
				for(int row = 0; row < SIZE; row++){
					for(int blockCol = 0; blockCol < SLICE; blockCol++){
						for(int col = 0; col < SIZE; col++){
							outa << A[blockRow * SLICE + blockCol][row * SIZE + col] << " ";
							outb << B[blockRow * SLICE + blockCol][col * SIZE + row] << " ";
							outc << C[blockRow * SLICE + blockCol][row * SIZE + col] << " ";
						}
					}
					outa << endl;
					outb << endl;
					outc << endl;
				}
			}

			if(vm["output"].as<std::string>() == "file"){
				ofstream fouta("a.txt");
				ofstream foutb("b.txt");
				ofstream foutc("c.txt");

				fouta << outa.str();
				foutb << outb.str();
				foutc << outc.str();
			} else{
				cout << "==A==" << endl;
				cout << outa.str() << endl;
				cout << "==B==" << endl;
				cout << outb.str() << endl;
				cout << "==C==" << endl;
				cout << outc.str() << endl;
			}
		}

	} catch(Error& e) {
		cerr << "Error: " << e.what() << "(" << e.err() << ")" << endl;
	}

	for(int i = 0; i < SLICE * SLICE; i++){
		delete[] A[i];
		delete[] B[i];
	}
	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}

void logTime(std::string message) {
	gettimeofday(&tpend, NULL);
	float timeuse = (1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec) / 1000000.0;
	cout << "[" << timeuse << "] " << message << endl;
}

void showFlops(int rank){
	float timeuse = (1000000 * (tpend.tv_sec - tpcalc.tv_sec) + tpend.tv_usec - tpcalc.tv_usec) / 1000000.0;
	float gflops = pow(rank, 3) / 536870912 / timeuse;
	cout << format("%1% GFLOPS / %2%s Computing Time") % gflops % timeuse << endl;
}
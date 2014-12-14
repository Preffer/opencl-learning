__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void prod(const int SIZE, __read_only image2d_t A1, __read_only image2d_t A2, __read_only image2d_t B1, __read_only image2d_t B2, __write_only image2d_t C) {
 
	int col = get_global_id(0);
	int row = get_global_id(1);

	float sum = 0;

	for (int i = 0; i < SIZE; i++) {
		int2 coordA = (int2)(i, row);
		int2 coordB = (int2)(col, i);
		sum += read_imagef(A1, sampler, coordA).x * read_imagef(B1, sampler, coordB).x;
		sum += read_imagef(A2, sampler, coordA).x * read_imagef(B2, sampler, coordB).x;
	}
	write_imagef(C, (int2)(col, row), sum);
}

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void prod(__read_only image2d_t A, __read_only image2d_t B, __write_only image2d_t C, const int R) {
 
 	int col = get_global_id(0);
	int row = get_global_id(1);

	float sum = 0;

	for (int i = 0; i < R; i++) {
		sum += read_imagef(A, sampler, (int2)(i, row)).x * read_imagef(B, sampler, (int2)(col, i)).x;
	}

	write_imagef(C, (int2)(col, row), sum);
}
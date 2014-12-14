__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void prod(__read_only image2d_t A1, __read_only image2d_t A2, __read_only image2d_t A3, __read_only image2d_t A4, __read_only image2d_t B1, __read_only image2d_t B2, __read_only image2d_t B3, __read_only image2d_t B4, __write_only image2d_t C1, __write_only image2d_t C2, __write_only image2d_t C3, __write_only image2d_t C4, const int SIZE) {
 
	int col = get_global_id(0);
	int row = get_global_id(1);
	int block = get_global_id(2);

	float sum = 0;

	switch (block) {
		case 0:
			for (int i = 0; i < SIZE; i++) {
				int2 coordA = (int2)(i, row);
				int2 coordB = (int2)(col, i);
				sum += read_imagef(A1, sampler, coordA).x * read_imagef(B1, sampler, coordB).x;
				sum += read_imagef(A2, sampler, coordA).x * read_imagef(B3, sampler, coordB).x;
			}
			write_imagef(C1, (int2)(col, row), sum);
			break;

		case 1:
			for (int i = 0; i < SIZE; i++) {
				int2 coordA = (int2)(i, row);
				int2 coordB = (int2)(col, i);
				sum += read_imagef(A1, sampler, coordA).x * read_imagef(B2, sampler, coordB).x;
				sum += read_imagef(A2, sampler, coordA).x * read_imagef(B4, sampler, coordB).x;
			}
			write_imagef(C2, (int2)(col, row), sum);
			break;

		case 2:
			for (int i = 0; i < SIZE; i++) {
				int2 coordA = (int2)(i, row);
				int2 coordB = (int2)(col, i);
				sum += read_imagef(A3, sampler, coordA).x * read_imagef(B1, sampler, coordB).x;
				sum += read_imagef(A4, sampler, coordA).x * read_imagef(B3, sampler, coordB).x;
			}
			write_imagef(C3, (int2)(col, row), sum);
			break;

		case 3:
			for (int i = 0; i < SIZE; i++) {
				int2 coordA = (int2)(i, row);
				int2 coordB = (int2)(col, i);
				sum += read_imagef(A3, sampler, coordA).x * read_imagef(B2, sampler, coordB).x;
				sum += read_imagef(A4, sampler, coordA).x * read_imagef(B4, sampler, coordB).x;
			}
			write_imagef(C4, (int2)(col, row), sum);
			break;
	}
}
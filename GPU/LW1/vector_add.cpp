#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024
#define BILLION  1000000000L;
using namespace std;
cl_int errcode;
cl_event write_event[2];



void print_clbuild_errors(cl_program program,cl_device_id device)
	{
		cout<<"Program Build failed\n";
		size_t length;
		char buffer[2048];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		cout<<"--- Build log ---\n "<<buffer<<endl;
		exit(1);
	}

unsigned char ** read_file(const char *name) {
  size_t size;
  unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  FILE* fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s",name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  unsigned char **outputstr=(unsigned char **)malloc(sizeof(unsigned char *));
  *outputstr= (unsigned char *)malloc(size);
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s",name);
    exit(-1);
  }

  if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  printf("file size %d\n",size);
  printf("-------------------------------------------\n");
  snprintf((char *)*outputstr,size,"%s\n",*output);
  printf("%s\n",*outputstr);
  printf("-------------------------------------------\n");
  return outputstr;
}
void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}


void checkError(int status, const char *msg) {
	if(status!=CL_SUCCESS)
		printf("%s\n",msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

int main()
{
     char char_buffer[STRING_BUFFER_LEN];
     cl_platform_id platform;
     cl_device_id device;
     cl_context context;
     cl_context_properties context_properties[] =
     {
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
     };
     cl_command_queue queue;
     cl_program program;
     cl_kernel kernel;



//--------------------------------------------------------------------
const unsigned N = 40960000; //0;
float *input_a;//=(float *) malloc(sizeof(float)*N);
float *input_b;//=(float *) malloc(sizeof(float)*N);
float *output=(float *) malloc(sizeof(float)*N);
float *ref_output=(float *) malloc(sizeof(float)*N);
cl_mem input_a_buf; // num_devices elements
cl_mem input_b_buf; // num_devices elements
cl_mem output_buf; // num_devices elements
int status;

struct timespec startc, stopc, startbuf, stopbuf;
	time_t start,end;
	double diff;
	double diffc;

    time (&start);
     clGetPlatformIDs(1, &platform, NULL);
     clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

     context_properties[1] = (cl_context_properties)platform;
     clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
     context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
     queue = clCreateCommandQueue(context, device, 0, NULL);

     unsigned char **opencl_program=read_file("vector_add.cl");
     program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
     if (program == NULL)
	{
         printf("Program creation failed\n");
         return 1;
	}
     int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	 if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
     kernel = clCreateKernel(program, "vector_add", NULL);
 // Input buffers.

 /*
    input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
       N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");
*/
	  clock_gettime( CLOCK_REALTIME,&startbuf);
		input_a_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR|CL_MEM_READ_ONLY,
       N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR|CL_MEM_READ_ONLY,
        N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");

	  clock_gettime( CLOCK_REALTIME,&stopbuf);
		diffc= stopbuf.tv_sec-startbuf.tv_sec + (double)(stopbuf.tv_nsec -startbuf.tv_nsec)/(double)BILLION;
		printf("Buffers took %.8f seconds to be created\n\n", diffc);
    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
	  cl_event kernel_event,finish_event;
		//Maping
		input_a = (float *)clEnqueueMapBuffer(queue, input_a_buf, CL_TRUE,
					CL_MAP_WRITE,0, N* sizeof(float), 0, NULL, &write_event[0],&errcode);
			checkError(errcode, "Failed to map input A");

			input_b = (float *)clEnqueueMapBuffer(queue, input_b_buf, CL_TRUE,
					CL_MAP_WRITE, 0,N* sizeof(float), 0, NULL, &write_event[1],&errcode);
			checkError(errcode, "Failed to map input B");
			// Map to host memory
				//output = (float *)clEnqueueMapBuffer(queue, output_buf, CL_TRUE,
				//		CL_MAP_READ, 0,N* sizeof(float),  0, NULL, NULL,&errcode);
				//checkError(errcode, "Failed to map output");

size_t size;
				// Wait for a specific event



		time (&start);
		clock_gettime( CLOCK_REALTIME,&startc);
		for(unsigned j = 0; j < N; ++j) {
		      input_a[j] = rand_float();
		      input_b[j] = rand_float();
		      ref_output[j] = input_a[j] + input_b[j];
		      //printf("ref %f\n",ref_output[j]);
		    }
		time (&end);
		clock_gettime( CLOCK_REALTIME,&stopc);

		diff = difftime (end,start);
		diffc= stopc.tv_sec-startc.tv_sec + (double)(stopc.tv_nsec -startc.tv_nsec)/(double)BILLION;
			printf ("CPU took %.8lf seconds to generate the vectors.\n\n", diffc );

	/*
    status = clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE,
        0, N* sizeof(float), input_a, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue, input_b_buf, CL_FALSE,
        0, N* sizeof(float), input_b, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");
*/
    // Set kernel arguments.
    unsigned argi = 0;
	  clock_gettime( CLOCK_REALTIME,&startc);
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3");


		clEnqueueUnmapMemObject(queue,input_a_buf,input_a,0,NULL,NULL);
		clEnqueueUnmapMemObject(queue,input_b_buf,input_b,0,NULL,NULL);

    const size_t global_work_size = N;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        &global_work_size, NULL, 2, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");
		status=clWaitForEvents(1,&kernel_event);
			checkError(status, "Failed  wait");
			// Profile Events
			clGetEventProfilingInfo(kernel_event,CL_PROFILING_COMMAND_START,8,&start,&size);
			clGetEventProfilingInfo(kernel_event,CL_PROFILING_COMMAND_END,8,&end,&size);

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
        0, N* sizeof(float), output, 1, &kernel_event, &finish_event);

   time (&end);
	 clock_gettime( CLOCK_REALTIME,&stopc);
   diff = difftime (end,start);
	 diffc= stopc.tv_sec-startc.tv_sec + (double)(stopc.tv_nsec -startc.tv_nsec)/(double)BILLION;
   diff++;
	 printf ("GPU took %.8lf seconds to run.\n\n", diffc );
// Verify results.
bool pass = true;

for(unsigned j = 0; j < N && pass; ++j) {
			//	if((input_b[j]-1.0) > 1.0e-5f) {
			if(fabsf(output[j] - ref_output[j]) > 1.0e-5f) {
        printf("Failed verification @ index %d\nOutput: %f\nReference: %f\n",
            j, output[j], ref_output[j]);
				//printf("%d:%f\n",j,input_a[j]);
				pass = false;
      }
}
float sum_vect=0;
for(unsigned int i=0; i< N; i++) {
	sum_vect+=output[i];
}

printf("Average of the vector is = %f\n\n",sum_vect/N);
    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
clReleaseKernel(kernel);
clReleaseCommandQueue(queue);
clReleaseMemObject(input_a_buf);
clReleaseMemObject(input_b_buf);
clReleaseMemObject(output_buf);
clReleaseProgram(program);
clReleaseContext(context);

size_t max_work_group_size;
     clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),&max_work_group_size,NULL);
     printf("%-40s = %d\n\n", "CL_DEVICE_MAX_WORK_GROUP_SIZE", max_work_group_size);

//--------------------------------------------------------------------






     clFinish(queue);

     return 0;
}

#include<stdio.h>
#include<fstream>
//////////////////////////////////////////////////////////////
__global__ void Conv3x3_2d(const double *g_input,const double *g_weight3x3_2d,double *g_output3x3_2d) { 
    __shared__ double s_weight3x3_2d[9];    //using shared mem for only weight
    for(int i=0;i<9;i++){
        s_weight3x3_2d[i] = g_weight3x3_2d[i];
    }
    double sum=0;
    int x= blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y*224+x; //for c[i] = sum;

    for(int z=0;z<300;z++) {
      sum=0;    //reset sum 
      for(int m=0;m<3;m++){   //3x3
          for(int n=0;n<3;n++){
              sum += g_input[z*226*226 + (y+m)*226+(x+n)] * s_weight3x3_2d[m*3 +n];   //convolution
          }
      }
      g_output3x3_2d[i+ z*(224)*(224)] =sum;    //return sum result to output
    }
}


////////////////////////////////////////////////////////////
__global__ void Conv1x1_3d(const double *g_input,const double *g_weight1x1_3d,double *g_output1x1_3d){
    __shared__ double s_weight1x1_3d[1*1*300];    //using shared mem for only weight
    for(int i=0;i<1*1*300;i+=2){
        s_weight1x1_3d[i]=g_weight1x1_3d[i];
        s_weight1x1_3d[i+1]=g_weight1x1_3d[i+1];
    }
     double sum=0;
    int x= blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int i= z*224*224+y*224+x;

    for(int a=0;a<900;a++) {    //output channel
      sum=0;    //reset for next convolution
      for(int v=0;v<300;v++){
        for(int m=0;m<1;m++){   //1x1
         for(int n=0;n<1;n++){
             if(a %2 ==0){
              sum += g_input[(z+v)*226*226+(y+m+1)*226+(x+n+1)] *1; // s_weight1x1_3d[v+m+n];   //convolution
             }
             else{
              sum += g_input[(z+v)*226*226+(y+m+1)*226+(x+n+1)] * -1; // -s_weight1x1_3d[v+m+n];   //convolution
             }
         }
       }
      }
      g_output1x1_3d[i+ a*224*224] =sum;   //return sum result to output
    }
}

/////////////////////////////////////////////////////////////
__global__ void Conv3x3_3d(const double *g_input,const double *g_weight3x3_3d,double *g_output3x3_3d){
      __shared__ double s_weight3x3_3d[3*3*300];    //use shared mem
    for(int i=0;i<3*3*300;i+=2){        //use loop unrolling 
        s_weight3x3_3d[i]=g_weight3x3_3d[i];
        s_weight3x3_3d[i+1]=g_weight3x3_3d[i+1];
    }
     double sum=0;
    int x= blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int i= z*224*224+y*224+x;

    for(int a=0;a<900;a++) {  //output channel
      sum=0;    //reset for next convolution
      for(int v=0;v<300;v++){
        for(int m=0;m<3;m++){ //3x3
         for(int n=0;n<3;n++){   
             sum += g_input[(z+v)*226*226 + (y+m)*226+(x+n)] * s_weight3x3_3d[v*3*3 + m*3 +n]; //convolution
         }
       }
      }
      g_output3x3_3d[i+ a*224*224] =sum;   //return sum result to output
    }
}

//////////////////////////////////////////////////////////////
void input(double * input)
{
    for(int z=0;z<300;z++)
        for(int i=0;i<226;i++)
            for(int j=0;j<226;j++)
            {
                if(i == 0 || j == 0 || j == 225 || i == 225)
                    input[z*226*226+i*226+j] = 0;
                else
                    input[z*226*226+i*226+j] = 1;
            }
}

void Init_weight3x3_3d(double * weight)
{
    for(int k=0;k<900;k++)
        for(int z=0;z<300;z++)
            for(int i=0;i<3;i++)
                for(int j=0;j<3;j++){
                    if((i*3 + j) % 2 == 0)
                        weight[k*3*3*300 + z*3*3+i*3+j] = 1;
                    else
                        weight[k*3*3*300 + z*3*3+i*3+j] = -1;
                }
}

void Init_weight3x3_2d(double * weight)
{
    for(int i=0;i<9;i++)
        weight[i] = 1;
}

void Init_weight1x1_3d(double * weight)
{
    for(int k = 0;k<900;k++)
        for(int z=0;z<300;z++){
            if(k % 2 == 0)
                weight[z] = 1;	//짝수
            else
                weight[z] = -1;	//홀수
        }
}

void save_file(double * output3x3_3d,double * output1x1_3d,double * output3x3_2d)
{
    FILE * fp = fopen("3d_Result3x3.txt","w");
    FILE * fp2 = fopen("3d_Result1x1.txt","w");
    FILE * fp3 = fopen("2d_Result3x3.txt","w");
    for(int i=0;i<224*224*900;i++)
    {
        fprintf(fp,"%f\n",output3x3_3d[i]);
        fprintf(fp2,"%f\n",output1x1_3d[i]);
    }
    for(int i=0;i<224*224*300;i++)
        fprintf(fp3,"%f\n",output3x3_2d[i]);

    fclose(fp3);
    fclose(fp2);
    fclose(fp);
}
int main(){
    
    double * input_3d = (double*)malloc(sizeof(double)*226*226*300);    //input Feature Map
    
    double * output3x3_2d = (double*)malloc(sizeof(double)*224*224*300);//Output Feature Map
    double * output1x1_3d = (double*)malloc(sizeof(double)*224*224*900);
    double * output3x3_3d = (double*)malloc(sizeof(double)*224*224*900);

    double * weight3x3_2d = (double*)malloc(sizeof(double)*3*3);		//Weight
    double * weight1x1_3d = (double*)malloc(sizeof(double)*1*1*300*900);
    double * weight3x3_3d = (double*)malloc(sizeof(double)*3*3*300*900);
    
    double * g_input, * g_output3x3_3d, * g_output1x1_3d, * g_output3x3_2d, * g_weight3x3_3d, * g_weight1x1_3d, * g_weight3x3_2d;
    
    cudaEvent_t start, stop3x3_3d, stop1x1_3d, stop3x3_2d;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop3x3_3d);
    cudaEventCreate(&stop1x1_3d);
    cudaEventCreate(&stop3x3_2d);
    //Initialization Input Feature Map & Weight
    input(input_3d);
    Init_weight3x3_3d(weight3x3_3d);
    Init_weight1x1_3d(weight1x1_3d);
    Init_weight3x3_2d(weight3x3_2d);
    
    cudaMalloc((void**)&g_input,sizeof(double)*226*226*300);
    cudaMalloc((void**)&g_output3x3_3d,sizeof(double)*224*224*900);
    cudaMalloc((void**)&g_output1x1_3d,sizeof(double)*224*224*900);
    cudaMalloc((void**)&g_output3x3_2d,sizeof(double)*224*224*300);
    cudaMalloc((void**)&g_weight3x3_3d,sizeof(double)*3*3*300*900);
    cudaMalloc((void**)&g_weight1x1_3d,sizeof(double)*1*1*300*900);
    cudaMalloc((void**)&g_weight3x3_2d,sizeof(double)*3*3);
    
    cudaMemcpy(g_input,input_3d,sizeof(double)*226*226*300,cudaMemcpyHostToDevice);
    cudaMemcpy(g_weight3x3_3d,weight3x3_3d,sizeof(double)*3*3*300*900,cudaMemcpyHostToDevice);
    cudaMemcpy(g_weight1x1_3d,weight1x1_3d,sizeof(double)*1*1*300*900,cudaMemcpyHostToDevice);
    cudaMemcpy(g_weight3x3_2d,weight3x3_2d,sizeof(double)*3*3,cudaMemcpyHostToDevice);

    /*
    Project
    Block 및 Grid 선언 자유, 주어진 3개의 Kernel Conv3x3_3d, Conv1x1_3d, Conv3x3_2d를 구현(Kernel명 및 Argument 유지)
    가능한 빠른 Performance를 가지는 Kernel을 구현할 것

    결과는 Text File을 통해서 확인, cudaEvent 관련 코드는 성능 측정을 위한 코드이니 수정하지 말것
	
	Kernel 별 배점
	Conv3x3_2d = 20%
	Conv1x1_3d = 35%
	Conv3x3_3d = 45%
    */
////////////////////////////////////////아래의 3개 Kernel을 구현 ///////////////////////////////////
    dim3 DimGrid(7,7);    //7by7 for 7*32 = 224 
    dim3 DimBlock(32,32,1); //32*32 = 1024 use max thread
    cudaEventRecord(start);
    Conv3x3_2d<<<DimGrid,DimBlock>>>(g_input,g_weight3x3_2d,g_output3x3_2d);
    cudaEventRecord(stop3x3_2d);
	
    Conv1x1_3d<<<DimGrid,DimBlock>>>(g_input,g_weight1x1_3d,g_output1x1_3d);
    cudaEventRecord(stop1x1_3d);
	
    Conv3x3_3d<<<DimGrid,DimBlock>>>(g_input,g_weight3x3_3d,g_output3x3_3d); 
    cudaEventRecord(stop3x3_3d);

    cudaEventSynchronize(stop3x3_3d);    
/////////////////////////////////////////////////////////////////////////////////////////////////

    float milliseconds[3]={0};
    cudaEventElapsedTime(&milliseconds[0],start,stop3x3_2d);
    cudaEventElapsedTime(&milliseconds[1],stop3x3_2d,stop1x1_3d);
    cudaEventElapsedTime(&milliseconds[2],stop1x1_3d,stop3x3_3d);
    printf("Execution Time \n Convolution3x3_2d : %f\n Convolution1x1_3d : %f\n Convolution3x3_3d : %f\n",milliseconds[0],milliseconds[1],milliseconds[2]);
    
    cudaMemcpy(output3x3_3d,g_output3x3_3d,sizeof(double)*224*224*900,cudaMemcpyDeviceToHost);
    cudaMemcpy(output1x1_3d,g_output1x1_3d,sizeof(double)*224*224*900,cudaMemcpyDeviceToHost);
    cudaMemcpy(output3x3_2d,g_output3x3_2d,sizeof(double)*224*224*300,cudaMemcpyDeviceToHost);
  
    save_file(output3x3_3d,output1x1_3d,output3x3_2d);
    cudaFree(g_input);
    cudaFree(g_weight3x3_3d);
    cudaFree(g_weight3x3_2d);
    cudaFree(g_weight1x1_3d);
    cudaFree(g_output3x3_3d);
    cudaFree(g_output3x3_2d);
    cudaFree(g_output1x1_3d);

    free(output3x3_3d);
    free(output1x1_3d);
    free(output3x3_2d);
    free(input_3d);
    free(weight3x3_3d);
    free(weight1x1_3d);
    free(weight3x3_2d);
}

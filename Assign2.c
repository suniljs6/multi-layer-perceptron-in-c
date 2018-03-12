#include <stdio.h>
#include <string.h>
#include <math.h>
#include<stdlib.h>
#include<time.h>
#include<limits.h>
int n;
double sigmoid(double x){
	double y,exp_value;
	exp_value=exp(-x);
	y=1/(1+exp_value);
	return y;
}

double sigmoid_bar(double value){
	double result = sigmoid(value);
	double res1 = result*(1-result);
	return res1;
}


int main(){
        printf("enter number of neurons in the hidden layer");	
	scanf("%d",&n);
	double X2[20];
	double z[20];
	double Wji[20][n];
	double Wkj[20][10];
	double D_Wkj_ofNew[20][20];
	double D_Wji_ofNew[20][20];
	double D_Wkj[20][10];
	double D_Wji[20][n];
	double netj[20];
	double netk[20];
		
	srand(time(NULL));
	for(int i=0; i<17; i++){
		for(int j=0; j<n; j++){
			Wji[i][j] = ((double)(rand()%3-1)/100.0);
		}
	}
	for(int j=0; j<n+1; j++){
		for(int k=0; k<10; k++){
			Wkj[j][k] = ((double)(rand()%5+1)/100.0);
		}
	}


	FILE *lf = fopen("train1.txt", "r");
	double A[3000][50];
	while(getc(lf)!= EOF){
		for(int i=0;i<2216;i++){
			for(int j=0;j<17;j++){
				fscanf(lf,"%lf",&A[i][j]);
			}
		}
	}

	int epho=0;
	while( epho < 100){
		for(int i=0; i< 2216; i++){
			double X[20];
			for(int j=1; j<17; j++){
				X[j-1] = A[i][j];
			}
			int class_label = (int)A[i][0];// assigning the label value
			double T[15];
			for(int j=0; j<10; j++){
				T[j]=0.0;
			}	
			T[class_label-1]=1.0;

			double X1_bias[20];
			X1_bias[0]=1.0;
			for(int j=1;j<17;j++){
				X1_bias[j]=X[j-1];
			}
			for(int j=0;j<n;j++){
				double sum=0;
				for(int i=0;i<17;i++){
					sum=sum+X1_bias[i]*Wji[i][j];
				}
				netj[j]=sum;
				X2[j]=sigmoid(sum);//values on hidden layer
			}

			double X2_bias[20];
			X2_bias[0]=1.0;
			for(int k=1;k<n;k++){
				X2_bias[k]=X2[k-1];
			}

			for(int k=0;k<10;k++){
				double sum1=0;
				for(int j=0;j<n+1;j++){
					sum1=sum1+X2_bias[j]*Wkj[j][k];  
				}
				netk[k]=sum1;
				z[k]=sigmoid(sum1);//values on output
			}
			double delta_output [20]; 
			for(int k=0; k<10; k++){
				delta_output[k] =  ( T[k]-z[k] ) * sigmoid_bar(netk[k]);//error on output
				//delta_output[k] =  ( T[k]-z[k] );
			}		  
			X2_bias[0]=1.0;
			for(int k=1;k<n;k++){
				X2_bias[k]=X2[k-1];
			}
			for(int j=0; j<n+1; j++){
				for(int k=0; k<10; k++){       
					D_Wkj[j][k] = 0.001 * X2_bias[j] * delta_output[k];//DELTA_Wkj of hidden and output
				}
			}
			double delta_hidden[10]; 
			for(int j=1; j<n+1; j++){
				double sum = 0;
				for(int r=0; r<10; r++){
					sum = sum+( delta_output[r] * Wkj[j][r] * sigmoid_bar(netj[j-1]) );
				}
				delta_hidden[j-1] = sum;//error on hidden
			}
			X1_bias[0]=1.0;
			int h_index;
			for(int j=1;j<17;j++){
				X1_bias[j]=X[j-1];
			}
			for(int j=0; j<n; j++){
				for(int i=0; i<17; i++){
					D_Wji[i][j] = 0.001*X1_bias[i]*delta_hidden[j];//DELTA_Wji of input and hidden
				}
			}
			for(int l=0; l<17; l++){
				for(int j=0; j<n; j++)
					Wji[l][j] = Wji[l][j] + D_Wji[l][j];//updating weights of Wji
			}
			for(int j=0; j<n+1; j++){
				for(int k=0; k<10; k++)
					Wkj[j][k] = Wkj[j][k] + D_Wkj[j][k];//updating weights of Wkj
			}
		}        

		epho++; 
	}
	FILE *f1= fopen("test.txt", "r");
	double Atest[3000][50];
	while(getc(f1)!= EOF){
		for(int i=0;i<998;i++){
			for(int j=0;j<17;j++){
				fscanf(f1,"%lf",&Atest[i][j]);
			}
		}
	}
	int label[1000];
	double test_out[1000][20];
	double X1_bias[20];
	double Xtest[20];
	int target_label[1000];
	for(int i=0;i<998;i++){  
		for(int j=1; j<17; j++){
			Xtest[j-1] = Atest[i][j];
		}
		
		target_label[i]=(int)Atest[i][0];
		X1_bias[0]=1.0;
		for(int j=1;j<17;j++){
			X1_bias[j]=Xtest[j-1];
		}
		for(int j=0;j<n;j++){
			double sum=0;
			for(int l=0;l<17;l++){
				sum=sum+X1_bias[l]*Wji[l][j];
			}
			X2[j]=sigmoid(sum);
		}             
		double X2_bias[20];
		X2_bias[0]=1.0;
		for(int k=1;k<n;k++){
			X2_bias[k]=X2[k-1];
		}
		for(int k=0;k<10;k++){
			double sum1=0;
			for(int j=0;j<n+1;j++){
				sum1=sum1+X2_bias[j]*Wkj[j][k];  
			}
			z[k]=sigmoid(sum1);
		}
		double max=INT_MIN * 1.0;
		int index;
		printf("%d:",i);
		for(int p=0;p<10;p++){
			printf("%lf ",z[p]);
			if(z[p] > max){
				max = z[p];
				index = p;
			}
		}
		printf("label:%d\n",index+1);
		label[i]=index+1;
	}
	int count =0;      
	for(int i=0;i<998;i++){
		if(label[i]==target_label[i]){
			count++;
		}
		
	}
	printf("count :%d\n",count);
	float percent = (float)((float)count/998.00)*100.0; 
	printf("accuracy:%f\n",percent);  
	return 0;
}

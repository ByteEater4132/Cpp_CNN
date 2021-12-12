#include<iostream>
#include"matiax.h"

using namespace std;
using namespace matiax_tool;
constexpr auto input_num = 10;
constexpr auto hidden_num = 4;
constexpr auto output_num = 2;
constexpr auto study_num = 0.1;

typedef struct train_data
{
	float data[input_num][1];
	struct train_data* next;
}TrainData;

float Input[input_num][1];
float Input_T[1][input_num];
float t[output_num][1] = { {0.99},{0.01} };
float m_ij[hidden_num][input_num];
float m_jk[output_num][hidden_num];
float m_jk_T[hidden_num][output_num];
float X_hidden[hidden_num][1];
float X_output[output_num][1];
float O_hidden[hidden_num][1];
float O_hidden_T[1][hidden_num];
float O_output[output_num][1];
float O_output_T[1][output_num];
float e_output[output_num][1];
float e_output_[output_num][1];
float e_hidden[hidden_num][1];
float e_hidden_[hidden_num][1];
float delta_m_jk[output_num][hidden_num];
float delta_m_ij[hidden_num][input_num];
float S_K[output_num][1];
float S_K_1[output_num][1];
float S_J[hidden_num][1];
float S_J_1[output_num][1];
float E_jk[output_num][1];
float E_ij[hidden_num][1];

int Init(void);
int get_data(float* in, float* out);
int train_cnn(void);

int main(void)
{
	init_M(&m_ij[0][0], hidden_num, input_num);
	init_M(&m_jk[0][0], output_num, hidden_num);

	cout << "w_ij:\n";
	print_matiax(&m_ij[0][0], hidden_num, input_num);
	cout << endl;
	cout << "w_jk:\n";
	print_matiax(&m_jk[0][0], output_num, hidden_num);
	cout << endl;

	//print_matiax(&Input[0][0], input_num, 1);
	print_matiax(&t[0][0], output_num, 1);
	

	for (int count = 0; count < 10; count++)
	{
		Init();
		cout << "Input:\n";
		print_matiax(&Input[0][0], input_num, 1);
		for (int count1 = 0; count1 < 5; count1++)
		{
			if (train_cnn())
			{
				print_matiax(&O_output[0][0], output_num, 1);
				cout << endl;
			}
		}
	}
	
	
	/*
	//测试矩阵点乘函数take();
	float A1[3][3] = { {1,2,3},{4,5,6},{7,8,9} };
	float A2[3][2] = { {1,1},{1,2},{2,1} };
	float* retx = (float*)malloc(3 * 2 * sizeof(float));
	//float* a;
	take(&A1[0][0], 3, 3, &A2[0][0], 3, 2, retx);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			printf("%-6.1f", *(retx + i * 2 + j));
		}
		cout << endl;
	}
	printf("%.1f\n", Sigmoid(0));

	//测试权值初始化函数init_M();
	float m_ij[hidden_num][input_num];
	float m_jk[output_num][hidden_num];
	init_M(&m_ij[0][0], hidden_num, input_num);
	init_M(&m_jk[0][0], output_num, hidden_num);
	// 函数功能正常
	cout << "Mij:\n";
	for (int i = 0; i < hidden_num * input_num; i++)
	{
		cout << *(&m_ij[0][0] + i) << "  ";
		if ((i + 1) % input_num == 0)
			cout << endl;
	}

	//测试矩阵转置函数matiax_T();
	float m_T[input_num][hidden_num];
	float m_jk_T[hidden_num][output_num];
	matiax_T(&m_ij[0][0], hidden_num, input_num, &m_T[0][0]);
	matiax_T(&m_jk[0][0], hidden_num, output_num, &m_jk_T[0][0]);
	cout << "Mij_T:\n";
	for (int i = 0; i < hidden_num * input_num; i++)
	{
		cout << *(&m_T[0][0] + i) << "  ";
		if ((i + 1) % hidden_num == 0)
			cout << endl;
	}
	

	//测试调整输入值区间函数reinput();
	float Input[input_num][1];
	for (int i = 0; i < input_num; i++)
	{
		*(&Input[0][0] + i) = rand() % 255;
		cout << *(&Input[0][0] + i) << "  ";
	}
	cout << endl;
	reInput(&Input[0][0], input_num, 0, 255);
	
	
	//测试matiax_sig()；计算部分
	float X_hidden[hidden_num][1];
	float O_hidden[hidden_num][1];
	float X_output[output_num][1];
	float O_output[output_num][1];
	float e_output[output_num][1];
	float e_hidden[hidden_num][1];
	float delta_m_jk[output_num][hidden_num];
	float detal_m_ij[hidden_num][input_num];
	//设定目标值，此处为随机设置
	float t[output_num][1] = { {0.9},{0.1} };

	take(&m_ij[0][0], hidden_num, input_num, &Input[0][0], input_num, 1, &X_hidden[0][0]);
	cout << "Input:\n";
	print_matiax(&Input[0][0], input_num, 1);
	cout << "m_ij:\n";
	print_matiax(&m_ij[0][0], hidden_num, input_num);
	cout << "m_jk:\n";
	print_matiax(&m_jk[0][0], output_num, hidden_num);
	cout << "X_hidden:\n";
	print_matiax(&X_hidden[0][0], hidden_num, 1);
	matiax_sig(&X_hidden[0][0], hidden_num, &O_hidden[0][0]);
	cout << "O_hidden:\n";
	print_matiax(&O_hidden[0][0], hidden_num, 1);
	take(&m_jk[0][0], output_num, hidden_num, &O_hidden[0][0], hidden_num, 1, &X_output[0][0]);
	matiax_sig(&X_output[0][0], output_num, &O_output[0][0]);
	cout << "X_output:\n";
	print_matiax(&X_output[0][0], output_num, 1);
	cout << "O_output:\n";
	print_matiax(&O_output[0][0], output_num, 1);
	
	reduce(&t[0][0], &O_output[0][0], output_num,1, &e_output[0][0]);
	cout << "t:\n";
	print_matiax(&t[0][0], output_num, 1);
	cout << "e_output:\n";
	print_matiax(&e_output[0][0], output_num, 1);
	take(&m_jk_T[0][0], hidden_num, output_num, &e_output[0][0], output_num, 1, &e_hidden[0][0]);
	cout << "e_hidden:\n";
	print_matiax(&e_hidden[0][0], hidden_num, 1);

	float e_output_[output_num][1];
	float S_K[output_num][1];
	float E_jk[output_num][1];
	float O_output_T[1][output_num];
	matiax_T(&O_output[0][0], output_num, 1, &O_output_T[0][0]);
	C_take(&e_output[0][0], output_num, 1, -1, &e_output_[0][0]);
	S_k(&m_jk[0][0], output_num, hidden_num, &O_output[0][0], output_num, 1, &S_K[0][0]);
	hadamard_take(&e_output_[0][0], output_num, 1, &S_K[0][0], output_num,1,&E_jk[0][0]);
	take(&E_jk[0][0], output_num, 1, &O_output_T[0][0], 1, output_num, &delta_m_jk[0][0]);
	C_take(&delta_m_jk[0][0], output_num, hidden_num, study_num, &delta_m_jk[0][0]);
	cout << "delta_m_jk:\n";
	print_matiax(&delta_m_jk[0][0], output_num, hidden_num);

	reduce(&m_jk[0][0], &delta_m_jk[0][0], output_num, hidden_num, &m_jk[0][0]);
	cout << "new m_jk:\n";
	print_matiax(&m_jk[0][0], output_num, hidden_num);

	take(&m_jk[0][0], output_num, hidden_num, &O_hidden[0][0], hidden_num, 1, &X_output[0][0]);
	matiax_sig(&X_output[0][0], output_num, &O_output[0][0]);
	cout << "X_output:\n";
	print_matiax(&X_output[0][0], output_num, 1);
	cout << "O_output:\n";
	print_matiax(&O_output[0][0], output_num, 1);
	*/
	return 0;
}

int Init(void)
{
	
	
	for (int i = 0; i < input_num; i++)
	{
		if (i <= (input_num / 2))
		{
			*(&Input[0][0] + i) = (rand() % 127)+127;
			//cout << *(&Input[0][0] + i) << "  ";
		}
		else
		{
			*(&Input[0][0] + i) = rand() % 127;
			//cout << *(&Input[0][0] + i) << "  ";
		}
	}
	//cout << endl;
	reInput(&Input[0][0], input_num, 0, 254);

	return 1;
}

int get_data(float* in, float* out)
{
	
}

int train_cnn(void)
{
	//cout << "w_ij:\n";
	//print_matiax(&m_ij[0][0], hidden_num, input_num);
	//cout << endl;
	//cout << "w_jk:\n";
	//print_matiax(&m_jk[0][0], output_num, hidden_num);
	//cout << endl;

	matiax_T(&m_jk[0][0], hidden_num, output_num, &m_jk_T[0][0]);
	matiax_T(&Input[0][0], input_num, 1, &Input_T[0][0]);

	take(&m_ij[0][0], hidden_num, input_num, &Input[0][0], input_num, 1, &X_hidden[0][0]);

	matiax_sig(&X_hidden[0][0], hidden_num, &O_hidden[0][0]);
	matiax_T(&O_hidden[0][0], hidden_num, 1, &O_hidden_T[0][0]);
	cout << "O_hidden:\n";
	print_matiax(&O_hidden[0][0], hidden_num, 1);

	take(&m_jk[0][0], output_num, hidden_num, &O_hidden[0][0], hidden_num, 1, &X_output[0][0]);

	matiax_sig(&X_output[0][0], output_num, &O_output[0][0]);
	matiax_T(&O_output[0][0], output_num, 1, &O_output_T[0][0]);
	cout << "O_output:\n";
	print_matiax(&O_output[0][0], output_num, 1);

	reduce(&t[0][0], &O_output[0][0], output_num, 1, &e_output[0][0]);
	cout << "e_output:\n";
	print_matiax(&e_output[0][0], output_num, 1);
	C_take(&e_output[0][0], output_num, 1, -1, &e_output_[0][0]);
	cout << "e_output_:\n";
	print_matiax(&e_output_[0][0], output_num, 1);

	take(&m_jk_T[0][0], hidden_num, output_num, &e_output[0][0], output_num, 1, &e_hidden[0][0]);
	C_take(&e_hidden[0][0], hidden_num, 1, -1, &e_hidden_[0][0]);


	S_jk(&m_jk[0][0], output_num, hidden_num, &O_output[0][0], output_num, 1, &S_K[0][0]);
	cout << "S_K:\n";
	print_matiax(&S_K[0][0], output_num, 1);
	
	
	one_reduce(&S_K[0][0], output_num, 1, &S_K_1[0][0]);
	cout << "S_K_1:\n";
	print_matiax(&S_K_1[0][0], output_num, 1);

	

	hadamard_take(&e_output_[0][0], output_num, 1, &S_K[0][0], output_num, 1, &E_jk[0][0]);
	hadamard_take(&E_jk[0][0], output_num, 1, &S_K_1[0][0], output_num, 1, &E_jk[0][0]);

	

	cout << "E_jk:\n";
	print_matiax(&E_jk[0][0], output_num, 1);
	take(&E_jk[0][0], output_num, 1, &O_hidden_T[0][0], 1, hidden_num, &delta_m_jk[0][0]);
	C_take(&delta_m_jk[0][0], output_num, hidden_num, study_num, &delta_m_jk[0][0]);


	


	S_jk(&m_ij[0][0], hidden_num, input_num, &O_hidden[0][0], hidden_num, 1, &S_J[0][0]);
	cout << "S_J:\n";
	print_matiax(&S_J[0][0], hidden_num, 1);

	one_reduce(&S_J[0][0], hidden_num, 1, &S_J_1[0][0]);
	cout << "S_J_1:\n";
	print_matiax(&S_J_1[0][0], hidden_num, 1);

	hadamard_take(&e_hidden_[0][0], hidden_num, 1, &S_K[0][0], hidden_num, 1, &E_ij[0][0]);
	hadamard_take(&E_ij[0][0], hidden_num, 1, &S_J_1[0][0], hidden_num, 1, &E_ij[0][0]);
	take(&E_ij[0][0], hidden_num, 1, &Input_T[0][0], 1, input_num, &delta_m_ij[0][0]);
	C_take(&delta_m_ij[0][0], hidden_num, input_num, study_num, &delta_m_ij[0][0]);

	
	
	cout << "delta w_ij:\n";
	print_matiax(&delta_m_ij[0][0], hidden_num, input_num);
	cout << endl;
	cout << "delta w_jk:\n";
	print_matiax(&delta_m_jk[0][0], output_num, hidden_num);
	cout << endl;

	reduce(&m_ij[0][0], &delta_m_ij[0][0], hidden_num, input_num, &m_ij[0][0]);
	reduce(&m_jk[0][0], &delta_m_jk[0][0], output_num, hidden_num, &m_jk[0][0]);

	
		cout << "w_ij:\n";
		print_matiax(&m_ij[0][0], hidden_num, input_num);
		cout << endl;
		cout << "w_jk:\n";
		print_matiax(&m_jk[0][0], output_num, hidden_num);
		cout << endl;
	

	return 1;
}
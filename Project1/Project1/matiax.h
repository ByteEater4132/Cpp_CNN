#pragma once
#include<iostream>
#include<stdlib.h>
#include<math.h>
using namespace std;
namespace matiax_tool
{
    void print_matiax(float* m, int i, int j);
    float* take(float* a1, int i1, int j1, float* a2, int i2, int j2, float* retx);
    float* hadamard_take(float* a1, int i1, int j1, float* a2, int i2, int j2, float* retx);
    float* C_take(float* m, int i, int j, float c, float* r);
    float* reduce(float* a1, float* a2, int i,int j, float* r);
    float* one_reduce(float* a, int i, int j, float* r);
    float Sigmoid(float i);
    float* init_M(float* m,int i,int j);
    float* matiax_T(float* m, int i, int j,float* r);
    float* reInput(float* m, int i,int in_up,int _dw);
    float* matiax_sig(float* m, int i, float* r);
    float* S_jk(float* m, int i1, int j1, float* o, int i2, int j2,float* r);
}
// 矩阵的点乘
float* matiax_tool::take(float* a1, int i1, int j1, float* a2, int i2, int j2,float *retx)
{
    if (j1 != i2) return 0;
        float sum;
        for (int i = 0; i < i1; i++)
        {
            for (int j = 0; j < j2; j++)
            {
                sum = 0;
                for (int k = 0; k < j1; k++)
                {
                    //printf("%.1f * %.1f\n", *(a1 + i * j1 + k), *(a2 + k * j2 + j));
                    sum += (*(a1 + i * j1 + k)) * (*(a2 + k * j2 + j));
                }
                retx[i * j2 + j] = sum;
                //printf("%4.1f\n", sum);
                //printf("%4.1f\n", *(&retx[0] + i * j2 + j));
            }
            //printf("\n");
        }
    return retx;
}

float* matiax_tool::hadamard_take(float* a1, int i1, int j1, float* a2, int i2, int j2, float* retx)
{
    if (i1 != i2 || j1 != j2)
        return NULL;
    for (int count = 0; count < i1 * j1; count++)
    {
        *(retx + count) = *(a1 + count) * (*(a2 + count));
    }
    return retx;
}

float* matiax_tool::C_take(float* m, int i, int j, float c, float* r)
{
    for (int count = 0; count < i * j; count++)
    {
        *(r + count) = *(m + count) * c;
    }
    return r;
}

// sigmoid()
float matiax_tool::Sigmoid(float i)
{
    float a;
    a = 1 / (1 + exp(-i));
    return a;
}

//初始化权值矩阵
float* matiax_tool::init_M(float* m, int i, int j)
{
    float temp;
    float rand_max = 1000 * pow(j, -0.5);
    int rand_max_int = 2 * (int)rand_max;
    srand(time(0));
    //printf("%f %d \n", rand_max, rand_max_int);
    for (int k = 0; k < i * j; k++)
    {
        temp = (float)((rand() % rand_max_int) - (rand_max_int / 2)) / 1000;
        //std::cout << temp << std::endl;
        *(m + k) = temp;
    }
    return m;
}
float* matiax_tool::matiax_T(float* m, int i, int j,float* r)
{
    for(int h=0;h<i;h++)
        for (int l = 0; l < j; l++)
        {
            // r[l][h] = m[h][l]
            *(r + l * i + h) = *(m + h * j + l);
        }
    return r;
}

//调正输入值区间为（0，1）
float* matiax_tool::reInput(float* m, int i,int in_up,int in_dw)
{
    in_dw = in_dw - in_up;
    for (int count = 0; count < i; count++)
    {
        *(m + count) = (*(m + count) - in_up) / in_dw;
        if (*(m + count) == 0)
            *(m + count) = 0.01;
        else if (*(m + count) == 1)
        {
            *(m + count) = 0.99;
        }
        //cout << *(m + count) << "  ";
    }
    //cout << endl;
    return m;
}

//矩阵的sigmoid运算;
float* matiax_tool::matiax_sig(float* m, int i, float* r)
{
    for (int count = 0; count < i; count++)
    {
        *(r + count) = Sigmoid(*(m + count));
    }
    return r;
}

void matiax_tool::print_matiax(float* m, int i, int j)
{
    for (int count = 0; count < i * j; count++)
    {
        cout << *(m + count) << "    ";
        if ((count + 1) % j == 0)
            cout << endl;
    }
}

float* matiax_tool::reduce(float* a1, float* a2, int i,int j, float* r)
{
    for (int count = 0; count < i*j; count++)
    {
        *(r + count) = *(a1 + count) - *(a2 + count);
    }
    return r;
}

float* matiax_tool::one_reduce(float* a, int i, int j, float* r)
{
    for (int count = 0; count < i * j; count++)
    {
        *(r + count) = 1 - (*(a + count));
    }
    return r;
}

float* matiax_tool::S_jk(float* m, int i1, int j1, float* o, int i2, int j2,float* r)
{
    float sum;
    for (int count_k = 0; count_k< i1; count_k++)
    {
        sum = 0;
        for (int count_j = 0; count_j < j1; count_j++)
        {
            sum += *(m + count_j * j1 + count_k) * (*(o + count_j));
        }
        *(r + count_k) = Sigmoid(sum);
    }
    return r;
}
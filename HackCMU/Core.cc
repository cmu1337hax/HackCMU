/*
 * Core.cc
 *
 *  Created on: Sep 21, 2018
 *      Author: Noah
 */
#include <iostream>
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>
#include <windows.h>
#include <cmath>
#include <assert.h>
#include <vector>
using namespace std;

struct Layer
{
	int size;
	vector<float> nodes;
	vector<float> coefficients;
};

struct Connection
{
	Layer in;
	Layer out;
	vector<vector<float>> edges;
};

Layer input;
Layer hidden [4];
Connection weights [5];
Layer output;
static const char allOut [] = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
							   'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','y','z',
							   '1','2','3','4','5','6','7','8','9','0','.',',','!','?'};
static const int possible = 66;
Layer makeLayer(int size)
{
	Layer output;
	output.size = size;
	output.nodes = vector<float>();
	output.coefficients = vector<float>();
	for(int i = 0; i < size; i++)
	{
		output.coefficients.push_back(1);
	}
	return output;
}

Connection connectLayers(Layer in, Layer out)
{
	Connection output;
	output.in = in;
	output.out = out;
	output.edges = vector<vector<float>>();
	for(int a = 0; a < in.size; a++)
	{
		vector<float> row(0,out.size);
		output.edges.push_back(row);
		for(int b = 0; b < out.size; b++)
		{
			output.edges[a].push_back(1);
		}
	}
	return output;
}

int setup()
{
	input = makeLayer(64);
	hidden[0] = makeLayer(36);
	weights[0] = connectLayers(input,hidden[0]);
	hidden[1] = makeLayer(36);
	weights[1] = connectLayers(hidden[0],hidden[1]);
	hidden[2] = makeLayer(36);
	weights[2] = connectLayers(hidden[1],hidden[2]);
	hidden[3] = makeLayer(36);
	weights[3] = connectLayers(hidden[2],hidden[3]);
	output = makeLayer(possible);
	weights[4] = connectLayers(hidden[3],output);
	return 0;
}

float actFunc(float x)
{
	return pow((1+exp(-x)),-1);
}

float weightedSum(Connection c, int outdex)
{
	int output = 0;
	for(int i = 0; i < c.in.size; i++)
	{
		output += c.in.nodes[i]*c.edges[i][outdex];
	}
	return output;
}

float adjust(Connection c, Layer expected)
{
	assert(c.out.size == expected.size);
	int output = 0;
	for(int i = 0; i < c.out.size; i++)
	{
		output += pow((expected.nodes[i]*expected.coefficients[i])-(c.out.nodes[i]*c.out.coefficients[i]),2);
	}
	return 0.5 * output;
}

void propagate(Connection arr [], int n)
{
	for(int c = 0; c < n; c++)
	{
		for(int i = 0; i < arr[c].out.size; i++)
		{
			arr[c].out.nodes[i] = arr[c].in.coefficients[i] * actFunc(weightedSum(arr[c],i));
		}
	}
}

void backpropagate(Connection arr[], Layer expected, int n)
{
	Layer target = expected;
	for(int c = n; c >= 0; c--)
	{
		for(int i = 0; i < arr[c].in.size; i++)
		{
			for(int o = 0; o < arr[c].out.size; o++)
			{
				arr[c].in.coefficients[i] *= adjust(arr[c],target);
			}
		}
		target = arr[c].in;
	}
}
Layer expectation(char c)
{
	Layer out = makeLayer(possible);
	for(int i = 0; i < possible; i++)
	{
		if(allOut[i] == c)
		{
			out.nodes[i] = 1;
		}
		else
		{
			out.nodes[i] = 0;
		}
	}
	return output;
}

int indexOfOutput(char c)
{
	for(int i = 0; i < possible; i++)
	{
		if(allOut[i] == c)
		{
			return i;
		}
	}
	return -1;
}

float train(vector<float> img, char c)
{
	cout << "Training start...\n";
	input.nodes = img;
	cout << "Input accepted...\n";
	propagate(weights, 5);
	cout << "Propagation done...\n";
	backpropagate(weights,expectation(c),5);
	cout << "Backpropagation done...\n";
	cout << "Closeness: ";
	cout << 1 - weights[4].out.nodes[indexOfOutput(c)];
	return 1 - weights[4].out.nodes[indexOfOutput(c)];
}

char guess(vector<float> img,float thresh)
{
	input.nodes = img;
	propagate(weights,5);
	int max = 0;
	for(int i = 1; i < output.size;i++)
	{
		if(output.nodes[i] >= thresh && output.nodes[i] > output.nodes[max])
		{
			max = i;
		}
	}
	return allOut[max];
}

void run(vector<float> img)
{
	ofstream blah;
	blah.open("output.txt");
	blah << guess(img, .80);
	blah.close();

}

//string serialize(Layer l)
//{
//	string output;
//	for(int i = 0; i < l.size;i++)
//	{
//		output.append(to_string(l.coefficients[i]));
//		output.append(" ");
//	}
//	output.append("=");
//	return output;
//}
//
//char* serialize(Connection c)
//{
//	char* output;
//	for(int i = 0; i < c.in.size;i++)
//	{
//		for(int o = 0; o < c.out.size;o++)
//		{
//			output += (to_string(c.edges[i][o])).c_str();
//			output += (" ");
//		}
//	output += ("=");
//	}
//	return output;
//}
//
//Layer toLayer(char* str, int n)
//{
//	Layer output = makeLayer(n);
//	char* split = strtok(str," ");
//	for(int i = 0; i < n; i++)
//	{
//		output.coefficients[i] = (float) atof(split[i]);
//	}
//
//}
//
//bool write()
//{
//	ofstream saveData;
//	saveData.open("saveData.bin", ios::out | ios::trunc | ios::binary);
//	if(saveData.is_open())
//	{
//		saveData.write(serialize(input),input.size*sizeof(float));
//		for(int l = 0; l < 4; l++)
//		{
//			saveData.write(serialize(hidden[l]),hidden[l].size*sizeof(float));
//		}
//		for(int c = 0; c < 5; c++)
//		{
//			saveData.write(serialize(weights[c]),36*36*sizeof(float));
//		}
//		saveData.close();
//		cout << "Data Saved!";
//		return true;
//	}
//	else
//	{
//		cout << "Data failed to open!";
//		return false;
//	}
//}
//
//bool read()
//{
//	ifstream saveData;
//	saveData.open("saveData.bin", ios::in | ios::binary);
//	if(saveData.is_open())
//	{
//		char* temp;
//		saveData.read(temp, input.size*sizeof(float));
//		input = toLayer(temp, input.size);
//		for(int l = 0; l < 4; l++)
//		{
//			saveData.read(temp,hidden[l].size*sizeof(float));
//			hidden[l] = toLayer(temp,hidden[l].size);
//		}
//		saveData.close();
//		cout << "Data read successfully";
//		return true;
//	}
//	else
//	{
//		cout << "Data read failed";
//		return false;
//	}
//}

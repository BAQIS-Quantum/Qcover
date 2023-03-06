#include<cstdio>
#include<cstdlib>
#include<string>
#include<fstream>
using namespace std;

int main(){
	string t = "test.txt";
	ifstream graph;
	graph.open(t.c_str());
	string g;
	while(graph >> g)
	{
		string s = "python downloadGraph.py " + g + " > " + g;  //download zip, every zip include a txt
		// string s = "unzip " + g  //unzip zips to txt to get graph instance
		// string s = "rm -r " + g  // delete zips
		system(s.c_str());
	}
	return 0;
}

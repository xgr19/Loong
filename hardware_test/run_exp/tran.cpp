#include<bits/stdc++.h>
using namespace std;
char ch[391];
int fc=0;
double num[197];
signed main () {
	ifstream inFile;
	freopen("ac.log","r",stdin);
//	freopen("result.json","w",stdout);
	printf("{\n");
	while (scanf("%s",ch)!=-1) {
		int len=strlen(ch);
		ch[len-5]=0;
		printf("%s\n",ch);
		string x=ch;
////		cout<<x<<endl; 
		string y="MNNConvert -f ONNX --modelFile ./ONNX/"+x+".onnx --MNNModel ./MNN/"+x+".mnn --bizCode biz >tran.log";
//		system(y.c_str());
////		sleep(1);
		string z="MNNV2Basic.out ./MNN/"+x+".mnn 10000 0 0 1 > ./time/"+x+".log";
		system(z.c_str());
//		string g="~/tmp/time/"+x+".log";
//		FILE * fp2 = fopen(g.c_str(), "r");
//		fc=5;
//		putchar('"');
//		printf("%s",x.c_str());
//		putchar('"');
//		putchar(':');
//		while (fscanf(fp2,"%lf",&num[++fc])!=EOF);
//		printf("%lf,%lf,%lf,%lf",num[fc-3],num[fc-2],num[fc-1],num[fc]);
//		while (fscanf(fp2,"%[^\n]\n",pd[++fc])!=EOF);
//		printf("\"%s\":%[^\n]\n",pd[fc-1]);
	}
	printf("}");
} 

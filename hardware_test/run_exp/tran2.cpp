#include<bits/stdc++.h>
using namespace std;
char ch[391],pd[9999][1191];
int fc=0,p[129],usd[12][5123],pot[12];
double ps[12][5123];
string pg[12][5123];
map<string,double> ma;
double num[197],nc;
void get_read(char *ch,int *p){
	int pc=0,pp=0,state=0,op=0;
	p[0]=0;
	while (ch[pc]){
		op=('0'<=ch[pc]&&ch[pc]<='9');
		if (op==0) {
			if (state==1) pp++,p[pp]=0,state=0;
			pc++;
		}else {
			state=1;
			p[pp]=p[pp]*10+ch[pc]-48;
			pc++;
		}
	}
}
signed main () {
	ifstream inFile;
	freopen("ac.log","r",stdin);
	freopen("result_.txt","w",stdout);
	while (scanf("%s",ch)!=-1) {
		int len=strlen(ch);
		ch[len-5]=0;
		cerr<<ch<<endl;
		p[0]=0;
//		printf("%s\n",ch);
		get_read(ch,p);
//		cerr<<p[1]<<" "<<p[2]<<endl; 
		string x=ch;
		string g="./time/"+x+".log";
		FILE * fp = fopen(g.c_str(), "r");
		if (fp==NULL) {
			cerr<<"sb"<<endl;
		}
//		printf("test:%s\n",g.c_str());
		fc=5;
//		putchar('"');
//		printf("%s",x.c_str());
//		putchar('"');
//		putchar(':');
		memset(pd,0,sizeof pd);
		while(!feof(fp)) {
      		fgets(pd[++fc],1024,fp);
      		//printf("%s",pd[fc]);
   		}
   		nc=0;
//		printf("%s",pd[fc-1]);
		sscanf(pd[fc-1]+5,"%lf",&nc);
//		sscanf(pd[12]+25,"%lf",&nc);
//		cerr<<nc<<endl;
//		ma[x]=1000.0/nc*p[2]*44;
		printf("%s :%.6lf\n",x.c_str(),nc);
		//		der[pp[0]]=max(der[pp[0]],1000.0/nc*pp[1]*44);
//		printf("%.6lf,",nc);
//		printf("%lf\n",nc);
//		while (fscanf(fp2,"%s",&pd[++fc])!=EOF);
//			cerr<<pd[fc]<<endl;
//		printf("%lf,%lf,%lf,%lf",num[fc-3],num[fc-2],num[fc-1],num[fc]);
//		break;
	}
//	printf("ÍÌÍÂÂÊmax:\n");
//	for (int i=0;i<8;i++) printf("%d batch_size:%d\n",i,pot[i]);
//	for (int i=0;i<8;i++)
//	 for (int j=0;j<5123;j++)
//	  if (usd[i][j]) {
//	  	printf("%s %.4lf\n",pg[i][j].c_str(),ps[i][j]);
//	  }
//	for (auto Z:ma) {
//		printf("%s %.4lf\n",Z.first.c_str(),Z.second);
//	}
//	printf("}");
} 

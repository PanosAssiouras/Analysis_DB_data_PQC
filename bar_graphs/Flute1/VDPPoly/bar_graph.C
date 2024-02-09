#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdio>


using namespace std;

TF1 *f1, *f2;
double finter(double *x, double*par) {
   return TMath::Abs(f1->EvalPar(x,par) - f2->EvalPar(x,par));
}


int sort_string_vector(vector<string> &stringVec)
{
 for (vector<string>::size_type i = 0; i != stringVec.size(); ++i)
    {
    // Sorting the string vector
    sort(stringVec.begin(), stringVec.end());
    // Ranged Based loops. This requires a C++11 Compiler also
    // If you don't have a C++11 Compiler you can use a standard
    // for loop to print your vector.

    cout << stringVec[i] << endl;

}
 return 0;
}
void Convert_TestDataFile_To_RootTree(TString TextDataName, TString RootDataName)
{
   TFile *f = new TFile(RootDataName,"RECREATE");
   TTree *T = new TTree("TRee","data from ascii file");
   TNtuple data("data","IV","Voltage:Cback");
   cout<<TextDataName<<endl;
   cout<<"root"<<endl;
	 std:: ifstream inputFile(TextDataName);
	  std::string line="";
    getline(inputFile,line);
    string VoltageString, CbackString ;
    double Voltage, Cback , Cint ;
	  while(getline(inputFile,line)){
         if ( line[0]=='C' || line[0]=='D' || line[0]=='V'|| !line.length())
           {  continue; }
     else{
       stringstream ss(line);
       getline(ss,VoltageString,',');
       getline(ss,CbackString,',');
       double Voltage=std::stod(VoltageString);
       double Cback=std::stod(CbackString);
       cout<<Voltage<<","<<Cback<<endl;
  //    sscanf(line.c_str(), "%lf %lf %lf", &Voltage, &Cback , &Cint);
        data.Fill(Voltage,Cback);

  }

   }
   data.Write();
   T->Write();
   f->Write();
   f->Close();
}

int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }
    while ((dirp = readdir(dp)) != NULL) {
		if((string(dirp->d_name)=="Cback_intersection.txt")||(string(dirp->d_name)=="TextToRoot.C"||(string(dirp->d_name)=="Cback@300V.txt"))
    ||(string(dirp->d_name)==".")||(string(dirp->d_name)==".."||string(dirp->d_name).find(".root")!=-1)){continue;}
        files.push_back(string(dirp->d_name));
      //  cout<<string(dirp->d_name)<<endl;
    }
    closedir(dp);
    return 0;
}


int Comparison()
{

    TH1F *h1= new TH1F("h1","Measured at Demokritos",12,0,12);
    h1->SetFillColor(kRed);

    TH1F *h2 = new TH1F("h2","Measured at Perugia",12,0,12);
    h2->SetFillColor(kBlue);

    TH1F *h3 = new TH1F("h3","Measured at Brown",12,0,12);
    h3->SetFillColor(kGreen);

    TCanvas *cs = new TCanvas("cs","cs",10,10,700,900);
     gStyle->SetOptStat(kFALSE);
     gStyle->SetTitleFontSize(0.025);
     gStyle->SetTitleAlign(23);



    string dir = string(".");  // Set the folder for search (".") for parent file
    vector<string> files = vector<string>(); // This vector will contain the names of each file in the folder after getdir function
    getdir(dir,files);

    sort_string_vector(files);

    cout<<files.size()<<endl;
        for (unsigned int k = 0; k<files.size();k++) {

            TGraph *gr1=new TGraph();
            TGraph *gr2=new TGraph();
            size_t pos1= files[k].find(".root");
            if(pos1==-1){
                size_t pos= files[k].find(".csv");
                cout << files[k] << endl;
                if(pos!=-1){
                  string r=files[k];
                  r=r.replace(r.begin()+pos,r.end(),".root");
                  char *root_name = new char[pos + 1];
                  std::strcpy(root_name, r.c_str());
                  Convert_TestDataFile_To_RootTree(files[k],r);
                  TFile* in_file=new TFile(root_name);
                  String names, center;
                  float average,std_dev;
                  float* row_content;
                  TNtuple* data=(TNtuple*) in_file->GetObjectChecked("data","TNtuple");
                  for(int irow=0; irow<data->GetEntries();++irow)
                  {   data->GetEntry(irow);
                      row_content=data->GetArgs();
                      names=row_content[0];
                      center=row_content[1]
                      average=row_content[2];
                      std_dev=row_content[3];
                      cout<<Reg<<","<<C_Exp<<","<<C_Java<<","<<C_TCAD<<endl;
                      if *center=="Demokritos"){
                        h1->SetBinContent(names,);
                        h1->GetXaxis()->SetBinLabel(Reg,Regions[Reg-1].c_str());
                        h2->SetBinContent(Reg,C_Java);
                        h2->GetXaxis()->SetBinLabel(Reg,Regions[Reg-1].c_str());
                        h3->SetBinContent(Reg,C_TCAD);
                        h3->GetXaxis()->SetBinLabel(Reg,Regions[Reg-1].c_str());
                      }


                    }
                }








using namespace std;



int Comparison()
{

   vector<string> Regions;
  Regions.push_back("1[0.133]");
  Regions.push_back("2[0.142]");
  Regions.push_back("3[0.125]");
  Regions.push_back("4[0.121]");
  Regions.push_back("5[0.233]");
  Regions.push_back("6[0.242]");
  Regions.push_back("7[0.225]");
  Regions.push_back("8[0.221]");
  Regions.push_back("9[0.333]");
  Regions.push_back("10[0.342]");
  Regions.push_back("11[0.325]");
  Regions.push_back("12[0.321]");



  TH1F *h1= new TH1F("h1","Experimental",12,0,12);
  h1->SetFillColor(kRed);

  TH1F *h2 = new TH1F("h2","Laplace solver",12,0,12);
  h2->SetFillColor(kBlue);

  TH1F *h3 = new TH1F("h3","TCAD simulations",12,0,12);
  h3->SetFillColor(kGreen);

  TCanvas *cs = new TCanvas("cs","cs",10,10,700,900);
   gStyle->SetOptStat(kFALSE);
   gStyle->SetTitleFontSize(0.025);
   gStyle->SetTitleAlign(23);
  //TText T; T.SetTextFont(42); T.SetTextAlign(21);
//  cs->Divide(2,2);


 std::ifstream inputFile1("Comparison_Cint.txt");

 string line;
 getline(inputFile1, line);
 cout<<line<<endl;
 double C_Exp,C_Java,C_TCAD;
 int Reg=0;

 while(getline(inputFile1, line)) {
    if (!line.length())
       continue;
    sscanf(line.c_str(), " %i %lf %lf %lf ",&Reg, &C_Exp ,&C_Java,&C_TCAD);
  //  Cbs_exp.push_back(Cb_per_cm*1e+12);
      cout<<Reg<<","<<C_Exp<<","<<C_Java<<","<<C_TCAD<<endl;
   h1->SetBinContent(Reg,C_Exp);
   h1->GetXaxis()->SetBinLabel(Reg,Regions[Reg-1].c_str());
   h2->SetBinContent(Reg,C_Java);
   h2->GetXaxis()->SetBinLabel(Reg,Regions[Reg-1].c_str());
   h3->SetBinContent(Reg,C_TCAD);
   h3->GetXaxis()->SetBinLabel(Reg,Regions[Reg-1].c_str());

 }

   THStack *hs = new THStack("hs","Comparison of the Backplane Capacitance for the MSSD_FZ120P" );
   hs->SetMinimum(0);
   hs->SetMaximum(2.0);

   TLegend *leg = new TLegend(0.75, 0.65, 0.999, 1.0);
   gStyle->SetLegendBorderSize(1);
   gStyle->SetLegendTextSize(0.025);


   cs->cd();
   hs->Add(h1);
   hs->Add(h3);
   hs->Add(h2);
   hs->Draw("nostackb");
   hs->GetXaxis()->SetTitle("Region name [w/p]");
   hs->GetYaxis()->SetTitle("Capacitance [pF/cm]");
  /* leg->SetTextColor(kBlue);;
   leg->SetTextAlign(12);
   leg->SetTextAlign(12);
   leg->AddEntry(h1, "Experimental", "lp");
   leg->AddEntry(h2, "Numerical Laplace solution", "lp");
   leg->AddEntry(h3, "TCAD smulations", "lp");*/
   gPad->BuildLegend(0.75,0.75,0.95,0.95,"");


    return 0;
}

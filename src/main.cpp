#include "MLP.h"
#include "mnist.h"

#include <iostream>
#include <vector>
#include <initializer_list>

//rand
#include <cstdlib>
#include <ctime> 

using namespace Eigen;
using namespace std;


template <typename T>
void print(const char* name, const vector<T> &v) {
  cout<<name<<":"<<endl;
  for(size_t i=0; i<v.size(); i++) {
    cout<<i<<":"<<endl;
    cout<<v[i]<<endl;
  }
}
template <typename T>
void print(const char* name, const T &t) {
  cout<<name<<":"<<endl;
  cout<<t<<endl;
}

void sample(vector<MLP::sample> &trainset, initializer_list<num> in_, initializer_list<num> out_) {
  ArrayXn in = Map<const ArrayXn>(in_.begin(), in_.size());
  ArrayXn out = Map<const ArrayXn>(out_.begin(), out_.size());
  MLP::sample goal(in, out);
  trainset.push_back(goal);
}

void tryall(MLP net) {
  for(int i=0; i<4; i++) {
    ArrayXn input(2);
    input << i/2, i%2;
    print("input",input);
    print("output",net.eval(input));
  }
}

int doxor() {
  ArrayXi layers(3);
  layers << 2, 6, 1;
  
  MLP net(layers, 1);
  
  vector<MLP::sample> tset;
  sample(tset, {0,0}, {0.1});
  sample(tset, {0,1}, {0.9});
  sample(tset, {1,0}, {0.9});
  sample(tset, {1,1}, {0.1});
  
  print("WEIGHTS", net.weights);
  
  tryall(net);
  
  for(int i=0; i<1000; i++) net.rprop(tset, 0.1, 1.2, 0.5);
  //for(int i=0; i<10000; i++) net.train(tset, 0.1);
  
  print("WEIGHTS", net.weights);
  
  tryall(net);
  return 0;
}
int main(int argc, char **argv) {
  srand((unsigned int) time(0));
  return mnist(argc, argv);
}
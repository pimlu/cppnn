
#include "MLP.h"

#include <iostream>
#include <fstream>
#include <vector>

#include <memory>
#include <stdexcept>
#include <cstdint>
#include <utility>

//sleep
#include <unistd.h>

using namespace std;
using namespace Eigen;

typedef pair<vector<uint8_t>, vector<MatrixXn>> MnistSet;

static_assert(sizeof(char) == 1, "char size isn't 1 byte");

//reads big endian integers
uint32_t idxint(istream &ifs) {
  uint32_t n=0;
  for(int i=0;i<4;i++) {
    n<<=8;
    n+=ifs.get();
  }
  return n;
}
//reads out label bytes
vector<uint8_t> readidx1(istream &ifs, const vector<uint32_t> &dims, string fname) {
  if(dims.size() != 1) throw runtime_error(fname+" has the wrong dimensions");
  vector<uint8_t> ret;
  ret.reserve(dims[0]);
  for(uint32_t i=0; i<dims[0]; i++) {
    ret.push_back(ifs.get());
  }
  return ret;
}

typedef Matrix<uint8_t, Dynamic, Dynamic, RowMajor> RowMatrixi8;
//reads out 2d matrices from idx.  normalizes to [0,1]
vector<MatrixXn> readidx3(istream &ifs, const vector<uint32_t> &dims, string fname) {
  if(dims.size() != 3) throw runtime_error(fname+" has the wrong dimensions");
  vector<MatrixXn> ret;
  ret.reserve(dims[0]);
  size_t size = dims[1]*dims[2];
  char *buffer = new char[dims[0]*size];
  ifs.read(buffer, dims[0]*size);
  for(uint32_t i=0; i<dims[0]; i++) {
    Map<RowMatrixi8> map(reinterpret_cast<uint8_t*>(buffer+i*size), dims[1], dims[2]);
    ret.push_back(map.cast<num>()/(num)(1<<8));
  }
  return ret;
}
//reads/validates header.  returns the dimensions it got from the header
vector<uint32_t> idxheader(istream &ifs, string fname) {
  uint32_t magic = 0x000008ff;
  uint32_t mask = 0xff;
  uint32_t fmagic = idxint(ifs);
  if((magic ^ fmagic) & ~mask) throw runtime_error(fname+" is not a valid file");
  int dims = fmagic & mask;
  vector<uint32_t> ret;
  for(int i=0; i<dims; i++) {
    ret.push_back(idxint(ifs));
  }
  return ret;
}
//prints out MNIST pictures (used for testing)
void printmat(MatrixXn mat, num thresh) {
  int r = mat.rows(), c = mat.cols();
  for(int i=0; i<r; i++) {
    for(int j=0; j<c; j++) {
      cout<< (mat(i,j)>thresh?"##":"  ");
    }
    cout<<endl;
  }
}
//unused
template<typename I, typename O>
vector<O> vmap(const vector<I> &v,const function<O(I)> &f) {
  vector<O> ret;
  ret.reserve(v.size());
  for(size_t i=0; i<v.size(); i++) ret.push_back(f(v[i]));
  return ret;
}
//converts from MNIST raws to samples compatible with MLP
vector<MLP::sample> convert(MnistSet &data) {
  ArrayXn labels[10];
  for(int i=0; i<10; i++) {
    ArrayXn label(10);
    label.fill(0.1);
    label(i) = 0.9;
    labels[i] = label;
  }
  vector<MLP::sample> ret;
  size_t n = data.first.size();
  ret.reserve(n);
  for(size_t i=0; i<n; i++) {
    MatrixXn &in = data.second[i];
    ret.push_back(MLP::sample(Map<VectorXn>(in.data(), in.size()), labels[data.first[i]]));
  }
  data.first.empty();
  data.second.empty();
  return ret;
}

//percent that it gets right accoding to best guess
num score(MLP &net, vector<MLP::sample> &testset) {
  size_t correct = 0;
  for(size_t i=0; i<testset.size(); i++) {
    int guess, actual;
    net.eval(testset[i].first).maxCoeff(&guess);
    testset[i].second.maxCoeff(&actual);
    //cout<<testset[i].second.size()<<"guess " <<guess<<"; actual "<<actual<<endl;
    if(guess==actual) correct++;
  }
  return correct/(num)testset.size();
}

void test(MLP &net, vector<MLP::sample> &testset, const char* name) {
  cout<<"testing..."<<endl;
  num accuracy = score(net, testset);
  cout<<name<<" accuracy is "<<accuracy<<endl;
}

int mnist(int argc, char **argv) {
  if(argc < 2) {
    cout<<"need data dir as an argument"<<endl;
    return 1;
  }
  enum IFILE { TRNLBL, TRNDAT, TSTLBL, TSTDAT };
  string files[] = {"train-labels.idx1-ubyte", "train-images.idx3-ubyte",
    "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte"};
  istream *ifs[4];
  vector<uint32_t> dims[4];
  cout<<"reading headers..."<<endl;
  for(size_t i=0; i<4; i++) {
    files[i] = argv[1]+("/"+files[i]);
    ifs[i] = new ifstream(files[i], ios::binary);
    dims[i] = idxheader(*ifs[i], files[i]);
  }
  cout<<"converting MNIST data..."<<endl;
  #define READF(d, n) readidx##d(*ifs[n], dims[n], files[n])
  #define SET(ind) MnistSet(READF(1, ind), READF(3, ind+1))
  MnistSet traindat = SET(0);
  MnistSet testdat = SET(2);
  cout<<"preprocessing... TODO"<<endl;
  cout<<"converting matrices to net samples..."<<endl;
  /*for(size_t i=0; i<traindat.first.size(); i++) {
    printmat(traindat.second[i], 0.5);
    cout<<"this is a "<<(int)traindat.first[i]<<endl;
    usleep(1000000);
  }*/
  vector<MLP::sample> trainset = convert(traindat);
  vector<MLP::sample> testset = convert(testdat);
  ArrayXi layers(3);
  layers << 28*28, 6*6, 10;
  cout<<"initializing net..."<<endl;
  MLP net(layers, 0.5);
  const int ITERS = 50;
  const size_t BATCH = 700;
  cout<<"training for "<<ITERS<<" iterations... minibatch size "<<BATCH<<endl;
  for(int i=0; i<ITERS; i++) {
    if(i%10==0) test(net, testset, "test");
    cout<<"iteration "<<i<<endl;
    net.rprop(trainset, 0.1, 1.2, 0.5, BATCH);
  }
  test(net, trainset, "train");
  test(net, testset, "test");
  
  for(size_t i=0; i<4; i++) delete ifs[i];
  return 0;
}
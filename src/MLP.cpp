#include "MLP.h"

using std::size_t;
using std::vector;

using namespace Eigen;

MLP::MLP(ArrayXi layers, num rval) : layers(layers) {
  for(size_t i=1; i<layers.size(); i++) {
    weights.push_back(MatrixXn::Random(layers[i], layers[i-1]+1) * rval);
  }
}

ArrayXn MLP::eval(ArrayXn input) {
  outputs.clear();
  inputs.clear();
  input.conservativeResize(layers[0]+1);
  input(layers[0]) = 1;
  inputs.push_back(input);
  for(size_t i=0; i<weights.size(); i++) {
    //sum net inputs
    ArrayXn nets = weights[i]*input.matrix();
    //squash for activation
    ArrayXn activations = 1/(1+(-nets).exp());
    outputs.push_back(activations);
    //add bias
    activations.conservativeResize(layers[i+1]+1);
    activations(layers[i+1]) = 1;
    inputs.push_back(activations);
    input = activations; //reuse for next layer
  }
  ArrayXn ret = outputs.back();
  //ret.conservativeResize(ret.size()-1);
  return ret;
}

//zeroes the weight partial differentials
void MLP::resetpdiffs() {
  if(pdiffs.size() == 0) {
    for(auto const &weight: weights) {
      pdiffs.push_back(MatrixXn(weight.rows(),weight.cols()));
    }
  }
  for(auto &pdiff: pdiffs) pdiff.setZero();
}

void MLP::applypdiffs(num eta) {
  for(size_t i=0; i<weights.size(); i++) {
    weights[i] -= eta*pdiffs[i];
  }
}

MatrixXn mtable(VectorXn rows, RowVectorXn cols) {
  return rows.replicate(1,cols.size()).cwiseProduct(
    cols.replicate(rows.size(),1)
  );
}

//performs backprop once on a single sample (adds to pdiffs)
void MLP::bprop(const sample &goal) {
  const ArrayXn &input = goal.first, &target = goal.second;
  ArrayXn out = eval(input);
  
  //error for each neuron is 1/2*(target-out)^2
  //dE/dOut = -(target-out) = out-target
  //dOut/dNet = out(1-out)
  
  //we need deltas, which are the parts of the partial derivates we reuse for the chain rule
  //the delta for a neuron n is the partial derivative of the total error with respect to the net input of the neuron
  
  //calculate the topmost set of deltas
  VectorXn deltas = (out-target)*out*(1-out);
  //cout << outputs.size() << " " << weights.size() << " " << pdiffs.size() << endl;
  for(size_t i=weights.size(); i--;) {
    //𝛿 x Ii
    //dE/dW for each weight is the delta of the neuron it goes to times the input of the weight
    MatrixXn table = mtable(deltas, inputs[i].transpose());
    
    //Ni+1 x Ii
    //or Oi+1 x Ii
    pdiffs[i] += table; //+0.000001*weights[i].array().cube().matrix();
    
    //no need to calculate deltas for the input layer
    if(i>0) {
      //use our current deltas to find the deltas further back using the chain rule
      VectorXn newdeltas(layers[i]);//ND
      //ND[i] = dE/dNet[i] = dE/dOut[i]*dOut/dNet[i]
      //this is a vector with one element for each neuron
      //each element in dE/dOut[i] is the sum of pair product of its weight and
      //  the associated delta in the next layer for all the next layer's neurons
      //1 x 𝛿  *  𝛿 x Ii = 1 x Ii
      ArrayXn dEdOut = deltas.transpose()*weights[i];
      dEdOut.conservativeResize(layers[i]); //no delta necessary for the bias
      ArrayXn output = outputs[i-1];
      ArrayXn dOutdNet = output*(1-output);
      
      newdeltas = dEdOut*dOutdNet;
      
      deltas = newdeltas;
    }
  }
}
void MLP::train(const std::vector<sample> &trainset, num eta) {
  resetpdiffs();
  for(const sample &s: trainset) bprop(s);
  applypdiffs(eta/trainset.size());
}
void MLP::rprop(const std::vector<sample> &trainset, num etai, num etap, num etan) {
  rprop(trainset, etai, etap, etan, -1);
}
void MLP::rprop(const std::vector<sample> &trainset, num etai, num etap, num etan, size_t batch) {
  resetpdiffs();
  if(batch == -1) for(const sample &s: trainset) bprop(s);
  else {
    for(size_t i=0; i<batch; i++) {
      bprop(trainset[std::rand()%trainset.size()]);
    }
  }
  if(pdextra.size() == 0) {
    for(auto const &weight: weights) {
      pdextra.push_back(MatrixXn(weight.rows(),weight.cols()));
      pdextra.back().fill(etai);//TODO what if it becomes 0 later?
    }
  }
  for(size_t i=0; i<pdextra.size(); i++) {
    MatrixXn etaps = pdextra[i]*etap;
    MatrixXn etans = pdextra[i]*(-etan);
    //use select to perform conditional based on sign change
    pdextra[i] = (pdextra[i].cwiseProduct(pdiffs[i]).array()>0).select(etaps, etans);
    weights[i] -= pdextra[i];
  }
}
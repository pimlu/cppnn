#pragma once

#include <Eigen/Dense>

#include <vector>

using num = double;
typedef Eigen::Matrix<num, Eigen::Dynamic, Eigen::Dynamic> MatrixXn;
typedef Eigen::Matrix<num, Eigen::Dynamic, 1> VectorXn;
typedef Eigen::Matrix<num, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorXn;
typedef Eigen::Array<num, Eigen::Dynamic, Eigen::Dynamic> ArrayXXn;
typedef Eigen::Array<num, Eigen::Dynamic, 1> ArrayXn;

class MLP {
  
  std::vector<MatrixXn> pdiffs;
  std::vector<MatrixXn> pdextra;
public:
  typedef std::pair<ArrayXn, ArrayXn> sample;
  Eigen::ArrayXi layers;
  std::vector<MatrixXn> weights;//weights to layer i
  std::vector<VectorXn> outputs;//output of layer i
  std::vector<VectorXn> inputs;//input to layer i
  
  MLP(Eigen::ArrayXi layers, num rval);
  ArrayXn eval(ArrayXn input);
  void resetpdiffs();
  void applypdiffs(num eta);
  void bprop(const sample &target);
  void train(const std::vector<sample> trainset, num eta);
  void rprop(const std::vector<sample> trainset, num etai, num etap, num etan);
};
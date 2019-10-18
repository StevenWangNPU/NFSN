function [z1,a1,a2] = forward_NN_new(X,w_h,b_h,w_o,b_o)
% Forward Propagation
% m hidden num
% X:d*n matrix
% hidden_weights: d*hidden_num
% hidden_bias vector: 1*m
% output_weights: c*1
[n,d] = size(X);
z1 = X * w_h  + b_h;
% activation 
a1 = relu_f(z1,0);
% output = hidden_z1 * output_weights + output_bias;
a2 = a1 * w_o  + b_o;
% a2 = 1./(1+exp(-z2));

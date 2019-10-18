function RMSE = RNNFS_newp(X,y,test_x,test_y,m,iter_num,lr,gamma,p,selected_num)
% hidden_num = reduced num
% hidden_weights: d*hidden_num
% hidden_bias vector: 1*m
% output_weights: m*1
[n,d] = size(X);
% initialization:
init_hidden_weights = 2 * rand(d,m)-1;
init_hidden_bias = 0.1*ones(1,m);
init_output_weights = 2 * rand(m,1)-1;
init_output_bias = 0.1 * ones(1);
for i = 1:iter_num
[hidden_z1,hidden_a1,output] = forward_NN_new(X,init_hidden_weights,init_hidden_bias,init_output_weights,init_output_bias);
[hidden_weights,hidden_bias,output_weights,output_bias] = BP_NN_newp(X,y,output,hidden_z1,hidden_a1,init_hidden_weights,init_output_weights,init_hidden_bias,init_output_bias,lr,gamma,p);
init_hidden_weights = hidden_weights;
init_hidden_bias = hidden_bias;
init_output_weights = output_weights;
init_output_bias = output_bias;
%     if mod(i,10)==0
%         fprintf(['Training Steps: ', num2str(i), '/', num2str(iter_num) '\n']);
%     end
end
% feature selection and prediction value
WW = init_hidden_weights.^ 2;
W_weight = sum(WW, 2);                             % sum the element row-by-row
[~, index_sorted_features] = sort(-W_weight);      % sort them from the largest to the smallest    
init_hidden_weights(index_sorted_features(selected_num+1:end),:) = 0;
[~,~,prediction] = forward_NN_new(test_x,init_hidden_weights,init_hidden_bias,init_output_weights,init_output_bias);

% calculate test MSE error
RMSE = sqrt(sum((prediction - test_y).^2)/length(prediction));
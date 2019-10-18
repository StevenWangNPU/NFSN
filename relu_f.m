function y = relu_f(x, flag)
[n,d] = size(x);
y = zeros(n,d);
if flag == 0
    y(x>0) = x(x>0);
else
    y(x>0) = 1;
end
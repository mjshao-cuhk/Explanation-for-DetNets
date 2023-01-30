function x = FW_DetNet(x_In,HTH,Hy, L, ss , alpha , gamma1 ,gamma3  ) 

 



x= x_In;
y_x = x_In;
for i=1:L
    grad_f = 2*(HTH*y_x - Hy);
    x_buff = y_x - ss(i)* grad_f;
    temp0 = gamma1(i)*x_buff;
    temp1 = gamma3(i)*(x_buff-2);
    temp2 = gamma3(i)*(x_buff+2);
    ply_x = 2*sigmoid(temp0) +2*sigmoid(temp1) +2*sigmoid(temp2)-3;
    y_x = x + alpha(i) *(ply_x - x);
    x = y_x;
end

end


function z = sigmoid(x)
    z = 1./(1+exp(-x));
end

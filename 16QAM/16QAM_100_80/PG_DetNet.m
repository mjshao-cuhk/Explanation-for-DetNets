function x = PG_DetNet(x_ini, HTH, HTy, iter, grad_ss, extra_ss, gamma1, gamma3)



x = x_ini;
y_x = x_ini;

for ii = 1:iter

    grad_f = 2*(HTH*y_x-HTy);

    x_buff = y_x - grad_ss(ii)*grad_f;

    temp1 = tanh(0.5*gamma3(ii)*(x_buff+2));
    temp2 = tanh(0.5*gamma3(ii)*(x_buff-2));
    temp3 = tanh(0.5*gamma1(ii)*x_buff);

    ply_x = temp1 + temp2 + temp3;

    y_x = ply_x + extra_ss(ii)*(ply_x-x);
    x = ply_x;

end
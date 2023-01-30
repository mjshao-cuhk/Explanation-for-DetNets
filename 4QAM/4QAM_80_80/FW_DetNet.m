function x = FW_DetNet(x_ini, HTH, HTy, iter, grad_ss, wk, gamma)



x = x_ini;
y_x = x_ini;

for ii = 1:iter

    grad_f = 2*(HTH*y_x-HTy);

    x_buff = y_x - grad_ss(ii)*grad_f;
    temp = 2*gamma(ii)*x_buff;
    ply_x = 2*(1./(1+exp(-temp)))-1;

    y_x = x + wk(ii)*(ply_x-x);
    x = y_x;

end
function x = PG_DetNet(x_ini, HTH, HTy, iter, grad_ss, extra_ss, gamma)



x = x_ini;
y_x = x_ini;

for ii = 1:iter

    grad_f = 2*(HTH*y_x-HTy);

    x_buff = y_x - grad_ss(ii)*grad_f;
    temp = 2*gamma(ii)*x_buff;
    ply_x = 2*(1./(1+exp(-temp)))-1;

    y_x = ply_x + extra_ss(ii)*(ply_x-x);
    x = ply_x;

end
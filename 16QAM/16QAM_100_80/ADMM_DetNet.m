function x = ADMM_DetNet(x_ini, HTH, HTy, Layers, ILbar, tau, gamma1, gamma3)

[~,n] = size(HTH);
z = x_ini;
u = zeros(n, 1);

for ii = 1:Layers
    temp1 = HTH + tau(ii)*eye(n);
    temp2 = HTy + u + tau(ii)*z;
    x = temp1\temp2;

    temp = x - u*ILbar(ii);
    %z = tanh(temp/gamma(ii)*0.5);
    z = tanh(temp/gamma1(ii)*0.5) + tanh((temp+2)/gamma3(ii)*0.5) + tanh((temp-2)/gamma3(ii)*0.5);

    u = u + tau(ii)*(z-x);
end


